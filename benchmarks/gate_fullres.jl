using MPI; MPI.Init()
rank = MPI.Comm_rank(MPI.COMM_WORLD)
nranks = MPI.Comm_size(MPI.COMM_WORLD)
using CUDA; CUDA.device!(rank)
using NCCL, Oceananigans, Breeze, AtmosphericProfilesLibrary
using Oceananigans.Units
using Oceananigans.Grids: znode
using Oceananigans.TimeSteppers: time_step!
Oceananigans.defaults.FloatType = Float32
NCCLExt = Base.get_extension(Oceananigans, :OceananigansNCCLExt)
arch = NCCLExt.NCCLDistributed(GPU(); partition=Partition(nranks, 1))

function gate_vgrid(zt; dz0=50, dzp=100, dzt=300)
    z1,z2,z3 = 1275,5100,18000; zf=[0.0]; z=0.0
    while z<zt
        a=clamp((z-z1)/(z2-z1),0,1); b=clamp((z-z2)/(z3-z2),0,1)
        dz=dz0+a*(dzp-dz0)+b*(dzt-dzp); z=min(z+dz,zt); push!(zf,z)
    end; zf
end
zf = gate_vgrid(27000); Nz=length(zf)-1

grid = RectilinearGrid(arch, size=(2048,2048,Nz), x=(0,204800), y=(0,204800), z=zf,
    halo=(5,5,5), topology=(Periodic,Periodic,Bounded))
constants = ThermodynamicConstants()
ref = ReferenceState(grid, constants, surface_pressure=101200, potential_temperature=298)
dynamics = AnelasticDynamics(ref)
microphysics = SaturationAdjustment(equilibrium=MixedPhaseEquilibrium())

FT = eltype(grid)
ρᵣ = ref.density; cᵖᵈ = constants.dry_air.heat_capacity
T₀ = AtmosphericProfilesLibrary.GATE_III_T(FT)
qᵗ₀ = AtmosphericProfilesLibrary.GATE_III_q_tot(FT)
u₀ = AtmosphericProfilesLibrary.GATE_III_u(FT)
∂t_T = AtmosphericProfilesLibrary.GATE_III_dTdt(FT)
∂t_qᵗ = AtmosphericProfilesLibrary.GATE_III_dqtdt(FT)

∂t_ρθ = Field{Nothing,Nothing,Center}(grid)
∂t_ρqᵉ = Field{Nothing,Nothing,Center}(grid)
set!(∂t_ρθ, z -> ∂t_T(z)); set!(∂t_ρθ, ρᵣ * cᵖᵈ * ∂t_ρθ)
set!(∂t_ρqᵉ, z -> ∂t_qᵗ(z)); set!(∂t_ρqᵉ, ρᵣ * ∂t_ρqᵉ)

zˢ, zᵗ = 19000.0, 27000.0
@inline function sponge_fn(i, j, k, grid, clock, fields, p)
    z = znode(i, j, k, grid, Center(), Center(), Face())
    mask = clamp((z - p.zˢ) / (p.zᶜ - p.zˢ), 0, 1)
    @inbounds ρw = fields.ρw[i, j, k]
    return -p.λ * mask * ρw
end
sponge = Forcing(sponge_fn, discrete_form=true, parameters=(; λ=0.1, zˢ, zᶜ=(zˢ+zᵗ)/2))

forcing = (ρw=sponge, ρθ=Forcing(∂t_ρθ), ρqᵉ=Forcing(∂t_ρqᵉ))
Ts = 299.88
bcs = (ρθ=FieldBoundaryConditions(bottom=BulkSensibleHeatFlux(coefficient=1.1e-3, surface_temperature=Ts)),
       ρqᵉ=FieldBoundaryConditions(bottom=BulkVaporFlux(coefficient=1.2e-3, surface_temperature=Ts)),
       ρu=FieldBoundaryConditions(bottom=Breeze.BulkDrag(coefficient=1.2e-3)),
       ρv=FieldBoundaryConditions(bottom=Breeze.BulkDrag(coefficient=1.2e-3)))

model = AtmosphereModel(grid; dynamics, microphysics, advection=WENO(order=5),
    coriolis=FPlane(latitude=8.5), forcing, boundary_conditions=bcs)

ϵ() = rand() - 0.5
set!(model, T=(x,y,z)->T₀(z)+0.5*ϵ()*(z<2000), qᵗ=(x,y,z)->qᵗ₀(z)+1e-4*ϵ()*(z<2000), u=(x,y,z)->u₀(z))

for _ in 1:3; time_step!(model, 0.5); end
CUDA.synchronize(); MPI.Barrier(MPI.COMM_WORLD)

times = Float64[]
for trial in 1:3
    N = 3; t = @elapsed begin
        for _ in 1:N; time_step!(model, 0.5); end
        CUDA.synchronize(); MPI.Barrier(MPI.COMM_WORLD)
    end
    push!(times, 1000t/N)
end

if rank == 0
    best = round(minimum(times), digits=1)
    all_t = join(round.(times, digits=1), ", ")
    nx_per = 2048 ÷ nranks
    result = "GATE_fullres_$(nranks)x1: $best ms/step  (trials: $all_t)  [2048x2048x$(Nz), $(nx_per)x2048x$(Nz)/GPU]"
    println(result)
    open("/tmp/gate_fullres_$(nranks).txt", "w") do f; println(f, result); end
end
MPI.Finalize()
