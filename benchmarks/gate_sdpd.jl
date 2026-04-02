using MPI; MPI.Init()
rank = MPI.Comm_rank(MPI.COMM_WORLD)
using CUDA; CUDA.device!(rank)
using NCCL, Oceananigans, Breeze, AtmosphericProfilesLibrary
using Oceananigans.Units
using Oceananigans.Grids: znode
using Oceananigans.TimeSteppers: time_step!
Oceananigans.defaults.FloatType = Float32
NCCLExt = Base.get_extension(Oceananigans, :OceananigansNCCLExt)
arch = NCCLExt.NCCLDistributed(GPU(); partition=Partition(8, 1))

function gate_vgrid(zt; dz0=50, dzp=100, dzt=300)
    z1, z2, z3 = 1275, 5100, 18000
    zf = [0.0]; z = 0.0
    while z < zt
        a = clamp((z - z1) / (z2 - z1), 0, 1)
        b = clamp((z - z2) / (z3 - z2), 0, 1)
        dz = dz0 + a * (dzp - dz0) + b * (dzt - dzp)
        z = min(z + dz, zt)
        push!(zf, z)
    end
    return zf
end

zf = gate_vgrid(27000)
Nz = length(zf) - 1

grid = RectilinearGrid(arch, size=(2048, 2048, Nz), x=(0, 204800), y=(0, 204800), z=zf,
                       halo=(5, 5, 5), topology=(Periodic, Periodic, Bounded))

constants = ThermodynamicConstants()
ref = ReferenceState(grid, constants, surface_pressure=101200, potential_temperature=298)
dynamics = AnelasticDynamics(ref)
microphysics = SaturationAdjustment(equilibrium=MixedPhaseEquilibrium())

FT = eltype(grid)
T_prof = AtmosphericProfilesLibrary.GATE_III_T(FT)
qt_prof = AtmosphericProfilesLibrary.GATE_III_q_tot(FT)
u_prof = AtmosphericProfilesLibrary.GATE_III_u(FT)
dTdt_prof = AtmosphericProfilesLibrary.GATE_III_dTdt(FT)
dqdt_prof = AtmosphericProfilesLibrary.GATE_III_dqtdt(FT)

rho_ref = ref.density
cpd = constants.dry_air.heat_capacity

tend_theta = Field{Nothing, Nothing, Center}(grid)
tend_qe = Field{Nothing, Nothing, Center}(grid)
set!(tend_theta, z -> dTdt_prof(z))
set!(tend_theta, rho_ref * cpd * tend_theta)
set!(tend_qe, z -> dqdt_prof(z))
set!(tend_qe, rho_ref * tend_qe)

@inline function sponge_fn(i, j, k, grid, clock, fields, p)
    z = znode(i, j, k, grid, Center(), Center(), Face())
    mask = clamp((z - p.zs) / (p.zc - p.zs), 0, 1)
    @inbounds rw = fields.ρw[i, j, k]
    return -p.lam * mask * rw
end

sponge = Forcing(sponge_fn, discrete_form=true, parameters=(; lam=0.1, zs=19000.0, zc=23000.0))

forcing = (ρw=sponge, ρθ=Forcing(tend_theta), ρqᵉ=Forcing(tend_qe))

Ts = 299.88
bcs = (ρθ  = FieldBoundaryConditions(bottom=BulkSensibleHeatFlux(coefficient=1.1e-3, surface_temperature=Ts)),
       ρqᵉ = FieldBoundaryConditions(bottom=BulkVaporFlux(coefficient=1.2e-3, surface_temperature=Ts)),
       ρu  = FieldBoundaryConditions(bottom=Breeze.BulkDrag(coefficient=1.2e-3)),
       ρv  = FieldBoundaryConditions(bottom=Breeze.BulkDrag(coefficient=1.2e-3)))

model = AtmosphereModel(grid; dynamics, microphysics, advection=WENO(order=5),
                        coriolis=FPlane(latitude=8.5), forcing, boundary_conditions=bcs)

eps() = rand() - 0.5
set!(model, T=(x,y,z) -> T_prof(z) + 0.5 * eps() * (z < 2000),
            qᵗ=(x,y,z) -> qt_prof(z) + 1e-4 * eps() * (z < 2000),
            u=(x,y,z) -> u_prof(z))

# Warmup
for _ in 1:3; time_step!(model, 0.5); end
CUDA.synchronize(); MPI.Barrier(MPI.COMM_WORLD)

# Measure at various fixed dt
for dt in [0.5, 1.0, 2.0, 3.0, 4.0]
    N = 5
    t = @elapsed begin
        for _ in 1:N; time_step!(model, dt); end
        CUDA.synchronize()
        MPI.Barrier(MPI.COMM_WORLD)
    end
    ms = round(1000t / N, digits=1)
    sdpd = round(dt / (t / N), digits=1)
    if rank == 0
        open("/tmp/gate_sdpd.txt", "a") do f
            println(f, "dt=$(dt)s: $(ms) ms/step, SDPD=$(sdpd)")
        end
    end
end

MPI.Finalize()
