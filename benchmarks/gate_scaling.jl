# GATE III deep tropical convection — weak scaling benchmark
#
# Usage:
#   1 GPU:   julia --project gate_scaling.jl [microphysics]
#   N GPUs:  mpiexec -n N julia --project gate_scaling.jl [microphysics] [px,py]
#
# microphysics: "satadj" (default), "1M" (NonEquilibrium 1-moment mixed-phase)
# px,py: partition (default: N,1)
#
# Each GPU gets 400×400 horizontal grid points with stretched vertical grid (~150 levels).

using MPI
MPI.Init()
rank = MPI.Comm_rank(MPI.COMM_WORLD)
nranks = MPI.Comm_size(MPI.COMM_WORLD)

using CUDA
CUDA.device!(rank % length(CUDA.devices()))

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: znode
using Oceananigans.TimeSteppers: time_step!
using Breeze
using AtmosphericProfilesLibrary
Oceananigans.defaults.FloatType = Float32

# ── Parse arguments ──────────────────────────────────────────
micro_type = length(ARGS) >= 1 ? ARGS[1] : "satadj"

if length(ARGS) >= 2
    px, py = parse.(Int, split(ARGS[2], ","))
else
    px, py = nranks, 1
end

# ── Architecture ──────────────────────────────────────────────
if nranks > 1
    using NCCL
    NCCLExt = Base.get_extension(Oceananigans, :OceananigansNCCLExt)
    arch = NCCLExt.NCCLDistributed(GPU(); partition=Partition(px, py))
else
    arch = GPU()
end

# ── Stretched vertical grid (GATE standard) ───────────────────
function gate_vertical_grid(zᵗ; Δz⁰=50, Δzᵖ=100, Δzᵗ=300)
    z₁, z₂, z₃ = 1275, 5100, 18000
    z_faces = [0.0]
    z = 0.0
    while z < zᵗ
        α = clamp((z - z₁) / (z₂ - z₁), 0, 1)
        β = clamp((z - z₂) / (z₃ - z₂), 0, 1)
        Δz = Δz⁰ + α * (Δzᵖ - Δz⁰) + β * (Δzᵗ - Δzᵖ)
        z = min(z + Δz, zᵗ)
        push!(z_faces, z)
    end
    return z_faces
end

zᵗ = 27000
zˢ = 19000
z_faces = gate_vertical_grid(zᵗ)
Nz = length(z_faces) - 1

# ── Grid: 400×400 per GPU ─────────────────────────────────────
Nx_per_gpu = 400
Ny_per_gpu = 400
Nx_total = Nx_per_gpu * px
Ny_total = Ny_per_gpu * py
dx = 100.0  # m
Lx = Nx_total * dx
Ly = Ny_total * dx

grid = RectilinearGrid(arch;
    size = (Nx_total, Ny_total, Nz),
    x = (0, Lx), y = (0, Ly), z = z_faces,
    halo = (5, 5, 5),
    topology = (Periodic, Periodic, Bounded))

# ── Reference state ───────────────────────────────────────────
constants = ThermodynamicConstants()
reference_state = ReferenceState(grid, constants,
    surface_pressure = 101200,
    potential_temperature = 298)
dynamics = AnelasticDynamics(reference_state)

# ── GATE profiles ─────────────────────────────────────────────
FT = eltype(grid)
T₀ = AtmosphericProfilesLibrary.GATE_III_T(FT)
qᵗ₀ = AtmosphericProfilesLibrary.GATE_III_q_tot(FT)
u₀ = AtmosphericProfilesLibrary.GATE_III_u(FT)

# ── Microphysics ──────────────────────────────────────────────
if micro_type == "1M"
    using CloudMicrophysics
    BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
    microphysics = BreezeCloudMicrophysicsExt.OneMomentCloudMicrophysics(FT;
        cloud_formation = NonEquilibriumCloudFormation(nothing, nothing))
    label = "GATE_1M"
else
    microphysics = SaturationAdjustment(equilibrium=MixedPhaseEquilibrium())
    label = "GATE_satadj"
end

# ── Forcings ──────────────────────────────────────────────────
ρᵣ = reference_state.density
cᵖᵈ = constants.dry_air.heat_capacity
∂t_T = AtmosphericProfilesLibrary.GATE_III_dTdt(FT)
∂t_qᵗ = AtmosphericProfilesLibrary.GATE_III_dqtdt(FT)

∂t_ρe_ls = Field{Nothing, Nothing, Center}(grid)
∂t_ρqᵗ_ls = Field{Nothing, Nothing, Center}(grid)
set!(∂t_ρe_ls, z -> ∂t_T(z))
set!(∂t_ρe_ls, ρᵣ * cᵖᵈ * ∂t_ρe_ls)
set!(∂t_ρqᵗ_ls, z -> ∂t_qᵗ(z))
set!(∂t_ρqᵗ_ls, ρᵣ * ∂t_ρqᵗ_ls)

# Sponge layer
@inline function sponge_damping(i, j, k, grid, clock, fields, p)
    z = znode(i, j, k, grid, Center(), Center(), Face())
    mask = clamp((z - p.zˢ) / (p.zᶜ - p.zˢ), 0, 1)
    @inbounds ρw = fields.ρw[i, j, k]
    return -p.λ * mask * ρw
end
sponge = Forcing(sponge_damping, discrete_form=true,
    parameters=(; λ=1/10, zˢ, zᶜ=(zᵗ + zˢ)/2))

if micro_type == "1M"
    forcing = (ρw = sponge, ρθ = Forcing(∂t_ρe_ls), ρqᵛ = Forcing(∂t_ρqᵗ_ls))
else
    forcing = (ρw = sponge, ρθ = Forcing(∂t_ρe_ls), ρqᵉ = Forcing(∂t_ρqᵗ_ls))
end

# ── Surface BCs ───────────────────────────────────────────────
T_surface = 299.88
ρθ_bcs = FieldBoundaryConditions(bottom=BulkSensibleHeatFlux(coefficient=1.1e-3, surface_temperature=T_surface))
ρqᵗ_bcs = FieldBoundaryConditions(bottom=BulkVaporFlux(coefficient=1.2e-3, surface_temperature=T_surface))
ρu_bcs = FieldBoundaryConditions(bottom=Breeze.BulkDrag(coefficient=1.2e-3))
ρv_bcs = FieldBoundaryConditions(bottom=Breeze.BulkDrag(coefficient=1.2e-3))
if micro_type == "1M"
    boundary_conditions = (ρθ=ρθ_bcs, ρqᵛ=ρqᵗ_bcs, ρu=ρu_bcs, ρv=ρv_bcs)
else
    boundary_conditions = (ρθ=ρθ_bcs, ρqᵉ=ρqᵗ_bcs, ρu=ρu_bcs, ρv=ρv_bcs)
end

# ── Model ─────────────────────────────────────────────────────
model = AtmosphereModel(grid; dynamics, microphysics, advection=WENO(order=5),
    coriolis=FPlane(latitude=8.5), forcing, boundary_conditions)

# ── Initial conditions ────────────────────────────────────────
δT, δqᵗ, zδ = 0.5, 1e-4, 2000.0
ϵ() = rand() - 0.5
Tᵢ(x, y, z) = T₀(z) + δT * ϵ() * (z < zδ)
qᵗᵢ(x, y, z) = qᵗ₀(z) + δqᵗ * ϵ() * (z < zδ)
uᵢ(x, y, z) = u₀(z)
set!(model, T=Tᵢ, qᵗ=qᵗᵢ, u=uᵢ)

# ── Warmup ────────────────────────────────────────────────────
dt = 0.5
for _ in 1:5; time_step!(model, dt); end
CUDA.synchronize()
nranks > 1 && MPI.Barrier(MPI.COMM_WORLD)

# ── Timed trials ──────────────────────────────────────────────
times = Float64[]
for trial in 1:3
    N = 10
    t = @elapsed begin
        for _ in 1:N; time_step!(model, dt); end
        CUDA.synchronize()
        nranks > 1 && MPI.Barrier(MPI.COMM_WORLD)
    end
    push!(times, 1000t / N)
end

if rank == 0
    best = round(minimum(times), digits=1)
    all_t = join(round.(times, digits=1), ", ")
    result = "$(label)_$(px)x$(py): $best ms/step  (trials: $all_t)"
    println(result)
    open("/tmp/gate_$(label)_$(px)x$(py).txt", "w") do f
        println(f, result)
    end
end

MPI.Finalize()
