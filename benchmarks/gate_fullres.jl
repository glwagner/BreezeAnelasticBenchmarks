# GATE III GigaLES full resolution (2048×2048×181) benchmark
#
# Usage:
#   mpiexec -n N julia --project gate_fullres.jl [microphysics] [precision]
#
# microphysics: "satadj" (default), "1M", "1M_mixed"
# precision:    "f32" (default), "f64"

using MPI; MPI.Init()
rank = MPI.Comm_rank(MPI.COMM_WORLD)
nranks = MPI.Comm_size(MPI.COMM_WORLD)
using CUDA; CUDA.device!(rank)
using NCCL, Oceananigans, Breeze, AtmosphericProfilesLibrary
using Oceananigans.Units
using Oceananigans.Grids: znode
using Oceananigans.TimeSteppers: time_step!

micro_type = length(ARGS) >= 1 ? ARGS[1] : "satadj"
precision  = length(ARGS) >= 2 ? ARGS[2] : "f32"
FT = precision == "f64" ? Float64 : Float32

Oceananigans.defaults.FloatType = FT
NCCLExt = Base.get_extension(Oceananigans, :OceananigansNCCLExt)
arch = NCCLExt.NCCLDistributed(GPU(); partition=Partition(nranks, 1))

# ── GATE stretched vertical grid ─────────────────────────────
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

# ── Grid ──────────────────────────────────────────────────────
grid = RectilinearGrid(arch, size=(2048, 2048, Nz), x=(0, 204800), y=(0, 204800), z=zf,
                       halo=(5, 5, 5), topology=(Periodic, Periodic, Bounded))

# ── Dynamics ──────────────────────────────────────────────────
constants = ThermodynamicConstants()
ref = ReferenceState(grid, constants, surface_pressure=101200, potential_temperature=298)
dynamics = AnelasticDynamics(ref)

# ── Microphysics ──────────────────────────────────────────────
if micro_type == "1M"
    using CloudMicrophysics
    BreezeExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
    microphysics = BreezeExt.OneMomentCloudMicrophysics(FT;
        cloud_formation = NonEquilibriumCloudFormation(nothing, nothing))
    label = "1M"
elseif micro_type == "1M_mixed"
    using CloudMicrophysics
    CMP = CloudMicrophysics.Parameters
    BreezeExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
    microphysics = BreezeExt.OneMomentCloudMicrophysics(FT;
        cloud_formation = NonEquilibriumCloudFormation(nothing, CMP.CloudIce(FT)))
    label = "1M_mixed"
else
    microphysics = SaturationAdjustment(equilibrium=MixedPhaseEquilibrium())
    label = "satadj"
end

# ── Profiles & forcings ──────────────────────────────────────
rho_ref = ref.density
cpd = constants.dry_air.heat_capacity
T_prof    = AtmosphericProfilesLibrary.GATE_III_T(FT)
qt_prof   = AtmosphericProfilesLibrary.GATE_III_q_tot(FT)
u_prof    = AtmosphericProfilesLibrary.GATE_III_u(FT)
dTdt_prof = AtmosphericProfilesLibrary.GATE_III_dTdt(FT)
dqdt_prof = AtmosphericProfilesLibrary.GATE_III_dqtdt(FT)

tend_theta = Field{Nothing, Nothing, Center}(grid)
tend_moist = Field{Nothing, Nothing, Center}(grid)
set!(tend_theta, z -> dTdt_prof(z))
set!(tend_theta, rho_ref * cpd * tend_theta)
set!(tend_moist, z -> dqdt_prof(z))
set!(tend_moist, rho_ref * tend_moist)

@inline function sponge_fn(i, j, k, grid, clock, fields, p)
    z = znode(i, j, k, grid, Center(), Center(), Face())
    mask = clamp((z - p.zs) / (p.zc - p.zs), 0, 1)
    @inbounds rw = fields.ρw[i, j, k]
    return -p.lam * mask * rw
end
sponge = Forcing(sponge_fn, discrete_form=true, parameters=(; lam=0.1, zs=19000.0, zc=23000.0))

# Moisture forcing key depends on microphysics
if micro_type in ("1M", "1M_mixed")
    forcing = (ρw=sponge, ρθ=Forcing(tend_theta), ρqᵛ=Forcing(tend_moist))
else
    forcing = (ρw=sponge, ρθ=Forcing(tend_theta), ρqᵉ=Forcing(tend_moist))
end

# ── Surface BCs ───────────────────────────────────────────────
Ts = 299.88
theta_bcs = FieldBoundaryConditions(bottom=BulkSensibleHeatFlux(coefficient=1.1e-3, surface_temperature=Ts))
moist_bcs = FieldBoundaryConditions(bottom=BulkVaporFlux(coefficient=1.2e-3, surface_temperature=Ts))
u_bcs = FieldBoundaryConditions(bottom=Breeze.BulkDrag(coefficient=1.2e-3))
v_bcs = FieldBoundaryConditions(bottom=Breeze.BulkDrag(coefficient=1.2e-3))

if micro_type in ("1M", "1M_mixed")
    boundary_conditions = (ρθ=theta_bcs, ρqᵛ=moist_bcs, ρu=u_bcs, ρv=v_bcs)
else
    boundary_conditions = (ρθ=theta_bcs, ρqᵉ=moist_bcs, ρu=u_bcs, ρv=v_bcs)
end

# ── Model ─────────────────────────────────────────────────────
model = AtmosphereModel(grid; dynamics, microphysics, advection=WENO(order=5),
                        coriolis=FPlane(latitude=8.5), forcing, boundary_conditions)

eps() = rand() - 0.5
set!(model, T=(x, y, z) -> T_prof(z) + FT(0.5) * eps() * (z < 2000),
            u=(x, y, z) -> u_prof(z))

# Also set moisture if satadj (uses qᵗ)
if !(micro_type in ("1M", "1M_mixed"))
    set!(model, qᵗ=(x, y, z) -> qt_prof(z) + FT(1e-4) * eps() * (z < 2000))
end

# ── Warmup ────────────────────────────────────────────────────
for _ in 1:3; time_step!(model, 0.5); end
CUDA.synchronize(); MPI.Barrier(MPI.COMM_WORLD)

# ── Timed trials ──────────────────────────────────────────────
times = Float64[]
for trial in 1:3
    N = 3
    t = @elapsed begin
        for _ in 1:N; time_step!(model, 0.5); end
        CUDA.synchronize(); MPI.Barrier(MPI.COMM_WORLD)
    end
    push!(times, 1000t / N)
end

if rank == 0
    best = round(minimum(times), digits=1)
    all_t = join(round.(times, digits=1), ", ")
    nx_per = 2048 ÷ nranks
    tag = "GATE_$(label)_$(uppercase(precision))_$(nranks)x1"
    result = "$tag: $best ms/step  (trials: $all_t)  [2048x2048x$(Nz), $(nx_per)x2048x$(Nz)/GPU]"
    println(result)
    open("/tmp/gate_fullres_result.txt", "a") do f
        println(f, result)
    end
end

MPI.Finalize()
