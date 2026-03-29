# Vanilla Oceananigans baroclinic instability benchmark
# Based on GB-25 sharding test but using standard MPI Distributed (no Reactant)

using Logging
disable_logging(Logging.Warn)

using MPI
MPI.Init()

using Oceananigans
using Oceananigans: GPU, Distributed, Partition, time_step!
using Oceananigans.Units
using SeawaterPolynomials
using Printf
using Random

use_nccl = "--nccl" in ARGS
if use_nccl
    using NCCL
    using Oceananigans
    const NCCLDistributed = Base.get_extension(Oceananigans, :OceananigansNCCLExt).NCCLDistributed
end

FT = Float64
Oceananigans.defaults.FloatType = FT

Ngpus = MPI.Comm_size(MPI.COMM_WORLD)
rank = MPI.Comm_rank(MPI.COMM_WORLD)

# Grid: weak scaling in x, base 64×64×4 per GPU
Nx_per_gpu = parse(Int, get(ENV, "NX_PER_GPU", "64"))
Ny = parse(Int, get(ENV, "NY", "64"))
Nz = parse(Int, get(ENV, "NZ", "4"))
H = 8  # halo size for WENO5

Nx = Nx_per_gpu * Ngpus

arch = if use_nccl
    NCCLDistributed(GPU(); partition = Partition(Ngpus, 1))
else
    Distributed(GPU(); partition = Partition(Ngpus, 1))
end

z = Oceananigans.Grids.ExponentialDiscretization(Nz, -4000, 0; scale=1000)

grid = LatitudeLongitudeGrid(arch;
    size = (Nx, Ny, Nz),
    halo = (H, H, H),
    z,
    latitude = (-80, 80),
    longitude = (0, 360),
)

model = HydrostaticFreeSurfaceModel(grid;
    free_surface = SplitExplicitFreeSurface(substeps=30),
    buoyancy = SeawaterBuoyancy(equation_of_state = SeawaterPolynomials.TEOS10EquationOfState(FT)),
    tracers = (:T, :S),
    coriolis = HydrostaticSphericalCoriolis(),
    momentum_advection = WENOVectorInvariant(order=5),
    tracer_advection = WENO(order=5),
    timestepper = :QuasiAdamsBashforth2,
)

model.clock.last_Δt = 1.0

Nt = parse(Int, get(ENV, "NT", "10"))

comm_backend = use_nccl ? "NCCL" : "MPI"
if rank == 0
    println("Baroclinic instability benchmark ($comm_backend): Ngpus=$Ngpus Nx=$Nx Ny=$Ny Nz=$Nz Nt=$Nt")
end

MPI.Barrier(MPI.COMM_WORLD)

# Warmup
elapsed1 = @elapsed begin
    for _ in 1:Nt
        time_step!(model, 1.0)
    end
end
MPI.Barrier(MPI.COMM_WORLD)

# Trial 1
elapsed2 = @elapsed begin
    for _ in 1:Nt
        time_step!(model, 1.0)
    end
end
MPI.Barrier(MPI.COMM_WORLD)

# Trial 2
elapsed3 = @elapsed begin
    for _ in 1:Nt
        time_step!(model, 1.0)
    end
end
MPI.Barrier(MPI.COMM_WORLD)

if rank == 0
    @printf("Nt=%d  Warmup:  %.3f seconds (%.1f ms/step)\n", Nt, elapsed1, 1000elapsed1/Nt)
    @printf("Nt=%d  Trial 1: %.3f seconds (%.1f ms/step)\n", Nt, elapsed2, 1000elapsed2/Nt)
    @printf("Nt=%d  Trial 2: %.3f seconds (%.1f ms/step)\n", Nt, elapsed3, 1000elapsed3/Nt)
end
