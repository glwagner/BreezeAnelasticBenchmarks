# MPI vs NCCL comparison for NonhydrostaticModel
#
# Usage (requires GPU-aware MPI):
#   mpiexec -n 1 julia --project benchmarks/compare_mpi_nccl.jl   # 1-GPU baseline
#   mpiexec -n 2 julia --project benchmarks/compare_mpi_nccl.jl   # 2-GPU comparison
#   mpiexec -n 4 julia --project benchmarks/compare_mpi_nccl.jl   # 4-GPU comparison
#
# On Perlmutter:
#   srun -n 2 --gpus-per-node=2 julia --project benchmarks/compare_mpi_nccl.jl
#
# Set USE_NCCL=0 to skip NCCL, USE_MPI=0 to skip MPI.

using Logging
disable_logging(Logging.Warn)

using MPI
MPI.Init()

rank = MPI.Comm_rank(MPI.COMM_WORLD)
Ngpus = MPI.Comm_size(MPI.COMM_WORLD)

using CUDA
CUDA.device!(rank)

using Oceananigans
using Oceananigans.TimeSteppers: time_step!
using Printf

use_mpi  = get(ENV, "USE_MPI", "1") != "0"
use_nccl = get(ENV, "USE_NCCL", "1") != "0"

Nt = 100  # steps per trial
Nx_per_gpu = 50
Ny = 400
Nz = 80

function make_model(arch)
    Nx = Nx_per_gpu * (Ngpus > 1 ? Ngpus : 1)
    grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(1, 1, 1),
                           halo=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b,
                                closure=ScalarDiffusivity(ν=200, κ=200))
    set!(model, u=1, b=(x, y, z) -> z)
    return model
end

function bench(model, label)
    # Warmup
    for _ in 1:Nt; time_step!(model, 0.01); end
    CUDA.synchronize()
    MPI.Barrier(MPI.COMM_WORLD)

    # Trial 1
    t1 = @elapsed begin
        for _ in 1:Nt; time_step!(model, 0.01); end
        CUDA.synchronize()
        MPI.Barrier(MPI.COMM_WORLD)
    end

    # Trial 2
    t2 = @elapsed begin
        for _ in 1:Nt; time_step!(model, 0.01); end
        CUDA.synchronize()
        MPI.Barrier(MPI.COMM_WORLD)
    end

    ms = round(1000 * min(t1, t2) / Nt, digits=2)
    rank == 0 && @printf("%-20s | GPUs=%d | %8.2f ms/step\n", label, Ngpus, ms)
    return ms
end

rank == 0 && println("="^60)
rank == 0 && @printf("NonhydrostaticModel + Centered(2) + diffusion\n")
rank == 0 && @printf("Grid: %dx%dx%d per GPU, Float64\n", Nx_per_gpu, Ny, Nz)
rank == 0 && println("="^60)

if Ngpus == 1
    # Non-distributed baseline
    model = make_model(GPU())
    bench(model, "1-GPU (baseline)")
else
    # MPI path
    if use_mpi
        arch_mpi = Distributed(GPU(); partition=Partition(Ngpus, 1))
        model_mpi = make_model(arch_mpi)
        bench(model_mpi, "MPI")
    end

    # NCCL path
    if use_nccl
        using NCCL
        NCCLExt = Base.get_extension(Oceananigans, :OceananigansNCCLExt)
        arch_nccl = NCCLExt.NCCLDistributed(GPU(); partition=Partition(Ngpus, 1))
        model_nccl = make_model(arch_nccl)
        bench(model_nccl, "NCCL")
    end
end

MPI.Finalize()
