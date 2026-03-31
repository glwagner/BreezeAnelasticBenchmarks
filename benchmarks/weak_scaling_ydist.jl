using MPI
MPI.Init()

rank = MPI.Comm_rank(MPI.COMM_WORLD)
Ngpus = MPI.Comm_size(MPI.COMM_WORLD)

using CUDA
CUDA.device!(rank)

using NCCL
using Oceananigans
using Oceananigans.TimeSteppers: time_step!
using Printf

NCCLExt = Base.get_extension(Oceananigans, :OceananigansNCCLExt)

# Distribute in y only: Partition(1, Ngpus)
arch = NCCLExt.NCCLDistributed(GPU(); partition = Partition(1, Ngpus))

# ERF-like: 50 x 400 x 80 per GPU, weak scaling in y
Nx = 50
Ny = 400 * Ngpus
grid = RectilinearGrid(arch,
                       size = (Nx, Ny, 80),
                       extent = (1, Ngpus, 1),
                       halo = (1, 1, 1),
                       topology = (Periodic, Periodic, Bounded))

model = NonhydrostaticModel(grid,
                            buoyancy = BuoyancyTracer(),
                            tracers = :b,
                            closure = ScalarDiffusivity(ν=200, κ=200))

set!(model, u=1, v=0, w=0, b=(x, y, z) -> z)

# Warmup
for _ in 1:100
    time_step!(model, 0.01)
end
CUDA.synchronize()
MPI.Barrier(MPI.COMM_WORLD)

# Benchmark
N = 100
t1 = @elapsed begin
    for _ in 1:N; time_step!(model, 0.01); end
    CUDA.synchronize(); MPI.Barrier(MPI.COMM_WORLD)
end

t2 = @elapsed begin
    for _ in 1:N; time_step!(model, 0.01); end
    CUDA.synchronize(); MPI.Barrier(MPI.COMM_WORLD)
end

t = min(t1, t2)

if rank == 0
    ms = round(1000t / N, digits=2)
    # 1-GPU baseline for 50x400x80 F64 NonhydrostaticModel: 5.79 ms
    eff = round(100 * 5.79 / ms, digits=1)
    @printf("Y_DIST | GPUs=%d | Partition(1,%d) | 50x%dx80/GPU | %.2f ms/step | eff=%.1f%%\n",
            Ngpus, Ngpus, 400, ms, eff)
end

MPI.Finalize()
