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
arch = NCCLExt.NCCLDistributed(GPU(); partition = Partition(Ngpus, 1))

Nx = 1024 * Ngpus
grid = RectilinearGrid(arch, Float32,
                       size = (Nx, 1024, 128),
                       x = (0, Ngpus),
                       y = (0, 1),
                       z = (0, 1),
                       halo = (3, 3, 3),
                       topology = (Periodic, Periodic, Bounded))

model = NonhydrostaticModel(grid;
                            advection = WENO(order=5),
                            buoyancy = BuoyancyTracer(),
                            tracers = :b)

set!(model, u=1, v=0, w=0, b=(x, y, z) -> z)

for _ in 1:10
    time_step!(model, 0.001)
end
CUDA.synchronize()
MPI.Barrier(MPI.COMM_WORLD)

N = 10
CUDA.synchronize()
MPI.Barrier(MPI.COMM_WORLD)

t = @elapsed begin
    for _ in 1:N
        time_step!(model, 0.001)
    end
    CUDA.synchronize()
    MPI.Barrier(MPI.COMM_WORLD)
end

if rank == 0
    ms = round(1000t / N, digits=1)
    eff = round(100 * 357.1 / ms, digits=1)
    @printf("LARGE_SCALING | GPUs=%d | 1024x1024x128/GPU F32 | WENO5 | %.1f ms/step | eff=%.1f%%\n",
            Ngpus, ms, eff)
end

MPI.Finalize()
