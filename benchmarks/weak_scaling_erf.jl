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

Nx = 50 * Ngpus
grid = RectilinearGrid(arch,
                       size = (Nx, 400, 80),
                       x = (0, 21e3 * Ngpus),
                       y = (0, 168e3),
                       z = (0, 20e3),
                       halo = (1, 1, 1),
                       topology = (Periodic, Periodic, Bounded))

model = NonhydrostaticModel(grid,
                            buoyancy = BuoyancyTracer(),
                            tracers = :b,
                            closure = ScalarDiffusivity(ν=200, κ=200))

set!(model, u=1, v=0, w=0, b=(x, y, z) -> 1e-4 * z)

for _ in 1:30
    time_step!(model, 0.1)
end
CUDA.synchronize()
MPI.Barrier(MPI.COMM_WORLD)

N = 20
CUDA.synchronize()
MPI.Barrier(MPI.COMM_WORLD)

t = @elapsed begin
    for _ in 1:N
        time_step!(model, 0.1)
    end
    CUDA.synchronize()
    MPI.Barrier(MPI.COMM_WORLD)
end

if rank == 0
    ms = round(1000t / N, digits=2)
    eff = round(100 * 6.85 / ms, digits=1)
    @printf("WEAK_SCALING_ERF | GPUs=%d | %dx400x80/GPU | Centered+diffusion | %.2f ms/step | efficiency=%.1f%%\n",
            Ngpus, 50, ms, eff)
end

MPI.Finalize()
