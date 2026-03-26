using MPI
MPI.Init()

using BreezeAnelasticBenchmarks
using Oceananigans.Units
using Printf
using CUDA

rank = MPI.Comm_rank(MPI.COMM_WORLD)

# ERF config at 400x400x80 on single GPU
model = setup_supercell_erf(GPU(); Nx=400, Ny=400, Nz=80,
                             Lx=168kilometers, Ly=168kilometers, Lz=20kilometers)

# Warmup
run_benchmark!(model)
CUDA.synchronize()

# Trials
t1 = @elapsed (run_benchmark!(model); CUDA.synchronize())
t2 = @elapsed (run_benchmark!(model); CUDA.synchronize())

if rank == 0
    @info "ERF config 400x400x80 single GPU"
    @info @sprintf("Trial 1: %.3f seconds (%.1f ms/step)", t1, 1000t1/10)
    @info @sprintf("Trial 2: %.3f seconds (%.1f ms/step)", t2, 1000t2/10)
end
