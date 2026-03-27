# Suppress "malformed environment entry" warnings from Cray MPICH multi-node runs
using Logging
disable_logging(Logging.Warn)

using MPI
MPI.Init()

using BreezeAnelasticBenchmarks
using Oceananigans.Units
using Printf

use_nccl = "--nccl" in ARGS
if use_nccl
    using NCCL
    using Oceananigans.DistributedComputations
    const NCCLDistributed = Base.get_extension(Oceananigans, :OceananigansNCCLExt).NCCLDistributed
end

FT = if "--float-type" in ARGS
    i = findfirst(==("--float-type"), ARGS)
    Dict("Float32" => Float32, "Float64" => Float64)[ARGS[i+1]]
else
    Float32
end

Ngpus = MPI.Comm_size(MPI.COMM_WORLD)
rank  = MPI.Comm_rank(MPI.COMM_WORLD)

# x-only partition for simplicity and best scaling
arch = if use_nccl
    NCCLDistributed(GPU(); partition = Partition(Ngpus, 1))
else
    Distributed(GPU(); partition = Partition(Ngpus, 1))
end

# Per-GPU grid size (configurable via environment variables)
Nx_per_gpu = parse(Int, get(ENV, "NX_PER_GPU", "200"))
Ny = parse(Int, get(ENV, "NY_PER_GPU", "200"))
Lx_per_gpu = Nx_per_gpu / 200 * 84kilometers
Ly = Ny / 200 * 84kilometers

Nx = Nx_per_gpu * Ngpus
Lx = Lx_per_gpu * Ngpus

if rank == 0
    comm_backend = use_nccl ? "NCCL" : "MPI"
    println("Compressible weak scaling benchmark ($comm_backend): Ngpus=$Ngpus FT=$FT Nx=$Nx Ny=$Ny Nx_per_gpu=$Nx_per_gpu")
end

model = setup_supercell_compressible(arch; FT, Nx, Ny, Lx, Ly)

Nt = parse(Int, get(ENV, "NT", "10"))

MPI.Barrier(MPI.COMM_WORLD)

# Warmup (includes any remaining compilation)
elapsed1 = @elapsed run_benchmark!(model, Nt)
MPI.Barrier(MPI.COMM_WORLD)

elapsed2 = @elapsed run_benchmark!(model, Nt)
MPI.Barrier(MPI.COMM_WORLD)

elapsed3 = @elapsed run_benchmark!(model, Nt)
MPI.Barrier(MPI.COMM_WORLD)

if rank == 0
    @printf("Nt=%d  Warmup:  %.3f seconds (%.1f ms/step)\n", Nt, elapsed1, 1000elapsed1/Nt)
    @printf("Nt=%d  Trial 1: %.3f seconds (%.1f ms/step)\n", Nt, elapsed2, 1000elapsed2/Nt)
    @printf("Nt=%d  Trial 2: %.3f seconds (%.1f ms/step)\n", Nt, elapsed3, 1000elapsed3/Nt)
end
