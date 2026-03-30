# Unified weak scaling benchmark for all Breeze supercell configurations.
#
# Usage:
#   srun julia --project=. benchmarks/run_benchmark.jl --config erf
#   srun julia --project=. benchmarks/run_benchmark.jl --config compressible --nccl
#   srun julia --project=. benchmarks/run_benchmark.jl --config weno
#
# Environment variables: NT, NX_PER_GPU, NY_PER_GPU, PARTITION_X_ONLY, RX, RY, FLOAT_TYPE

using Logging
disable_logging(Logging.Warn)

using MPI
MPI.Init()

using BreezeAnelasticBenchmarks
using Oceananigans.Units
using Printf

# Parse config
config = "compressible"
for (i, arg) in enumerate(ARGS)
    arg == "--config" && (config = ARGS[i+1])
end

use_nccl = "--nccl" in ARGS

if use_nccl
    using NCCL
    using Oceananigans
    const NCCLDistributed = Base.get_extension(Oceananigans, :OceananigansNCCLExt).NCCLDistributed
end

FT = if "--float-type" in ARGS
    i = findfirst(==("--float-type"), ARGS)
    Dict("Float32" => Float32, "Float64" => Float64)[ARGS[i+1]]
else
    Float32
end

Ngpus = MPI.Comm_size(MPI.COMM_WORLD)
rank = MPI.Comm_rank(MPI.COMM_WORLD)

# Partition
if config == "erf"
    # ERF supports 2D partition
    function compute_partition(Ngpus)
        for Ry in [8, 4, 2, 1]
            Ngpus % Ry != 0 && continue
            80 % Ry != 0 && continue
            Rx = Ngpus ÷ Ry
            Ny_g = parse(Int, get(ENV, "NY_PER_GPU", "200")) * Ry
            Ny_g % Rx != 0 && continue
            return Rx, Ry
        end
        return Ngpus, 1
    end

    if haskey(ENV, "RX") && haskey(ENV, "RY")
        Rx = parse(Int, ENV["RX"])
        Ry = parse(Int, ENV["RY"])
    elseif get(ENV, "PARTITION_X_ONLY", "") == "1"
        Rx, Ry = Ngpus, 1
    else
        Rx, Ry = compute_partition(Ngpus)
    end
else
    Rx, Ry = Ngpus, 1
end

arch = if use_nccl
    NCCLDistributed(GPU(); partition = Partition(Rx, Ry))
else
    Distributed(GPU(); partition = Partition(Rx, Ry))
end

# Grid size
if config == "weno"
    Nx_per_gpu = parse(Int, get(ENV, "NX_PER_GPU", "400"))
    Ny_per_gpu = parse(Int, get(ENV, "NY_PER_GPU", "400"))
    Lx_per_gpu = Nx_per_gpu / 400 * 168kilometers
    Ly_per_gpu = Ny_per_gpu / 400 * 168kilometers
else
    Nx_per_gpu = parse(Int, get(ENV, "NX_PER_GPU", "50"))
    Ny_per_gpu = parse(Int, get(ENV, "NY_PER_GPU", "400"))
    Lx_per_gpu = Nx_per_gpu / 200 * 84kilometers
    Ly_per_gpu = Ny_per_gpu / 200 * 84kilometers
end

Nx = Nx_per_gpu * Rx
Ny = Ny_per_gpu * Ry
Lx = Lx_per_gpu * Rx
Ly = Ly_per_gpu * Ry
Lz = 20kilometers

# Warm up CUDA kernel cache on rank 0 with a small single-GPU model.
# This populates the shared filesystem cache so all ranks can reuse it
# without Lustre contention during the distributed model construction.
if rank == 0
    using CUDA
    warmup_model = if config == "weno"
        setup_supercell(GPU(); FT, Nx=8, Ny=8, Nz=8, Lx=Lx_per_gpu, Ly=Ly_per_gpu, Lz)
    elseif config == "erf"
        setup_supercell_erf(GPU(); FT, Nx=8, Ny=8, Nz=8, Lx=Lx_per_gpu, Ly=Ly_per_gpu, Lz)
    else
        setup_supercell_compressible(GPU(); FT, Nx=8, Ny=8, Nz=8, Lx=Lx_per_gpu, Ly=Ly_per_gpu, Lz)
    end
    run_benchmark!(warmup_model, 1)
    GC.gc(true); CUDA.reclaim()
    println("CUDA kernel cache warmed up on rank 0")
end
MPI.Barrier(MPI.COMM_WORLD)

comm_backend = use_nccl ? "NCCL" : "MPI"
if rank == 0
    println("$config benchmark ($comm_backend): Ngpus=$Ngpus Rx=$Rx Ry=$Ry FT=$FT Nx=$Nx Ny=$Ny")
end

# Setup model
model = if config == "weno"
    setup_supercell(arch; FT, Nx, Ny, Lx, Ly, Lz)
elseif config == "erf"
    setup_supercell_erf(arch; FT, Nx, Ny, Lx, Ly, Lz)
elseif config == "compressible"
    setup_supercell_compressible(arch; FT, Nx, Ny, Lx, Ly, Lz)
else
    error("Unknown config: $config. Use weno, erf, or compressible.")
end

Nt = parse(Int, get(ENV, "NT", "100"))

MPI.Barrier(MPI.COMM_WORLD)

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
