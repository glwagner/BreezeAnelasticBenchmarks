using MPI
MPI.Init()

using BreezeAnelasticBenchmarks
using Oceananigans.Units
using Printf

FT = if "--float-type" in ARGS
    i = findfirst(==("--float-type"), ARGS)
    Dict("Float32" => Float32, "Float64" => Float64)[ARGS[i+1]]
else
    Float32
end

Ngpus = MPI.Comm_size(MPI.COMM_WORLD)
rank  = MPI.Comm_rank(MPI.COMM_WORLD)

# x-only partition for simplicity and best scaling
arch = Distributed(GPU(); partition = Partition(Ngpus, 1))

# 200×200×80 per GPU (weak scaling extends in x)
Nx_per_gpu = 200
Ny = 200
Lx_per_gpu = 84kilometers
Ly = 84kilometers

Nx = Nx_per_gpu * Ngpus
Lx = Lx_per_gpu * Ngpus

if rank == 0
    @info "Compressible weak scaling benchmark" Ngpus FT Nx Ny Nx_per_gpu
end

model = setup_supercell_compressible(arch; FT, Nx, Ny, Lx, Ly)

MPI.Barrier(MPI.COMM_WORLD)

# Warmup (includes any remaining compilation)
elapsed1 = @elapsed run_benchmark!(model)
MPI.Barrier(MPI.COMM_WORLD)

elapsed2 = @elapsed run_benchmark!(model)
MPI.Barrier(MPI.COMM_WORLD)

elapsed3 = @elapsed run_benchmark!(model)
MPI.Barrier(MPI.COMM_WORLD)

if rank == 0
    @info @sprintf("Warmup:  %.3f seconds", elapsed1)
    @info @sprintf("Trial 1: %.3f seconds", elapsed2)
    @info @sprintf("Trial 2: %.3f seconds", elapsed3)
end
