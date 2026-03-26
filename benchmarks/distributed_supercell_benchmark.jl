using MPI
MPI.Init()

using Breeze
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
arch  = Distributed(GPU(); partition = Partition(Ngpus, 1))

Nx_per_gpu = 400
Lx_per_gpu = 168kilometers

if rank == 0
    @info "Weak scaling benchmark" Ngpus FT Nx_per_gpu
end

model = setup_supercell(arch; FT,
                         Nx = Nx_per_gpu * Ngpus,
                         Ny = 400, Nz = 80,
                         Lx = Lx_per_gpu * Ngpus)

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
    @info @sprintf("Nt=%d  Warmup:  %.3f seconds (%.1f ms/step)", Nt, elapsed1, 1000elapsed1/Nt)
    @info @sprintf("Nt=%d  Trial 1: %.3f seconds (%.1f ms/step)", Nt, elapsed2, 1000elapsed2/Nt)
    @info @sprintf("Nt=%d  Trial 2: %.3f seconds (%.1f ms/step)", Nt, elapsed3, 1000elapsed3/Nt)
end
