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

# 2D partition: extend in both x and y
# Per GPU: 200×200×80 (= 400×400×80 per 4-GPU node)
# Choose Rx, Ry to be as square as possible
# Also need: Ny_global % Rx == 0 for the distributed Poisson solver
function compute_partition(Ngpus)
    for Ry in [8, 4, 2, 1]
        Ngpus % Ry != 0 && continue
        80 % Ry != 0 && continue  # Nz must be divisible by Ry
        Rx = Ngpus ÷ Ry
        Ny_g = parse(Int, get(ENV, "NY_PER_GPU", "200")) * Ry
        Ny_g % Rx != 0 && continue  # Ny_global must be divisible by Rx
        return Rx, Ry
    end
    return Ngpus, 1  # fallback to x-only
end
# Allow explicit partition via environment variables
if haskey(ENV, "RX") && haskey(ENV, "RY")
    Rx = parse(Int, ENV["RX"])
    Ry = parse(Int, ENV["RY"])
elseif get(ENV, "PARTITION_X_ONLY", "") == "1"
    Rx, Ry = Ngpus, 1
else
    Rx, Ry = compute_partition(Ngpus)
end

arch = Distributed(GPU(); partition = Partition(Rx, Ry))

# Per-GPU grid size (configurable via environment variables)
Nx_per_gpu = parse(Int, get(ENV, "NX_PER_GPU", "200"))
Ny_per_gpu = parse(Int, get(ENV, "NY_PER_GPU", "200"))
Lx_per_gpu = Nx_per_gpu / 200 * 84kilometers
Ly_per_gpu = Ny_per_gpu / 200 * 84kilometers

Nx = Nx_per_gpu * Rx
Ny = Ny_per_gpu * Ry
Lx = Lx_per_gpu * Rx
Ly = Ly_per_gpu * Ry

if rank == 0
    @info "ERF-equivalent weak scaling benchmark" Ngpus Rx Ry FT Nx Ny Nx_per_gpu Ny_per_gpu
end

model = setup_supercell_erf(arch; FT, Nx, Ny, Lx, Ly)

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
