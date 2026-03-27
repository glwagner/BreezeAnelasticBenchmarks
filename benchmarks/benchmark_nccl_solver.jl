using MPI
MPI.Init()

using CUDA
using NCCL
using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: DistributedFourierTridiagonalPoissonSolver,
                                            reconstruct_global_grid
using Oceananigans.Solvers: solve!

# The NCCL extension is loaded automatically when both Oceananigans and NCCL are loaded.
# Access the extension module to get NCCLDistributedFFTSolver.
const NCCLExt = Base.get_extension(Oceananigans, :OceananigansNCCLExt)
const NCCLDistributedFFTSolver = NCCLExt.NCCLDistributedFFTSolver

rank = MPI.Comm_rank(MPI.COMM_WORLD)
Ngpus = MPI.Comm_size(MPI.COMM_WORLD)
Ngpus < 2 && error("Need at least 2 GPUs")

FT = Float32
Nsolves = 30

# ============================================================
# Setup: Create distributed grid and solver
# ============================================================

rank == 0 && println("=== NCCL vs MPI Pressure Solver Benchmark ===")
rank == 0 && println("  GPUs: $Ngpus")

arch = Distributed(GPU(); partition = Partition(Ngpus, 1))
Nx_per_gpu = 200
grid = RectilinearGrid(arch,
                       size = (Nx_per_gpu * Ngpus, 200, 80),
                       x = (0, 84e3 * Ngpus), y = (0, 84e3), z = (0, 20e3),
                       halo = (1, 1, 1),
                       topology = (Periodic, Periodic, Bounded))

global_grid = reconstruct_global_grid(grid)

# Create MPI solver (baseline)
mpi_solver = DistributedFourierTridiagonalPoissonSolver(global_grid, grid)

# Create NCCL solver (wraps MPI solver with NCCL comms)
nccl_solver = NCCLDistributedFFTSolver(mpi_solver)

p = CenterField(grid)
rank == 0 && println("  Local grid: $(size(grid))")
rank == 0 && println("  NCCL version: $(NCCL.version())")

# ============================================================
# Phase 1: Correctness test — NCCL vs MPI should be identical
# ============================================================

rank == 0 && println("\n=== Correctness Test ===")

# Fill RHS with deterministic data
parent(mpi_solver.storage.zfield) .= CUDA.randn(FT, size(mpi_solver.storage.zfield)...)
CUDA.synchronize()

# Save the RHS for reuse
rhs_copy = copy(parent(mpi_solver.storage.zfield))

# Solve with MPI
p .= 0
parent(mpi_solver.storage.zfield) .= rhs_copy
solve!(p, mpi_solver)
CUDA.synchronize()
p_mpi = copy(parent(p))

# Solve with NCCL (same solver internals, different communication)
p .= 0
parent(nccl_solver.solver.storage.zfield) .= rhs_copy
solve!(p, nccl_solver)
CUDA.synchronize()
p_nccl = copy(parent(p))

# Compare
max_diff = maximum(abs.(p_nccl .- p_mpi))
if rank == 0
    if max_diff == 0
        println("  PASS: Results are bitwise identical")
    elseif max_diff < eps(FT)
        @printf("  PASS: Results match to machine precision (max diff: %.2e)\n", max_diff)
    else
        @printf("  FAIL: Results differ (max diff: %.2e)\n", max_diff)
    end
end

# ============================================================
# Phase 2: MPI solver benchmark (baseline)
# ============================================================

rank == 0 && println("\n=== MPI Solver Benchmark ===")

# Warmup
for _ in 1:5
    parent(mpi_solver.storage.zfield) .= rhs_copy
    solve!(p, mpi_solver)
end
CUDA.synchronize()
MPI.Barrier(MPI.COMM_WORLD)

t_mpi = CUDA.@elapsed begin
    for _ in 1:Nsolves
        parent(mpi_solver.storage.zfield) .= rhs_copy
        solve!(p, mpi_solver)
    end
end
MPI.Barrier(MPI.COMM_WORLD)

if rank == 0
    @printf("  MPI solver (%d solves): %.4f s (%.2f ms/solve)\n",
            Nsolves, t_mpi, 1000t_mpi / Nsolves)
end

# ============================================================
# Phase 3: NCCL solver benchmark
# ============================================================

rank == 0 && println("\n=== NCCL Solver Benchmark ===")

# Warmup
for _ in 1:5
    parent(nccl_solver.solver.storage.zfield) .= rhs_copy
    solve!(p, nccl_solver)
end
CUDA.synchronize()
MPI.Barrier(MPI.COMM_WORLD)

t_nccl = CUDA.@elapsed begin
    for _ in 1:Nsolves
        parent(nccl_solver.solver.storage.zfield) .= rhs_copy
        solve!(p, nccl_solver)
    end
end
MPI.Barrier(MPI.COMM_WORLD)

if rank == 0
    @printf("  NCCL solver (%d solves): %.4f s (%.2f ms/solve)\n",
            Nsolves, t_nccl, 1000t_nccl / Nsolves)
end

# ============================================================
# Summary
# ============================================================

if rank == 0
    println("\n" * "="^60)
    println("SUMMARY ($Ngpus GPUs, $(Nx_per_gpu)x200x80 per GPU)")
    println("="^60)
    @printf("  MPI solver:   %.2f ms/solve\n", 1000t_mpi / Nsolves)
    @printf("  NCCL solver:  %.2f ms/solve\n", 1000t_nccl / Nsolves)
    @printf("  Speedup:      %.2fx\n", t_mpi / t_nccl)
    println()
    @printf("  Expected: ~2.4x (36 ms → 15 ms) based on sync_device! elimination\n")
end

rank == 0 && println("\nDone.")
MPI.Finalize()
