using MPI
MPI.Init()

using CUDA
using Printf

using BreezeAnelasticBenchmarks
using Breeze
using Oceananigans
using Oceananigans.Units
using Oceananigans.Solvers: solve!
using Oceananigans.BoundaryConditions: fill_halo_regions!

rank = MPI.Comm_rank(MPI.COMM_WORLD)
Ngpus = MPI.Comm_size(MPI.COMM_WORLD)

if Ngpus != 1
    rank == 0 && @error "This profiling script must run on exactly 1 GPU"
    MPI.Finalize()
    exit(1)
end

FT = Float32

# Use smaller grid if needed to avoid OOM with pool=none
Nx = parse(Int, get(ENV, "PROFILE_NX", "400"))
Ny = parse(Int, get(ENV, "PROFILE_NY", "400"))
Nz = parse(Int, get(ENV, "PROFILE_NZ", "80"))
Lx = Nx / 400 * 168kilometers
Ly = Ny / 400 * 168kilometers
Lz = 20kilometers
println("Grid: $Nx × $Ny × $Nz")

# ============================================================
# Phase A: Profile non-distributed model alone (no memory competition)
# ============================================================

println("=== Phase A: Non-distributed model (GPU only) ==="); flush(stdout)
model_gpu = setup_supercell(GPU(); Nx, Ny, Nz, Lx, Ly, Lz)
println("  Model created."); flush(stdout)

# Warmup
time_step!(model_gpu, 0.1)
CUDA.synchronize()

# Full time step
t_gpu = CUDA.@elapsed begin
    for _ in 1:10
        time_step!(model_gpu, 0.1)
    end
end
@printf("  10 time steps:        %.4f s\n", t_gpu)

# Solver
solver_gpu = model_gpu.pressure_solver
println("  Solver type: ", typeof(solver_gpu).name.name)

# Time just the pressure solver (need to set source term first)
using Oceananigans.Architectures: on_architecture
rhs_data = on_architecture(GPU(), randn(FT, size(solver_gpu.source_term)...))
solver_gpu.source_term .= Complex{FT}.(rhs_data)
p_gpu = CenterField(model_gpu.grid)

CUDA.synchronize()
t_solve_gpu = CUDA.@elapsed begin
    for _ in 1:30
        solve!(p_gpu, solver_gpu)
    end
end
@printf("  Solver only (30x):    %.4f s\n", t_solve_gpu)

# Halo fills
CUDA.synchronize()
t_halo_gpu = CUDA.@elapsed begin
    for _ in 1:90  # ~3 per stage × 3 stages × 10 steps
        fill_halo_regions!(model_gpu.velocities)
    end
end
@printf("  Momentum halo (90x):  %.4f s\n", t_halo_gpu)

CUDA.synchronize()
t_phalo_gpu = CUDA.@elapsed begin
    for _ in 1:30
        fill_halo_regions!(p_gpu)
    end
end
@printf("  Pressure halo (30x):  %.4f s\n", t_phalo_gpu)

# Free GPU model
model_gpu = nothing
solver_gpu = nothing
p_gpu = nothing
GC.gc()
CUDA.reclaim()

# ============================================================
# Phase B: Profile distributed model alone
# ============================================================

println("\n=== Phase B: Distributed model (1 GPU) ==="); flush(stdout)
arch_dist = Distributed(GPU(); partition=Partition(1, 1))
model_dist = setup_supercell(arch_dist; Nx, Ny, Nz, Lx, Ly, Lz)
println("  Model created."); flush(stdout)

# Warmup
MPI.Barrier(MPI.COMM_WORLD)
time_step!(model_dist, 0.1)
CUDA.synchronize()
MPI.Barrier(MPI.COMM_WORLD)

# Full time step
t_dist = CUDA.@elapsed begin
    for _ in 1:10
        time_step!(model_dist, 0.1)
    end
end
@printf("  10 time steps:        %.4f s\n", t_dist)

# Solver
solver_dist = model_dist.pressure_solver
println("  Solver type: ", typeof(solver_dist).name.name)

# Time the pressure solver
rhs_dist = on_architecture(GPU(), randn(FT, size(solver_dist.source_term)...))
solver_dist.source_term .= Complex{FT}.(rhs_dist)
p_dist = CenterField(model_dist.grid)

CUDA.synchronize()
t_solve_dist = CUDA.@elapsed begin
    for _ in 1:30
        solve!(p_dist, solver_dist)
    end
end
@printf("  Solver only (30x):    %.4f s\n", t_solve_dist)

# Halo fills
CUDA.synchronize()
MPI.Barrier(MPI.COMM_WORLD)
t_halo_dist = CUDA.@elapsed begin
    for _ in 1:90
        fill_halo_regions!(model_dist.velocities)
    end
end
@printf("  Momentum halo (90x):  %.4f s\n", t_halo_dist)

CUDA.synchronize()
MPI.Barrier(MPI.COMM_WORLD)
t_phalo_dist = CUDA.@elapsed begin
    for _ in 1:30
        fill_halo_regions!(p_dist)
    end
end
@printf("  Pressure halo (30x):  %.4f s\n", t_phalo_dist)

# ============================================================
# Phase C: FFT comparison
# ============================================================

println("\n=== Phase C: FFT performance ===")

Nx, Ny, Nz = size(model_dist.grid)
Nc = Complex{FT}
data_3d = CUDA.rand(Nc, Nx, Ny, Nz)
data_3d_copy = copy(data_3d)

# Batched 2D periodic FFT (what non-distributed solver does internally)
plan_2d = CUDA.CUFFT.plan_fft!(data_3d, [1, 2])
CUDA.synchronize()
t_fft2d = CUDA.@elapsed begin
    for _ in 1:60
        plan_2d * data_3d
    end
end
@printf("  Batched 2D FFT (60x):          %.4f s\n", t_fft2d)

# Two separate 1D FFTs
plan_1d_x = CUDA.CUFFT.plan_fft!(data_3d, [1])
plan_1d_y = CUDA.CUFFT.plan_fft!(data_3d, [2])
CUDA.synchronize()
t_fft1d = CUDA.@elapsed begin
    for _ in 1:30
        plan_1d_x * data_3d
        plan_1d_y * data_3d
    end
end
@printf("  Two 1D FFTs (30×2=60):         %.4f s\n", t_fft1d)
@printf("  FFT overhead (1D vs 2D):       %.2fx\n", t_fft1d / t_fft2d)

# Reshaped 1D FFT (what distributed y-FFT actually does)
data_reshaped = reshape(data_3d_copy, Ny, Nx, Nz)
plan_1d_y_reshaped = CUDA.CUFFT.plan_fft!(data_reshaped, [1])
CUDA.synchronize()
t_fft1d_reshape = CUDA.@elapsed begin
    for _ in 1:60
        plan_1d_y_reshaped * data_reshaped
    end
end
@printf("  1D FFT reshaped (60x):         %.4f s\n", t_fft1d_reshape)

# ============================================================
# Phase D: MPI and CUDA sync overhead
# ============================================================

println("\n=== Phase D: MPI and sync overhead ===")

CUDA.synchronize()
t_barrier = @elapsed begin
    for _ in 1:1000
        MPI.Barrier(MPI.COMM_WORLD)
    end
end
@printf("  MPI.Barrier (1000x):        %.4f s  (%.1f μs each)\n", t_barrier, t_barrier * 1000)

t_sync = @elapsed begin
    for _ in 1:1000
        CUDA.synchronize()
    end
end
@printf("  CUDA.synchronize (1000x):   %.4f s  (%.1f μs each)\n", t_sync, t_sync * 1000)

# ============================================================
# Summary
# ============================================================

println("\n" * "="^60)
println("SUMMARY")
println("="^60)
@printf("  Non-distributed 10 steps:    %.4f s\n", t_gpu)
@printf("  Distributed 10 steps:        %.4f s\n", t_dist)
@printf("  Overhead ratio:              %.2fx\n", t_dist / t_gpu)
@printf("  Absolute overhead:           %.4f s\n", t_dist - t_gpu)
println()
@printf("  Solver (30x): GPU=%.4f  Dist=%.4f  diff=%.4f s\n",
        t_solve_gpu, t_solve_dist, t_solve_dist - t_solve_gpu)
@printf("  Momentum halo (90x): GPU=%.4f  Dist=%.4f  diff=%.4f s\n",
        t_halo_gpu, t_halo_dist, t_halo_dist - t_halo_gpu)
@printf("  Pressure halo (30x): GPU=%.4f  Dist=%.4f  diff=%.4f s\n",
        t_phalo_gpu, t_phalo_dist, t_phalo_dist - t_phalo_gpu)

total_component_overhead = (t_solve_dist - t_solve_gpu) +
                           (t_halo_dist - t_halo_gpu) +
                           (t_phalo_dist - t_phalo_gpu)
total_overhead = t_dist - t_gpu
@printf("\n  Component overhead sum: %.4f s\n", total_component_overhead)
@printf("  Total overhead:        %.4f s\n", total_overhead)
@printf("  Unaccounted:           %.4f s\n", total_overhead - total_component_overhead)

MPI.Finalize()
