using MPI
MPI.Init()

using CUDA
using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.Solvers: solve!, FourierTridiagonalPoissonSolver
using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: DistributedFourierTridiagonalPoissonSolver,
                                            reconstruct_global_grid,
                                            transpose_y_to_x!,
                                            transpose_x_to_y!,
                                            pack_buffer_y_to_x!,
                                            unpack_buffer_x_from_y!,
                                            sync_device!

rank = MPI.Comm_rank(MPI.COMM_WORLD)
Ngpus = MPI.Comm_size(MPI.COMM_WORLD)

FT = Float32

# Grid size per GPU (weak scaling)
Nx_per_gpu = 200
Ny = 200
Nz = 80
Lx_per_gpu = 84kilometers
Ly = 84kilometers
Lz = 20kilometers

Nsolves = 30  # ≈ 3 RK3 stages × 10 timesteps

# ============================================================
# Phase A: Non-distributed solver (1 GPU reference)
# Only run on rank 0 when Ngpus == 1
# ============================================================

if Ngpus == 1
    println("=== Phase A: Non-distributed FourierTridiagonalPoissonSolver (1 GPU) ===")
    grid = RectilinearGrid(GPU(),
                           size = (Nx_per_gpu, Ny, Nz),
                           x = (0, Lx_per_gpu), y = (0, Ly), z = (0, Lz),
                           halo = (1, 1, 1),
                           topology = (Periodic, Periodic, Bounded))

    solver = FourierTridiagonalPoissonSolver(grid)
    println("  Solver type: ", typeof(solver).name.name)
    println("  Grid: ", size(grid))

    # Fill source term with random data on GPU
    solver.source_term .= CUDA.randn(Complex{FT}, size(solver.source_term)...)
    p = CenterField(grid)

    # Warmup
    solve!(p, solver)
    CUDA.synchronize()

    # Benchmark
    CUDA.synchronize()
    t = CUDA.@elapsed begin
        for _ in 1:Nsolves
            solve!(p, solver)
        end
    end
    @printf("  %d solves: %.4f s  (%.2f ms/solve)\n", Nsolves, t, 1000t/Nsolves)

    # Also time components
    CUDA.synchronize()
    t_fft = CUDA.@elapsed begin
        for _ in 1:Nsolves
            for transform! in solver.transforms.forward
                transform!(solver.source_term, solver.buffer)
            end
            for transform! in solver.transforms.backward
                transform!(solver.storage, solver.buffer)
            end
        end
    end
    @printf("  FFTs only (%d×forward+backward): %.4f s\n", Nsolves, t_fft)
end

# ============================================================
# Phase B: Distributed solver
# ============================================================

if Ngpus > 1
    rank == 0 && println("=== Phase B: DistributedFourierTridiagonalPoissonSolver ($Ngpus GPUs) ===")

    arch = Distributed(GPU(); partition = Partition(Ngpus, 1))
    Nx = Nx_per_gpu * Ngpus
    Lx = Lx_per_gpu * Ngpus

    grid = RectilinearGrid(arch,
                           size = (Nx, Ny, Nz),
                           x = (0, Lx), y = (0, Ly), z = (0, Lz),
                           halo = (1, 1, 1),
                           topology = (Periodic, Periodic, Bounded))

    global_grid = reconstruct_global_grid(grid)
    solver = DistributedFourierTridiagonalPoissonSolver(global_grid, grid)

    rank == 0 && println("  Solver type: ", typeof(solver).name.name)
    rank == 0 && println("  Local grid: ", size(grid))
    rank == 0 && println("  Global grid: ", size(global_grid))

    # Fill source term
    parent(solver.storage.zfield) .= CUDA.randn(Complex{FT}, size(solver.storage.zfield)...)
    p = CenterField(grid)

    # Warmup
    MPI.Barrier(MPI.COMM_WORLD)
    solve!(p, solver)
    CUDA.synchronize()
    MPI.Barrier(MPI.COMM_WORLD)

    # Benchmark
    CUDA.synchronize()
    MPI.Barrier(MPI.COMM_WORLD)
    t = CUDA.@elapsed begin
        for _ in 1:Nsolves
            solve!(p, solver)
        end
    end
    MPI.Barrier(MPI.COMM_WORLD)

    if rank == 0
        @printf("  %d solves: %.4f s  (%.2f ms/solve)\n", Nsolves, t, 1000t/Nsolves)
    end

    # Time individual components
    storage = solver.storage
    buffer = solver.buffer

    # Time just the transposes (MPI communication)
    CUDA.synchronize()
    MPI.Barrier(MPI.COMM_WORLD)
    t_transpose = CUDA.@elapsed begin
        for _ in 1:Nsolves
            # Forward transposes (that actually do MPI for slab-x)
            transpose_y_to_x!(storage)
            transpose_x_to_y!(storage)
            # Backward transposes
            transpose_y_to_x!(storage)
            transpose_x_to_y!(storage)
        end
    end
    MPI.Barrier(MPI.COMM_WORLD)

    if rank == 0
        @printf("  Transposes only (4×%d): %.4f s  (%.2f ms per transpose)\n",
                Nsolves, t_transpose, 1000t_transpose/(4*Nsolves))
    end

    # Time just the FFTs (no transpose)
    CUDA.synchronize()
    t_fft = CUDA.@elapsed begin
        for _ in 1:Nsolves
            solver.plan.forward.y!(parent(storage.yfield), buffer.y)
            solver.plan.forward.x!(parent(storage.xfield), buffer.x)
            solver.plan.backward.x!(parent(storage.xfield), buffer.x)
            solver.plan.backward.y!(parent(storage.yfield), buffer.y)
        end
    end

    if rank == 0
        @printf("  FFTs only (4×%d): %.4f s  (%.2f ms per FFT)\n",
                Nsolves, t_fft, 1000t_fft/(4*Nsolves))
    end

    # Time just one Alltoallv round-trip (the raw MPI cost)
    using MPI: VBuffer, Alltoallv!

    CUDA.synchronize()
    MPI.Barrier(MPI.COMM_WORLD)
    t_alltoallv = CUDA.@elapsed begin
        for _ in 1:Nsolves
            pack_buffer_y_to_x!(storage.xybuff, storage.yfield)
            sync_device!(arch)
            Alltoallv!(VBuffer(storage.xybuff.send, storage.counts.xy),
                       VBuffer(storage.xybuff.recv, storage.counts.xy),
                       storage.comms.xy)
            unpack_buffer_x_from_y!(storage.xfield, storage.yfield, storage.xybuff)
        end
    end
    MPI.Barrier(MPI.COMM_WORLD)

    if rank == 0
        @printf("  Single transpose breakdown (%d):\n", Nsolves)
        @printf("    pack+sync+alltoallv+unpack: %.4f s  (%.2f ms each)\n",
                t_alltoallv, 1000t_alltoallv/Nsolves)
    end

    # Time just sync_device
    CUDA.synchronize()
    t_sync = @elapsed begin
        for _ in 1:1000
            sync_device!(arch)
        end
    end
    if rank == 0
        @printf("  sync_device! (1000×): %.4f s  (%.1f μs each)\n", t_sync, t_sync*1000)
    end
end

rank == 0 && println("\nDone.")
MPI.Finalize()
