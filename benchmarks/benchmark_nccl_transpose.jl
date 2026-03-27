using MPI
MPI.Init()

using CUDA
using Printf
using NCCL
using Oceananigans
using Oceananigans.Units
using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: DistributedFourierTridiagonalPoissonSolver,
                                            reconstruct_global_grid,
                                            TransposableField,
                                            transpose_y_to_x!,
                                            transpose_x_to_y!,
                                            pack_buffer_y_to_x!,
                                            unpack_buffer_x_from_y!,
                                            sync_device!
using Oceananigans.Solvers: solve!
using MPI: Alltoall!, UBuffer

rank = MPI.Comm_rank(MPI.COMM_WORLD)
Ngpus = MPI.Comm_size(MPI.COMM_WORLD)
Ngpus < 2 && error("Need at least 2 GPUs")

FT = Float32
Nreps = 100

# ============================================================
# Phase 1: Create NCCL communicator from MPI
# ============================================================

rank == 0 && println("=== Phase 1: NCCL Communicator Setup ===")

# Create NCCL UniqueID on rank 0, broadcast via MPI
if rank == 0
    nccl_id = NCCL.UniqueID()
    # Extract raw bytes as a plain Vector{UInt8}
    id_bytes = Vector{UInt8}(undef, 128)
    id_ref = Ref(nccl_id.internal)
    unsafe_copyto!(pointer(id_bytes), Ptr{UInt8}(pointer_from_objref(id_ref)), 128)
else
    id_bytes = Vector{UInt8}(undef, 128)
end

MPI.Bcast!(id_bytes, MPI.COMM_WORLD; root=0)

# Reconstruct UniqueID on all ranks
nccl_internal = ntuple(i -> id_bytes[i], Val(128))
nccl_id_all = NCCL.UniqueID(nccl_internal)
nccl_comm = NCCL.Communicator(Ngpus, nccl_id_all, rank)

rank == 0 && println("  NCCL communicator created: $(Ngpus) ranks")
rank == 0 && println("  NCCL version: $(NCCL.version())")

# ============================================================
# Setup: Create distributed grid and solver
# ============================================================

rank == 0 && println("\n=== Setup ===")

arch = Distributed(GPU(); partition = Partition(Ngpus, 1))
Nx = 200 * Ngpus
grid = RectilinearGrid(arch,
                       size = (Nx, 200, 80),
                       x = (0, 84e3 * Ngpus), y = (0, 84e3), z = (0, 20e3),
                       halo = (1, 1, 1),
                       topology = (Periodic, Periodic, Bounded))

global_grid = reconstruct_global_grid(grid)
solver = DistributedFourierTridiagonalPoissonSolver(global_grid, grid)
storage = solver.storage

parent(storage.yfield) .= CUDA.randn(Complex{FT}, size(storage.yfield)...)

rank == 0 && println("  Local grid: $(size(grid))")
rank == 0 && println("  Data size: $(size(storage.yfield))")

# ============================================================
# Get buffer info for manual transposes
# ============================================================

xy_counts = storage.counts.xy
xy_comm = storage.comms.xy
count_per_rank = xy_counts[1]  # all equal for uniform partition

# Create per-rank NCCL buffers
total_elements = length(parent(storage.yfield))
chunk_size = total_elements ÷ Ngpus

nccl_send_bufs = [CUDA.zeros(Complex{FT}, chunk_size) for _ in 1:Ngpus]
nccl_recv_bufs = [CUDA.zeros(Complex{FT}, chunk_size) for _ in 1:Ngpus]

rank == 0 && println("  Chunk size per rank: $(chunk_size) elements ($(chunk_size * sizeof(Complex{FT}) / 1e6) MB)")

# ============================================================
# Benchmark 1: Current MPI Alltoall transpose (baseline)
# ============================================================

rank == 0 && println("\n=== Benchmark: MPI Alltoall Transpose ===")

# Warmup
for _ in 1:5
    transpose_y_to_x!(storage)
    transpose_x_to_y!(storage)
end
CUDA.synchronize()
MPI.Barrier(MPI.COMM_WORLD)

t_mpi = CUDA.@elapsed begin
    for _ in 1:Nreps
        transpose_y_to_x!(storage)
        transpose_x_to_y!(storage)
    end
end
MPI.Barrier(MPI.COMM_WORLD)

if rank == 0
    @printf("  MPI Alltoall round-trip (%d reps): %.4f s (%.2f ms each)\n",
            Nreps, t_mpi, 1000t_mpi/Nreps)
end

# ============================================================
# Benchmark 2: NCCL Send/Recv transpose
# ============================================================

rank == 0 && println("\n=== Benchmark: NCCL Send/Recv Transpose ===")

send_buf = storage.xybuff.send
recv_buf = storage.xybuff.recv

# Warmup NCCL
for _ in 1:5
    pack_buffer_y_to_x!(storage.xybuff, storage.yfield)
    CUDA.synchronize()
    NCCL.groupStart()
    for r in 0:Ngpus-1
        offset = r * count_per_rank
        send_view = view(send_buf, (offset+1):(offset+count_per_rank))
        recv_view = view(recv_buf, (offset+1):(offset+count_per_rank))
        NCCL.Send(send_view, nccl_comm; peer=r)
        NCCL.Recv!(recv_view, nccl_comm; peer=r)
    end
    NCCL.groupEnd()
    unpack_buffer_x_from_y!(storage.xfield, storage.yfield, storage.xybuff)
end
CUDA.synchronize()
MPI.Barrier(MPI.COMM_WORLD)

# Benchmark: NCCL with sync_device (apples-to-apples with MPI)
t_nccl_sync = CUDA.@elapsed begin
    for _ in 1:Nreps
        # Forward: y → x
        pack_buffer_y_to_x!(storage.xybuff, storage.yfield)
        CUDA.synchronize()  # Same as MPI path for fair comparison
        NCCL.groupStart()
        for r in 0:Ngpus-1
            offset = r * count_per_rank
            send_view = view(send_buf, (offset+1):(offset+count_per_rank))
            recv_view = view(recv_buf, (offset+1):(offset+count_per_rank))
            NCCL.Send(send_view, nccl_comm; peer=r)
            NCCL.Recv!(recv_view, nccl_comm; peer=r)
        end
        NCCL.groupEnd()
        unpack_buffer_x_from_y!(storage.xfield, storage.yfield, storage.xybuff)

        # Reverse: x → y (same pattern)
        pack_buffer_y_to_x!(storage.xybuff, storage.yfield)
        CUDA.synchronize()
        NCCL.groupStart()
        for r in 0:Ngpus-1
            offset = r * count_per_rank
            send_view = view(send_buf, (offset+1):(offset+count_per_rank))
            recv_view = view(recv_buf, (offset+1):(offset+count_per_rank))
            NCCL.Send(send_view, nccl_comm; peer=r)
            NCCL.Recv!(recv_view, nccl_comm; peer=r)
        end
        NCCL.groupEnd()
        unpack_buffer_x_from_y!(storage.xfield, storage.yfield, storage.xybuff)
    end
end
MPI.Barrier(MPI.COMM_WORLD)

if rank == 0
    @printf("  NCCL with sync (%d reps): %.4f s (%.2f ms each)\n",
            Nreps, t_nccl_sync, 1000t_nccl_sync/Nreps)
end

# Benchmark: NCCL WITHOUT sync_device (the real advantage — stream-native)
t_nccl_nosync = CUDA.@elapsed begin
    for _ in 1:Nreps
        pack_buffer_y_to_x!(storage.xybuff, storage.yfield)
        # NO sync_device! — NCCL is stream-ordered, reads from same stream
        NCCL.groupStart()
        for r in 0:Ngpus-1
            offset = r * count_per_rank
            send_view = view(send_buf, (offset+1):(offset+count_per_rank))
            recv_view = view(recv_buf, (offset+1):(offset+count_per_rank))
            NCCL.Send(send_view, nccl_comm; peer=r)
            NCCL.Recv!(recv_view, nccl_comm; peer=r)
        end
        NCCL.groupEnd()
        unpack_buffer_x_from_y!(storage.xfield, storage.yfield, storage.xybuff)

        pack_buffer_y_to_x!(storage.xybuff, storage.yfield)
        NCCL.groupStart()
        for r in 0:Ngpus-1
            offset = r * count_per_rank
            send_view = view(send_buf, (offset+1):(offset+count_per_rank))
            recv_view = view(recv_buf, (offset+1):(offset+count_per_rank))
            NCCL.Send(send_view, nccl_comm; peer=r)
            NCCL.Recv!(recv_view, nccl_comm; peer=r)
        end
        NCCL.groupEnd()
        unpack_buffer_x_from_y!(storage.xfield, storage.yfield, storage.xybuff)
    end
end
MPI.Barrier(MPI.COMM_WORLD)

if rank == 0
    @printf("  NCCL no sync (%d reps):   %.4f s (%.2f ms each)\n",
            Nreps, t_nccl_nosync, 1000t_nccl_nosync/Nreps)
end

# ============================================================
# Benchmark 3: Full solver comparison
# ============================================================

rank == 0 && println("\n=== Benchmark: Full Pressure Solver ===")

Nsolves = 30
p = CenterField(grid)

# MPI solver (current)
parent(solver.storage.zfield) .= CUDA.randn(Complex{FT}, size(solver.storage.zfield)...)
solve!(p, solver)  # warmup
CUDA.synchronize()
MPI.Barrier(MPI.COMM_WORLD)

t_solver_mpi = CUDA.@elapsed begin
    for _ in 1:Nsolves
        solve!(p, solver)
    end
end
MPI.Barrier(MPI.COMM_WORLD)

if rank == 0
    @printf("  MPI solver (%d solves): %.4f s (%.2f ms/solve)\n",
            Nsolves, t_solver_mpi, 1000t_solver_mpi/Nsolves)
end

# ============================================================
# Summary
# ============================================================

if rank == 0
    println("\n" * "="^60)
    println("SUMMARY ($Ngpus GPUs)")
    println("="^60)
    @printf("  MPI Alltoall round-trip:  %.2f ms\n", 1000t_mpi/Nreps)
    @printf("  NCCL (with sync):        %.2f ms  (%.1fx vs MPI)\n",
            1000t_nccl_sync/Nreps, t_mpi/t_nccl_sync)
    @printf("  NCCL (no sync):          %.2f ms  (%.1fx vs MPI)\n",
            1000t_nccl_nosync/Nreps, t_mpi/t_nccl_nosync)
    @printf("  MPI solver:              %.2f ms/solve\n", 1000t_solver_mpi/Nsolves)
    println()
    @printf("  GPU memcpy floor:        ~1.0 ms (from earlier benchmarks)\n")
end

rank == 0 && println("\nDone.")
MPI.Finalize()
