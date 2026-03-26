using MPI
MPI.Init()

using CUDA
using Printf
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
using MPI: VBuffer, Alltoallv!, Alltoall!

rank = MPI.Comm_rank(MPI.COMM_WORLD)
Ngpus = MPI.Comm_size(MPI.COMM_WORLD)
Ngpus < 2 && error("Need at least 2 GPUs")

FT = Float32
Nreps = 100  # more reps for accurate timing

# Create distributed grid and solver storage
arch = Distributed(GPU(); partition=Partition(Ngpus, 1))
Nx = 200 * Ngpus
grid = RectilinearGrid(arch,
                       size = (Nx, 200, 80),
                       x = (0, 84e3 * Ngpus), y = (0, 84e3), z = (0, 20e3),
                       halo = (1, 1, 1),
                       topology = (Periodic, Periodic, Bounded))

global_grid = reconstruct_global_grid(grid)
solver = DistributedFourierTridiagonalPoissonSolver(global_grid, grid)
storage = solver.storage

# Fill with data
parent(storage.yfield) .= CUDA.randn(Complex{FT}, size(storage.yfield)...)

rank == 0 && println("=== Transpose Strategy Benchmark ($Ngpus GPUs) ===")
rank == 0 && println("Data size per rank: $(size(storage.yfield))")

# ============================================================
# Strategy 1: Current Alltoallv (baseline)
# ============================================================

CUDA.synchronize()
MPI.Barrier(MPI.COMM_WORLD)
t1 = CUDA.@elapsed begin
    for _ in 1:Nreps
        transpose_y_to_x!(storage)
        transpose_x_to_y!(storage)
    end
end
MPI.Barrier(MPI.COMM_WORLD)
if rank == 0
    @printf("  1. Alltoallv (current):    %.4f s  (%.2f ms per round-trip)\n", t1, 1000t1/Nreps)
end

# ============================================================
# Strategy 2: Alltoall (equal-size chunks, simpler than Alltoallv)
# ============================================================

# For equal partitions, we can use Alltoall instead of Alltoallv
# Each rank sends count/Ngpus elements to each other rank
send_buf = storage.xybuff.send
recv_buf = storage.xybuff.recv
xy_counts = storage.counts.xy
xy_comm = storage.comms.xy

count_per_rank = xy_counts[1]  # all equal for uniform partition

CUDA.synchronize()
MPI.Barrier(MPI.COMM_WORLD)
t2 = CUDA.@elapsed begin
    for _ in 1:Nreps
        pack_buffer_y_to_x!(storage.xybuff, storage.yfield)
        sync_device!(arch)
        Alltoall!(MPI.UBuffer(send_buf, count_per_rank), MPI.UBuffer(recv_buf, count_per_rank), xy_comm)
        unpack_buffer_x_from_y!(storage.xfield, storage.yfield, storage.xybuff)
        # reverse direction
        pack_buffer_y_to_x!(storage.xybuff, storage.yfield)
        sync_device!(arch)
        Alltoall!(MPI.UBuffer(send_buf, count_per_rank), MPI.UBuffer(recv_buf, count_per_rank), xy_comm)
        unpack_buffer_x_from_y!(storage.xfield, storage.yfield, storage.xybuff)
    end
end
MPI.Barrier(MPI.COMM_WORLD)
if rank == 0
    @printf("  2. Alltoall (equal-size):   %.4f s  (%.2f ms per round-trip)\n", t2, 1000t2/Nreps)
end

# ============================================================
# Strategy 3: Non-blocking Isend/Irecv (manual point-to-point)
# ============================================================

CUDA.synchronize()
MPI.Barrier(MPI.COMM_WORLD)
t3 = CUDA.@elapsed begin
    for _ in 1:Nreps
        # Forward: y → x
        pack_buffer_y_to_x!(storage.xybuff, storage.yfield)
        sync_device!(arch)

        requests = MPI.Request[]
        offset = 0
        for r in 0:Ngpus-1
            count = xy_counts[r+1]
            send_view = view(send_buf, (offset+1):(offset+count))
            recv_view = view(recv_buf, (offset+1):(offset+count))
            push!(requests, MPI.Irecv!(recv_view, xy_comm; source=r))
            push!(requests, MPI.Isend(send_view, xy_comm; dest=r))
            offset += count
        end
        MPI.Waitall(requests)
        unpack_buffer_x_from_y!(storage.xfield, storage.yfield, storage.xybuff)

        # Reverse: x → y (reuse same buffers for simplicity)
        pack_buffer_y_to_x!(storage.xybuff, storage.yfield)
        sync_device!(arch)

        requests = MPI.Request[]
        offset = 0
        for r in 0:Ngpus-1
            count = xy_counts[r+1]
            send_view = view(send_buf, (offset+1):(offset+count))
            recv_view = view(recv_buf, (offset+1):(offset+count))
            push!(requests, MPI.Irecv!(recv_view, xy_comm; source=r))
            push!(requests, MPI.Isend(send_view, xy_comm; dest=r))
            offset += count
        end
        MPI.Waitall(requests)
        unpack_buffer_x_from_y!(storage.xfield, storage.yfield, storage.xybuff)
    end
end
MPI.Barrier(MPI.COMM_WORLD)
if rank == 0
    @printf("  3. Isend/Irecv (manual):   %.4f s  (%.2f ms per round-trip)\n", t3, 1000t3/Nreps)
end

# ============================================================
# Strategy 4: NCCL Send/Recv
# ============================================================

try
    using NCCL

    nccl_id = rank == 0 ? NCCL.UniqueID() : NCCL.UniqueID()
    # Broadcast the unique ID from rank 0
    id_bytes = reinterpret(UInt8, [nccl_id.internal])
    MPI.Bcast!(id_bytes, MPI.COMM_WORLD; root=0)
    nccl_id = NCCL.UniqueID(reinterpret(NTuple{128, UInt8}, id_bytes)[1])

    nccl_comm = NCCL.Communicator(Ngpus, nccl_id, rank)

    CUDA.synchronize()
    MPI.Barrier(MPI.COMM_WORLD)
    t4 = CUDA.@elapsed begin
        for _ in 1:Nreps
            # Forward: y → x
            pack_buffer_y_to_x!(storage.xybuff, storage.yfield)
            CUDA.synchronize()

            NCCL.groupStart()
            offset = 0
            for r in 0:Ngpus-1
                count = xy_counts[r+1]
                send_view = view(send_buf, (offset+1):(offset+count))
                recv_view = view(recv_buf, (offset+1):(offset+count))
                NCCL.Recv!(recv_view, nccl_comm; peer=r)
                NCCL.Send(send_view, nccl_comm; peer=r)
                offset += count
            end
            NCCL.groupEnd()
            unpack_buffer_x_from_y!(storage.xfield, storage.yfield, storage.xybuff)

            # Reverse: x → y
            pack_buffer_y_to_x!(storage.xybuff, storage.yfield)
            CUDA.synchronize()

            NCCL.groupStart()
            offset = 0
            for r in 0:Ngpus-1
                count = xy_counts[r+1]
                send_view = view(send_buf, (offset+1):(offset+count))
                recv_view = view(recv_buf, (offset+1):(offset+count))
                NCCL.Recv!(recv_view, nccl_comm; peer=r)
                NCCL.Send(send_view, nccl_comm; peer=r)
                offset += count
            end
            NCCL.groupEnd()
            unpack_buffer_x_from_y!(storage.xfield, storage.yfield, storage.xybuff)
        end
    end
    MPI.Barrier(MPI.COMM_WORLD)
    if rank == 0
        @printf("  4. NCCL Send/Recv:         %.4f s  (%.2f ms per round-trip)\n", t4, 1000t4/Nreps)
    end
catch e
    rank == 0 && @printf("  4. NCCL: FAILED (%s)\n", sprint(showerror, e))
end

# ============================================================
# Strategy 5: Raw bandwidth test (CUDA memcpy baseline)
# ============================================================

data_size = length(send_buf)
temp = CUDA.zeros(eltype(send_buf), data_size)

CUDA.synchronize()
t5 = CUDA.@elapsed begin
    for _ in 1:Nreps
        copyto!(temp, send_buf)
        copyto!(send_buf, temp)
    end
end
bytes_moved = 2 * data_size * sizeof(eltype(send_buf)) * Nreps
bw = bytes_moved / t5 / 1e9
if rank == 0
    @printf("  5. GPU memcpy baseline:    %.4f s  (%.2f ms per round-trip, %.1f GB/s)\n", t5, 1000t5/Nreps, bw)
end

# ============================================================
# Summary
# ============================================================

if rank == 0
    println("\n=== Summary ===")
    @printf("  Alltoallv (current): %.2f ms\n", 1000t1/Nreps)
    @printf("  Alltoall:            %.2f ms (%.1fx vs current)\n", 1000t2/Nreps, t1/t2)
    @printf("  Isend/Irecv:         %.2f ms (%.1fx vs current)\n", 1000t3/Nreps, t1/t3)
    @printf("  GPU memcpy:          %.2f ms (theoretical floor)\n", 1000t5/Nreps)
end

rank == 0 && println("\nDone.")
MPI.Finalize()
