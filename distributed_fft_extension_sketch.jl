#=
Sketch: OceananigansDistributedFFTExt — async distributed FFT for Poisson solver

Goal: Replace the synchronous transpose-based distributed FFT in
DistributedFourierTridiagonalPoissonSolver with an async pipeline
that overlaps MPI communication with FFT computation.

Current approach (synchronous, slab-x Z-stretched):
  y-FFT → [pack → sync → Alltoall → unpack] → x-FFT → tridiag → x-IFFT → [pack → sync → Alltoall → unpack] → y-IFFT

Proposed approach (async, slab-x Z-stretched):
  y-FFT → [Isend/Irecv (non-blocking)] → x-FFT on arrived chunks → tridiag → x-IFFT → [Isend/Irecv] → y-IFFT

Key insight: For slab-x with Rx ranks, each rank sends Rx-1 chunks and keeps 1 local chunk.
The local chunk can be FFT'd immediately while waiting for remote chunks. As remote chunks
arrive, they can be FFT'd incrementally.

Architecture:
  OceananigansDistributedFFTExt (extension package)
  └── AsyncDistributedFourierTridiagonalPoissonSolver
      ├── async_transpose!  (non-blocking MPI + incremental FFT)
      ├── solve!            (pipelined forward FFT → tridiag → backward FFT)
      └── AsyncTransposableField (double-buffered for overlap)
=#

module OceananigansDistributedFFTExt

using Oceananigans
using Oceananigans.Solvers
using Oceananigans.DistributedComputations
using MPI
using CUDA

#####
##### Core type: replaces DistributedFourierTridiagonalPoissonSolver
#####

struct AsyncDistributedFFTSolver{G, L, B, P, R, S, β, C}
    plan :: P
    global_grid :: G
    local_grid :: L
    batched_tridiagonal_solver :: B
    source_term :: R
    storage :: S
    buffer :: β
    # NEW: communication state for async MPI
    comm_state :: C
end

#####
##### Communication state for async transpose
#####

struct AsyncCommState{SB, RB, RQ, CK}
    send_buffers :: SB    # Vector of per-rank send buffers (GPU arrays)
    recv_buffers :: RB    # Vector of per-rank recv buffers (GPU arrays)
    requests :: RQ        # MPI request storage
    chunk_sizes :: CK     # Elements per rank
end

function AsyncCommState(storage, arch, Rx)
    # Pre-allocate per-rank send/recv buffers for non-blocking MPI
    T = eltype(parent(storage.yfield))
    total = length(parent(storage.yfield))
    chunk = total ÷ Rx

    send_buffers = [CUDA.zeros(T, chunk) for _ in 1:Rx]
    recv_buffers = [CUDA.zeros(T, chunk) for _ in 1:Rx]
    requests = MPI.Request[]
    chunk_sizes = fill(chunk, Rx)

    return AsyncCommState(send_buffers, recv_buffers, requests, chunk_sizes)
end

#####
##### Async transpose: non-blocking send/recv with local shortcut
#####

"""
    async_transpose_y_to_x!(storage, comm_state, comm)

Non-blocking transpose from y-local to x-local configuration.

Algorithm:
1. Pack data into per-rank chunks
2. Local chunk: copy directly (no MPI)
3. Remote chunks: Isend + Irecv (non-blocking)
4. Return immediately — caller must Waitall before using remote data

This allows the caller to do useful work (e.g., FFT on the local chunk)
while remote data is in flight.
"""
function async_transpose_y_to_x!(storage, comm_state, comm, my_rank, Rx)
    pack_into_chunks!(comm_state.send_buffers, storage.yfield, Rx)
    CUDA.synchronize()  # Ensure pack is complete before MPI

    empty!(comm_state.requests)

    for r in 0:Rx-1
        if r == my_rank
            # Local: direct copy, no MPI
            copyto!(comm_state.recv_buffers[r+1], comm_state.send_buffers[r+1])
        else
            # Remote: non-blocking send/recv
            push!(comm_state.requests,
                  MPI.Irecv!(comm_state.recv_buffers[r+1], comm, source=r))
            push!(comm_state.requests,
                  MPI.Isend(comm_state.send_buffers[r+1], comm, dest=r))
        end
    end

    return nothing  # Caller must MPI.Waitall(comm_state.requests) later
end

"""
Wait for async transpose to complete, then unpack.
"""
function complete_transpose_y_to_x!(storage, comm_state, Rx)
    MPI.Waitall(comm_state.requests)
    unpack_from_chunks!(storage.xfield, comm_state.recv_buffers, Rx)
    return nothing
end

#####
##### Pipelined solve: overlap communication with computation
#####

"""
    solve!(x, solver::AsyncDistributedFFTSolver)

Pipelined solve for slab-x Z-stretched:

1. y-FFT (fully local, no communication)
2. Start async transpose y→x (non-blocking MPI)
3. While transpose in flight: FFT the local chunk immediately
4. When remote chunks arrive: FFT them
5. Tridiagonal solve (all data now in x-local spectral space)
6. Start async transpose x→y (non-blocking MPI)
7. While transpose in flight: IFFT the local chunk
8. When remote chunks arrive: IFFT them
9. y-IFFT (fully local)

This overlaps MPI latency with FFT computation.
"""
function Oceananigans.Solvers.solve!(x, solver::AsyncDistributedFFTSolver)
    arch = Oceananigans.Architectures.architecture(solver)
    storage = solver.storage
    buffer = solver.buffer
    cs = solver.comm_state

    Rx = arch.ranks[1]
    my_rank = arch.local_rank
    comm = storage.comms.xy

    # ── Forward pass ──

    # Step 1: y-FFT (local, y is not distributed in slab-x)
    solver.plan.forward.y!(parent(storage.yfield), buffer.y)

    # Step 2: Start async transpose y→x
    async_transpose_y_to_x!(storage, cs, comm, my_rank, Rx)

    # Step 3: FFT the local chunk immediately (don't wait for MPI)
    local_chunk = cs.recv_buffers[my_rank + 1]
    fft_chunk!(local_chunk, solver.plan.forward.x!)

    # Step 4: Wait for remote chunks + FFT them
    complete_transpose_y_to_x!(storage, cs, Rx)
    solver.plan.forward.x!(parent(storage.xfield), buffer.x)

    # Step 5: Tridiagonal solve in x-local space
    parent(solver.source_term) .= parent(storage.xfield)
    Oceananigans.Solvers.solve!(storage.xfield,
                                 solver.batched_tridiagonal_solver,
                                 solver.source_term)

    # ── Backward pass ──

    # Step 6: x-IFFT
    solver.plan.backward.x!(parent(storage.xfield), buffer.x)

    # Step 7: Start async transpose x→y
    async_transpose_x_to_y!(storage, cs, comm, my_rank, Rx)

    # Step 8: Wait + unpack
    complete_transpose_x_to_y!(storage, cs, Rx)

    # Step 9: y-IFFT (local)
    solver.plan.backward.y!(parent(storage.yfield), buffer.y)

    # Copy real component
    Oceananigans.Utils.launch!(arch, solver.local_grid, :xyz,
                                _copy_real_component!, x, parent(storage.zfield))

    return x
end

#####
##### Pack/unpack helpers (per-rank chunks for non-blocking MPI)
#####

"""
Pack a 3D field into per-rank 1D chunks for non-blocking send.
Unlike the current Alltoall approach which packs everything into one buffer,
this creates separate buffers per rank to enable incremental processing.
"""
function pack_into_chunks!(chunks, field, Rx)
    # Each chunk contains the portion of data destined for rank r
    # Layout depends on the transpose direction
    # For y→x: split along y dimension
    Ny_local = size(field, 2)
    chunk_ny = Ny_local ÷ Rx

    for r in 1:Rx
        j_start = (r-1) * chunk_ny + 1
        j_end = r * chunk_ny
        # Pack field[:, j_start:j_end, :] into chunks[r]
        _pack_chunk!(chunks[r], parent(field), j_start, j_end)
    end
    return nothing
end

# GPU kernel for packing
@kernel function _pack_chunk_kernel!(chunk, field, j_start, Nx, chunk_ny, Nz)
    idx = @index(Global)
    # Convert linear index to (i, j_local, k)
    i = ((idx - 1) % Nx) + 1
    j_local = (((idx - 1) ÷ Nx) % chunk_ny) + 1
    k = ((idx - 1) ÷ (Nx * chunk_ny)) + 1
    j = j_start + j_local - 1
    @inbounds chunk[idx] = field[i, j, k]
end

#####
##### Extension point: construct from existing solver
#####

"""
    AsyncDistributedFFTSolver(global_grid, local_grid; kw...)

Construct an async distributed FFT solver. Drop-in replacement for
DistributedFourierTridiagonalPoissonSolver with pipelined MPI.
"""
function AsyncDistributedFFTSolver(global_grid, local_grid, planner_flag=FFTW.PATIENT;
                                    tridiagonal_formulation=nothing)

    # Reuse the existing constructor logic for plans, eigenvalues, etc.
    # Just wrap the result with async communication state
    base_solver = DistributedFourierTridiagonalPoissonSolver(global_grid, local_grid,
                                                              planner_flag;
                                                              tridiagonal_formulation)

    Rx, _, _ = local_grid.architecture.ranks
    comm_state = AsyncCommState(base_solver.storage, local_grid.architecture, Rx)

    return AsyncDistributedFFTSolver(base_solver.plan,
                                      base_solver.global_grid,
                                      base_solver.local_grid,
                                      base_solver.batched_tridiagonal_solver,
                                      base_solver.source_term,
                                      base_solver.storage,
                                      base_solver.buffer,
                                      comm_state)
end

#####
##### Alternative: NCCL-based transpose (future work)
#####

#=
For even better intra-node performance, replace MPI with NCCL:

using NCCL

function nccl_transpose_y_to_x!(storage, nccl_comm, Rx)
    NCCL.groupStart()
    for r in 0:Rx-1
        NCCL.Send(send_chunk[r], nccl_comm; peer=r)
        NCCL.Recv!(recv_chunk[r], nccl_comm; peer=r)
    end
    NCCL.groupEnd()
    # NCCL operations are async on the GPU stream — no CPU sync needed
end

Benefits:
- NCCL uses NVLink directly (no MPI/GTL overhead)
- Operations are async on the GPU stream (no sync_device! needed)
- GroupStart/GroupEnd batches all send/recv into one operation
- Avoids the Alltoallv vs Alltoall performance cliff we discovered
=#

end # module

#####
##### How to register as an Oceananigans extension
#####

#=
In the extension's Project.toml:

[deps]
Oceananigans = "9e8cae18-63c1-5223-a75c-80ca9d6e9a09"
MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"

Then in Oceananigans' Project.toml, add:

[extensions]
OceananigansDistributedFFTExt = ["MPI", "CUDA"]

The extension would override the solver construction:

# In Oceananigans' DistributedComputations.jl:
function fft_poisson_solver(local_grid::DistributedRectilinearGrid,
                            global_grid::GridWithFourierTridiagonalSolver)
    # Check if async extension is loaded
    if _has_async_fft_extension()
        return AsyncDistributedFFTSolver(global_grid, local_grid)
    else
        return DistributedFourierTridiagonalPoissonSolver(global_grid, local_grid)
    end
end
=#

#####
##### Performance model: expected improvement
#####

#=
Current (synchronous, slab-x with Alltoall):
  Per solve = y-FFT + [pack + sync + Alltoall + unpack] + x-FFT + tridiag
            + x-IFFT + [pack + sync + Alltoall + unpack] + y-IFFT

  Measured on 2 GPUs (200×200×80):
    Alltoall round-trip: 1.85 ms
    FFT (per call): 0.85 ms
    Tridiag: ~5 ms
    Total: ~36 ms/solve

Async approach:
  Per solve = y-FFT + [Isend/Irecv + local_FFT (overlapped)] + remote_FFT + tridiag
            + x-IFFT + [Isend/Irecv + local_IFFT (overlapped)] + remote_IFFT + y-IFFT

  Expected: overlap ~1.85 ms of MPI latency with ~0.85 ms of FFT
  → save ~1.7 ms per transpose × 2 transposes = ~3.4 ms
  → ~32.6 ms/solve (modest 10% improvement)

The bigger win is for MULTI-NODE where Alltoall latency is ~8.7 ms:
  → overlap 8.7 ms with 0.85 ms FFT → save ~0.85 ms per transpose
  → Actually limited by: max(MPI_time, FFT_time) instead of MPI_time + FFT_time

NCCL approach (no sync_device! needed):
  Per solve = y-FFT + [NCCL grouped send/recv (GPU-async)] + x-FFT + tridiag + ...
  → eliminate sync_device! entirely (NCCL is GPU-stream-native)
  → estimated save: 2 × sync_device overhead per solve
=#
