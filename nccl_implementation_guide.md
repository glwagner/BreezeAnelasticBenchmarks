# NCCL Distributed FFT Solver — Implementation Guide

## Overview

Replace the MPI `Alltoall`-based transpose in Oceananigans'
`DistributedFourierTridiagonalPoissonSolver` with NCCL `Send`/`Recv`.
NCCL operations are GPU-stream-native (no `sync_device!` / `CUDA.synchronize()`
needed), eliminating the pipeline stalls that dominate the current solver cost.

## What we know from benchmarking

On Derecho A100s with the current optimized MPI solver (200×200×80/GPU, slab-x):

| Component | 1 GPU | 2 GPU | Notes |
|-----------|-------|-------|-------|
| Full solver | 8.4 ms | 36.2 ms | 4.3x overhead |
| MPI Alltoall transpose (×2) | — | ~3.7 ms | Fast after Alltoallv→Alltoall fix |
| Pipeline stalls from sync_device! | — | ~25 ms | **Dominant cost** |
| FFTs (×4) | — | ~3.4 ms | |
| Tridiag solve | ~5 ms | ~5 ms | Same |

The `sync_device!` before each `Alltoall` drains the GPU pipeline, forcing all
previously queued kernels to complete. With 2 transposes per solve, that's 2 full
pipeline flushes per solve × 3 RK3 stages × 100 timesteps.

## The fix: NCCL is stream-native

```julia
# Current (MPI): requires sync_device! before collective
pack_buffer!(buffer, field)
sync_device!(arch)            # ← PIPELINE FLUSH — this is the bottleneck
Alltoall!(UBuffer(send, n), UBuffer(recv, n), comm)
unpack_buffer!(field, buffer)

# Proposed (NCCL): no sync needed — stream-ordered
pack_buffer!(buffer, field)
# NO sync_device! — NCCL reads from same CUDA stream
NCCL.groupStart()
for r in 0:Rx-1
    NCCL.Send(send_chunk[r], nccl_comm; peer=r)
    NCCL.Recv!(recv_chunk[r], nccl_comm; peer=r)
end
NCCL.groupEnd()
# NCCL enqueues on the CUDA stream — unpack kernel waits automatically
unpack_buffer!(field, buffer)
```

## Step-by-step implementation

### Step 1: NCCL communicator from MPI

Create NCCL communicators that mirror the MPI subcommunicators used for transposes.
The `TransposableField` has two MPI subcommunicators: `comms.xy` and `comms.yz`.
For slab-x, only `comms.xy` is used.

```julia
using NCCL
using MPI

function create_nccl_comm_from_mpi(mpi_comm)
    nranks = MPI.Comm_size(mpi_comm)
    rank = MPI.Comm_rank(mpi_comm)

    # Rank 0 creates unique ID
    if rank == 0
        nccl_id = NCCL.UniqueID()
        id_bytes = Vector{UInt8}(undef, 128)
        unsafe_copyto!(pointer(id_bytes),
                       Ptr{UInt8}(pointer_from_objref(Ref(nccl_id.internal))), 128)
    else
        id_bytes = Vector{UInt8}(undef, 128)
    end

    # Broadcast via MPI (one-time cost at solver construction)
    MPI.Bcast!(id_bytes, mpi_comm; root=0)

    # All ranks reconstruct the UniqueID and create NCCL comm
    nccl_internal = ntuple(i -> id_bytes[i], Val(128))
    nccl_id_all = NCCL.UniqueID(nccl_internal)
    return NCCL.Communicator(nranks, nccl_id_all, rank)
end
```

**Note:** The `ntuple(i -> id_bytes[i], Val(128))` creates a `NTuple{128, UInt8}`.
NCCL.UniqueID expects `NTuple{128, UInt8}` but the internal type may be
`NTuple{128, Int8}`. Check the NCCL.jl source and cast if needed.

### Step 2: NCCL transpose function

Replace the generated `transpose!` functions in `distributed_transpose.jl`.
The key change: no `sync_device!`, use NCCL grouped Send/Recv.

```julia
function nccl_transpose!(pack_buffer!, unpack_buffer!,
                          buffer, fromfield, tofield,
                          counts, nccl_comm, Rx)
    # Pack data into 1D buffer (GPU kernel)
    pack_buffer!(buffer, fromfield)

    # NCCL grouped send/recv — stream-ordered, no sync needed
    count_per_rank = counts[1]  # all equal for uniform partition
    NCCL.groupStart()
    for r in 0:Rx-1
        offset = r * count_per_rank
        send_view = view(buffer.send, (offset+1):(offset+count_per_rank))
        recv_view = view(buffer.recv, (offset+1):(offset+count_per_rank))
        NCCL.Send(send_view, nccl_comm; peer=r)
        NCCL.Recv!(recv_view, nccl_comm; peer=r)
    end
    NCCL.groupEnd()

    # Unpack (GPU kernel — implicitly waits for NCCL to complete on same stream)
    unpack_buffer!(tofield, fromfield, buffer)
    return nothing
end
```

### Step 3: Wire into the solver

Create a new solver type or modify the existing one to use NCCL transposes.
The simplest approach: add NCCL comm to the solver struct and dispatch.

```julia
# In the DistributedFourierTridiagonalPoissonSolver constructor:
nccl_xy_comm = create_nccl_comm_from_mpi(storage.comms.xy)

# In _slab_x_solve!: replace transpose calls
# Before:
#   transpose_y_to_x!(storage)   # uses MPI Alltoall
# After:
#   nccl_transpose_y_to_x!(storage, nccl_xy_comm, Rx)
```

For the slab-x optimized solve (`_slab_x_solve!`), the full function becomes:

```julia
function _slab_x_solve_nccl!(x, solver)
    arch    = architecture(solver)
    storage = solver.storage
    buffer  = solver.buffer
    nccl_comm = solver.nccl_comm
    Rx = arch.ranks[1]

    # Forward: y-FFT (local) → NCCL transpose → x-FFT → tridiag
    solver.plan.forward.y!(parent(storage.yfield), buffer.y)

    nccl_transpose!(pack_buffer_y_to_x!, unpack_buffer_x_from_y!,
                     storage.xybuff, storage.yfield, storage.xfield,
                     storage.counts.xy, nccl_comm, Rx)

    solver.plan.forward.x!(parent(storage.xfield), buffer.x)

    parent(solver.source_term) .= parent(storage.xfield)
    solve!(storage.xfield, solver.batched_tridiagonal_solver, solver.source_term)

    # Backward: x-IFFT → NCCL transpose → y-IFFT
    solver.plan.backward.x!(parent(storage.xfield), buffer.x)

    nccl_transpose!(pack_buffer_x_to_y!, unpack_buffer_y_from_x!,
                     storage.xybuff, storage.xfield, storage.yfield,
                     storage.counts.xy, nccl_comm, Rx)

    solver.plan.backward.y!(parent(storage.yfield), buffer.y)

    launch!(arch, solver.local_grid, :xyz,
            _copy_real_component!, x, parent(storage.zfield))
    return x
end
```

### Step 4: Benchmark

Use `benchmarks/benchmark_pressure_solver.jl` as a template. The key comparison:

```julia
# MPI baseline
t_mpi = CUDA.@elapsed begin
    for _ in 1:30
        solve!(p, solver_mpi)
    end
end

# NCCL version
t_nccl = CUDA.@elapsed begin
    for _ in 1:30
        solve!(p, solver_nccl)
    end
end
```

Expected improvement: 36 ms → ~15-20 ms per solve on 2 GPUs (intra-node).

## Files to modify

### In Oceananigans (`glw/optimize-distributed-solver` branch):

1. **`src/DistributedComputations/distributed_fft_tridiagonal_solver.jl`**
   - Add `nccl_comm` field to the solver struct (or a wrapper struct)
   - Create NCCL comm in constructor
   - Use NCCL transpose in `_slab_x_solve!`

2. **`src/DistributedComputations/distributed_transpose.jl`**
   - Add `nccl_transpose!` function
   - Or: add NCCL path inside the existing `$transpose!` generated functions

3. **`src/DistributedComputations/transposable_field.jl`**
   - Store NCCL comm alongside MPI comm

### Alternatively, as an extension:

Create `OceananigansNCCLExt` that overrides the solver construction:

```
OceananigansNCCLExt/
├── Project.toml           # depends on Oceananigans, NCCL, CUDA
├── src/
│   └── OceananigansNCCLExt.jl
└── test/
    └── runtests.jl
```

The extension would:
1. Define `NCCLDistributedFFTSolver` wrapping the base solver + NCCL comm
2. Override `solve!` to use NCCL transposes
3. Provide a constructor: `NCCLDistributedFFTSolver(global_grid, local_grid)`

## NCCL.jl API reference

Based on the installed version (NCCL 2.28.3):

```julia
using NCCL

# Create communicator
id = NCCL.UniqueID()
comm = NCCL.Communicator(nranks, id, rank)

# Point-to-point (must be inside group for multi-peer)
NCCL.groupStart()
NCCL.Send(sendbuf::CuArray, comm; peer=dest_rank)
NCCL.Recv!(recvbuf::CuArray, comm; peer=src_rank)
NCCL.groupEnd()

# Collectives
NCCL.Allreduce!(sendbuf, recvbuf, op, comm)
NCCL.Broadcast!(buf, comm; root=0)
NCCL.Reduce!(sendbuf, recvbuf, op, comm; root=0)
NCCL.Allgather!(sendbuf, recvbuf, comm)
NCCL.ReduceScatter!(sendbuf, recvbuf, op, comm)

# Communicator info
NCCL.rank(comm)
NCCL.size(comm)
```

**Important:** NCCL operations are enqueued on the default CUDA stream.
They are ordered with respect to other operations on the same stream.
No explicit synchronization is needed between GPU kernels and NCCL calls
on the same stream.

## Potential issues

1. **NCCL on Slingshot (inter-node):** NCCL needs the OFI network plugin for
   Slingshot. The CUDA.jl artifact version of NCCL may not include it.
   Fix: set `NCCL_NET_OFI_PLUGIN_PATH` or use system NCCL.

2. **NCCL.jl stream support:** Check if `Send`/`Recv!` accept a `stream` keyword.
   If not, they use the default stream, which is fine for the basic implementation
   but limits the multi-stream overlap (Phase 5 in the plan).

3. **UniqueID serialization:** The `NCCL.UniqueID` internal type is
   `NTuple{128, Int8}` (not `UInt8`). The MPI broadcast needs to handle this
   correctly. Use `reinterpret` or raw byte copy.

4. **Multiple NCCL comms:** For pencil decomposition (Ry > 1), need two NCCL
   communicators (matching the MPI yz and xy subcommunicators). Each requires
   its own UniqueID broadcast.

## Expected results

| Metric | MPI (current) | NCCL (expected) | Improvement |
|--------|--------------|-----------------|-------------|
| sync_device! per solve | 2 | 0 | Eliminated |
| Transpose latency (2 GPU, NVLink) | 1.85 ms | <0.5 ms | 3-4x |
| Pipeline stall overhead | ~25 ms | ~0 ms | Eliminated |
| **Solver total (2 GPU)** | **36 ms** | **~15 ms** | **~2.4x** |
| **Solver total (8 GPU, 2 nodes)** | **75 ms** | **~40 ms** | **~1.9x** |
