# Plan: NCCL-based Distributed FFT for Oceananigans

## Goal

Replace the MPI-based distributed FFT transpose in Oceananigans with NCCL,
eliminating GPU pipeline stalls and MPI overhead. Build as an Oceananigans
extension package that is a drop-in replacement for the existing
`DistributedFourierTridiagonalPoissonSolver`.

## Why NCCL

| Property | MPI (current) | NCCL |
|----------|--------------|------|
| GPU sync before comm | Required (`sync_device!`) | Not needed (stream-native) |
| Intra-node transport | GTL → NVLink (indirect) | NVLink (direct) |
| Inter-node transport | Slingshot via GTL | Slingshot via NCCL net plugin |
| Alltoall performance | 1.85 ms (intra), 8.7 ms (inter) | Expected: <1 ms (intra) |
| Pipeline stalls | Yes (every transpose) | No (GPU-async) |
| Collective optimization | MPI implementation dependent | NVIDIA-optimized for GPU topology |

The key advantage: NCCL operations are enqueued on the CUDA stream like kernel launches.
No `sync_device!` / `CUDA.synchronize()` is needed before or after. The GPU pipeline
runs uninterrupted.

## Architecture

```
OceananigansNCCLExt/
├── Project.toml
├── src/
│   ├── OceananigansNCCLExt.jl       # Extension entry point
│   ├── nccl_communicator.jl          # NCCL comm setup from MPI ranks
│   ├── nccl_transpose.jl            # GPU-stream-native transpose
│   ├── nccl_distributed_solver.jl   # Solver that uses NCCL transposes
│   └── nccl_transposable_field.jl   # Double-buffered field for overlap
├── test/
│   ├── test_nccl_transpose.jl       # Correctness tests
│   └── test_nccl_solver.jl          # Solver validation against MPI version
└── benchmarks/
    ├── benchmark_nccl_transpose.jl   # Compare NCCL vs MPI transpose
    └── benchmark_nccl_solver.jl      # Full solver comparison
```

## Implementation phases

### Phase 1: NCCL communicator from MPI (1-2 days)

NCCL needs its own communicator, initialized from a UniqueID broadcast via MPI.
This only happens once at solver construction time.

```julia
using NCCL
using MPI

function create_nccl_comm(mpi_comm)
    nranks = MPI.Comm_size(mpi_comm)
    rank = MPI.Comm_rank(mpi_comm)

    # Rank 0 creates unique ID, broadcasts to all
    if rank == 0
        id = NCCL.UniqueID()
        id_bytes = reinterpret(UInt8, Ref(id.internal))
    else
        id_bytes = zeros(UInt8, 128)
    end
    MPI.Bcast!(id_bytes, mpi_comm; root=0)

    # All ranks create communicator from shared ID
    nccl_id = NCCL.UniqueID(reinterpret(NTuple{128, UInt8}, id_bytes)[1])
    return NCCL.Communicator(nranks, nccl_id, rank)
end
```

**Deliverable:** Working NCCL communicator creation from MPI world comm + subcommunicators.

**Test:** Verify all ranks can send/recv a small CuArray via NCCL.

### Phase 2: NCCL transpose (2-3 days)

Replace the `pack → sync_device! → Alltoall → unpack` pattern with:

```julia
function nccl_transpose_y_to_x!(storage, nccl_comm, Rx, my_rank, chunks)
    pack_into_chunks!(chunks.send, storage.yfield, Rx)
    # NO sync_device! needed — NCCL reads from same stream

    NCCL.groupStart()
    for r in 0:Rx-1
        NCCL.Send(chunks.send[r+1], nccl_comm; peer=r)
        NCCL.Recv!(chunks.recv[r+1], nccl_comm; peer=r)
    end
    NCCL.groupEnd()
    # NCCL ops are enqueued on the default CUDA stream
    # They will complete before any subsequent kernel on the same stream

    unpack_from_chunks!(storage.xfield, chunks.recv, Rx)
    return nothing
end
```

Key design decisions:
- **Per-rank chunk buffers** (not one flat buffer like Alltoall): enables incremental processing
- **No sync_device!**: NCCL is stream-ordered, so the pack kernel completes before NCCL reads
- **No Waitall**: NCCL operations on the same stream are ordered — the unpack kernel
  implicitly waits for all recv to complete

**Deliverable:** `nccl_transpose_y_to_x!` and `nccl_transpose_x_to_y!` that match
the behavior of the existing MPI-based transposes.

**Test:** Compare output of NCCL transpose vs MPI transpose for random input data.
Verify bit-exact results.

### Phase 3: NCCL-based solver (2-3 days)

Build `NCCLDistributedFFTSolver` that uses the NCCL transposes inside the
slab-x optimized solve:

```julia
function solve!(x, solver::NCCLDistributedFFTSolver)
    storage = solver.storage
    buffer = solver.buffer
    nccl_comm = solver.nccl_comm
    chunks = solver.chunks

    # Forward: y-FFT → NCCL transpose → x-FFT → tridiag
    solver.plan.forward.y!(parent(storage.yfield), buffer.y)
    nccl_transpose_y_to_x!(storage, nccl_comm, Rx, my_rank, chunks)
    solver.plan.forward.x!(parent(storage.xfield), buffer.x)

    parent(solver.source_term) .= parent(storage.xfield)
    solve!(storage.xfield, solver.batched_tridiagonal_solver, solver.source_term)

    # Backward: x-IFFT → NCCL transpose → y-IFFT
    solver.plan.backward.x!(parent(storage.xfield), buffer.x)
    nccl_transpose_x_to_y!(storage, nccl_comm, Rx, my_rank, chunks)
    solver.plan.backward.y!(parent(storage.yfield), buffer.y)

    launch!(arch, solver.local_grid, :xyz,
            _copy_real_component!, x, parent(storage.zfield))
    return x
end
```

**Deliverable:** Working solver that passes correctness tests against the MPI version.

**Test:** Solve Poisson equation on distributed grid, compare solution against
single-GPU reference to machine precision.

### Phase 4: Benchmarking (1-2 days)

Run the pressure solver isolation benchmark with NCCL vs MPI:
- 2 GPUs (1 node, NVLink)
- 4 GPUs (1 node, NVLink)
- 8 GPUs (2 nodes, Slingshot)
- 20 GPUs (5 nodes)

Compare:
- ms/solve
- Transpose time
- Pipeline stall overhead (should be ~0 for NCCL)

Also run full Breeze weak scaling (ERF-like anelastic) to measure end-to-end impact.

### Phase 5: Overlap communication with computation (2-3 days)

With NCCL, we can go further: use CUDA streams to overlap the transpose
communication with FFT computation on local data.

```julia
function pipelined_solve!(x, solver::NCCLDistributedFFTSolver)
    # Create a separate stream for communication
    comm_stream = CUDA.CuStream()

    # Forward pass
    solver.plan.forward.y!(parent(storage.yfield), buffer.y)

    # On comm_stream: start packing + NCCL transpose
    CUDA.@sync comm_stream begin
        pack_into_chunks!(chunks.send, storage.yfield, Rx)
        NCCL.groupStart()
        for r in 0:Rx-1
            NCCL.Send(chunks.send[r+1], nccl_comm; peer=r, stream=comm_stream)
            NCCL.Recv!(chunks.recv[r+1], nccl_comm; peer=r, stream=comm_stream)
        end
        NCCL.groupEnd()
    end

    # On default stream: FFT the local chunk while remote data is in flight
    local_chunk = chunks.recv[my_rank + 1]  # Already available (local copy)
    fft_local_chunk!(local_chunk, solver.plan.forward.x!)

    # Sync comm_stream → default stream (wait for remote data)
    CUDA.wait(comm_stream)

    # FFT the remote chunks
    fft_remote_chunks!(chunks.recv, solver.plan.forward.x!, my_rank, Rx)

    # ... tridiag, backward pass with same pattern ...
end
```

This overlaps the NCCL transfer time with FFT computation on the local chunk.
For 2 GPUs, the local chunk is 50% of the data, so we can overlap ~50% of the
FFT with the full NCCL transfer.

**Deliverable:** Pipelined solver with stream-based overlap.

### Phase 6: Pencil decomposition support (3-5 days)

Extend from slab-x to full pencil (Rx, Ry) decomposition:
- Need two NCCL subcommunicators (yz and xy, matching current MPI subcommunicators)
- The Z-stretched algorithm with pencil needs 4 NCCL transposes instead of 2
- Overlap opportunities are greater with 4 transposes

### Phase 7: Integration into Oceananigans (2-3 days)

Package as a proper Julia extension:
- Register in Oceananigans' `Project.toml` extensions
- Auto-detect NCCL availability and use it when present
- Fallback to MPI when NCCL is not available
- Add CI tests

## Expected performance

Based on our measurements:

| Metric | Current (MPI Alltoall) | Expected (NCCL) | Source |
|--------|----------------------|-----------------|--------|
| Transpose (2 GPU, NVLink) | 1.85 ms | <0.5 ms | NCCL direct NVLink |
| sync_device! overhead | 2× per solve | 0 | NCCL is stream-native |
| Pipeline stalls | ~30 ms/solve | ~0 | No CPU-GPU sync |
| **Solver total (2 GPU)** | **36.2 ms** | **~15-20 ms** | 2x improvement |
| **Solver total (8 GPU, inter-node)** | **74.9 ms** | **~40-50 ms** | 1.5-2x improvement |

The biggest win is eliminating the pipeline stalls from `sync_device!` calls.
Even with MPI Alltoall at 1.85 ms, the measured solver time is 36 ms because
of pipeline serialization. NCCL eliminates this entirely.

## Dependencies

- `NCCL.jl` (v0.5+) — already installed, provides `Communicator`, `Send`, `Recv!`, `groupStart`, `groupEnd`
- `CUDA.jl` (v5+) — for stream management
- `MPI.jl` — still needed for NCCL communicator initialization (UniqueID broadcast)

## Risks and mitigations

1. **NCCL.jl API completeness**: The Julia wrapper may not expose all needed features
   (e.g., stream selection for Send/Recv). Mitigation: contribute to NCCL.jl or use ccall.

2. **Slingshot compatibility**: NCCL on Cray Slingshot requires the NCCL OFI plugin.
   On Derecho, NCCL's bundled artifacts may not include this. Mitigation: use system NCCL
   or build with Slingshot support.

3. **Multi-node initialization**: NCCL UniqueID must be broadcast via MPI before NCCL
   comms are created. Our earlier attempt at NCCL in the transpose benchmark failed on
   the UniqueID broadcast (MPI.Buffer type mismatch). Mitigation: fix the serialization.

4. **Correctness**: The NCCL transpose must produce bit-identical results to MPI.
   Mitigation: comprehensive comparison tests before benchmarking.

## Timeline

| Phase | Effort | Cumulative |
|-------|--------|------------|
| 1. NCCL communicator | 1-2 days | 1-2 days |
| 2. NCCL transpose | 2-3 days | 3-5 days |
| 3. NCCL solver | 2-3 days | 5-8 days |
| 4. Benchmarking | 1-2 days | 6-10 days |
| 5. Stream overlap | 2-3 days | 8-13 days |
| 6. Pencil decomposition | 3-5 days | 11-18 days |
| 7. Oceananigans integration | 2-3 days | 13-21 days |

**MVP (phases 1-4): ~1-2 weeks** for a working NCCL solver with benchmarks.
**Full implementation (all phases): ~3 weeks.**
