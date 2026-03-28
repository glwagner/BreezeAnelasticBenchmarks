# Kernel Fusion Sketch: Eliminating Pack/Unpack Overhead

## Profiling Evidence (2 GPUs, 1024×1024×128 F32, WENO5, 3 timesteps)

The remaining distributed overhead after NCCL + async + comm_stream optimizations:

| Source | ms/step | What it does |
|--------|---------|-------------|
| Halo buffer ops (`broadcast_kernel_cartesian`) | 30.1 | `fill_send_buffers!` + `recv_from_buffers!` |
| Transpose pack/unpack | 26.4 | `pack_buffer_y_to_x!` etc. for pressure solver |
| NCCL on default stream | 8.0 | Sync single-field fills |
| Event sync overhead | ~13.5 | Waiting for comm_stream |
| **Total overhead** | **~78** | vs 357 ms 1-GPU baseline → 435 ms 2-GPU |

NCCL communication itself (81 ms/step) is almost entirely hidden on comm_stream.
The bottleneck is **56 ms of data shuffling kernels** on the default stream.

## What the pack/unpack kernels do

### Halo buffer ops (30.1 ms, 630 calls = 105/step)

`fill_send_buffers!` copies a strip of the field interior to a flat 1D send buffer:
```julia
# From distributed_transpose.jl / communication_buffers.jl
# Conceptually for west halo, Hx=3:
send_buffer[1:Hx, :, :] .= field[Nx-Hx+1:Nx, :, :]
```

`recv_from_buffers!` copies the flat 1D recv buffer into the halo region:
```julia
field[1-Hx:0, :, :] .= recv_buffer[1:Hx, :, :]
```

Each is a simple memory copy on the GPU — launched as a `broadcast_kernel_cartesian`
(KernelAbstractions broadcast). Average: 286 μs per kernel.

### Transpose pack/unpack (26.4 ms, 72 calls = 12/step)

`pack_buffer_y_to_x!` reorders a 3D field into a 1D buffer for alltoall:
```julia
@kernel function _pack_buffer_y_to_x!(xybuff, yfield, N)
    i, j, k = @index(Global, NTuple)
    Nx, _, Nz = N
    @inbounds xybuff.send[i + Nx * (k-1 + Nz * (j-1))] = yfield[i, j, k]
end
```

Average: 1.4-4.4 ms per kernel (larger data: 1024×512×128 Complex{Float32}).

## Fusion Strategy 1: Fused halo pack into tendency kernel

**Idea:** When computing tendencies near the boundary, simultaneously write the
result to both the tendency field AND the send buffer. The boundary tendencies
ARE the data that needs to be communicated.

**Current flow:**
```
compute_Gu!(grid, :xyz)       # writes Gu[1:Nx, 1:Ny, 1:Nz]
fill_send_buffers!(Gu, bufs)  # copies Gu[Nx-Hx+1:Nx, :, :] → bufs.east.send
NCCL Send/Recv
recv_from_buffers!(Gu, bufs)  # copies bufs.west.recv → Gu[1-Hx:0, :, :]
```

**Fused flow:**
```
compute_Gu!(grid, :interior)  # Gu[Hx+1:Nx-Hx, :, :] only (no boundary)
compute_Gu_and_pack!(grid, :boundary, bufs)  # Gu boundary + bufs.send simultaneously
NCCL Send/Recv
recv_from_buffers!(Gu, bufs)  # still needed (remote data)
```

**The fused kernel:**
```julia
@kernel function _compute_Gu_and_pack_east!(Gu, bufs_east_send, grid, ...)
    i_local, j, k = @index(Global, NTuple)
    i = Nx - Hx + i_local  # boundary strip indices

    # Compute tendency (same as normal compute_Gu)
    @inbounds Gu[i, j, k] = -div_Uc(i, j, k, grid, ...) + ...

    # Simultaneously pack into send buffer
    @inbounds bufs_east_send[i_local + Hx * (j-1 + Ny * (k-1))] = Gu[i, j, k]
end
```

**Saves:** 30.1 ms of fill_send_buffers! (the copy is free — done inside the compute kernel)

**Difficulty:** HIGH. Requires modifying every tendency kernel to optionally accept
send buffer arguments and write to them for boundary cells. The existing KernelAbstractions
`launch!` with `exclude_periphery` already splits interior/boundary. The boundary kernels
(from `compute_buffer_tendencies!`) could be augmented to also pack.

**Approach:** Override `compute_buffer_tendencies!` in the NCCL extension to use
fused pack+compute kernels. This avoids modifying Oceananigans core.

## Fusion Strategy 2: Zero-copy halo communication

**Idea:** Instead of copying field data to separate send buffers, have NCCL read
directly from the field's halo region. This eliminates pack entirely.

**Problem:** NCCL Send/Recv operates on contiguous memory. The halo strip
`field[Nx-Hx+1:Nx, :, :]` is NOT contiguous in memory (it's strided — the
field is stored column-major with full Nx stride in the first dimension).

For the **west/east halos** (x-direction), the strip IS contiguous if Hx
columns are adjacent in memory. In column-major (Julia/Fortran order):
- `field[Nx-Hx+1:Nx, j, k]` for fixed j,k is contiguous (Hx values)
- But across j and k, the values are strided

So we'd need either:
a) NCCL to support strided buffers (it doesn't), or
b) Reorganize field memory layout so halo strips are contiguous, or
c) Use `reshape`/`reinterpret` tricks if the data happens to be contiguous

For slab-x partitioning (only x is distributed), only west/east halos
need communication. The strip `field[1:Hx, 1:Ny, 1:Nz]` in Julia
column-major IS contiguous: it's the first Hx elements of each column,
and columns are contiguous. So `field[1:Hx, :, :]` is a contiguous block
of `Hx * Ny * Nz` elements!

**This means we CAN do zero-copy for slab-x:**
```julia
# Instead of:
fill_send_buffers!(c, bufs, grid, Val(:west_and_east))  # copy to flat buffer
NCCL.Send(bufs.east.send, ...)

# Do:
east_strip = view(parent(field), Nx-Hx+1:Nx, 1:Ny, 1:Nz)  # contiguous!
west_strip = view(parent(field), 1:Hx, 1:Ny, 1:Nz)         # contiguous!
NCCL.Send(east_strip, ...)  # send directly from field memory
NCCL.Recv!(west_strip, ...)  # recv directly into field halo
```

Wait — `east_strip` is `parent(field)[Nx-Hx+1:Nx, :, :]`. Is this contiguous?
In column-major order, elements are stored as [i + Nx*(j-1) + Nx*Ny*(k-1)].
The strip i=Nx-Hx+1:Nx, j=1:Ny, k=1:Nz has indices:
  (Nx-Hx+1) + Nx*(0) + Nx*Ny*(0),  (Nx-Hx+2) + ...,  ..., Nx + Nx*(0) + ...,
  (Nx-Hx+1) + Nx*(1) + ...,  ...

These are NOT contiguous — there's a gap of (Nx-Hx) between the end of one
row and the start of the next. Only the FIRST Hx elements of each column
(the west halo) are contiguous if the field starts at index 1.

Actually, for the WEST halo: `field[1:Hx, 1:Ny, 1:Nz]` — the elements
1,2,...,Hx of column j=1, then Hx elements of column j=2, etc. In memory:
  indices 1..Hx, Nx+1..Nx+Hx, 2Nx+1..2Nx+Hx, ...
These are NOT contiguous (strided by Nx).

So zero-copy doesn't work for the standard memory layout. It would require
a "halo-contiguous" layout where halo strips are stored contiguously.

## Fusion Strategy 3: Eliminate separate pack with in-place send

**Idea:** The send buffer IS the halo region of the field. Instead of copying
field interior → send buffer, the field's data layout stores halo data in
a format that NCCL can send directly.

This requires changing `CommunicationBuffers` to point into the field's
own data array, with appropriate offsets. It's a data structure change in
Oceananigans.

## Recommendation

**Strategy 1 (fused pack+compute) is the most practical:**
- 30.1 ms savings (halo buffer ops)
- Can be implemented in the NCCL extension by overriding `compute_buffer_tendencies!`
- Doesn't require changing Oceananigans data structures

**Strategy 2 (zero-copy) would save ~30 ms but requires contiguous memory layout.**
Not feasible with current column-major field storage.

**Strategy 3 is the cleanest but requires Oceananigans core changes.**

**For the transpose pack/unpack (26.4 ms):** This can't be fused with computation
because it's a data redistribution (alltoall) that must complete before the
next FFT can start. The pack kernel reorders data for the transpose — it's
inherent to the algorithm. Reducing this requires either:
- cuFFTMp (handles redistribution internally, not available from Julia), or
- Pencil decomposition that reduces per-transpose data volume
