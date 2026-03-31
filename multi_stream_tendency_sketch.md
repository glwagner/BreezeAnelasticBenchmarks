# Multi-Stream Tendency Computation Sketch

## Motivation

For small grids (especially hydrostatic models with thin z), individual tendency
kernels don't saturate the GPU. Running independent kernels on separate CUDA
streams enables concurrent execution.

Measured tendency kernel times on A100 (WENO5, 5 tracers):

| Grid | Cells | Per-kernel time | GPU occupancy | Multi-stream benefit? |
|------|-------|----------------|--------------|----------------------|
| 50×400×10 | 200K | 79 μs | ~23% of SMs | Yes — room for 3-4× concurrency |
| 50×400×80 | 1.6M | 303 μs | ~100% of SMs | Marginal |

## Proposed API

Split `compute_interior_tendency_contributions!` into:

```julia
compute_momentum_tendencies!(model, params)   # Gu, Gv, Gw on momentum_stream
compute_scalar_tendencies!(model, params)     # Gc for each tracer on scalar_stream
```

These are independent — momentum tendencies don't read tracer tendencies and vice
versa. They CAN read the same field values (u, v, w, b) but only read, not write.

## Implementation: 2-stream tendency computation

```julia
function compute_tendencies!(model, callbacks)
    grid = model.grid
    arch = architecture(grid)

    interior_params = interior_tendency_kernel_parameters(arch, grid)
    active_cells_map = get_active_cells_map(grid, Val(:interior))

    # Launch momentum tendencies on the default stream
    compute_momentum_tendencies!(model, interior_params; active_cells_map)

    # Launch scalar tendencies on a separate stream (concurrent with momentum)
    scalar_stream = get_scalar_stream(arch)  # cached CUDA stream
    with_stream(scalar_stream) do
        compute_scalar_tendencies!(model, interior_params; active_cells_map)
    end

    # Sync both streams before proceeding
    CUDA.synchronize(scalar_stream)

    # Communication + buffer tendencies (existing async overlap mechanism)
    complete_communication_and_compute_buffer!(model, grid, arch)

    return nothing
end
```

### `with_stream` helper

```julia
function with_stream(f, stream::CuStream)
    old = CUDA.stream()
    CUDA.stream!(stream)
    try
        f()
    finally
        CUDA.stream!(old)
    end
end
```

Note: `CUDA.stream!` changes the default stream for the current task. All
KernelAbstractions launches within the `do` block would use the scalar_stream.

### Buffer tendencies (after halo sync)

```julia
function compute_buffer_tendencies!(model)
    # ... existing buffer parameter computation ...

    # Same split for buffer tendencies
    compute_momentum_tendencies!(model, buffer_params)

    scalar_stream = get_scalar_stream(arch)
    with_stream(scalar_stream) do
        compute_scalar_tendencies!(model, buffer_params)
    end
    CUDA.synchronize(scalar_stream)
end
```

## Implementation: 3-stream with pipelined communication

The full pipeline overlaps communication with compute across independent field groups:

```julia
function compute_tendencies!(model, callbacks)
    arch = architecture(model.grid)
    interior_params = interior_tendency_kernel_parameters(arch, model.grid)

    #=
    Timeline (3 streams):

    comm_stream:    [scalar NCCL ----]             [velocity NCCL ----]
    default:        [velocity interior Gu,Gv,Gw] → [wait vel halos] → [velocity buffer Gu,Gv,Gw]
    scalar_stream:  [wait scalar halos] → [scalar full Gc×5]
                              ↑ overlaps with velocity interior ↑      ↑ overlaps with velocity NCCL ↑
    =#

    # Scalar halos started in pipelined rk3_substep! (Phase 1)
    # Velocity halos started in pipelined rk3_substep! (Phase 3)

    # Default stream: velocity interior tendencies (don't need velocity halos)
    compute_momentum_tendencies!(model, interior_params)

    # Scalar stream: wait for scalar halos, compute full scalar tendencies
    scalar_stream = get_scalar_stream(arch)
    with_stream(scalar_stream) do
        synchronize_communication!(model.tracers)  # scalar halos from pipelined RK3
        compute_scalar_tendencies!(model, :xyz)     # full domain (interior + buffer)
    end

    # Default stream: wait for velocity halos, compute velocity buffer
    synchronize_communication!(model.velocities)
    buffer_params = buffer_tendency_kernel_parameters(model.grid, arch)
    compute_momentum_tendencies!(model, buffer_params)

    # Wait for scalar stream
    CUDA.synchronize(scalar_stream)

    return nothing
end
```

## Expected Impact

### Small grid (50×400×10, hydrostatic-like)

8 tendency kernels at 79 μs each = 632 μs serial.
With 2 streams (3 velocity + 5 scalar concurrent): ~max(3×79, 5×79) = 395 μs.
**Savings: ~237 μs per tendency call × 3 substages = 0.7 ms/step.**

### Medium grid (50×400×80)

8 kernels at 303 μs each = 2.4 ms serial.
With 2 streams: kernels mostly saturate GPU, limited benefit.
**Savings: ~0.2 ms/step** (from partial overlap at kernel boundaries).

### Large grid (1024×1024×128)

Kernels fully saturate GPU. Multi-stream has no benefit for compute.
But communication overlap (pipelined RK3) still helps.

### Additional benefit: communication overlap

The 3-stream pipeline overlaps:
- Scalar NCCL with velocity interior tendencies (already done by pipelined RK3)
- Velocity NCCL with scalar full tendencies (**NEW** — this is the key win)

For 5 tracers at 303 μs each = 1.5 ms of scalar tendencies that can hide
velocity NCCL latency. This helps at ALL grid sizes.

## Compatibility

- Works with both MPI and NCCL (stream management is CUDA-level)
- For CPU: no streams needed, just sequential execution
- The momentum/scalar split is independently useful for code clarity
- The `with_stream` helper is minimal infrastructure

## Files to modify

1. `compute_nonhydrostatic_tendencies.jl` — split into momentum + scalar functions
2. `compute_nonhydrostatic_buffer_tendencies.jl` — same split
3. `interleave_communication_and_computation.jl` — 3-stream pipeline
4. Add `scalar_stream` to `Distributed` architecture (or `NCCLCommunicator`)
