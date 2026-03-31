# Strategy: Accelerate the Distributed Pressure Solver

## Problem Statement

The distributed `DistributedFourierTridiagonalPoissonSolver` has two major
overhead sources identified by nsight profiling (50×400×80 grid, 2 GPUs):

1. **FFT strategy mismatch (3.55 ms, 53% of overhead):** The distributed solver
   uses three separate 1D FFT plans (x, y, z) even when some directions are
   fully local. The non-distributed solver uses a single batched multi-dimensional
   FFT that is 6× faster on GPU.

2. **Sync points (3.26 ms, 49% of overhead):** The solver has 5 synchronization
   points per RK3 substage where the GPU idles waiting for communication.

These are **backend-agnostic** problems — they affect both MPI and NCCL equally.
The fixes belong in Oceananigans core, not in the NCCL extension.

## Current Solver Flow (slab-x, Z-stretched)

```
_slab_x_solve!:
  y-FFT(yfield)              ← 1D FFT along dim 2 (LOCAL, no comm needed)
  transpose_y_to_x!(storage) ← pack → SYNC → alltoall → unpack
  x-FFT(xfield)              ← 1D FFT along dim 1 (after transpose, now local)
  tridiag_solve(xfield)      ← local batched tridiagonal
  x-IFFT(xfield)
  transpose_x_to_y!(storage) ← pack → SYNC → alltoall → unpack
  y-IFFT(yfield)
```

The y-FFT operates on fully local data (y is not partitioned in slab-x).
Yet it uses a 1D plan that transforms one y-pencil at a time — very
inefficient on GPU compared to a batched plan.

## Why the Non-Distributed Solver is Faster

For `(Periodic, Periodic, Bounded)` topology, the non-distributed solver
detects this as a "batchable" topology and creates:

```julia
# Non-distributed (plan_transforms.jl, line 116):
forward_periodic_plan = plan_forward_transform(storage, Periodic(), [1, 2])  # batched 2D FFT!
forward_bounded_plan  = plan_forward_transform(storage, Bounded(),  [3])     # 1D DCT in z
```

**One cuFFT call** transforms both x and y simultaneously. The GPU processes
the entire 3D array in a single batched operation.

The distributed solver creates:

```julia
# Distributed (plan_distributed_transforms.jl, lines 13-31):
forward_plan_x = plan_forward_transform(parent(storage.xfield), Periodic(), [1])  # 1D in x
forward_plan_y = plan_forward_transform(parent(storage.yfield), Periodic(), [2])  # 1D in y
forward_plan_z = plan_forward_transform(parent(storage.zfield), Bounded(),  [3])  # 1D in z
```

**Three separate cuFFT calls**, each operating on one dimension. The y-FFT
in particular is inefficient: it's a 1D plan along dim 2, meaning cuFFT must
batch across dims 1 and 3, which is a strided access pattern.

### Measured cost (nsight, 50×400×80 grid)

| Solver | FFT total | FFT calls/step |
|--------|----------|---------------|
| Non-distributed | 0.71 ms | 24 |
| Distributed | 4.26 ms | 501 |

The distributed solver launches **21× more FFT kernels** for the same per-GPU
grid size, because each 1D plan launches many small kernels instead of one
large batched kernel.

## Proposed Fix 1: Reshape y-FFT for Contiguous Memory Access

**The problem is NOT about batching y+z** — the z-direction uses a tridiagonal
solver (not FFT) in the slab-x Z-stretched path. The z-FFT plan isn't even
used in `_slab_x_solve!`.

**The actual problem:** The 1D FFT along dim 2 generates 55× more GPU kernel
launches than the non-distributed solver because cuFFT decomposes a strided
dim-2 FFT into many small kernels.

Nsight data (50×400×80 grid, 50 steps):

| | 1-GPU non-distributed | 2-GPU distributed |
|---|---|---|
| FFT kernel calls | 900 (18/step) | 49,200 (492/step/GPU) |
| Avg kernel time | 33 μs | 8 μs |
| FFT total | 30.1 ms | 409.6 ms |

The 2-GPU solver launches 492 FFT kernels per step because the y-FFT plan
along dim 2 of a 50×400×80 array creates many tiny kernels. cuFFT must do
50×80 = 4000 independent transforms of length 400, each accessing strided
memory (stride = Nx = 50 elements between consecutive y-values).

**The fix:** Reshape the y-field to put dim 2 first (contiguous in memory),
then plan the FFT along dim 1 — exactly what the non-distributed solver does
for Bounded y-topology.

### Current code (`plan_distributed_transforms.jl`):

```julia
# For Periodic y on GPU (our case): plans along dim 2 (strided)
forward_plan_y = plan_forward_transform(parent(storage.yfield), Periodic(), [2], planner_flag)
```

This generates many small kernels because dim 2 is non-contiguous.

### For Bounded y on GPU (already has the fix!):

```julia
# Reshapes to put y first, then plans along dim 1 (contiguous)
rs_size    = reshaped_size(grids[2])   # (Ny, Nx, Nz)
rs_storage = reshape(parent(storage.yfield), rs_size)
forward_plan_y = plan_forward_transform(rs_storage, Bounded(), [1], planner_flag)
```

### Proposed change:

```julia
# Apply the same reshape strategy to Periodic y on GPU:
if arch isa GPU
    rs_size    = reshaped_size(grids[2])   # (Ny, Nx, Nz)
    rs_storage = reshape(parent(storage.yfield), rs_size)
    forward_plan_y  = plan_forward_transform(rs_storage, topo[2](), [1], planner_flag)
    backward_plan_y = plan_backward_transform(rs_storage, topo[2](), [1], planner_flag)
    y_dims = [2]  # DiscreteTransform still reports dim 2 for correct permutation
else
    forward_plan_y  = plan_forward_transform(parent(storage.yfield), topo[2](), [2], planner_flag)
    backward_plan_y = plan_backward_transform(parent(storage.yfield), topo[2](), [2], planner_flag)
    y_dims = [2]
end
```

This is a **4-line change** in `plan_distributed_transforms.jl`.

### Expected Improvement

With contiguous-memory layout, cuFFT should generate ~10× fewer kernels
(similar to the Bounded case which already uses this strategy). The y-FFT
cost should drop from the current ~1.5 ms to ~0.15 ms per step.

**Estimated total FFT savings: ~2.5 ms/step** on the 50×400×80 grid.

Note: the x-FFT after transpose operates on the transposed xfield where
dim 1 IS contiguous — it doesn't have this problem. The x-FFT already
plans along dim 1.

### UPDATE: Deeper analysis shows the reshape alone may not help

The nsight data shows that the 2-GPU `regular_fft<400>` kernel is **identical**
to the 1-GPU kernel (same template parameters). The difference is purely in
call count: 48,000 vs 300 for 50 steps.

The 1-GPU non-distributed solver uses a **batched 2D FFT plan along dims [1,2]**,
which cuFFT executes as 80 batches of a 50×400 2D transform. cuFFT fuses the
row and column transforms into a single efficient operation.

The distributed solver does a **1D FFT along dim 2** only, which cuFFT sees as
4000 independent length-400 transforms. Even with optimal batching, this is
fundamentally less efficient than the 2D plan because cuFFT can't fuse the
row/column transforms.

**The distributed solver cannot do a batched 2D (x,y) FFT** because x and y
data live on different grids — the y-FFT happens on `yfield` (before transpose)
and the x-FFT happens on `xfield` (after transpose). They have different sizes
and memory layouts.

The reshape-to-contiguous strategy may still help reduce kernel launches from
480 to ~60 per step (by improving cuFFT's batching efficiency), but it cannot
achieve the 1-GPU batched 2D performance.

**Fundamental options to close the gap:**
1. **Accept the 1D FFT overhead** (~3.5 ms on 50×400×80) — at large grids the
   overhead is a smaller fraction of total compute
2. **Communication-avoiding solver** (multigrid, CG with FFT preconditioner) —
   eliminates transpose communication entirely
3. **cuFFTMp** — NVIDIA's distributed FFT library handles the transpose internally
   with optimized NVSHMEM communication, but is not accessible from Julia

## Proposed Fix 2: Overlap Velocity Halo with Tracer Advance

**Start the velocity halo communication needed by the pressure solver
BEFORE the pressure solver is called.**

Current flow per RK3 substage:
```
_rk3_substep_field!(tracers)           ← local GPU kernel
_rk3_substep_field!(velocities)        ← local GPU kernel (exclude_periphery=true)
compute_pressure_correction!:
  fill_halo_regions!(velocities)       ← SYNC WAIT (~0.25 ms)
  solve_for_pressure!(...)
```

Proposed:
```
_rk3_substep_field!(tracers)
_rk3_substep_field!(velocities)
fill_halo_regions!(velocities, async=true)  ← start comm on comm_stream
fill_halo_regions!(tracers, async=true)     ← also start (already done by pipelined RK3)
compute_pressure_correction!:
  synchronize_communication!(velocities)    ← wait (shorter wait — comm started earlier)
  solve_for_pressure!(...)
```

This overlaps the velocity halo transfer with the tracer halo transfer and
any CPU-side overhead between `_rk3_substep_field!` and the solver.

**Estimated savings: ~0.2 ms** (the overlap window is small — just the gap
between the RK3 kernel and the solver entry).

This is already partially implemented in `nccl_pipelined_rk3.jl` for tracer
halos. Extending it to velocity halos requires care: `exclude_periphery=true`
means the boundary cells were NOT written by the RK3 kernel, so we need to
determine if communicating stale boundary data before make_pressure_correction
is acceptable. (It is NOT — the solver needs the RK3-stepped interior data
with fresh halos from the RK3-stepped boundary data on the neighbor. Since
`exclude_periphery=true` didn't write the boundary, the boundary data is from
the PREVIOUS substage's `make_pressure_correction`, which is the most recent
complete update. So the halos being sent are valid for the solver's needs.)

**UPDATE:** On reflection, this needs more careful analysis of what data the
solver actually reads from the halo cells. If it computes the divergence using
halos, and the halos contain pressure-corrected values from the previous
substage, that should be correct for the RK3 prediction step.

## Proposed Fix 3: Reduce Solver Sync Points

**Eliminate redundant halo fills in the pressure correction.**

Current `compute_pressure_correction!`:
```julia
fill_halo_regions!(model.velocities, ...)   ← 3 fields, SYNC
solve_for_pressure!(p, ...)                  ← 2 transpose SYNCs
fill_halo_regions!(p)                        ← 1 field, SYNC
```

The post-solver `fill_halo_regions!(p)` communicates the solved pressure.
But `make_pressure_correction!` computes `u -= ∂p/∂x` which only reads
interior pressure values (plus one halo cell for the derivative). If the
halo is only 1 cell, the pressure correction near the boundary needs the
neighbor's pressure at one cell.

**Question:** Is this halo fill actually needed? If the grid has `halo=(1,1,1)`
and the pressure correction reads `p[i±1,j,k]`, then yes — the halo must be
fresh. But if we defer the pressure halo fill to `update_state!` (which fills
all halos anyway), we save one sync point per substage.

This requires verifying that the pressure correction kernel doesn't read beyond
what's available without the post-solver halo fill. For `halo=(1,1,1)`, the
derivative `∂xᶠᶜᶜ(i,j,k,grid,p) = (p[i,j,k] - p[i-1,j,k]) / Δx` reads
`p[i-1]` which at the boundary `i=1` reads `p[0]` — the halo. So the fill
IS needed.

**Savings if applicable:** ~0.25 ms per substage × 3 = 0.75 ms/step.

## Proposed Fix 4: Merge Transpose Sync into a Single Wait

**Combine the two transpose sync points in the solver into fewer waits.**

Current solver flow:
```
y-FFT → [pack → SYNC₁ → alltoall → unpack] → x-FFT → tridiag → x-IFFT → [pack → SYNC₂ → alltoall → unpack] → y-IFFT
```

With async communication (MPI Isend/Irecv or NCCL on comm_stream):
```
y-FFT → [pack → async_send₁] → ... overlap ... → [wait₁ → unpack] → x-FFT → tridiag → x-IFFT → [pack → async_send₂] → ... overlap ... → [wait₂ → unpack] → y-IFFT
```

The problem: there's nothing to overlap with between the async send and the wait
for the forward transpose. The x-FFT NEEDS the transposed data. Similarly for
the backward transpose.

**However:** The tridiagonal solve between the forward and backward transposes
takes ~0.2 ms. If we start the backward transpose's pack BEFORE the tridiag
(by packing from the pre-tridiag data, then re-packing after)... no, that
doesn't work because the data changes.

**Conclusion:** The two solver transpose sync points are **causally irreducible**.
Each FFT step requires its input data to be fully available. No fix possible
without changing the solver algorithm (e.g., communication-avoiding solver).

## Summary: Priority-Ordered Recommendations

| Fix | Backend | Savings | Difficulty | Where |
|-----|---------|---------|-----------|-------|
| **1. Batched local FFT** | Both | ~3 ms | Medium | `plan_distributed_transforms.jl` |
| 2. Overlap velocity halo | Both | ~0.2 ms | Low | `nonhydrostatic_rk3_substep.jl` |
| 3. Defer pressure halo | Both | ~0.75 ms | Medium | `pressure_correction.jl` |
| 4. Merge transpose syncs | N/A | 0 | N/A | Causally irreducible |

**Fix 1 is the clear priority.** It's a 3 ms/step improvement on small grids
(36% → ~55% efficiency for NonhydrostaticModel), and proportionally helps large
grids too. It requires modifying only `plan_distributed_transforms.jl` — no
changes to the solver algorithm or communication pattern.

All fixes are backend-agnostic and benefit both MPI and NCCL equally.
