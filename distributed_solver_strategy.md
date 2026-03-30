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

## Proposed Fix 1: Batched Local FFT for Slab-X

**Modify `plan_distributed_transforms` to detect slab-x and use a batched
plan for the local y-direction.**

For slab-x (Ry=1, only x is partitioned):
- y is fully local → can batch the y-FFT
- z is fully local → can batch the z-FFT
- Only x requires a transpose

The fix: instead of separate 1D plans for y and z, create a **single batched
plan for (y, z)** or at minimum use the non-distributed plan for the y-FFT.

### Implementation

In `plan_distributed_transforms.jl`, detect slab-x and use batched plans:

```julia
function plan_distributed_transforms(global_grid, storage::TransposableField, planner_flag)
    topo = topology(global_grid)
    arch = architecture(global_grid)
    Rx, Ry, _ = arch.ranks

    grids = (storage.zfield.grid, storage.yfield.grid, storage.xfield.grid)

    # x-FFT always needs separate plan (operates after transpose)
    forward_plan_x  = plan_forward_transform(parent(storage.xfield), topo[1](), [1], planner_flag)
    backward_plan_x = plan_backward_transform(parent(storage.xfield), topo[1](), [1], planner_flag)

    if Ry == 1  # slab-x: y and z are fully local
        # Use batched plan for local directions (same strategy as non-distributed solver)
        local_topo = (topo[2], topo[3])  # (Periodic, Bounded) for typical case
        local_periodic_dims = [d+1 for (d, t) in enumerate(local_topo) if t == Periodic]  # [2]
        local_bounded_dims  = [d+1 for (d, t) in enumerate(local_topo) if t == Bounded]   # [3]

        # Batched y-FFT: plan along dim 2 with batching across dims 1,3
        # This is the same as the non-distributed solver's periodic plan
        if !isempty(local_periodic_dims)
            forward_plan_yz_periodic  = plan_forward_transform(parent(storage.yfield),
                                                                Periodic(), local_periodic_dims, planner_flag)
            backward_plan_yz_periodic = plan_backward_transform(parent(storage.yfield),
                                                                 Periodic(), local_periodic_dims, planner_flag)
        end

        if !isempty(local_bounded_dims)
            forward_plan_yz_bounded  = plan_forward_transform(parent(storage.yfield),
                                                               Bounded(), local_bounded_dims, planner_flag)
            backward_plan_yz_bounded = plan_backward_transform(parent(storage.yfield),
                                                                Bounded(), local_bounded_dims, planner_flag)
        end

        # ... construct forward/backward operations using batched plans ...
    else
        # Pencil decomposition: fall back to separate 1D plans (current behavior)
        # ...
    end
end
```

### Expected Improvement

The y-FFT currently costs ~1.5 ms (of the 4.26 ms total). With a batched plan,
it would drop to ~0.3 ms (similar to the non-distributed solver's (x,y) FFT cost
divided by 2). **Estimated savings: ~1.2 ms per step** on the small grid.

The z-FFT (DCT for Bounded) would also benefit from batching if combined with
the y-FFT into a single `plan_forward_transform(storage, Bounded(), [3])` call
that batches across x and y.

Combined with the reduced kernel launch count (501 → ~100), the total FFT cost
should approach the non-distributed solver's 0.71 ms. **Estimated savings: ~3 ms.**

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
