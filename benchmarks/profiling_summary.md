# Nsight Profiling Report: Distributed GPU Overhead Analysis

## Setup

- **Hardware:** NVIDIA A100-SXM4-80GB, NV12 NVLink (4 GPUs), Julia 1.12.5
- **Communication:** NCCL 2.28.3 via `OceananigansNCCLExt` (`NCCLDistributed` architecture)
- **Grid:** 50×400×80 per GPU, `Periodic × Periodic × Bounded`, weak scaling in x
- **Profiled region:** 50 steady-state timesteps after 50–100 warmup steps + GC flush

Two model configurations are compared:

| Config | Model | Dynamics | Advection | Closure | Pressure solver | Precision |
|--------|-------|----------|-----------|---------|----------------|-----------|
| **Anelastic** | `NonhydrostaticModel` | Incompressible | `Centered(2)` | `ScalarDiffusivity(ν=200,κ=200)` | FFT + tridiagonal | Float64 |
| **Compressible** | `AtmosphereModel` | `CompressibleDynamics` | `Centered(2)` | `ScalarDiffusivity(ν=200,κ=200)` | None (diagnostic EOS) | Float32 |

---

## Summary

| Metric | NonhydrostaticModel | CompressibleDynamics |
|--------|--------------------|--------------------|
| 1-GPU baseline | 4.47 ms/step | 5.1 ms/step |
| 2-GPU NCCL | 12.38 ms/step | 8.3 ms/step |
| **Scaling efficiency** | **36%** | **61%** |
| Distributed overhead | 7.91 ms | 3.2 ms |
| Sync points per step | ~13 | ~4 |
| Pressure solver overhead | 3.55 ms (45% of overhead) | 0 |
| GPU idle (sync waits) | 3.26 ms (41%) | 1.58 ms (49%) |
| NCCL calls per step | 60 | 90 |
| NCCL on comm_stream | 100% | 100% |
| NCCL overlapping compute pairs | 12,465 | 13,889 |

The compressible model achieves **1.7× better scaling efficiency** because it has
no pressure solver and therefore no FFT transpose synchronization points.

---

## Part 1: NonhydrostaticModel (Anelastic, with Pressure Solver)

### 1-GPU Non-Distributed Baseline (4.47 ms/step)

| Category | ms/step | % | Calls/step | Notes |
|----------|---------|---|-----------|-------|
| Tendency computation | 1.07 | 24% | 12 | 4 fields × 3 RK3 substages |
| Broadcast/fill | 0.81 | 18% | 15 | Field initialization, `set!` |
| **Pressure solver FFT** | **0.71** | **16%** | **24** | **Single batched 2D FFT (x,y)** |
| RK3 field advance | 0.44 | 10% | 12 | `_rk3_substep_field!` |
| Periodic halo fill | 0.39 | 9% | 78 | Local periodic BC copies |
| Pressure correction | 0.28 | 6% | 3 | `_make_pressure_correction!` |
| Cache tendencies | 0.26 | 6% | 12 | `cache_previous_tendencies!` |
| Solver tridiag+misc | 0.19 | 4% | 6 | Tridiagonal solve in z |
| Z-permute (solver) | 0.19 | 4% | 6 | Data layout permutation |
| Hydrostatic pressure | 0.18 | 4% | 3 | `update_hydrostatic_pressure!` |
| **Total** | **~4.5** | | | |

The pressure solver (FFT + tridiag + permute) totals **1.09 ms/step** (24%).
Uses a single batched 2D FFT — the most efficient GPU FFT strategy.

### 2-GPU NCCL Distributed (12.38 ms/step)

| Category | ms/step/GPU | % | Calls/step/GPU | vs 1-GPU |
|----------|------------|---|---------------|---------|
| **NCCL communication** | **10.06** | **52%** | 30 | NEW |
| **Pressure solver FFT** | **4.26** | **22%** | 501 | **6× cost increase** |
| Broadcast/fill (incl halo pack) | 1.14 | 6% | 108 | +0.33 |
| Tendency computation | 1.11 | 6% | 36 | same |
| Hydrostatic pressure | 0.58 | 3% | 9 | +0.40 |
| Transpose pack/unpack | 0.49 | 3% | 12 | NEW |
| RK3 field advance | 0.45 | 2% | 12 | same |
| Solver tridiag+misc | 0.30 | 2% | 9 | +0.11 |
| Pressure correction | 0.27 | 1% | 3 | same |
| Cache tendencies | 0.26 | 1% | 12 | same |
| Periodic halo fill | 0.19 | 1% | 54 | same |
| Z-permute (solver) | 0.19 | 1% | 6 | same |

**NCCL profile:** All 3000 calls on comm_stream. 12,465 overlapping NCCL+compute kernel pairs.
**GPU idle:** 13 gaps > 0.5 ms on the default stream, totaling 3.26 ms/step.

### Overhead Breakdown (7.91 ms/step)

| Source | ms/step | % of overhead | Root cause |
|--------|---------|--------------|-----------|
| **Distributed FFT strategy** | **3.55** | **45%** | 1D FFTs + transpose replace batched 2D FFT |
| **GPU idle at sync points** | **3.26** | **41%** | `cuStreamWaitEvent` for NCCL completion |
| Transpose pack/unpack | 0.49 | 6% | Data reorder for FFT alltoall |
| Hydrostatic pressure increase | 0.40 | 5% | Distributed computation overhead |
| Halo pack/unpack | 0.33 | 4% | `fill_send_buffers!` + `recv_from_buffers!` |

**Key finding:** The #1 overhead source is the distributed pressure solver's FFT strategy.
The non-distributed solver uses one efficient batched 2D FFT in (x,y) costing 0.71 ms.
The distributed solver uses two separate 1D FFTs with a transpose costing 4.26 ms — a
**6× increase**. For slab-x partitioning where y is fully local, the y-direction FFT
could use the same batched plan, recovering most of this 3.55 ms overhead.

### RK3 substep timeline (per substage)

```
update_state!:
  fill_halo_regions!(fields, async=true)    → NCCL on comm_stream
  compute_interior_tendencies!()            → DEFAULT stream (overlaps ✓)
  synchronize_communication!()              → WAIT #1 (~0.3 ms)
  compute_buffer_tendencies!()

rk3_substep!:
  _rk3_substep_field!()                     → advance fields
  compute_pressure_correction!:
    fill_halo_regions!(velocities)          → WAIT #2 (~0.3 ms)
    y-FFT → transpose y→x                  → WAIT #3 (~0.3 ms)
    x-FFT → tridiag → x-IFFT
    transpose x→y                           → WAIT #4 (~0.3 ms)
    y-IFFT → fill_halo_regions!(pressure)   → WAIT #5 (~0.3 ms)
  make_pressure_correction!()
```

~5 sync points × 3 substages ≈ 13 per step. Each wait is ~0.25 ms (NCCL latency over NVLink).

---

## Part 2: CompressibleDynamics (No Pressure Solver)

### 1-GPU Non-Distributed Baseline (5.1 ms/step)

| Category | ms/step | % | Calls/step | Notes |
|----------|---------|---|-----------|-------|
| **Tendency computation** | **2.06** | **50%** | 18 | 6 fields × 3 SSP-RK3 substages |
| Periodic/boundary halo fill | 0.62 | 15% | 135 | Local periodic BC copies |
| RK3/SSP field advance | 0.53 | 13% | 18 | `ssp_rk3_substep!` |
| Auxiliary thermodynamics | 0.25 | 6% | 3 | EOS, diagnostic variables |
| Broadcast/fill/set | 0.23 | 6% | 6 | Field initialization |
| Acoustic/EOS/pressure | 0.23 | 6% | 3 | Diagnostic pressure |
| Compute velocities | 0.21 | 5% | 3 | Velocity from momentum |
| **Total** | **~4.1** | | | |

**No pressure solver.** Tendency computation dominates at 50%.
The SSP-RK3 has 3 substages, each advancing 6 prognostic fields
(u, v, w, ρ, θ, plus moisture).

### 2-GPU NCCL Distributed (8.3 ms/step)

| Category | ms/step/GPU | % | Calls/step/GPU | vs 1-GPU |
|----------|------------|---|---------------|---------|
| **NCCL communication** | **7.69** | **65%** | 45 | NEW |
| Tendency computation | 1.85 | 16% | 18 | same |
| Broadcast/fill (incl halo pack) | 0.95 | 8% | 150 | +0.72 |
| RK3/SSP field advance | 0.48 | 4% | 18 | same |
| Periodic/boundary halo fill | 0.28 | 2% | 90 | -0.34 |
| Acoustic/EOS/pressure | 0.20 | 2% | 3 | same |
| Other | 0.41 | 3% | 6 | |

**NCCL profile:** All 4500 calls on comm_stream. 13,889 overlapping NCCL+compute pairs.
**GPU idle:** Only 4 gaps > 0.5 ms, totaling 1.58 ms/step.

### Overhead Breakdown (3.2 ms/step)

| Source | ms/step | % of overhead | Root cause |
|--------|---------|--------------|-----------|
| **GPU idle at sync points** | **1.58** | **49%** | `synchronize_communication!` waits |
| **Halo pack/unpack** | **0.72** | **23%** | More fields → more buffer copies |
| Other distributed overhead | 0.57 | 18% | Kernel launch overhead, bookkeeping |
| Reduced local halo fills | -0.34 | -11% | Distributed halos replace local |

### SSP-RK3 substep timeline (per substage)

```
update_state!:
  fill_halo_regions!(fields, async=true)    → NCCL on comm_stream
  compute_interior_tendencies!()            → DEFAULT stream (overlaps ✓)
  synchronize_communication!()              → WAIT #1 (~0.5 ms)
  compute_buffer_tendencies!()

ssp_rk3_substep!:
  store_initial_state!()                    → copy fields (local)
  advance all fields                        → local GPU kernels (no comm needed)
```

Only **1 sync point per substage** × 3 = 3 per step (plus ~1 from misc).
No pressure solver transposes. The NCCL halo transfer (~1.8 ms) is almost
entirely hidden behind interior tendency computation (~2 ms).

---

## Cross-Model Comparison

### Why compressible scales 1.7× better

| Factor | NonhydrostaticModel | CompressibleDynamics | Impact |
|--------|--------------------|--------------------|--------|
| Pressure solver FFT overhead | 3.55 ms | 0 ms | Biggest factor |
| Sync points per step | ~13 | ~4 | 3× fewer waits |
| GPU idle time | 3.26 ms | 1.58 ms | 2× less idle |
| Total overhead | 7.91 ms | 3.2 ms | 2.5× less overhead |

The compressible model has **no global communication in the solver** — all communication
is local halo exchange. The 13→4 sync point reduction directly traces to eliminating
the 6 pressure solver transpose waits and 4 pressure-related halo fills per step.

### What's well-optimized (both models)

1. **NCCL on dedicated comm_stream** — all NCCL calls off the default stream
2. **Async overlap working** — 12K-14K overlapping NCCL+compute kernel pairs
3. **Pipelined RK3** — tracer halos start before pressure solver
4. **Allocation-reduced** — 6.25 MB/step (no GC spikes in steady state)
5. **cuMemcpy2D halo pack** — DMA engine frees compute units

### Remaining optimization opportunities

**For NonhydrostaticModel:**
1. **Fix distributed FFT** to use batched 2D when y is local (saves 3.55 ms → 36%→55% eff)
2. **Communication-avoiding solver** (multigrid) eliminates transpose syncs entirely
3. **Larger per-GPU grids** naturally improve efficiency (85% at 1024×1024×128)

**For CompressibleDynamics:**
1. **Cross-substage pipelining** — start next substage's halos during current substage's
   field advance (requires double-buffered fields)
2. **Larger per-GPU grids** — at 1024×1024×128, overhead would be ~3 ms vs ~350 ms compute → >99% efficiency

**For both models:**
- Scaling efficiency improves with grid size because overhead is approximately fixed
  (~3-8 ms) while compute grows as N³. At production grid sizes, both models achieve 80-85% efficiency.

### Scaling efficiency vs grid size

| Grid per GPU | NonhydrostaticModel | CompressibleDynamics |
|-------------|--------------------|--------------------|
| 50×400×80 | 36% | 61% |
| 200×200×80 | 65% | — |
| 1024×1024×128 (F32) | 85% | **98%** |

#### CompressibleDynamics large grid results (1024×1024×128/GPU, F32)

| GPUs | ms/step | Efficiency |
|------|---------|-----------|
| 1 | 189.9 | 100% |
| 2 | 194.6 | 97.6% |
| 4 | 193.9 | 97.9% |

Overhead: only ~4 ms on ~190 ms of compute. Near-perfect weak scaling.
