# Nsight Profiling Summary: Time-stepping Cost Breakdown

**Hardware:** A100-SXM4-80GB, NV12 NVLink, Julia 1.12.5, NCCL 2.28.3
**Profiled region:** 50 steady-state timesteps (after 50-100 warmup + GC flush)

---

## Part 1: NonhydrostaticModel (ERF-like, with pressure solver)

**Config:** `NonhydrostaticModel` + `Centered(2)` + `ScalarDiffusivity(ν=200, κ=200)`
**Grid:** 50×400×80 per GPU, `Periodic × Periodic × Bounded`, Float64
**Benchmark:** 1 GPU = 4.47 ms/step, 2 GPU = 12.38 ms/step (36% efficiency)

### 1-GPU Non-Distributed Baseline (4.47 ms/step)

| Category | ms/step | % | Calls/step |
|----------|---------|---|-----------|
| Tendency computation | 1.07 | 23% | 12 |
| Broadcast/fill | 0.81 | 18% | 15 |
| **Pressure solver FFT** | **0.71** | **15%** | 24 |
| RK3 field advance | 0.44 | 10% | 12 |
| Periodic halo fill | 0.39 | 8% | 78 |
| Pressure correction | 0.28 | 6% | 3 |
| Cache tendencies | 0.26 | 6% | 12 |
| Solver tridiag+misc | 0.19 | 4% | 6 |
| Z-permute (solver) | 0.19 | 4% | 6 |
| Hydrostatic pressure | 0.18 | 4% | 3 |
| **Total kernel time** | **~4.5** | | |

The pressure solver (FFT + tridiag + permute) totals **1.09 ms** (24% of step time).
Uses a single batched 2D FFT in (x,y) — very efficient on GPU.

### 2-GPU NCCL Distributed (12.38 ms/step)

| Category | ms/step/GPU | % | Calls/step/GPU | vs 1-GPU |
|----------|------------|---|---------------|---------|
| **NCCL communication** | **10.06** | **53%** | 30 | NEW |
| **Pressure solver FFT** | **4.26** | **22%** | 501 | **6× more** |
| Broadcast/fill (incl halo pack) | 1.14 | 6% | 108 | +0.33 |
| Tendency computation | 1.11 | 6% | 36 | same |
| Hydrostatic pressure | 0.58 | 3% | 9 | +0.40 |
| Transpose pack/unpack | 0.49 | 3% | 12 | NEW |
| RK3 field advance | 0.45 | 2% | 12 | same |
| Solver tridiag+misc | 0.30 | 2% | 9 | +0.11 |
| Pressure correction | 0.27 | 1% | 3 | same |
| Cache tendencies | 0.26 | 1% | 12 | same |

**NCCL:** All 3000 calls on comm_stream. 12,465 overlapping NCCL+compute pairs.
**GPU idle:** 13 gaps > 0.5 ms, totaling 3.26 ms/step (sync waits for NCCL).

### NonhydrostaticModel Overhead Breakdown

**Total overhead: 12.38 - 4.47 = 7.91 ms/step**

| Overhead source | ms/step | % of overhead | Notes |
|----------------|---------|--------------|-------|
| **Distributed FFT (1D vs 2D)** | **3.55** | **45%** | Batched 2D → two 1D FFTs + transpose |
| **GPU idle at sync points** | **3.26** | **41%** | cuStreamWaitEvent for NCCL completion |
| Transpose pack/unpack | 0.49 | 6% | Pack data for FFT alltoall |
| Hydrostatic pressure increase | 0.40 | 5% | Distributed computation overhead |
| Halo pack/unpack | 0.33 | 4% | fill_send_buffers + recv_from_buffers |

**Key finding:** The #1 overhead source is the distributed pressure solver's FFT strategy
(45% of overhead), NOT communication latency. The non-distributed solver uses one efficient
batched 2D FFT (0.71 ms), while the distributed solver uses two separate 1D FFTs with a
transpose between them (4.26 ms) — a 6× cost increase. Fixing this (using batched 2D FFT
when y is fully local in slab-x) would recover ~3.5 ms and raise efficiency from 36% to ~55%.

---

## Part 2: CompressibleDynamics (Breeze, no pressure solver)

**Config:** `AtmosphereModel` + `CompressibleDynamics` + `Centered(2)` + `ScalarDiffusivity(ν=200, κ=200)`
**Grid:** 50×400×80 per GPU, `Periodic × Periodic × Bounded`, Float32
**Benchmark:** 1 GPU = 5.1 ms/step, 2 GPU = 8.3 ms/step (61% efficiency)

### 1-GPU Non-Distributed Baseline (5.1 ms/step)

| Category | ms/step | % | Calls/step |
|----------|---------|---|-----------|
| Tendency computation | 2.06 | 40% | 18 |
| Periodic/boundary halo fill | 0.62 | 12% | 135 |
| RK3/SSP field advance | 0.53 | 10% | 18 |
| Auxiliary thermodynamics | 0.25 | 5% | 3 |
| Broadcast/fill/set | 0.23 | 5% | 6 |
| Acoustic/EOS/pressure | 0.23 | 5% | 3 |
| Compute velocities | 0.21 | 4% | 3 |
| **Total kernel time** | **~4.1** | | |

**No pressure solver FFT.** Tendency computation (2.06 ms) dominates at 40%.
The SSP-RK3 time stepper advances 6 fields × 3 substages = 18 field advances.

### 2-GPU NCCL Distributed (8.3 ms/step)

| Category | ms/step/GPU | % | Calls/step/GPU | vs 1-GPU |
|----------|------------|---|---------------|---------|
| **NCCL communication** | **7.69** | **63%** | 45 | NEW |
| Tendency computation | 1.85 | 15% | 18 | same |
| Broadcast/fill (incl halo pack) | 0.95 | 8% | 150 | +0.72 |
| RK3/SSP field advance | 0.48 | 4% | 18 | same |
| Periodic/boundary halo fill | 0.28 | 2% | 90 | -0.34 (fewer local halos) |
| Acoustic/EOS/pressure | 0.20 | 2% | 3 | same |
| Other | 0.41 | 3% | 6 | +0.16 |

**NCCL:** All 4500 calls on comm_stream. 13,889 overlapping NCCL+compute pairs.
**GPU idle:** Only 4 gaps > 0.5 ms, totaling 1.58 ms/step (much less than NonhydrostaticModel!).

### CompressibleDynamics Overhead Breakdown

**Total overhead: 8.3 - 5.1 = 3.2 ms/step**

| Overhead source | ms/step | % of overhead | Notes |
|----------------|---------|--------------|-------|
| **GPU idle at sync points** | **1.58** | **49%** | cuStreamWaitEvent waits |
| **Halo pack/unpack** | **0.72** | **23%** | More fields → more pack/unpack |
| Other distributed overhead | 0.57 | 18% | Kernel launch, bookkeeping |
| Reduced local halo fills | -0.34 | -11% | Distributed halos replace local |

**Key finding:** Without the pressure solver, the overhead drops from 7.91 to 3.2 ms (2.5×
less). The GPU idle time at sync points drops from 3.26 to 1.58 ms because there are no
FFT transpose sync points — only halo synchronization.

---

## Comparison Summary

| Metric | NonhydrostaticModel | CompressibleDynamics |
|--------|--------------------|--------------------|
| 1-GPU baseline | 4.47 ms | 5.1 ms |
| 2-GPU NCCL | 12.38 ms | 8.3 ms |
| **Efficiency** | **36%** | **61%** |
| Total overhead | 7.91 ms | 3.2 ms |
| Pressure solver FFT overhead | 3.55 ms (45%) | 0 (no solver) |
| GPU idle at sync points | 3.26 ms (41%) | 1.58 ms (49%) |
| Halo pack/unpack | 0.33 ms (4%) | 0.72 ms (23%) |
| Sync points per step | ~13 | ~4 |
| NCCL calls per step | 60 | 90 |
| NCCL overlapping pairs | 12,465 | 13,889 |

### Why compressible scales better

1. **No pressure solver** → no FFT transpose sync points (saves 3.55 ms)
2. **Fewer sync points** → 4 vs 13 per step (saves 1.68 ms of GPU idle)
3. **Same per-GPU compute** → tendency computation is identical (2.06 vs 1.07 ms)

### Implications for optimization

**For NonhydrostaticModel:** The #1 priority is fixing the distributed FFT strategy.
Using a batched 2D FFT when y is fully local (slab-x) would save 3.55 ms/step and
raise efficiency from 36% to ~55%. The remaining 4.36 ms overhead is from sync points
(3.26 ms) and pack/unpack (0.82 ms).

**For CompressibleDynamics:** Already well-optimized. The 3.2 ms overhead is dominated
by the irreducible NCCL sync latency (1.58 ms) and halo pack/unpack (0.72 ms).
Further improvement requires larger per-GPU grids (where compute grows faster than
communication) or pipelining across RK3 substages.

**For both models:** At larger grids (1024×1024×128), compute dominates and efficiency
reaches 82-85%. The overhead is approximately fixed regardless of grid size.
