# Nsight Profiling Summary: Time-stepping Cost Breakdown

## Configuration

- **Model:** `NonhydrostaticModel` + `Centered(2)` + `ScalarDiffusivity(ν=200, κ=200)`
- **Grid:** 50×400×80 per GPU, `Periodic × Periodic × Bounded`
- **Precision:** Float64
- **Hardware:** A100-SXM4-80GB, NV12 NVLink
- **Profiled region:** 50 steady-state timesteps (after 50-100 warmup steps)

## 1-GPU Non-Distributed Baseline

**Wall clock: 4.47 ms/step** (benchmark), 7.25 ms/step (nsight, includes profiling overhead)

| Category | ms/step | % of total | Calls/step |
|----------|---------|-----------|-----------|
| Tendency computation | 1.07 | 15% | 12 |
| Broadcast/fill | 0.81 | 11% | 15 |
| Pressure solver FFT | 0.71 | 10% | 24 |
| RK3 field advance | 0.44 | 6% | 12 |
| Periodic halo fill | 0.39 | 5% | 78 |
| Pressure correction | 0.28 | 4% | 3 |
| Cache tendencies | 0.26 | 4% | 12 |
| Solver tridiag+misc | 0.19 | 3% | 6 |
| Z-permute (solver) | 0.19 | 3% | 6 |
| Hydrostatic pressure | 0.18 | 2% | 3 |

Note: kernel times sum to ~4.5 ms; the remainder is kernel launch overhead
and CPU-side orchestration.

## 2-GPU NCCL Distributed

**Wall clock: 12.38 ms/step** (benchmark)

| Category | ms/step/GPU | % of kernel time | Calls/step/GPU | 1-GPU comparison |
|----------|------------|-----------------|---------------|-----------------|
| **NCCL communication** | **10.06** | **53%** | 30 | NEW (0 in 1-GPU) |
| Pressure solver FFT | 4.26 | 22% | 501 | 6× more calls (distributed 1D FFTs) |
| Broadcast/fill (incl halo pack) | 1.14 | 6% | 108 | +0.33 (halo buffers) |
| Tendency computation | 1.11 | 6% | 36 | +0.04 (same per-GPU work) |
| Hydrostatic pressure | 0.58 | 3% | 9 | +0.40 |
| Transpose pack/unpack | 0.49 | 3% | 12 | NEW (FFT transpose) |
| RK3 field advance | 0.45 | 2% | 12 | same |
| Solver tridiag+misc | 0.30 | 2% | 9 | +0.11 |
| Pressure correction | 0.27 | 1% | 3 | same |
| Cache tendencies | 0.26 | 1% | 12 | same |
| Periodic halo fill | 0.19 | 1% | 54 | same |
| Z-permute (solver) | 0.19 | 1% | 6 | same |

## Overhead Analysis

**Total overhead: 12.38 - 4.47 = 7.91 ms/step**

| Overhead source | ms/step | % of overhead | How it arises |
|----------------|---------|--------------|--------------|
| GPU idle at sync points | 3.26 | 41% | `cuStreamWaitEvent` waiting for NCCL |
| Pressure solver FFT increase | 3.55 | 45% | 1D FFTs replace batched 2D FFT |
| Halo pack/unpack | 0.33 | 4% | `fill_send_buffers!` + `recv_from_buffers!` |
| Transpose pack/unpack | 0.49 | 6% | Pack for FFT alltoall transpose |
| Hydrostatic pressure increase | 0.40 | 5% | Distributed computation overhead |

### Breakdown of the 3.26 ms GPU idle time

13 gaps > 0.5 ms on the default stream, totaling 162.8 ms over 50 steps = 3.26 ms/step.
These are the `cuStreamWaitEvent` points where the default stream waits for NCCL:

| Sync point | Per substage | Per step (×3 RK3) | Avg wait |
|------------|-------------|-------------------|---------|
| `synchronize_communication!` | 1 | 3 | ~0.3 ms |
| Pressure solver y→x transpose | 1 | 3 | ~0.3 ms |
| Pressure solver x→y transpose | 1 | 3 | ~0.3 ms |
| Velocity halo fill (pressure corr.) | ~0.3 | ~1 | ~0.3 ms |

### Pressure solver FFT cost increase

The non-distributed solver uses a **single batched 2D FFT** in (x,y).
The distributed solver uses **two separate 1D FFTs** (y then x) with a
transpose between them. This changes the FFT cost from 0.71 to 4.26 ms/step.

The 1D FFTs have more overhead per transform (smaller batch size, more kernel
launches) and the transpose adds pack/unpack + NCCL communication.

## NCCL Communication Profile

All 3000 NCCL calls (60/step) run on the dedicated comm_stream.
**12,465 overlapping NCCL+compute kernel pairs** confirm async overlap works.

| Duration bucket | Calls | Total (ms) | Avg (μs) |
|----------------|-------|-----------|---------|
| < 50 μs | 1200 | 39 | 33 |
| 50-200 μs | 807 | 101 | 125 |
| 0.2-1 ms | 738 | 285 | 386 |
| 1-5 ms | 248 | 413 | 1,667 |
| > 5 ms | 7 | 168 | 23,951 |

The 7 calls > 5 ms (168 ms total) are NCCL initialization outliers.
Steady-state NCCL: 2993 calls, 838 ms, **280 μs average**.

## Key Takeaways

1. **The dominant overhead is the distributed FFT solver** (3.55 ms/step, 45% of overhead).
   The 1D FFT strategy is 6× more expensive than the batched 2D FFT.

2. **GPU idle time at sync points** is 3.26 ms/step (41% of overhead).
   This is the irreducible NCCL latency at 13 synchronization points per step.

3. **NCCL communication itself is well-hidden** — all on comm_stream, 12K+ overlapping
   compute pairs. The latency only shows when the GPU must wait for results.

4. **Pack/unpack is a small fraction** (0.82 ms combined, 10% of overhead).

5. **Tendency computation is the same** on 1 and 2 GPUs (1.07 vs 1.11 ms) — weak scaling
   of compute is essentially perfect.
