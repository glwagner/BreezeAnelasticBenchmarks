# Distributed GPU Performance Report

## Hardware and Software

- **GPUs:** 4× NVIDIA A100-SXM4-80GB
- **Interconnect:** NV12 NVLink between all GPU pairs
- **Communication:** NCCL 2.28.3 via `OceananigansNCCLExt` (`NCCLDistributed`)
- **Software:** Julia 1.12.5, Oceananigans (`glw/nccl-distributed-solver`), Breeze (`glw/distributed-tests`)
- **Profiling:** NVIDIA Nsight Systems 2025.3.2

## Benchmark Configurations

| Config | Model | Dynamics | Advection | Closure | Pressure solver | Precision |
|--------|-------|----------|-----------|---------|----------------|-----------|
| **Anelastic (ERF)** | `NonhydrostaticModel` | Incompressible | `Centered(2)` | `ScalarDiffusivity(ν=200,κ=200)` | FFT+tridiagonal | F64 |
| **Anelastic (WENO5)** | `NonhydrostaticModel` | Incompressible | `WENO(order=5)` | None | FFT+tridiagonal | F32 |
| **Compressible** | `AtmosphereModel` | `CompressibleDynamics` | `Centered(2)` | `ScalarDiffusivity(ν=200,κ=200)` | None | F32 |

All benchmarks use weak scaling in x via `Partition(Ngpus, 1)`. Timing uses
the best of 2 trials after a 100-step warmup. All measurements on a single
machine (intra-node NVLink only, no inter-node communication).

---

## Weak Scaling Results

### Small grid: 50×400×80 per GPU

| Config | 1 GPU | 2 GPUs | 4 GPUs | 1→2 eff | 2→4 eff |
|--------|-------|--------|--------|---------|---------|
| Anelastic ERF (F32) | 5.3 ms | 8.6 ms | 9.1 ms | 62% | 95% |
| Compressible (F32) | 5.4 ms | 8.5 ms | 8.2 ms | 64% | 104% |

### Anelastic small grid: NonhydrostaticModel + diffusion, 50×400×80 per GPU

| Config | 1 GPU | 2 GPUs | 4 GPUs | 1→2 eff | 2→4 eff |
|--------|-------|--------|--------|---------|---------|
| F64 (Oceananigans) | 5.79 ms | 12.47 ms | 12.46 ms | 46% | 100% |

### Large grid: 1024×1024×128 per GPU

| Config | 1 GPU | 2 GPUs | 4 GPUs | 1→2 eff | 2→4 eff |
|--------|-------|--------|--------|---------|---------|
| Anelastic WENO5 (F32) | 355.5 ms | 422.7 ms | 421.9 ms | 84% | 100% |
| Compressible (F32) | 189.9 ms | 194.6 ms | 193.9 ms | **98%** | **100%** |

---

## Nsight Profiling Analysis

Profiles collected over 50 steady-state timesteps after warmup + GC flush.

### NonhydrostaticModel (Anelastic): 50×400×80, F64

#### 1-GPU Baseline (5.79 ms/step)

| Category | ms/step | % | Calls/step |
|----------|---------|---|-----------|
| Tendency computation | 1.07 | 24% | 12 |
| Broadcast/fill | 0.81 | 18% | 15 |
| Pressure solver FFT | 0.71 | 16% | 24 |
| RK3 field advance | 0.44 | 10% | 12 |
| Periodic halo fill | 0.39 | 9% | 78 |
| Pressure correction | 0.28 | 6% | 3 |
| Cache tendencies | 0.26 | 6% | 12 |
| Solver tridiag+misc | 0.19 | 4% | 6 |
| Z-permute | 0.19 | 4% | 6 |
| Hydrostatic pressure | 0.18 | 4% | 3 |

Pressure solver total (FFT + tridiag + permute): **1.09 ms (24%)**. Uses a single
efficient batched 2D FFT in (x,y).

#### 2-GPU NCCL Distributed (12.47 ms/step)

| Category | ms/step/GPU | % | vs 1-GPU |
|----------|------------|---|---------|
| NCCL communication | 10.06 | 52% | NEW |
| Pressure solver FFT | 4.26 | 22% | **6× cost increase** |
| Broadcast/fill (incl halo) | 1.14 | 6% | +0.33 |
| Tendency computation | 1.11 | 6% | same |
| Hydrostatic pressure | 0.58 | 3% | +0.40 |
| Transpose pack/unpack | 0.49 | 3% | NEW |
| RK3 field advance | 0.45 | 2% | same |
| Other (tridiag, correction, cache, z-perm, local halos) | 1.21 | 6% | same |

NCCL: all 3000 calls on comm_stream; 12,465 overlapping NCCL+compute pairs.
GPU idle: 13 gaps > 0.5 ms totaling 3.26 ms/step.

#### Overhead Breakdown (6.68 ms/step)

| Source | ms/step | % of overhead |
|--------|---------|--------------|
| **Distributed FFT (1D vs 2D)** | **3.55** | **53%** |
| **GPU idle at 13 sync points** | **3.26** | **49%** |
| Transpose pack/unpack | 0.49 | 7% |
| Hydrostatic pressure increase | 0.40 | 6% |
| Halo pack/unpack | 0.33 | 5% |

Note: percentages exceed 100% because some overlap (NCCL idle partially
overlaps with the FFT cost increase).

**Key finding:** The distributed pressure solver's FFT strategy is the single
largest overhead source. The non-distributed solver uses one batched 2D FFT
(0.71 ms); the distributed solver uses two 1D FFTs with a transpose (4.26 ms).
For slab-x (y is local), the y-FFT could use the batched plan.

#### Per-substage sync point timeline

```
update_state!:
  fill_halo(fields, async=true)         → NCCL on comm_stream
  compute_interior_tendencies!()        → overlaps with NCCL ✓
  synchronize_communication!()          → WAIT #1
  compute_buffer_tendencies!()

rk3_substep!:
  advance fields
  compute_pressure_correction!:
    fill_halo(velocities)               → WAIT #2
    solve_for_pressure!:
      y-FFT → transpose y→x            → WAIT #3
      x-FFT → tridiag → x-IFFT
      transpose x→y                    → WAIT #4
      y-IFFT
    fill_halo(pressure)                 → WAIT #5
  make_pressure_correction!()
```

~5 sync points × 3 substages ≈ 13 per step.

---

### CompressibleDynamics: 50×400×80, F32

#### 1-GPU Baseline (5.4 ms/step)

| Category | ms/step | % | Calls/step |
|----------|---------|---|-----------|
| Tendency computation | 2.06 | 50% | 18 |
| Periodic/boundary halo | 0.62 | 15% | 135 |
| SSP-RK3 field advance | 0.53 | 13% | 18 |
| Auxiliary thermodynamics | 0.25 | 6% | 3 |
| Broadcast/fill/set | 0.23 | 6% | 6 |
| Acoustic/EOS/pressure | 0.23 | 6% | 3 |
| Compute velocities | 0.21 | 5% | 3 |

**No pressure solver FFT.** Tendency computation dominates at 50%.
6 prognostic fields × 3 SSP-RK3 substages = 18 field advances per step.

#### 2-GPU NCCL Distributed (8.5 ms/step)

| Category | ms/step/GPU | % | vs 1-GPU |
|----------|------------|---|---------|
| NCCL communication | 7.69 | 65% | NEW |
| Tendency computation | 1.85 | 16% | same |
| Broadcast/fill (incl halo) | 0.95 | 8% | +0.72 |
| SSP-RK3 field advance | 0.48 | 4% | same |
| Periodic/boundary halo | 0.28 | 2% | -0.34 |
| Other | 0.61 | 5% | |

NCCL: all 4500 calls on comm_stream; 13,889 overlapping NCCL+compute pairs.
GPU idle: 4 gaps > 0.5 ms totaling 1.58 ms/step.

#### Overhead Breakdown (3.1 ms/step)

| Source | ms/step | % of overhead |
|--------|---------|--------------|
| **GPU idle at 4 sync points** | **1.58** | **51%** |
| **Halo pack/unpack** | **0.72** | **23%** |
| Other distributed overhead | 0.80 | 26% |

#### Per-substage sync point timeline

```
update_state!:
  fill_halo(fields, async=true)         → NCCL on comm_stream
  compute_interior_tendencies!()        → overlaps with NCCL ✓
  synchronize_communication!()          → WAIT #1
  compute_buffer_tendencies!()

ssp_rk3_substep!:
  store_initial_state!()                → local copy
  advance all fields                    → local GPU kernels
```

Only **1 sync point per substage** × 3 = 3 per step (plus ~1 misc).

---

## Why Compressible Scales Better

| Factor | NonhydrostaticModel | CompressibleDynamics |
|--------|--------------------|--------------------|
| Has pressure solver | Yes (FFT + tridiagonal) | No |
| Sync points per step | ~13 | ~4 |
| GPU idle per step | 3.26 ms | 1.58 ms |
| FFT overhead | 3.55 ms | 0 ms |
| Total overhead | 6.68 ms | 3.1 ms |
| **Small grid efficiency** | **46–62%** | **62–64%** |
| **Large grid efficiency** | **84%** | **98%** |

The 13→4 sync point reduction comes from eliminating:
- 6 FFT transpose sync points (y→x, x→y per substage)
- 3 velocity halo fills before solver
- 1 pressure halo fill after solver

## NCCL Optimizations Applied

1. **Dedicated comm_stream** — all NCCL off the default stream
2. **Async overlap** — halo fills overlap with interior tendency computation
3. **Pipelined RK3** — tracer halos start before pressure solver
4. **Multi-field batching** — all sides in one NCCL group per field
5. **cuMemcpy2D halo pack** — DMA engine for strided copies
6. **Allocation-reduced** — 6.25 MB/step, no GC spikes in steady state
7. **sync_device! eliminated** — NCCL is stream-native

## Remaining Optimization Opportunities

**For NonhydrostaticModel:**
- Fix distributed FFT to use batched 2D when y is local (biggest single win: ~3.5 ms)
- Communication-avoiding solver (multigrid) would eliminate all FFT transpose syncs

**For CompressibleDynamics:**
- Already near-optimal at large grids (98% efficiency)
- Cross-substage pipelining could save ~1 ms at small grids

**For both:**
- Efficiency improves naturally with grid size (fixed ~3-7 ms overhead vs N³ compute)
- Multi-node testing needed (Slingshot/InfiniBand adds inter-node latency)

---

## Part 3: X-Distribution vs Y-Distribution

**Config:** NonhydrostaticModel + Centered(2) + diffusion, 50×400×80/GPU, F64, 2 GPUs

| Category | x-dist Partition(2,1) | y-dist Partition(1,2) | Difference |
|----------|----------------------|----------------------|-----------|
| NCCL | 10.06 ms | 7.65 ms | **-2.41** |
| FFT | 4.26 ms | 2.61 ms | **-1.65** |
| Tendencies | 1.11 ms | 1.17 ms | +0.06 |
| Broadcast/fill | 1.14 ms | 1.03 ms | -0.11 |
| Transpose pack | 0.49 ms | 0.82 ms | +0.33 |
| Other | 2.23 ms | 2.07 ms | -0.16 |
| **GPU idle gaps** | **3.26 ms (13 gaps)** | **1.41 ms (7 gaps)** | **-1.85** |

### Benchmark results

| Partition | 1 GPU | 2 GPUs | 4 GPUs | 2-GPU eff |
|-----------|-------|--------|--------|----------|
| `Partition(Ngpus, 1)` x-dist | 5.79 | 12.47 | 12.46 | 46% |
| `Partition(1, Ngpus)` y-dist | 5.79 | 10.29 | 9.80 | **56%** |

### FFT kernel launch comparison

| Direction | x-dist | y-dist |
|-----------|--------|--------|
| `regular_fft<400>` (y-FFT) | 480/step | — |
| `regular_fft<800>` (y-FFT after transpose) | — | 240/step |
| `regular_fft<100>` (x-FFT after transpose) | 6/step | — |
| `regular_fft<50>` (x-FFT local) | — | 6/step |
| `regular_fft<80>` (z-permute) | — | 6/step |

### Why y-dist is faster

1. **The x-FFT stays local and contiguous** (dim 1, 6 kernels/step vs 480 for x-dist's strided y-FFT). This saves 1.65 ms in FFT time.

2. **Fewer sync points** (7 vs 13 GPU idle gaps). For slab-y, the solver transposes y→x then x→y. The x-direction is local, so the x-FFT doesn't need communication. The y-FFT needs a transpose, but only in one direction (y is distributed).

3. **NCCL communication is faster** (7.65 vs 10.06 ms) because there's less total data to transfer — the y-halo has fewer cells (Nx × Nz = 50×80 = 4000) vs the x-halo (Ny × Nz = 400×80 = 32000).

### Implications

For grids where one horizontal dimension is much larger than the other (like 50×400×80), **distribute along the larger dimension** for better scaling. This keeps the smaller dimension local and contiguous for FFT.
