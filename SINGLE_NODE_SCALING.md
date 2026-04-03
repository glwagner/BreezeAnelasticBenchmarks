# Single-Node GPU Scaling Results

Benchmarks on a single node with 8× NVIDIA A100 SXM4 80 GB GPUs
connected via NVLink. All communication uses NCCL via the
`OceananigansNCCLExt` extension.

## Models Tested

| Model | Solver | Microphysics | Prognostic fields |
|-------|--------|-------------|-------------------|
| **HFSM** | SplitExplicit (30 substeps) | — | u, v, T, S, η |
| **NHM** | FFT pressure (distributed transposes) | — | u, v, w, T, S |
| **AM dry** | Anelastic FFT pressure | — | u, v, w, b |
| **GATE SatAdj** | Anelastic FFT pressure | SaturationAdjustment (mixed phase) | u, v, w, θ, qᵉ |
| **GATE 1M mixed** | Anelastic FFT pressure | 1-moment non-equilibrium (liquid + ice) | u, v, w, θ, qᵛ, qᶜˡ, qᶜⁱ, qʳ, qˢ |

HFSM and NHM use WENO(order=7); AM and GATE use WENO(order=5).
All use periodic horizontal boundary conditions.

---

## GATE III GigaLES — Full Resolution Performance

### 2048×2048×181 (stretched vertical grid, 100 m horizontal resolution)

> Requires ~96 GB in F32. Does not fit on a single A100.

**Step time (ms) and 4→8 GPU scaling**

| Microphysics | Fields | GPUs | F32 (ms) | F64 (ms) | F64/F32 |
|-------------|--------|------|---------|---------|---------|
| SatAdj | 5 | 4 (4×1) | 688.6 | 1187.6 | 1.72× |
| SatAdj | 5 | 8 (8×1) | 348.8 | 605.8 | 1.74× |
| 1M mixed | 9 | 4 (4×1) | 1128.8 | OOM | — |
| 1M mixed | 9 | 8 (8×1) | 576.3 | 1060.6 | 1.84× |

4→8 GPU efficiency: **97% (SatAdj F32)**, **96% (SatAdj F64)**, **98% (1M F32)**.

F64 1M mixed at 4 GPUs OOMs (9 fields × Float64 exceeds 80 GB/GPU).

### SDPD (Simulated Days Per Day) on 8 A100s

ms/step is nearly constant across dt because the anelastic pressure
solver dominates and its cost is independent of dt.

**SatAdj microphysics (F32)**

| dt (s) | ms/step | SDPD |
|--------|---------|------|
| 0.5 | 349.5 | 1.4 |
| 1.0 | 349.2 | 2.9 |
| 2.0 | 364.7 | 5.5 |
| 3.0 | 377.7 | 7.9 |
| 4.0 | 377.6 | **10.6** |

**1M mixed-phase microphysics**

| dt (s) | F32 ms/step | F32 SDPD | F64 ms/step | F64 SDPD |
|--------|------------|----------|------------|----------|
| 0.5 | 587.9 | 0.9 | 1069.8 | 0.5 |
| 1.0 | 575.8 | 1.7 | 1060.5 | 0.9 |
| 2.0 | 578.3 | 3.5 | 1076.7 | 1.9 |
| 3.0 | 580.6 | 5.2 | 1127.8 | 2.7 |
| 4.0 | 580.4 | **6.9** | 1127.8 | **3.5** |

At CFL-limited dt (3–4 s for 100 m resolution with ~20 m/s winds):
- **SatAdj F32: 8–11 SDPD**
- **1M mixed F32: 5–7 SDPD**
- **1M mixed F64: 3–4 SDPD**

---

## Strong Scaling

Fixed total grid, distributed across more GPUs.

### HFSM — 2800×2800×100 (SplitExplicit, no FFT solver)

| Partition | GPUs | ms/step | Speedup | Efficiency |
|-----------|------|---------|---------|------------|
| 1×1 | 1 | 802.4 | 1.00× | 100% |
| 2×1 | 2 | 411.5 | 1.95× | 97% |
| 1×2 | 2 | 410.3 | 1.96× | 98% |
| 4×1 | 4 | 211.8 | 3.79× | 95% |
| 2×2 | 4 | 211.4 | 3.80× | 95% |
| 1×4 | 4 | 208.5 | 3.85× | 96% |
| 8×1 | 8 | 108.7 | 7.38× | 92% |
| 4×2 | 8 | 110.3 | 7.27× | 91% |
| 2×4 | 8 | 107.7 | 7.45× | 93% |
| 1×8 | 8 | 105.6 | 7.60× | **95%** |

### NHM — 2400×2400×100 (FFT pressure solver)

| Partition | GPUs | ms/step | Speedup | Efficiency |
|-----------|------|---------|---------|------------|
| 1×1 | 1 | 2683.6 | 1.00× | 100% |
| 2×1 | 2 | 1645.8 | 1.63× | 82% |
| 1×2 | 2 | 1817.8 | 1.48× | 74% |
| 4×1 | 4 | 840.4 | 3.19× | 80% |
| 2×2 | 4 | 989.5 | 2.71× | 68% |
| 1×4 | 4 | 900.3 | 2.98× | 75% |
| 8×1 | 8 | 433.2 | 6.19× | **77%** |
| 4×2 | 8 | 497.5 | 5.39× | 67% |
| 2×4 | 8 | 479.9 | 5.59× | 70% |

### AM dry — 2400×2400×80 (Anelastic FFT solver)

| Partition | GPUs | ms/step | Speedup | Efficiency |
|-----------|------|---------|---------|------------|
| 1×1 | 1 | 1294.6 | 1.00× | 100% |
| 2×1 | 2 | 1058.8 | 1.22× | 61% |
| 1×2 | 2 | 1177.1 | 1.10× | 55% |
| 4×1 | 4 | 453.0 | 2.86× | 71% |
| 2×2 | 4 | 683.5 | 1.89× | 47% |
| 1×4 | 4 | 580.1 | 2.23× | 56% |
| 8×1 | 8 | 233.3 | 5.55× | **69%** |
| 4×2 | 8 | 350.0 | 3.70× | 46% |
| 2×4 | 8 | 543.6 | 2.38× | 30% |
| 1×8 | 8 | 285.2 | 4.54× | 57% |

---

## Weak Scaling

400×400 horizontal grid points per GPU (HFSM/NHM: Nz=100, AM/GATE: Nz≈150–181).

### Summary at 8 GPUs (best partition)

| Model | 1-GPU (ms) | 8×1 (ms) | Efficiency |
|-------|-----------|----------|------------|
| HFSM (SplitExplicit) | 16.7 | 18.7 | **89%** |
| NHM (FFT solver) | 74.8 | 97.5 | **77%** |
| AM dry (Anelastic FFT) | 45.6 | 64.7 | **71%** |

### HFSM — 400×400×100/GPU

| Partition | GPUs | ms/step | Efficiency |
|-----------|------|---------|------------|
| 1×1 | 1 | 16.7 | 100% |
| 2×1 | 2 | 18.6 | 90% |
| 1×2 | 2 | 18.4 | 91% |
| 4×1 | 4 | 19.4 | 86% |
| 2×2 | 4 | 21.7 | 77% |
| 1×4 | 4 | 20.9 | 80% |
| 8×1 | 8 | 18.7 | 89% |
| 4×2 | 8 | 22.5 | 74% |
| 2×4 | 8 | 22.0 | 76% |
| 1×8 | 8 | 20.8 | 80% |

### NHM — 400×400×100/GPU

| Partition | GPUs | ms/step | Efficiency |
|-----------|------|---------|------------|
| 1×1 | 1 | 74.8 | 100% |
| 2×1 | 2 | 95.6 | 78% |
| 1×2 | 2 | 101.2 | 74% |
| 4×1 | 4 | 96.4 | 78% |
| 2×2 | 4 | 116.2 | 64% |
| 1×4 | 4 | 98.9 | 76% |
| 8×1 | 8 | 97.5 | 77% |
| 4×2 | 8 | 117.4 | 64% |
| 2×4 | 8 | 114.2 | 66% |

### AM dry — 400×400×100/GPU

| Partition | GPUs | ms/step | Efficiency |
|-----------|------|---------|------------|
| 1×1 | 1 | 45.6 | 100% |
| 2×1 | 2 | 64.3 | 71% |
| 1×2 | 2 | 82.8 | 55% |
| 4×1 | 4 | 63.8 | 71% |
| 2×2 | 4 | 98.6 | 46% |
| 1×4 | 4 | 81.1 | 56% |
| 8×1 | 8 | 64.7 | 71% |
| 4×2 | 8 | 100.7 | 45% |
| 2×4 | 8 | 99.9 | 46% |

---

## F64 vs F32 Performance

Single GPU, GATE setup (400×400×~150).

| Microphysics | F32 (ms) | F64 (ms) | F64/F32 |
|-------------|---------|---------|---------|
| SatAdj (mixed phase) | 75.5 | 139.9 | 1.85× |
| 1M mixed phase | 142.0 | 272.8 | 1.92× |

The F64/F32 ratio of ~1.7–1.9× is less than the A100's theoretical 2:1
F32:F64 throughput ratio because NCCL communication (60–90% of step time)
is approximately constant — NVLink bandwidth is not saturated even at 2×
the bytes per transfer. The solver itself operates at the model precision
(ComplexF32 FFTs, F32 tridiagonal coefficients), with only the Poisson
eigenvalues stored in Float64 for numerical accuracy.

---

## Nsight Systems Profiling (8 GPUs)

GPU kernel time breakdown for 3 time steps.

### HFSM (2×4, 2800×2800×100)

| % GPU time | Kernel |
|-----------|--------|
| 89.5% | NCCL Send/Recv (halo communication) |
| 2.3% | `compute_hydrostatic_free_surface_Gu_` |
| 2.3% | `compute_hydrostatic_free_surface_Gv_` |
| 3.4% | `compute_hydrostatic_free_surface_Gc_` (T, S) |
| 0.1% | Split-explicit barotropic substeps |

### NHM (8×1, 1600×1600×100)

| % GPU time | Kernel |
|-----------|--------|
| 69.2% | NCCL Send/Recv (halo + FFT transpose) |
| 2.8% | `permutedims_kernel_` (FFT layout transpose) |
| 18.1% | Tendency kernels (Gu, Gv, Gw, Gc) |
| 0.5% | `vector_fft` (FFT compute) |
| 1.0% | Pack/unpack buffers |

Of the 69% NCCL time:
- **62%** is FFT transpose Alltoall (transfers >10 ms)
- **8%** is halo communication (transfers <10 ms)

### AM dry (8×1, 800×800×80)

| % GPU time | Kernel |
|-----------|--------|
| 91.6% | NCCL Send/Recv |
| 1.1% | `permutedims_kernel_` |
| 4.5% | Tendency kernels |
| 0.2% | FFT compute |

---

## Maximum Domain Size

At 100 m horizontal resolution with the GATE stretched vertical grid
(~181 levels), SatAdj microphysics, Float32:

| Per-GPU grid | Memory | Fits A100 80 GB? |
|-------------|--------|-------------------|
| 800×800×181 | 16 GB | Yes |
| 1000×1000×181 | 41 GB | Yes |
| 1200×1200×181 | 60 GB | Yes |
| 1400×1400×181 | 83 GB | No |

**Maximum on 8 A100s (8×1 partition): 9600×1200×181 = 960 km × 120 km × 27 km**

---

## Key Findings

1. **HFSM scales near-perfectly** (92–95% at 8 GPUs) because it avoids the
   FFT pressure solver entirely. Partition geometry has minimal impact.

2. **x-only partitions are optimal** for FFT-based models (NHM, AM, GATE).
   The distributed FFT transposes in y→x, so x-partition keeps y local.
   2D partitions add 10–20% overhead from additional communication.

3. **The transpose dispatch fix doubled efficiency** for FFT-based models.
   The original code dispatched 3 of 4 transpose directions to CPU-staged
   MPI instead of NCCL.

4. **Native `ncclAlltoAll` = grouped Send/Recv** in performance (444.7 vs
   441.5 ms at 8 GPUs). The bottleneck is data volume, not collective
   implementation.

5. **GATE GigaLES (2048×2048×181)** achieves 96–98% scaling from 4→8 GPUs
   across all microphysics and precision configurations.

6. **SDPD at full resolution on 8 A100s**: SatAdj F32 achieves **10.6 SDPD**;
   1M mixed-phase F32 achieves **6.9 SDPD**; 1M mixed-phase F64 achieves
   **3.5 SDPD** (all at dt=4s).

7. **F64/F32 ratio is 1.7–1.9×**, less than the theoretical 2× because
   NCCL communication (60–90% of step time) is approximately constant
   regardless of precision — NVLink bandwidth is not saturated.

---

## Platform Details

- **GPUs**: 8× NVIDIA A100-SXM4-80GB
- **Interconnect**: NVLink (intra-node)
- **CUDA**: 13.0
- **NCCL**: 2.28.3
- **Julia**: 1.12
- **Oceananigans**: v0.105+ (branch `glw/nccl-distributed-solver`)
- **Breeze**: branch `glw/distributed-tests`
- **Communication**: NCCL via `OceananigansNCCLExt` (no GPU-aware MPI required)
