# Perlmutter vs Derecho: Performance Investigation

## Summary

The reported "4x slowdown" on Derecho is **not a hardware issue**. Diagnostic benchmarks
show identical single-GPU performance on both systems (~0.61 s). The slowdown is caused
by the **Distributed architecture code path** being used even for single-GPU runs.

## Key Evidence

### Diagnostic benchmarks (non-distributed, single GPU)

| System     | GPU                    | Trial 1 | Trial 2 | Trial 3 |
|------------|------------------------|---------|---------|---------|
| Perlmutter | A100-SXM4-80GB (?)     | 0.615 s | 0.612 s | —       |
| Derecho    | A100-SXM4-40GB         | 0.614 s | 0.615 s | 0.615 s |
| Derecho    | (pool=none)            | 0.635 s | 0.620 s | 0.615 s |
| Derecho    | (pool=cuda)            | 0.614 s | 0.618 s | 0.615 s |

**Result: Performance is identical.** The CUDA memory pool setting makes no difference.

### Distributed benchmarks (Derecho weak scaling)

**Early runs (default memory pool, jobs 5608xxx):**

| GPUs | Nodes | Trial 1  | Trial 2  | Job     |
|------|-------|----------|----------|---------|
| 1    | 1     | 2.319 s  | 2.417 s  | 5608630 |
| 1    | 1     | 2.595 s  | 2.669 s  | 5608759 |
| 2    | 1     | 7.959 s  | 7.475 s  | 5608760 |
| 4    | 1     | 7.234 s  | 7.642 s  | 5608761 |
| 8    | 2     | 6.867 s  | 7.127 s  | 5608762 |

**Latest runs (jobs 5616xxx):**

| GPUs | Nodes | Pool    | Trial 1  | Trial 2  | Job     | Status |
|------|-------|---------|----------|----------|---------|--------|
| 1    | 1     | default | 2.687 s  | 2.677 s  | 5616519 | OK     |
| 1    | 1     | none    | 2.502 s  | 2.571 s  | 5616520 | OK     |
| 1    | 1     | default | 2.608 s  | 2.583 s  | 5616279 | OK     |
| 2    | 1     | default | —        | —        | 5616280 | FAILED |
| 2    | 1     | none    | 7.901 s  | 7.623 s  | 5616392 | OK     |
| 4    | 1     | none    | 7.011 s  | 6.759 s  | 5616391 | OK     |
| 4    | 1     | none    | 7.166 s  | 6.835 s  | 5616393 | OK     |
| 8    | 2     | default | —        | —        | 5616281 | FAILED |
| 8    | 2     | none    | —        | —        | —       | NOT RUN|

The 1-GPU distributed run (~2.6 s) is 4x slower than the 1-GPU non-distributed run (~0.61 s).

### CUDA-aware MPI requires `JULIA_CUDA_MEMORY_POOL=none`

The 2-GPU and 8-GPU runs without `JULIA_CUDA_MEMORY_POOL=none` failed with:
```
(GTL DEBUG) cuIpcGetMemHandle: invalid argument, CUDA_ERROR_INVALID_VALUE, line no 148
MPIError: Invalid count
```

The default CUDA.jl binned memory pool allocates memory that cannot be shared via
`cuIpcGetMemHandle` between MPI ranks. Setting `JULIA_CUDA_MEMORY_POOL=none` forces
direct CUDA allocations which are IPC-compatible. This is required for multi-GPU
CUDA-aware MPI on Derecho (and likely any Cray MPICH system using GTL).

Note: 1-GPU distributed runs work with any pool setting because no inter-rank IPC occurs.

**8-GPU run with `pool=none` has not been run yet.**

### Perlmutter weak scaling (distributed)

Results pending — no distributed benchmark results from Perlmutter are available for comparison.

## Root Cause Analysis (Updated 2026-03-26)

### Profiling results (job 5627505)

| Component | GPU | Distributed | Overhead |
|-----------|-----|------------|----------|
| 10 time steps | 0.997 s | 2.939 s | 2.95x |
| Solver only (30x) | 0.092 s | 0.236 s | +0.143 s |
| Momentum halo (90x) | 0.015 s | 0.023 s | +0.009 s |
| Pressure halo (30x) | 0.007 s | 0.007 s | 0 |
| **Unaccounted** | — | — | **+1.791 s (92%)** |

FFT comparison (60 calls):
| Batched 2D | Two 1D | Reshaped 1D |
|-----------|--------|-------------|
| 0.031 s | 0.287 s (9.3x!) | 0.012 s |

### The pressure solver is NOT the issue

Both distributed and non-distributed models use the same `FourierTridiagonalPoissonSolver`
(not `DistributedFourierTridiagonalPoissonSolver`). The Breeze dispatch in
`dynamics_pressure_solver()` creates a regular solver on the local grid.

### The real bottleneck: cumulative GPU sync stalls in time-stepping

92% of the overhead is NOT in the solver or isolated halo fills. It's in the
**cumulative `sync_device!` (CUDA.synchronize) calls** scattered throughout the
distributed time-stepping code.

On a `DistributedGrid`, even with 1 rank, each time step triggers:

1. **`update_state!`** calls `fill_halo_regions!(prognostic_fields, async=true)`
2. **`compute_velocities!`** calls `fill_halo_regions!` 3 times (density, momentum, velocities)
   - Each call triggers `fill_corners!` which calls `sync_device!(arch)`
   - `sync_device!` on Distributed GPU = `CUDA.synchronize()` = flush GPU pipeline
3. **`compute_pressure_correction!`** calls `fill_halo_regions!` twice more
4. Per stage total: ~5+ `sync_device!` calls × 3 RK3 stages × 10 timesteps = **150+ GPU syncs**

In isolation, `CUDA.synchronize()` costs only 1.5 μs. But in the actual time step, each
sync **interrupts a stream of async GPU kernel launches**, forcing the GPU to drain its
pipeline before proceeding. This breaks the GPU's ability to overlap kernel execution
and memory transfers, causing massive throughput loss.

The non-distributed code path has NO `sync_device!` calls in halo fills (no corners to
communicate), so the GPU pipeline runs uninterrupted.

### The weak scaling issue: 2+ GPUs are 3x slower than 1 GPU (distributed)

Going from 1 to 2 distributed GPUs jumps from ~2.4 s to ~7.5 s (3x increase).
This is a separate issue from the 4x Distributed overhead on 1 GPU.
Possible causes:
- Halo communication overhead (WENO5 requires 5-point halos)
- Pressure solver communication (global transpose for Fourier solve)
- Possible serialization or synchronization bottlenecks in the distributed path

## Hardware Comparison

| Spec                | Perlmutter (NERSC)          | Derecho (NCAR)              |
|---------------------|-----------------------------|-----------------------------|
| GPU model           | A100-SXM4-80GB (HBM2e)*    | A100-SXM4-40GB (HBM2)      |
| Memory bandwidth    | 2,039 GB/s*                 | 1,555 GB/s                  |
| GPUs per node       | 4                           | 4                           |
| NVLink bandwidth    | 600 GB/s                    | 600 GB/s                    |
| CPU                 | AMD Milan (1 socket/GPU node)| AMD Milan (1 socket/GPU node)|
| Interconnect        | HPE Slingshot-11            | HPE Slingshot-11            |
| Julia version       | 1.12.1                      | 1.12.5                      |
| CUDA runtime        | 12.x                        | 13.2.0                      |
| Job scheduler       | SLURM                       | PBS                         |
| MPI                 | Cray MPICH                  | Cray MPICH                  |

*Perlmutter uses `--constraint=gpu` (not `gpu&hbm80g`), so it may run on either 40GB
or 80GB nodes. The CLAUDE.md states 80GB. If Perlmutter used 80GB nodes, the 31% higher
memory bandwidth would benefit memory-bound kernels, but the diagnostic results show this
is not the source of the discrepancy.

**Key hardware difference:** Derecho has A100-40GB (1,555 GB/s bandwidth) while
Perlmutter likely has A100-80GB (2,039 GB/s bandwidth). This is a 31% bandwidth advantage
for Perlmutter, but since both systems achieve ~0.61 s on non-distributed benchmarks,
this difference is not the bottleneck for the current grid size.

## README Error

The README states "Both systems use NVIDIA A100-SXM4-80GB GPUs" but diagnostic output
(`diag-perf.o5615935`) confirms Derecho has **A100-SXM4-40GB**:
```
NVIDIA A100-SXM4-40GB, 580.65.06, 40960 MiB, 210 MHz, 1215 MHz
```

## Conclusions and Optimizations (2026-03-26)

### Root causes identified and fixed

1. **1-GPU distributed overhead (2.95x → 1.19x):** `fill_corners!` called `sync_device!`
   on every halo fill even with no corner neighbors. Fixed by early return.

2. **Multi-GPU pressure solver overhead (3.6x speedup on 2 GPUs):** Two issues:
   - The Z-stretched solver did 4 MPI transposes per solve; for slab-x (Ry=1) only
     2 are needed (tridiagonal can be solved in x-local space where z is fully local).
   - Cray MPICH's `Alltoallv` is **28x slower** than `Alltoall` for GPU buffers.
     Replacing with `Alltoall` for equal partitions was the single biggest win.

3. **Redundant halo fills in Breeze (9 per timestep removed):** Reference density
   (constant), potential temperature density (prognostic, already async-filled), and
   density/ρθ in compressible dynamics were filled unnecessarily.

### Pressure solver isolated benchmark (30 solves, 200×200×80/GPU)

| GPUs | Baseline | Optimized | Speedup |
|------|----------|-----------|---------|
| 1    | 7.9 ms/solve | 8.4 ms/solve | — |
| 2    | 129.5 ms/solve | 36.2 ms/solve | **3.6x** |
| 4    | 109.6 ms/solve | 37.7 ms/solve | **2.9x** |

### Transpose communication benchmark (round-trip, 200×200×80 ComplexF32)

| Strategy | 2 GPUs (NVLink) | 4 GPUs (NVLink) | 8 GPUs (Slingshot) |
|----------|----------------|-----------------|-------------------|
| `Alltoallv` (original) | 52.0 ms | 46.8 ms | 70.5 ms |
| **`Alltoall`** | **1.85 ms** | **2.20 ms** | **8.69 ms** |
| `Isend`/`Irecv` | 18.8 ms | 12.0 ms | 10.5 ms |
| GPU memcpy (floor) | 1.03 ms | 1.05 ms | 1.05 ms |

### Branches

- **Oceananigans** [`glw/optimize-distributed-solver`](https://github.com/CliMA/Oceananigans.jl/tree/glw/optimize-distributed-solver):
  fill_corners fix + Periodic y-FFT + slab-x tridiag + Alltoall
- **Oceananigans** [`glw/skip-empty-fill-corners`](https://github.com/CliMA/Oceananigans.jl/tree/glw/skip-empty-fill-corners):
  fill_corners fix only
- **Breeze** [`glw/distributed-tests`](https://github.com/NumericalEarth/Breeze.jl/tree/glw/distributed-tests):
  redundant halo fill removal + distributed solver dispatch

### Other findings

- `JULIA_CUDA_MEMORY_POOL=none` is **required** for multi-GPU CUDA-aware MPI
  (cuIpcGetMemHandle fails with default binned pool).
- Single-GPU non-distributed performance is identical on Perlmutter and Derecho (~0.61 s).
- Multi-node precompilation can deadlock; precompile on a single node first.

## Data Sources

- `diag-perf.o5615935` — Derecho diagnostic, default memory pool
- `diag-perf.o5615936` — Derecho diagnostic, pool=none
- `diag-perf.o5615937` — Derecho diagnostic, pool=cuda
- `supercell-weak.o5608630` — Derecho distributed 1 GPU (first run)
- `supercell-weak.o5608759` — Derecho distributed 1 GPU (second run)
- `supercell-weak.o5608760` — Derecho distributed 2 GPUs
- `supercell-weak.o5608761` — Derecho distributed 4 GPUs
- `supercell-weak.o5608762` — Derecho distributed 8 GPUs
- Perlmutter single-GPU results from README.md
- NCAR Derecho specs: https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/derecho/
- Perlmutter architecture: https://docs.nersc.gov/systems/perlmutter/architecture/
- NVIDIA A100 40GB vs 80GB: 40GB has HBM2 (1,555 GB/s), 80GB has HBM2e (2,039 GB/s)
