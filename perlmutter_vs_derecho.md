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

| GPUs | Nodes | Trial 1  | Trial 2  |
|------|-------|----------|----------|
| 1    | 1     | 2.319 s  | 2.417 s  |
| 1*   | 1     | 2.595 s  | 2.669 s  |
| 2    | 1     | 7.959 s  | 7.475 s  |
| 4    | 1     | 7.234 s  | 7.642 s  |
| 8    | 2     | 6.867 s  | 7.127 s  |

*Second 1-GPU run.

The 1-GPU distributed run (~2.4 s) is 4x slower than the 1-GPU non-distributed run (~0.61 s).

### Perlmutter weak scaling (distributed)

Results pending — no distributed benchmark results from Perlmutter are available for comparison.

## Root Cause Analysis

### The 4x slowdown: Distributed architecture overhead

The distributed benchmark script (`distributed_supercell_benchmark.jl`) always uses:
```julia
arch = Distributed(GPU(); partition=Partition(Ngpus, 1))
```

Even when `Ngpus=1`, this creates a full distributed architecture. The output confirms:
```
Warning: We are building a Distributed architecture on a single MPI rank.
```

The Distributed code path adds overhead from:
1. **DistributedFourierTridiagonalPoissonSolver** — the distributed pressure solver
   replaces the standard `FourierTridiagonalPoissonSolver`, adding MPI communication
   and potentially different algorithm paths even on a single rank
2. **MPI barriers and communication setup** — each time step includes MPI barrier
   synchronization and halo communication infrastructure
3. **Distributed boundary conditions** — `inject_halo_communication_boundary_conditions`
   modifies boundary conditions for inter-rank communication, adding overhead to halo fills

The non-distributed script (`supercell_benchmark.jl` / `diagnose_performance.jl`) uses
`GPU()` directly and achieves 0.61 s — matching Perlmutter exactly.

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

## Conclusions

1. **The 4x slowdown is a software issue, not hardware.** Single-GPU non-distributed
   performance is identical on both systems (~0.61 s for Float32).

2. **The Distributed architecture adds ~4x overhead on a single rank.** This overhead
   likely comes from the distributed pressure solver, halo communication infrastructure,
   and MPI synchronization.

3. **No Perlmutter distributed results exist for comparison.** The Perlmutter weak
   scaling jobs are still pending (NERSC account out of hours). It is likely that
   Perlmutter distributed runs would show the same ~4x overhead.

4. **The CUDA memory pool setting is irrelevant.** All three pool options (default/binned,
   none, cuda) produce identical benchmark times on Derecho.

## Recommended Next Steps

1. **Run non-distributed single-GPU benchmark on Perlmutter** using `supercell_benchmark.jl`
   to confirm matching performance (already done: 0.615 s).

2. **Run Perlmutter distributed benchmarks** to determine if the Distributed overhead
   is system-specific or inherent to the Oceananigans distributed code path.

3. **Profile the Distributed code path** to identify where the 4x overhead comes from:
   - Time the pressure solver separately (distributed vs non-distributed)
   - Time halo fills separately
   - Check if MPI barriers are adding latency

4. **Fix the weak scaling regression** (2+ GPUs being 3x slower than 1 distributed GPU)
   — this is likely the more impactful issue for the project goals.

5. **Correct the README** to reflect that Derecho has A100-SXM4-40GB GPUs.

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
