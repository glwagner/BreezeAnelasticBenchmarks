# Scaling Performance Plan

## Current Status (2026-03-26, updated)

### Optimizations applied

Four optimizations on Oceananigans `glw/optimize-distributed-solver` + Breeze `glw/distributed-tests`:
1. `fill_corners!` early return (no-corner case)
2. Periodic y-FFT along dim 2 (skip permutedims)
3. Slab-x tridiagonal in x-local space (4→2 transposes)
4. `Alltoall` instead of `Alltoallv` (28x faster on GPU)
Plus redundant halo fill removal in Breeze (3 per RK3 stage).

### Pressure solver isolated benchmark (optimized)

| GPUs | Baseline | Optimized | Speedup |
|------|----------|-----------|---------|
| 1    | 7.9 ms/solve | 8.4 ms/solve | — |
| 2    | 129.5 ms/solve | 36.2 ms/solve | 3.6x |
| 4    | 109.6 ms/solve | 37.7 ms/solve | 2.9x |

### Root Cause Analysis

The Distributed code path overhead on Derecho has two layers:

**Layer 1: Distributed overhead on 1 GPU (~4x, Derecho-specific)**

The `DistributedFourierTridiagonalPoissonSolver` (Z-stretched) uses a different FFT
strategy than the non-distributed `FourierTridiagonalPoissonSolver`:

- **Non-distributed:** Single batched 2D FFT in (x,y) simultaneously. Hits the
  `bounded_dims == ()` fast-path in `plan_transforms.jl:183` for (Periodic, Periodic)
  topology. Very efficient on GPU.

- **Distributed:** Two separate 1D FFTs (x then y), each with a memory reshape for the
  y-direction. The Z-stretched algorithm (`solve!` in
  `distributed_fft_tridiagonal_solver.jl:264`) also does 8 transpose operations (which
  are no-ops on 1 GPU since both yzbuffer and xybuffer are `nothing` for Partition(1,1))
  plus extra data copies.

On Perlmutter this difference costs only 0.03 s; on Derecho it costs 2.0 s.
Possible Derecho-specific factors:
- CUDA library version mismatch (system CUDA 12.9 loaded over CUDA.jl's bundled libraries)
- Different cuFFT behavior for 1D vs 2D batched transforms on this CUDA version
- MPI self-communication overhead in `fill_halo_regions!` (Cray MPICH on Derecho vs Perlmutter)
- Distributed halo fill infrastructure (DistributedCommunicationBCs on every field)

**Layer 2: Multi-GPU communication overhead (~3x on top of Layer 1)**

Going from 1→2 distributed GPUs adds ~3x (2.6 s → 7.6 s). This is the actual MPI
communication cost for:
- 8 MPI `Alltoallv!` calls per pressure solve (Z-stretched algorithm)
- Halo communication in `fill_halo_regions!` for momentum + pressure fields
- 3 RK3 stages per timestep × 10 timesteps = 30 pressure solves total

Scaling from 4→8 GPUs (across 2 nodes) shows no additional penalty, which suggests
intra-node NVLink communication is the bottleneck, not inter-node Slingshot.

## Plan

### Phase 1: Diagnose the 1-GPU distributed overhead

**Goal:** Identify exactly where the 2.0 s overhead comes from on Derecho.

1. Write a diagnostic script that times each component of a single time step:
   - `fill_halo_regions!` for momentum (distributed vs non-distributed)
   - `solve!` for the pressure solver (distributed vs non-distributed)
   - Individual FFTs (1D vs batched 2D)
   - MPI operations (barriers, alltoallv on 1 rank)

2. Test whether the overhead is in the pressure solver or elsewhere:
   - Replace `DistributedFourierTridiagonalPoissonSolver` with `FourierTridiagonalPoissonSolver`
     on 1 GPU (if possible) to isolate the solver overhead from the halo fill overhead.

3. Test CUDA library configuration:
   - Run with `LD_LIBRARY_PATH` cleared to use CUDA.jl's bundled libraries
   - Compare 1D vs 2D cuFFT performance directly

### Phase 2: Fix the 1-GPU distributed performance

**Goal:** Distributed 1-GPU should match non-distributed (~0.61 s).

Possible fixes (in order of likelihood):
1. **Use batched 2D FFT in distributed solver on 1 rank** — modify the distributed solve
   path to detect single-rank case and use the batched 2D FFT plan instead of two 1D plans.
2. **Fix halo fill self-communication** — ensure DistributedCommunicationBCs on 1 rank
   are optimized to skip MPI entirely.
3. **Fix CUDA library loading** — ensure consistent CUDA library versions on Derecho.

### Phase 3: Single-node weak scaling (2, 4 GPUs)

**Goal:** Achieve good intra-node scaling on Derecho (4 GPUs per node).

1. Run weak scaling suite with fixed 1-GPU baseline:
   - 1 GPU: target ~0.61 s (after Phase 2 fix)
   - 2 GPUs: measure actual communication overhead
   - 4 GPUs: full single node

2. Profile inter-GPU communication:
   - Time `Alltoallv!` operations separately from compute
   - Check if GPU-aware MPI (GTL) is actually transferring data via NVLink
   - Verify `MPICH_GPU_SUPPORT_ENABLED=1` is working correctly

3. Optimize if needed:
   - Overlap communication with computation
   - Reduce halo size if possible (WENO5 needs 5, but check other fields)

### Phase 4: Multi-node weak scaling (2, 4, 8, 16 nodes = 8, 16, 32, 64 GPUs)

**Goal:** Demonstrate weak scaling across Derecho GPU nodes.

1. Submit PBS jobs:
   - 2 nodes (8 GPUs): `select=2:ncpus=64:mpiprocs=4:ngpus=4`
   - 4 nodes (16 GPUs): `select=4:ncpus=64:mpiprocs=4:ngpus=4`
   - 8 nodes (32 GPUs): `select=8:ncpus=64:mpiprocs=4:ngpus=4`
   - 16 nodes (64 GPUs): `select=16:ncpus=64:mpiprocs=4:ngpus=4`

2. All runs require `JULIA_CUDA_MEMORY_POOL=none` for CUDA-aware MPI.

3. Measure scaling efficiency:
   - Ideal: constant time per 10 timesteps regardless of GPU count
   - Acceptable: <20% overhead at 64 GPUs
   - Report: time per timestep, communication fraction, scaling efficiency

### Phase 5: ERF comparison with corrected grid sizes

ERF uses 400×400×80 on 2 nodes (8 GPUs). The per-GPU grid depends on partition:

**Ry=1 (x-only partition):**
- Per GPU: 50×400×80
- Weak scaling: Nx = 50 × Ngpus, Ny = 400
- Constraint: Ny % Rx = 400 % Ngpus = 0 → Ngpus must divide 400

| GPUs | Nodes | Grid | Partition |
|------|-------|------|-----------|
| 1 | 1 | 50×400×80 | 1×1 |
| 2 | 1 | 100×400×80 | 2×1 |
| 4 | 1 | 200×400×80 | 4×1 |
| 8 | 2 | 400×400×80 | 8×1 |
| 16 | 4 | 800×400×80 | 16×1 |
| 20 | 5 | 1000×400×80 | 20×1 |
| 40 | 10 | 2000×400×80 | 40×1 |

**Ry=2 partition:**
- Per GPU: 100×200×80
- Weak scaling: Nx = 100 × Rx, Ny = 200 × Ry = 400
- Constraint: Ny_global % Rx = 400 % Rx = 0

| GPUs | Nodes | Grid | Partition |
|------|-------|------|-----------|
| 2 | 1 | 100×400×80 | 1×2 |
| 4 | 1 | 200×400×80 | 2×2 |
| 8 | 2 | 400×400×80 | 4×2 |
| 16 | 4 | 800×400×80 | 8×2 |
| 32 | 8 | 1600×400×80 | 16×2 |

Also run single-GPU reference at 400×400×80 for direct comparison with ERF 2-node result.

Submit via:
```bash
# Ry=1
qsub -v NGPUS=N,PARTITION_X_ONLY=1,NX_PER_GPU=50,NY_PER_GPU=400 -l select=... benchmarks/derecho_distributed_supercell_erf_benchmark.sh

# Ry=2
qsub -v NGPUS=N,RX=Rx,RY=2,NX_PER_GPU=100,NY_PER_GPU=200 -l select=... benchmarks/derecho_distributed_supercell_erf_benchmark.sh
```

### Phase 6: Documentation and comparison

1. Complete Perlmutter distributed benchmarks (when account hours available)
2. Update README with full scaling curves for both Ry=1 and Ry=2
3. Final performance report comparing both systems and both partitions

## Key Files

| File | Purpose |
|------|---------|
| `src/BreezeAnelasticBenchmarks.jl` | setup_supercell, run_benchmark! |
| `benchmarks/distributed_supercell_benchmark.jl` | MPI weak scaling script |
| `benchmarks/derecho_distributed_supercell_benchmark.sh` | PBS submission |
| `~/.julia/packages/Oceananigans/j2tCL/src/DistributedComputations/distributed_fft_tridiagonal_solver.jl` | Distributed pressure solver |
| `~/.julia/packages/Oceananigans/j2tCL/src/DistributedComputations/distributed_transpose.jl` | MPI transpose operations |
| `~/.julia/packages/Oceananigans/j2tCL/src/Solvers/fourier_tridiagonal_poisson_solver.jl` | Non-distributed pressure solver |

## Environment Requirements

```bash
module --force purge
module load ncarenv nvhpc cuda cray-mpich
export MPICH_GPU_SUPPORT_ENABLED=1
export JULIA_MPI_HAS_CUDA=true
export PALS_TRANSFER=false
export JULIA_CUDA_MEMORY_POOL=none  # Required for multi-GPU CUDA-aware MPI
```
