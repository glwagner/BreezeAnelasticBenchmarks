# BreezeAnelasticBenchmarks

Weak-scaling GPU benchmarks for anelastic supercell simulations
built with [Breeze.jl](https://github.com/NumericalEarth/Breeze.jl)
and [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl).

Multi-GPU distribution uses Oceananigans'
[distributed grid infrastructure](https://clima.github.io/OceananigansDocumentation/stable/grids#Distributed-grids).

## Package structure

This repo is a Julia package (`BreezeAnelasticBenchmarks`) that exports
`setup_supercell` and `run_benchmark!`. Precompilation via
[PrecompileTools.jl](https://github.com/JuliaLang/PrecompileTools.jl)
caches GPU kernel compilation so benchmark jobs start faster.

| Path | Description |
|------|-------------|
| `src/BreezeAnelasticBenchmarks.jl` | Package module with setup, benchmark, and precompile workloads |
| `benchmarks/supercell_benchmark.jl` | Single-GPU benchmark script |
| `benchmarks/supercell_benchmark.sh` | SLURM submission for single-GPU runs (Perlmutter) |
| `benchmarks/derecho_supercell_benchmark.sh` | PBS submission for single-GPU runs (Derecho) |
| `benchmarks/distributed_supercell_benchmark.jl` | MPI weak-scaling benchmark script |
| `benchmarks/distributed_supercell_benchmark.sh` | SLURM submission for multi-GPU runs (Perlmutter) |
| `benchmarks/derecho_distributed_supercell_benchmark.sh` | PBS submission for multi-GPU runs (Derecho) |
| `Project.toml` | Julia package dependencies |

## Benchmark configuration

The benchmark runs a DCMIP2016 supercell test case with Kessler microphysics,
WENO5 advection, and anelastic dynamics. Each GPU gets a 400 x 400 x 80 grid
(168 km x 168 km x 20 km). For weak scaling, the domain extends in x with
the number of GPUs via `Partition(Ngpus, 1)`, keeping per-GPU work constant.

Timing is for 10 time steps at `dt = 0.1 s`, three trials
(first is warmup that includes any remaining compilation).

## Running on Perlmutter (NERSC)

### Setup

```bash
module load julia/1.12.1
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Precompilation

Precompile on a GPU node (Perlmutter login nodes have GPUs) to cache
GPU kernel compilation:

```bash
julia --project=. -e 'using Pkg; Pkg.precompile()'
```

The precompile workload runs a small (8x8x8) version of the benchmark.
If MPI is initialized, only rank 0 runs the serial workload and all ranks
participate in a distributed workload.

### Single GPU

```bash
sbatch benchmarks/supercell_benchmark.sh
```

### Weak scaling

```bash
NGPUS=1 sbatch --nodes=1 benchmarks/distributed_supercell_benchmark.sh
NGPUS=2 sbatch --nodes=1 benchmarks/distributed_supercell_benchmark.sh
NGPUS=4 sbatch --nodes=1 benchmarks/distributed_supercell_benchmark.sh
NGPUS=8 sbatch --nodes=2 benchmarks/distributed_supercell_benchmark.sh
```

Perlmutter has 4 A100 GPUs per node, so 8 GPUs requires `--nodes=2`.

### Double precision

```bash
FLOAT_TYPE=Float64 sbatch benchmarks/supercell_benchmark.sh
```

## Running on Derecho (NCAR)

Derecho has 82 GPU nodes, each with 4 NVIDIA A100 GPUs and 128 AMD Milan cores.
It uses PBS (not SLURM) and Cray MPICH.

### Setup

```bash
module --force purge
module load ncarenv nvhpc cuda cray-mpich
julia +1.12 --project=. -e 'using Pkg; Pkg.instantiate()'
julia +1.12 --project=. -e 'using MPIPreferences; MPIPreferences.use_system_binary(vendor="cray")'
julia +1.12 --project=. -e 'using Pkg; Pkg.precompile()'
```

### Single GPU

```bash
qsub benchmarks/derecho_supercell_benchmark.sh
```

### Weak scaling

```bash
qsub -v NGPUS=1 -l select=1:ncpus=64:mpiprocs=1:ngpus=1:gpu_type=a100:mem=384GB benchmarks/derecho_distributed_supercell_benchmark.sh
qsub -v NGPUS=2 -l select=1:ncpus=64:mpiprocs=2:ngpus=2:gpu_type=a100:mem=384GB benchmarks/derecho_distributed_supercell_benchmark.sh
qsub -v NGPUS=4 -l select=1:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=384GB benchmarks/derecho_distributed_supercell_benchmark.sh
qsub -v NGPUS=8 -l select=2:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=384GB benchmarks/derecho_distributed_supercell_benchmark.sh
```

Derecho has 4 A100 GPUs per node, so 8 GPUs requires `select=2`.

### Double precision

```bash
FLOAT_TYPE=Float64 qsub benchmarks/derecho_supercell_benchmark.sh
```

See [Oceananigans on Derecho](https://github.com/CliMA/Oceananigans.jl/discussions/3669)
for additional setup guidance.

## Dependencies

Breeze is pinned to the
[`glw/distributed-tests`](https://github.com/NumericalEarth/Breeze.jl/tree/glw/distributed-tests)
branch, which includes fixes for distributed anelastic simulations
(halo communication for reduced-dimension fields and
`DistributedFourierTridiagonalPoissonSolver` for the anelastic pressure solve).

## Results

### Perlmutter (NERSC) — NVIDIA A100-SXM4-80GB, Julia 1.12.1

#### Single GPU

| Precision | Trial 1 | Trial 2 |
|-----------|---------|---------|
| Float32   | 0.615 s | 0.612 s |
| Float64   | 0.985 s | 0.987 s |

#### Weak scaling (Float32)

Results pending (account out of hours).

### Derecho (NCAR) — NVIDIA A100-SXM4-40GB, Julia 1.12.5

#### Single GPU (non-distributed)

| Memory Pool | Trial 1 | Trial 2 | Trial 3 |
|-------------|---------|---------|---------|
| default     | 0.614 s | 0.615 s | 0.615 s |
| none        | 0.635 s | 0.620 s | 0.615 s |
| cuda        | 0.614 s | 0.618 s | 0.615 s |

Single-GPU performance matches Perlmutter despite
40GB vs 80GB memory (1,555 vs 2,039 GB/s bandwidth).

#### Weak scaling — WENO5 supercell (400×400×80 per GPU, halo=5, x-only partition)

| GPUs | Nodes | Partition | Trial 1 | Trial 2 | Eff vs 1 GPU | Eff vs 2 nodes |
|------|-------|-----------|---------|---------|-------------|---------------|
| 1    | 1     | 1×1       | 0.887 s | 0.940 s | 100%        | —             |
| 2    | 1     | 2×1       | 7.655 s | 7.711 s | 12%         | —             |
| 4    | 1     | 4×1       | 7.106 s | 6.743 s | 14%         | —             |
| 8    | 2     | 8×1       | 7.483 s | 7.714 s | 12%         | 100%          |
| 16   | 4     | 16×1      | 10.295 s| 10.291 s| 9%          | 75%           |

#### Weak scaling — ERF-equivalent (anelastic, Centered(2), ScalarDiffusivity, halo=1)

ERF uses 400×400×80 on 2 nodes (8 GPUs). Per-GPU grid depends on partition:
- Ry=1 (x-only): 50×400×80 per GPU
- Ry=2: 100×200×80 per GPU

ERF uses 400×400×80 on 2 nodes (8 GPUs).

**1-GPU reference at 400×400×80: 0.614 s (61.4 ms/step)**

**Ry=1 (x-only partition, 50×400×80 per GPU)**

| GPUs | Nodes | Partition | Trial 2 | Eff vs 1 GPU | Eff vs 8 GPU |
|------|-------|-----------|---------|-------------|-------------|
| 1    | 1     | 1×1       | 0.547 s | 100%        | —           |
| 2    | 1     | 2×1       | 1.313 s | 42%         | —           |
| 4    | 1     | 4×1       | 0.965 s | 57%         | —           |
| 8    | 2     | 8×1       | 1.855 s | 29%         | 100%        |
| 16   | 4     | 16×1      | 1.707 s | 32%         | 109%        |
| 20   | 5     | 20×1      | 2.197 s | 25%         | 84%         |
| 40   | 10    | 40×1      | 2.363 s | 23%         | 79%         |

**Ry=2 partition (100×200×80 per GPU)**

| GPUs | Nodes | Partition | Trial 2 | Eff vs 1 GPU | Eff vs 8 GPU |
|------|-------|-----------|---------|-------------|-------------|
| 2    | 1     | 1×2       | 0.774 s | 71%         | —           |
| 4    | 1     | 2×2       | 1.375 s | 40%         | —           |
| 8    | 2     | 4×2       | 1.606 s | 34%         | 100%        |
| 16   | 4     | 8×2       | 2.031 s | 27%         | 79%         |
| 32   | 8     | 16×2      | 2.918 s | 19%         | 55%         |

Notes:
- Ry=2 at 2 GPUs achieves 71% efficiency — the y-decomposition avoids MPI transposes
  in the pressure solver (y is local for Ry≤1 in slab-x, but Ry=2 uses pencil decomposition).
- Ry=1 at 4 GPUs (57%) is the sweet spot for x-only intra-node.
- Multi-node scaling from 8→40 GPUs (Ry=1) maintains 79% relative efficiency.

## Optimizations

Four optimizations were developed during this benchmarking effort. These are
on the Oceananigans branch
[`glw/optimize-distributed-solver`](https://github.com/CliMA/Oceananigans.jl/tree/glw/optimize-distributed-solver)
and the Breeze branch
[`glw/distributed-tests`](https://github.com/NumericalEarth/Breeze.jl/tree/glw/distributed-tests).

### 1. Skip `fill_corners!` when no corner neighbors exist (Oceananigans)

`fill_corners!` called `sync_device!` (`CUDA.synchronize()`) on every distributed
halo fill, even when all corner connectivity is `nothing` (1 rank, slab decomposition).
This flushed the GPU pipeline ~300 times per 10 timesteps.

**Fix:** Early return when all four corner neighbors are `nothing`.
**Impact:** 1-GPU distributed overhead dropped from 2.95x to 1.19x.

### 2. Solve tridiagonal in x-local space for slab-x decomposition (Oceananigans)

The Z-stretched `DistributedFourierTridiagonalPoissonSolver` performed 4 MPI
transpose operations per pressure solve. For slab-x decomposition (`Partition(Ngpus, 1)`),
z is always fully local in x-local space, so the tridiagonal solve can happen
directly after the forward FFTs without transposing back to z-local first.

**Fix:** New `_slab_x_solve!` method that does `y-FFT → transpose → x-FFT → tridiag → x-IFFT → transpose → y-IFFT`.
**Impact:** 4 → 2 MPI transposes per solve = 2.1x solver speedup on 2 GPUs.

### 3. Use `Alltoall` instead of `Alltoallv` for equal partitions (Oceananigans)

Cray MPICH's `Alltoallv` is catastrophically slow for GPU buffers compared to
`Alltoall` (28x slower on 2 A100 GPUs over NVLink). Since most simulations use
equal partition sizes, the code now uses `Alltoall` when all chunk counts are equal.

**Fix:** Check if all counts are equal; use `MPI.Alltoall!` with `UBuffer` instead of `MPI.Alltoallv!` with `VBuffer`.
**Impact:** Transpose round-trip: 52 ms → 1.85 ms on 2 intra-node GPUs.

### 4. Remove redundant halo fills (Breeze)

Several `fill_halo_regions!` calls per RK3 stage were redundant:
- Reference density (constant, z-only, never changes)
- Potential temperature density (prognostic, already in async fill)
- Density and ρθ in `compute_auxiliary_dynamics_variables!` (already in async fill)

**Fix:** Removed 3 redundant fills per RK3 stage = 90 fewer fills per benchmark trial.

### Combined pressure solver improvement

Isolated pressure solver benchmark (30 solves, 200×200×80/GPU, Periodic×Periodic×Bounded):

| GPUs | Baseline | Optimized | Speedup |
|------|----------|-----------|---------|
| 1    | 7.9 ms/solve | 8.4 ms/solve | — |
| 2    | 129.5 ms/solve | 36.2 ms/solve | 3.6x |
| 4    | 109.6 ms/solve | 37.7 ms/solve | 2.9x |

See [`perlmutter_vs_derecho.md`](perlmutter_vs_derecho.md) and
[`scaling_plan.md`](scaling_plan.md) for detailed investigation notes.

## References

- [Silvestri, Wagner, et al. (2023)](https://arxiv.org/abs/2309.06662) --
  Oceananigans.jl: A library that achieves breakthrough resolution, memory
  and energy efficiency in global ocean simulations
