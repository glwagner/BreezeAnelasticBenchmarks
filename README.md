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

#### Weak scaling — ERF-equivalent (200×200×80 per GPU, Centered(2), ScalarDiffusivity, halo=1)

| GPUs | Nodes | Partition | Trial 1 | Trial 2 | Eff vs 1 GPU | Eff vs 2 nodes |
|------|-------|-----------|---------|---------|-------------|---------------|
| 1    | 1     | 1×1       | 0.655 s | 0.570 s | 100%        | —             |
| 2    | 1     | 2×1       | 3.181 s | 3.557 s | 16%         | —             |
| 4    | 1     | 4×1       | 3.053 s | 3.104 s | 18%         | —             |
| 8    | 2     | 8×1       | 4.088 s | 3.144 s | 18%         | 100%          |
| 16   | 4     | 2×8       | 5.760 s | 5.998 s | 10%         | 52%           |

Additional results with Ry=2 and higher node counts pending.

#### Performance analysis

**Root cause identified:** The distributed code path in Oceananigans calls
`sync_device!` (= `CUDA.synchronize()`) inside `fill_corners!` on every halo fill,
even when there are no corner neighbors (1 rank, or slab decomposition). This flushes
the GPU's async execution pipeline ~300 times per 10 timesteps, causing massive
throughput loss.

**Fix applied (local):** Added early return in `fill_corners!` when all corner
connectivity is `nothing`. Result: 1-GPU distributed overhead dropped from
2.95x to 1.19x (2.94 s → 1.19 s). This fix needs to be upstreamed to Oceananigans.

**Remaining issue:** Multi-GPU overhead is dominated by the pressure solver's
distributed transpose operations (`Alltoallv!`). The Z-stretched
`DistributedFourierTridiagonalPoissonSolver` requires 8 MPI transpose operations
per pressure solve × 3 RK3 stages × 10 timesteps = 240 transposes, each with
GPU sync + MPI collective. This communication cost is roughly constant regardless
of GPU count, explaining why 2-GPU and 8-GPU times are similar.

See [`perlmutter_vs_derecho.md`](perlmutter_vs_derecho.md) and
[`scaling_plan.md`](scaling_plan.md) for detailed analysis and next steps.

## References

- [Silvestri, Wagner, et al. (2023)](https://arxiv.org/abs/2309.06662) --
  Oceananigans.jl: A library that achieves breakthrough resolution, memory
  and energy efficiency in global ocean simulations
