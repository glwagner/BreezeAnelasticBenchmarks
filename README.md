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

#### Weak scaling (Float32, distributed)

| GPUs | Nodes | Trial 1 | Trial 2 |
|------|-------|---------|---------|
| 1    | 1     | 2.608 s | 2.583 s |
| 2    | 1     | 7.901 s | 7.623 s |
| 4    | 1     | 7.011 s | 6.759 s |
| 8    | 2     | 7.166 s | 6.835 s |

**Known issue:** The `Distributed` code path adds ~4x overhead even
on a single GPU (2.6 s vs 0.61 s). This overhead likely comes from
the `DistributedFourierTridiagonalPoissonSolver` and halo communication
infrastructure. Multi-GPU runs show an additional ~3x overhead (7 s vs 2.6 s)
from inter-GPU communication. Scaling from 4→8 GPUs across nodes is good.
See [`perlmutter_vs_derecho.md`](perlmutter_vs_derecho.md) for detailed analysis.

## References

- [Silvestri, Wagner, et al. (2023)](https://arxiv.org/abs/2309.06662) --
  Oceananigans.jl: A library that achieves breakthrough resolution, memory
  and energy efficiency in global ocean simulations
