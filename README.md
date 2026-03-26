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
| `benchmarks/supercell_benchmark.sh` | SLURM submission for single-GPU runs |
| `benchmarks/distributed_supercell_benchmark.jl` | MPI weak-scaling benchmark script |
| `benchmarks/distributed_supercell_benchmark.sh` | SLURM submission for multi-GPU runs |
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

## Dependencies

Breeze is pinned to the
[`glw/distributed-tests`](https://github.com/NumericalEarth/Breeze.jl/tree/glw/distributed-tests)
branch, which includes fixes for distributed anelastic simulations
(halo communication for reduced-dimension fields and
`DistributedFourierTridiagonalPoissonSolver` for the anelastic pressure solve).

## Results

All benchmarks run on NERSC Perlmutter (NVIDIA A100-SXM4-80GB GPUs),
Julia 1.12.1.

### Single GPU

| Precision | Trial 1 | Trial 2 |
|-----------|---------|---------|
| Float32   | 0.615 s | 0.612 s |
| Float64   | 0.985 s | 0.987 s |

### Weak scaling (Float32)

Results pending.

## References

- [Silvestri, Wagner, et al. (2023)](https://arxiv.org/abs/2309.06662) --
  Oceananigans.jl: A model that achieves breakthrough resolution, memory
  and energy efficiency in global ocean simulations (2023 Gordon Bell Prize
  finalist for Climate Modelling at SC23)
- [Wagner et al. (2025)](https://arxiv.org/abs/2601.10441) --
  Performance benchmarks for atmospheric simulations with Breeze.jl
