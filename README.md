# BreezeAnelasticBenchmarks

Weak-scaling GPU benchmarks for anelastic supercell simulations
built with [Breeze.jl](https://github.com/NumericalEarth/Breeze.jl)
and [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl).

Multi-GPU distribution uses Oceananigans'
[distributed grid infrastructure](https://clima.github.io/OceananigansDocumentation/stable/grids#Distributed-grids).

## Repository contents

| File | Description |
|------|-------------|
| `supercell_benchmark.jl` | Single-GPU supercell benchmark script |
| `supercell_benchmark.sh` | SLURM submission for single-GPU runs |
| `distributed_supercell_benchmark.jl` | MPI weak-scaling benchmark script |
| `distributed_supercell_benchmark.sh` | SLURM submission for multi-GPU runs |
| `Project.toml` | Julia project dependencies |

## Benchmark configuration

The benchmark runs a DCMIP2016 supercell test case with Kessler microphysics,
WENO5 advection, and anelastic dynamics. Each GPU gets a 400 × 400 × 80 grid
(168 km × 168 km × 20 km). For weak scaling, the domain extends in x with
the number of GPUs, keeping per-GPU work constant.

Timing is for 10 time steps of `Δt = 0.1 s` after a warmup pass
(which includes compilation).

## Running on Perlmutter (NERSC)

First, instantiate the project:

```bash
module load julia/1.12.1
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Single GPU:

```bash
sbatch supercell_benchmark.sh
```

Weak scaling (2 or 4 GPUs):

```bash
NGPUS=2 sbatch distributed_supercell_benchmark.sh
NGPUS=4 sbatch distributed_supercell_benchmark.sh
```

Set `FLOAT_TYPE=Float64` for double-precision runs:

```bash
FLOAT_TYPE=Float64 sbatch supercell_benchmark.sh
```

## Results

All benchmarks run on NERSC Perlmutter (NVIDIA A100 80 GB GPUs),
Julia 1.12.1, Breeze v0.4.3, Oceananigans v0.106.

### Single GPU

| Precision | Trial 1 | Trial 2 |
|-----------|---------|---------|
| Float32   | 0.611 s | 0.614 s |
| Float64   | 0.985 s | 0.987 s |

### Weak scaling (Float32)

Results pending.

## References

- [Silvestri, Wagner, et al. (2023)](https://arxiv.org/abs/2309.06662) —
  Oceananigans.jl: A model that achieves breakthrough resolution, memory
  and energy efficiency in global ocean simulations (2023 Gordon Bell Prize
  finalist for Climate Modelling at SC23)
- [Wagner et al. (2025)](https://arxiv.org/abs/2601.10441) —
  Performance benchmarks for atmospheric simulations with Breeze.jl
