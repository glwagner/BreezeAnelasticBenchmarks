# BreezeAnelasticBenchmarks

Weak-scaling benchmarks for [Breeze.jl](https://github.com/NumericalEarth/Breeze.jl)
anelastic supercell simulations on GPU clusters.

## Setup

Each GPU gets a 400×400×80 grid (168 km × 168 km × 20 km). The domain extends
in x with the number of GPUs, keeping per-GPU work constant.

## Running on Perlmutter (NERSC)

Single GPU:

```bash
sbatch supercell_benchmark.sh
```

Weak scaling (2 or 4 GPUs):

```bash
NGPUS=2 sbatch distributed_supercell_benchmark.sh
NGPUS=4 sbatch distributed_supercell_benchmark.sh
```

Set `FLOAT_TYPE=Float64` for double-precision benchmarks.
