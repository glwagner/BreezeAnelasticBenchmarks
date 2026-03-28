# BreezeAnelasticBenchmarks

Weak-scaling GPU benchmarks for atmospheric supercell simulations
built with [Breeze.jl](https://github.com/NumericalEarth/Breeze.jl)
and [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl)
using NCCL for GPU-to-GPU communication on
[NERSC Perlmutter](https://docs.nersc.gov/systems/perlmutter/)
(NVIDIA A100-SXM4-80GB, 4 per node, NVLink intra-node, Slingshot-11 inter-node).

![Scaling results](scaling_results.png)

## Benchmark configurations

Three test cases using the DCMIP2016 supercell with Kessler microphysics,
Tetens saturation vapor pressure, and SSP-RK3 time stepping.
Weak scaling extends the domain in x via `Partition(Ngpus, 1)`.

| Config | Dynamics | Advection | Closure | Halo | Pressure solve |
|--------|----------|-----------|---------|------|----------------|
| **WENO5 supercell** | `AnelasticDynamics` | `WENO(order=5)` | — | (5,5,5) | Distributed FFT + tridiagonal |
| **ERF-like anelastic** | `AnelasticDynamics` | `Centered(order=2)` | `ScalarDiffusivity(ν=200, κ=200)` | (1,1,1) | Distributed FFT + tridiagonal |
| **ERF-like compressible** | `CompressibleDynamics` | `Centered(order=2)` | `ScalarDiffusivity(ν=200, κ=200)` | (1,1,1) | None (diagnostic EOS) |

Per-GPU grid: **50×400×80** (matching ERF weak scaling where 400×400×80 is split across 8 GPUs).
Timing: 100 time steps at dt = 0.1 s, three trials (first is warmup).

## Results — Perlmutter (NERSC), NVIDIA A100-SXM4-80GB

### Compressible weak scaling (50×400×80/GPU, NT=100)

| GPUs | Nodes | MPI (ms/step) | NCCL (ms/step) | NCCL speedup |
|------|-------|---------------|----------------|--------------|
| 1    | 1     | 3.6           | —              | —            |
| 2    | 1     | 8.6           | 6.3            | 1.4x         |
| 4    | 1     | 10.0          | 5.5            | 1.8x         |
| 8    | 2     | pending       | pending        | —            |
| 16   | 4     | pending       | pending        | —            |

NCCL at 4 GPUs (5.5 ms) is close to single-GPU MPI (3.6 ms) — **65% scaling efficiency**
vs MPI's 36%. The `sync_device!` elimination in NCCL is the key factor.

### ERF anelastic weak scaling (50×400×80/GPU, NT=100)

| GPUs | Nodes | MPI (ms/step) | NCCL (ms/step) | NCCL speedup |
|------|-------|---------------|----------------|--------------|
| 1    | 1     | pending       | —              | —            |
| 2    | 1     | pending       | pending        | —            |
| 4    | 1     | pending       | pending        | —            |
| 8    | 2     | pending       | pending        | —            |

### NCCL pressure solver (isolated, 200×200×80/GPU)

| GPUs | NCCL (ms/solve) | MPI Derecho (ms/solve) | Speedup |
|------|-----------------|------------------------|---------|
| 1    | 0.95            | 8.4                    | 8.8x    |
| 2    | 2.82            | 36.2                   | 12.8x   |
| 4    | 2.95            | 37.7                   | 12.8x   |

## NCCL distributed communication

An NCCL extension for Oceananigans
([PR #5444](https://github.com/CliMA/Oceananigans.jl/pull/5444))
replaces all MPI-based GPU communication with
[NCCL](https://github.com/JuliaGPU/NCCL.jl), eliminating `sync_device!`
pipeline stalls and enabling GPU-stream-native communication.

Key features:
1. **`NCCLDistributed` architecture** — drop-in replacement for `Distributed`
2. **NCCL pressure solver transposes** — replaces `sync_device! + MPI.Alltoall` with
   NCCL grouped `Send`/`Recv` (stream-native, no pipeline stalls)
3. **`sync_device!` no-op** — NCCL operations are GPU-stream-ordered

### Usage

```julia
using NCCL  # triggers extension load
NCCLExt = Base.get_extension(Oceananigans, :OceananigansNCCLExt)
arch = NCCLExt.NCCLDistributed(GPU(); partition = Partition(Ngpus, 1))
# All halo fills and solver transposes automatically use NCCL
```

Or from the command line:
```bash
USE_NCCL=1 NGPUS=4 sbatch --nodes=1 benchmarks/distributed_supercell_erf_benchmark.sh
```

## Running on Perlmutter (NERSC)

### Setup

```bash
module load julia/1.12.1
module load nccl/2.29.2-cu13
export LD_PRELOAD=/usr/lib64/libstdc++.so.6

julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using CUDA; CUDA.set_runtime_version!(v"13.0")'
julia --project=. -e 'using MPIPreferences; MPIPreferences.use_system_binary()'
julia --project=. -e 'using Pkg; Pkg.precompile()'
```

### Weak scaling

```bash
# Compressible, 4 GPUs, NCCL
USE_NCCL=1 NGPUS=4 sbatch --nodes=1 benchmarks/distributed_supercell_compressible_benchmark.sh

# ERF anelastic, 8 GPUs, NCCL
USE_NCCL=1 NGPUS=8 sbatch --nodes=2 benchmarks/distributed_supercell_erf_benchmark.sh

# WENO5 anelastic, 8 GPUs, MPI
NGPUS=8 sbatch --nodes=2 benchmarks/distributed_supercell_benchmark.sh
```

Perlmutter has 4 A100-80GB GPUs per node.

## Package structure

| Path | Description |
|------|-------------|
| `src/BreezeAnelasticBenchmarks.jl` | Package module with setup functions and precompile workloads |
| `benchmarks/distributed_supercell_benchmark.jl` | WENO5 anelastic weak-scaling script |
| `benchmarks/distributed_supercell_erf_benchmark.jl` | ERF-like anelastic weak-scaling script |
| `benchmarks/distributed_supercell_compressible_benchmark.jl` | Compressible weak-scaling script |
| `benchmarks/benchmark_pressure_solver.jl` | Isolated pressure solver benchmark |
| `plot_scaling.jl` | CairoMakie scaling plot script |

## Dependencies

- **Breeze:** [`glw/distributed-tests`](https://github.com/NumericalEarth/Breeze.jl/tree/glw/distributed-tests)
- **Oceananigans:** [`glw/nccl-distributed-solver`](https://github.com/CliMA/Oceananigans.jl/tree/glw/nccl-distributed-solver)
  ([PR #5444](https://github.com/CliMA/Oceananigans.jl/pull/5444))
- **NCCL.jl Complex type support:** [JuliaGPU/NCCL.jl#67](https://github.com/JuliaGPU/NCCL.jl/pull/67)

## Derecho (NCAR) results — MPI with optimizations

Earlier benchmarks on [NCAR Derecho](https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/derecho/)
(A100-SXM4-40GB, Cray MPICH) using MPI with four optimizations
(fill_corners skip, slab-x tridiagonal, Alltoall, redundant halo removal):

**50×400×80 per GPU, MPI:**

| GPUs | Nodes | WENO5 anelastic | ERF anelastic | Compressible |
|------|-------|----------------|--------------|-------------|
| 1    | 1     | 60.0 ms/step   | 51.2 ms/step | 4.4 ms/step |
| 2    | 1     | 145.1          | 137.0        | 62.7        |
| 4    | 1     | 151.3          | 140.5        | 72.0        |
| 8    | 2     | —              | 172.4        | 95.4        |
| 16   | 4     | —              | 180.6        | 110.1       |
| 20   | 5     | —              | 204.9        | 123.7       |
| 40   | 10    | —              | 233.3        | 137.4       |

See [`perlmutter_vs_derecho.md`](perlmutter_vs_derecho.md) and
[`scaling_plan.md`](scaling_plan.md) for detailed investigation notes.

## References

- [Silvestri, Wagner, et al. (2023)](https://arxiv.org/abs/2309.06662) --
  Oceananigans.jl: A library that achieves breakthrough resolution, memory
  and energy efficiency in global ocean simulations
- [Wagner et al. (2025)](https://arxiv.org/abs/2601.10441) --
  Performance benchmarks for atmospheric simulations with Breeze.jl
