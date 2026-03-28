# BreezeAnelasticBenchmarks

Weak-scaling GPU benchmarks for atmospheric supercell simulations
built with [Breeze.jl](https://github.com/NumericalEarth/Breeze.jl)
(branch [`glw/distributed-tests`](https://github.com/NumericalEarth/Breeze.jl/tree/glw/distributed-tests))
and [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl)
(branch [`glw/optimize-distributed-solver`](https://github.com/CliMA/Oceananigans.jl/tree/glw/optimize-distributed-solver))
on [NCAR Derecho](https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/derecho/)
(NVIDIA A100-SXM4-40GB GPUs, 4 per node, NVLink intra-node, Slingshot-11 inter-node).

![Scaling results](scaling_results.png)

## Benchmark configurations

Three test cases are benchmarked, all using the DCMIP2016 supercell initial conditions
with Kessler microphysics, Tetens saturation vapor pressure, and SSP-RK3 time stepping.
Weak scaling extends the domain in x via `Partition(Ngpus, 1)`.

| Config | Dynamics | Advection | Closure | Halo | Pressure solve |
|--------|----------|-----------|---------|------|----------------|
| **WENO5 supercell** | `AnelasticDynamics` | `WENO(order=5)` | — | (5,5,5) | Distributed FFT + tridiagonal |
| **ERF-like anelastic** | `AnelasticDynamics` | `Centered(order=2)` | `ScalarDiffusivity(ν=200, κ=200)` | (1,1,1) | Distributed FFT + tridiagonal |
| **ERF-like compressible** | `CompressibleDynamics` | `Centered(order=2)` | `ScalarDiffusivity(ν=200, κ=200)` | (1,1,1) | None (diagnostic EOS) |

Two per-GPU grid sizes are tested:
- **400×400×80** per GPU (168 km × 168 km × 20 km) — the full supercell grid
- **50×400×80** per GPU — matching the ERF comparison where 400×400×80 is split across 8 GPUs (2 nodes)

Timing uses 100 time steps at dt = 0.1 s, three trials (first is warmup).

## Results

### Single GPU (NT=100, Derecho A100-SXM4-40GB)

| Config | Grid | Float32 (ms/step) | Float64 (ms/step) | F64/F32 |
|--------|------|-------------------|-------------------|---------|
| WENO5 + Kessler, anelastic | 400×400×80 | 95.4 | 148.5 | 1.56x |
| WENO5 + Kessler, anelastic | 50×400×80 | 60.0 | — | — |
| Centered(2) + diffusion, anelastic | 400×400×80 | 70.0 | — | — |
| Centered(2) + diffusion, anelastic | 50×400×80 | 53.7 | 64.9 | 1.21x |
| Centered(2) + diffusion, compressible | 400×400×80 | 19.7 | — | — |
| Centered(2) + diffusion, compressible | 50×400×80 | 5.2 | 5.4 | 1.04x |

### Weak scaling (Float32, x-only partition, NT=100, all optimizations applied)

**50×400×80 per GPU:**

| GPUs | Nodes | WENO5 anelastic | ERF anelastic | Compressible |
|------|-------|----------------|--------------|-------------|
| 1    | 1     | 60.0 ms/step   | 51.2 ms/step | 4.4 ms/step |
| 2    | 1     | 145.1          | 137.0        | 62.7        |
| 4    | 1     | 151.3          | 140.5        | 72.0        |
| 8    | 2     | pending        | 172.4        | 95.4        |
| 16   | 4     | pending        | 180.6        | 110.1       |
| 20   | 5     | pending        | 204.9        | 123.7       |
| 40   | 10    | pending        | 233.3        | 137.4       |

**400×400×80 per GPU:**

| GPUs | Nodes | WENO5 anelastic | ERF anelastic | Compressible |
|------|-------|----------------|--------------|-------------|
| 1    | 1     | 95.4 ms/step   | 70.0 ms/step | 19.7 ms/step |
| 2    | 1     | 462.2          | 381.5        | 244.6        |
| 4    | 1     | 491.6          | 467.9        | 203.5        |
| 8    | 2     | 666.6          | pending      | pending      |
| 16   | 4     | 765.9          | pending      | pending      |
| 20   | 5     | 803.3          | pending      | pending      |
| 40   | 10    | pending        | —            | —            |

### Multi-node scaling efficiency (relative to 2 nodes / 8 GPUs)

**50×400×80 per GPU:**

| GPUs | Nodes | WENO5 | ERF anelastic | Compressible |
|------|-------|-------|--------------|-------------|
| 8    | 2     | —     | 100%         | 100%        |
| 16   | 4     | —     | 96%          | 87%         |
| 20   | 5     | —     | 84%          | 77%         |
| 40   | 10    | —     | 74%          | 69%         |

**400×400×80 per GPU:**

| GPUs | Nodes | WENO5 | ERF anelastic | Compressible |
|------|-------|-------|--------------|-------------|
| 8    | 2     | 100%  | pending      | pending     |
| 16   | 4     | 87%   | pending      | pending     |
| 20   | 5     | 83%   | pending      | pending     |

### Perlmutter (NERSC) — NVIDIA A100-SXM4-80GB, Julia 1.12.1

**Compressible weak scaling (50×400×80 per GPU, NT=100, MPI):**

| GPUs | Nodes | ms/step |
|------|-------|---------|
| 1    | 1     | 3.5     |
| 2    | 1     | 8.6     |
| 4    | 1     | 9.8     |
| 8    | 2     | pending |
| 16   | 4     | pending |

Additional configs (ERF anelastic, WENO5, NCCL comparison) pending.

## Package structure

This repo is a Julia package (`BreezeAnelasticBenchmarks`) that exports
`setup_supercell`, `setup_supercell_erf`, `setup_supercell_compressible`,
and `run_benchmark!`.

| Path | Description |
|------|-------------|
| `src/BreezeAnelasticBenchmarks.jl` | Package module with setup functions and precompile workloads |
| `benchmarks/distributed_supercell_benchmark.jl` | WENO5 anelastic weak-scaling script |
| `benchmarks/distributed_supercell_erf_benchmark.jl` | ERF-like anelastic weak-scaling script |
| `benchmarks/distributed_supercell_compressible_benchmark.jl` | Compressible weak-scaling script |
| `benchmarks/benchmark_pressure_solver.jl` | Isolated pressure solver benchmark |
| `benchmarks/benchmark_transpose_strategies.jl` | MPI transpose strategy comparison |
| `plot_scaling.jl` | CairoMakie scaling plot script |

## Running on Derecho (NCAR)

### Setup

```bash
module --force purge
module load ncarenv nvhpc cuda cray-mpich
julia +1.12 --project=. -e 'using Pkg; Pkg.instantiate()'
julia +1.12 --project=. -e 'using MPIPreferences; MPIPreferences.use_system_binary(vendor="cray")'
julia +1.12 --project=. -e 'using Pkg; Pkg.precompile()'
```

### Weak scaling

All scripts support `NT`, `NX_PER_GPU`, `NY_PER_GPU`, `PARTITION_X_ONLY`, `RX`, `RY`,
and `FLOAT_TYPE` environment variables via `-v` on `qsub`.

```bash
# WENO5 anelastic, 4 GPUs, 100 steps
qsub -v NGPUS=4,NT=100 -l select=1:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=384GB \
    benchmarks/derecho_distributed_supercell_benchmark.sh

# ERF anelastic, 8 GPUs, x-only, 50×400×80/GPU
qsub -v NGPUS=8,PARTITION_X_ONLY=1,NX_PER_GPU=50,NY_PER_GPU=400,NT=100 \
    -l select=2:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=384GB \
    benchmarks/derecho_distributed_supercell_erf_benchmark.sh

# Compressible, 8 GPUs, 50×400×80/GPU
qsub -v NX_PER_GPU=50,NY_PER_GPU=400,NT=100 \
    -l select=2:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=384GB \
    benchmarks/derecho_distributed_supercell_compressible_benchmark.sh
```

Derecho has 4 A100 GPUs per node. Multi-GPU requires `JULIA_CUDA_MEMORY_POOL=none`.

## Running on Perlmutter (NERSC)

### Setup

```bash
module load julia/1.12.1
module load nccl/2.29.2-cu13
export LD_PRELOAD=/usr/lib64/libstdc++.so.6

julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using CUDA; CUDA.set_runtime_version!(v"13.0")'
julia --project=. -e 'using MPIPreferences; MPIPreferences.use_system_binary()'
```

Then edit `LocalPreferences.toml` to add GTL preload for CUDA-aware MPI:
```toml
[MPIPreferences]
preloads = ["/opt/cray/pe/mpich/default/gtl/lib/libmpi_gtl_cuda.so"]
preloads_env_switch = "MPICH_GPU_SUPPORT_ENABLED"
```

Precompile (login nodes have GPUs):
```bash
julia --project=. -e 'using Pkg; Pkg.precompile()'
```

### NCCL on Perlmutter

NCCL requires `nccl/2.29.2-cu13` (the cu13 version) and an `LD_PRELOAD` workaround
for libstdc++ compatibility. Both are set in the SLURM scripts automatically.
To use NCCL communication instead of MPI, pass `USE_NCCL=1`:

```bash
USE_NCCL=1 NGPUS=4 sbatch --nodes=1 benchmarks/distributed_supercell_erf_benchmark.sh
```

### Weak scaling

```bash
sbatch benchmarks/supercell_benchmark.sh                    # single GPU
NGPUS=4 sbatch --nodes=1 benchmarks/distributed_supercell_benchmark.sh  # 4 GPUs, MPI
NGPUS=8 sbatch --nodes=2 benchmarks/distributed_supercell_benchmark.sh  # 8 GPUs, MPI
USE_NCCL=1 NGPUS=8 sbatch --nodes=2 benchmarks/distributed_supercell_benchmark.sh  # 8 GPUs, NCCL
```

### Cray MPICH warning suppression

Multi-node jobs on Perlmutter generate millions of "malformed environment entry"
warnings from Julia's `Base.env.jl` due to Cray MPICH injecting corrupted
environment entries. The benchmark scripts suppress these via
`disable_logging(Logging.Warn)` and print timing to stdout instead of stderr.

## Dependencies

Breeze is pinned to the
[`glw/distributed-tests`](https://github.com/NumericalEarth/Breeze.jl/tree/glw/distributed-tests)
branch, which includes fixes for distributed anelastic simulations and redundant halo fill removal.

Oceananigans optimizations are on the
[`glw/optimize-distributed-solver`](https://github.com/CliMA/Oceananigans.jl/tree/glw/optimize-distributed-solver)
branch (applied locally to `~/.julia/packages/Oceananigans/` on Derecho).

## NCCL distributed communication (new)

An NCCL extension for Oceananigans replaces all MPI-based GPU communication
with [NCCL](https://github.com/JuliaGPU/NCCL.jl), eliminating `sync_device!`
pipeline stalls and enabling GPU-stream-native communication.

**PR:** [CliMA/Oceananigans.jl#5444](https://github.com/CliMA/Oceananigans.jl/pull/5444)
**NCCL.jl Complex type support:** [JuliaGPU/NCCL.jl#67](https://github.com/JuliaGPU/NCCL.jl/pull/67)

### Usage

```julia
using NCCL  # triggers extension load
NCCLExt = Base.get_extension(Oceananigans, :OceananigansNCCLExt)
arch = NCCLExt.NCCLDistributed(GPU(); partition = Partition(Ngpus, 1))
# All halo fills and solver transposes automatically use NCCL
```

### NCCL weak scaling results (A100-SXM4-80GB, NV12 NVLink)

NonhydrostaticModel + WENO5 + BuoyancyTracer, 200×200×80 per GPU:

| GPUs | ms/step | Scaling efficiency |
|------|---------|-------------------|
| 1    | 13.37   | 100% (baseline)   |
| 2    | 23.46   | 57%               |
| 4    | 21.58   | 62%               |

Pressure solver only (isolated):

| GPUs | NCCL (ms/solve) | MPI Derecho (ms/solve) |
|------|-----------------|------------------------|
| 1    | 0.95            | 8.4                    |
| 2    | 2.82            | 36.2                   |
| 4    | 2.95            | 37.7                   |

### NCCL extension features

1. **`NCCLDistributed` architecture** — drop-in replacement for `Distributed`, routes all
   GPU-to-GPU communication through NCCL instead of MPI
2. **NCCL pressure solver transposes** — replaces `sync_device! + MPI.Alltoall` with
   NCCL grouped `Send`/`Recv` (stream-native, no pipeline stalls)
3. **Multi-field batched halo fills** — packs all fields' send buffers, then one NCCL
   group for all `Send`/`Recv`, then unpacks all. Reduces NCCL kernel launches per timestep.
4. **`sync_device!` no-op** — NCCL operations are GPU-stream-ordered, so explicit
   CPU-GPU synchronization before communication is unnecessary

### Nsight Systems profiling (4 GPUs, WENO5, 10 timesteps)

| Category | % GPU time | Total (ms) |
|----------|-----------|-----------|
| NCCL communication | 27% | 295 |
| Tendencies (WENO5) | 26% | 286 |
| FFT (pressure solver) | 13% | 145 |
| Pack/Unpack buffers | ~5% | ~55 |
| Other | ~29% | ~315 |

NCCL steady-state: 0.24 ms/call average. 6 outlier calls > 2 ms (NCCL init) account
for 52% of total NCCL time.

## MPI optimizations (Derecho)

Four optimizations were developed during this benchmarking effort:

1. **Skip `fill_corners!` when no corner neighbors exist** — eliminates unnecessary
   `CUDA.synchronize()` calls. 1-GPU distributed overhead: 2.95x → 1.19x.

2. **Solve tridiagonal in x-local space for slab-x** — reduces MPI transposes
   from 4 to 2 per pressure solve. 2.1x solver speedup.

3. **Use `Alltoall` instead of `Alltoallv`** — Cray MPICH's `Alltoallv` is 28x slower
   than `Alltoall` for GPU buffers. Transpose round-trip: 52 ms → 1.85 ms on 2 GPUs.

4. **Remove redundant halo fills in Breeze** — 3 fewer fills per RK3 stage.

Combined pressure solver improvement: **3.6x on 2 GPUs** (129.5 → 36.2 ms/solve).

See [`perlmutter_vs_derecho.md`](perlmutter_vs_derecho.md) and
[`scaling_plan.md`](scaling_plan.md) for detailed investigation notes.

## References

- [Silvestri, Wagner, et al. (2023)](https://arxiv.org/abs/2309.06662) --
  Oceananigans.jl: A library that achieves breakthrough resolution, memory
  and energy efficiency in global ocean simulations
