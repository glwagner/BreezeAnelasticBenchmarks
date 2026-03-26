# BreezeAnelasticBenchmarks

Weak-scaling GPU benchmarks for Breeze.jl anelastic supercell simulations
on NERSC Perlmutter (NVIDIA A100 80 GB GPUs).

## Current state (2026-03-25)

### Package structure
- Repo is a Julia package (`BreezeAnelasticBenchmarks`) with PrecompileTools
- Exports `setup_supercell(arch; kw...)` and `run_benchmark!(model, Nt, dt)`
- Benchmark scripts in `benchmarks/` directory
- Breeze pinned to `glw/distributed-tests` branch (PR #441) for distributed support

### Completed
- Single-GPU Float32: ~0.61 s per 10 time steps
- Single-GPU Float64: ~0.99 s per 10 time steps
- 1-GPU distributed Float32: ~0.64 s per 10 time steps

### Pending
- 2, 4, 8-GPU weak scaling jobs submitted on overrun QOS
  (jobs 50565267, 50565274, 50565277, 50565278)
- Account m5176_g is out of debug/regular hours

### Bugs found and fixed
- Oceananigans v0.105: `inject_halo_communication_boundary_conditions` replaces
  `nothing` BCs on `Field{Nothing, Nothing, Center}` with DistributedCommunicationBC,
  causing BoundsError during halo fill. Fixed in Breeze `glw/distributed-tests` branch.
- Breeze main: `FourierTridiagonalPoissonSolver` doesn't handle `FullyConnected`
  topology on distributed grids. Fixed in Breeze `glw/distributed-tests` branch
  by dispatching to `DistributedFourierTridiagonalPoissonSolver`.

## Project setup

- Julia 1.12.1, `module load julia/1.12.1` on Perlmutter
- SLURM account: `m5176_g`, constraint: `gpu`
- GitHub: https://github.com/glwagner/BreezeAnelasticBenchmarks

## Benchmark design

- Weak scaling: each GPU gets 400 x 400 x 80 grid points (168 km x 168 km x 20 km)
- Domain extends in x with number of GPUs via `Partition(Ngpus, 1)`
- Physics: DCMIP2016 supercell, Kessler microphysics, WENO5 advection, anelastic dynamics
- Timing: 10 time steps at dt = 0.1 s, three trials (first is warmup/compilation)
- Distributed script uses MPI barriers between trials for synchronized timing

## Parent project

This repo was created from benchmarks in [Breeze.jl](https://github.com/NumericalEarth/Breeze.jl)
(`examples/` directory, branch `glw/erf-benchmarks`).
