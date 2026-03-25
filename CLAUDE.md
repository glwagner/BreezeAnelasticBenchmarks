# BreezeAnelasticBenchmarks

Weak-scaling GPU benchmarks for Breeze.jl anelastic supercell simulations
on NERSC Perlmutter (NVIDIA A100 80 GB GPUs).

## Current state (2026-03-25)

### Completed
- Single-GPU Float32 benchmark: ~0.61 s per 10 time steps
- Single-GPU Float64 benchmark: ~0.99 s per 10 time steps
- Results are from Breeze.jl repo runs (jobs 50542402, 50546374, 50547658)

### Pending SLURM jobs (submitted from this repo)
- Job 50551243: 1-GPU Float32 baseline (for consistent comparison)
- Job 50551235: 2-GPU Float32 weak scaling
- Job 50551238: 4-GPU Float32 weak scaling

These were pending in the debug queue as of session end. Check output files:
- `supercell_benchmark-50551243.out`
- `distributed_supercell_benchmark-50551235.out`
- `distributed_supercell_benchmark-50551238.out`

### TODO
- Check if the 3 pending jobs completed successfully
- Update the README weak-scaling table with distributed results
- The earlier distributed runs (jobs 50550380, 50550383) failed because
  MPI.jl was missing from the Breeze.jl examples/Project.toml. The runs
  from this repo should work since Project.toml here includes MPI.
- Consider adding Float64 weak-scaling runs

## Project setup

- Julia 1.12.1, `module load julia/1.12.1` on Perlmutter
- Project is already instantiated (Manifest.toml exists)
- SLURM account: `m5176_g`, QOS: `debug`, constraint: `gpu`
- GitHub: https://github.com/glwagner/BreezeAnelasticBenchmarks

## Benchmark design

- Weak scaling: each GPU gets 400 × 400 × 80 grid points (168 km × 168 km × 20 km)
- Domain extends in x with number of GPUs via `Partition(Ngpus, 1)`
- Physics: DCMIP2016 supercell, Kessler microphysics, WENO5 advection, anelastic dynamics
- Timing: 10 time steps at Δt = 0.1 s, three trials (first is warmup/compilation)
- Distributed script uses MPI barriers between trials for synchronized timing

## Parent project

This repo was created from benchmarks in [Breeze.jl](https://github.com/NumericalEarth/Breeze.jl)
(`examples/` directory, branch `glw/erf-benchmarks`).
