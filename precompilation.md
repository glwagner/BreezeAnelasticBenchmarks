# Precompilation on Perlmutter

## The problem

Julia uses just-in-time (JIT) compilation, specializing code for specific types at first use.
Without precompilation, each benchmark job spends 30–60+ minutes compiling before producing
any results. On multi-node jobs, this often exceeds the walltime limit.

## Key insight: Julia specializes on types, not values

Grid size doesn't matter — a precompile workload on an 8×8×8 grid caches the same compiled
code as a 400×400×80 grid. What matters is the **type combination**:

- **Architecture type**: `GPU`, `Distributed{..., MPI.Comm}`, `Distributed{..., NCCLCommunicator}`
- **Dynamics type**: `AnelasticDynamics` vs `CompressibleDynamics`
- **Advection type**: `WENO{5}` vs `Centered{2}`
- **Closure type**: `Nothing` vs `ScalarDiffusivity`

Each unique combination triggers separate compilation. Missing any combination means
the benchmark job will JIT-compile it at runtime.

## What must be precompiled

Three benchmark configs × three architectures = **9 combinations**:

| | GPU (serial) | MPI Distributed | NCCL Distributed |
|---|---|---|---|
| **WENO5 anelastic** | `setup_supercell` | `setup_supercell` | `setup_supercell` |
| **ERF anelastic** | `setup_supercell_erf` | `setup_supercell_erf` | `setup_supercell_erf` |
| **Compressible** | `setup_supercell_compressible` | `setup_supercell_compressible` | `setup_supercell_compressible` |

The `Distributed` and `NCCLDistributed` architectures are different types, so they need
separate precompilation even though the physics code is the same.

## How to precompile everything

### Step 1: Serial precompile on login node

```bash
module load julia/1.12.1 nccl/2.29.2-cu13
export LD_PRELOAD=/usr/lib64/libstdc++.so.6
julia --project=. -e 'using Pkg; Pkg.precompile()'
```

This caches the 3 GPU (serial) workloads. Perlmutter login nodes have GPUs.

### Step 2: Distributed precompile under MPI (2 GPUs)

```bash
sbatch benchmarks/perlmutter_precompile.sh
```

This runs `Pkg.precompile()` under `srun -n 2 --gpus 2`, which triggers the
distributed precompile workloads for both MPI and NCCL architectures.
Takes ~40 seconds once the serial cache exists.

### Why a single 2-GPU job covers everything

The precompile workload in `src/BreezeAnelasticBenchmarks.jl` runs all 9 combinations:

```julia
function precompile_all_configs!(arch; Nx, Ny, Nz, Lx, Ly, Lz)
    run_benchmark!(setup_supercell(arch; Nx, Ny, Nz, Lx, Ly, Lz), 1)
    run_benchmark!(setup_supercell_erf(arch; Nx, Ny, Nz, Lx, Ly, Lz), 1)
    run_benchmark!(setup_supercell_compressible(arch; Nx, Ny, Nz, Lx, Ly, Lz), 1)
end

# Called for GPU(), Distributed(GPU()), and NCCLDistributed(GPU())
```

The compiled code is cached in `~/.julia/compiled/v1.12/` and shared across all
subsequent jobs regardless of node count or grid size.

## Common pitfalls

1. **Editing installed packages invalidates the cache** — if you patch
   `~/.julia/packages/Oceananigans/...` or `~/.julia/packages/Breeze/...`,
   you must re-run `Pkg.precompile()` on the login node AND the distributed
   precompile job.

2. **Switching Oceananigans/Breeze/NCCL branches** — `Pkg.update()` or `Pkg.add()`
   changes the package slug, creating a new compilation target. Re-precompile after
   any branch switch.

3. **CUDA version changes** — loading `nccl/2.29.2-cu13` switches to `cudatoolkit/13.0`.
   If you previously precompiled with CUDA 12.9, the cache is invalidated. Always
   precompile with the same modules as the SLURM scripts.

4. **MPI precompile must run under srun** — `Pkg.precompile()` on the login node
   only triggers the serial workload (no `MPI.Initialized()`). The distributed
   workloads require an actual MPI job.

5. **NCCL precompile requires the NCCL module** — the precompile workload uses
   `try; using NCCL; ...` so it gracefully skips NCCL if the module isn't loaded,
   but then NCCL code paths won't be cached.
