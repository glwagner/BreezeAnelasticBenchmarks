#!/bin/bash
#SBATCH --job-name=precompile
#SBATCH --account=m5176_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=precompile-%j.out
#SBATCH --error=precompile-%j.err

# Distributed precompile: run Pkg.precompile() under MPI so the distributed
# workload in BreezeAnelasticBenchmarks gets cached for all future jobs.

module load julia/1.12.1

export MPICH_GPU_SUPPORT_ENABLED=1
export JULIA_CUDA_MEMORY_POOL=none

srun --ntasks=2 --gpus=2 --gpu-bind=none \
    julia --project=. -e '
        using MPI
        MPI.Init()
        using Pkg
        Pkg.precompile()
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        rank == 0 && println("Distributed precompile complete")
    ' 2>/dev/null
