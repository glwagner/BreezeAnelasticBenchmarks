#!/bin/bash
#SBATCH --job-name=diag-cf
#SBATCH --account=m5176_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=2
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=diag-cf-%j.out
#SBATCH --error=diag-cf-%j.err

module load julia/1.12.1
module load cray-mpich
module load nccl/2.29.2-cu13

export MPICH_GPU_SUPPORT_ENABLED=1
export JULIA_CUDA_MEMORY_POOL=none
export JULIA_PKG_PRECOMPILE_AUTO=0
export LD_LIBRARY_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"
export LD_PRELOAD=/usr/lib64/libstdc++.so.6

# Precompile first (single process)
julia --project=. -e 'using Pkg; Pkg.precompile()' 2>/dev/null

srun --ntasks=8 --gpus=8 --gpu-bind=none \
    julia --project=. benchmarks/diagnose_centerfield.jl
