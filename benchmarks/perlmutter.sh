#!/bin/bash
#SBATCH --job-name=breeze-bench
#SBATCH --account=m5176_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=breeze-bench-%j.out
#SBATCH --error=breeze-bench-%j.err

# Unified Perlmutter benchmark submission script.
#
# Usage:
#   CONFIG=compressible NGPUS=4 sbatch --nodes=1 benchmarks/perlmutter.sh
#   CONFIG=erf USE_NCCL=1 NGPUS=8 sbatch --nodes=2 benchmarks/perlmutter.sh
#   CONFIG=weno NGPUS=16 sbatch --nodes=4 benchmarks/perlmutter.sh

module load julia/1.12.1
module load cray-mpich
module load nccl/2.29.2-cu13

export MPICH_GPU_SUPPORT_ENABLED=1
export JULIA_CUDA_MEMORY_POOL=none
export JULIA_PKG_PRECOMPILE_AUTO=0

# GTL is loaded via MPIPreferences dlopen (no LD_PRELOAD of GTL).
# GTL needs libcudart.so.12 but nccl/2.29.2-cu13 loads CUDA 13.
# Add CUDA 12.9 lib path so GTL can find libcudart.so.12.
export LD_LIBRARY_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"

# libstdc++ preload fixes NCCL.jl compatibility with system Julia.
export LD_PRELOAD=/usr/lib64/libstdc++.so.6

# Precompile on a single process before srun to avoid
# multi-rank precompile contention and deadlocks
julia --project=. -e 'using Pkg; Pkg.precompile()' 2>/dev/null

NGPUS="${NGPUS:-1}"
CONFIG="${CONFIG:-compressible}"

NCCL_FLAG=""
[ "${USE_NCCL:-0}" = "1" ] && NCCL_FLAG="--nccl"

srun --ntasks="${NGPUS}" --gpus="${NGPUS}" --gpu-bind=none \
    julia --project=. benchmarks/run_benchmark.jl --config "$CONFIG" $NCCL_FLAG
