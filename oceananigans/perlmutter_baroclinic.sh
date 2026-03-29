#!/bin/bash
#SBATCH --job-name=baro-inst
#SBATCH --account=m5176_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=baroclinic-%j.out
#SBATCH --error=baroclinic-%j.err

# Vanilla Oceananigans baroclinic instability weak scaling benchmark
# Based on GB-25 sharding test but using standard MPI Distributed
#
# Usage:
#   NGPUS=4 sbatch --nodes=1 oceananigans/perlmutter_baroclinic.sh
#   NGPUS=8 sbatch --nodes=2 oceananigans/perlmutter_baroclinic.sh
#   USE_NCCL=1 NGPUS=8 sbatch --nodes=2 oceananigans/perlmutter_baroclinic.sh

module load julia/1.12.1
module load nccl/2.29.2-cu13

export MPICH_GPU_SUPPORT_ENABLED=1
export JULIA_CUDA_MEMORY_POOL=none
export LD_PRELOAD="/usr/lib64/libstdc++.so.6:${CRAY_MPICH_ROOTDIR}/gtl/lib/libmpi_gtl_cuda.so"
export LD_LIBRARY_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"

NGPUS="${NGPUS:-1}"
export NT="${NT:-10}"
export NX_PER_GPU="${NX_PER_GPU:-64}"
export NY="${NY:-64}"
export NZ="${NZ:-4}"

NCCL_FLAG=""
[ "${USE_NCCL:-0}" = "1" ] && NCCL_FLAG="--nccl"

srun --ntasks="${NGPUS}" --gpus="${NGPUS}" --gpu-bind=none \
    julia --project=. oceananigans/baroclinic_instability_benchmark.jl $NCCL_FLAG
