#!/bin/bash
#SBATCH --job-name=supercell-erf
#SBATCH --account=m5176_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=distributed_supercell_erf_benchmark-%j.out
#SBATCH --error=distributed_supercell_erf_benchmark-%j.err

# ERF-equivalent weak scaling: Centered(2), ScalarDiffusivity(ν=200,κ=200),
# 200×200×80 per GPU, 2D partition, halo=(1,1,1).
#
# Usage (pass --nodes and NGPUS on sbatch command line):
#   NGPUS=1  sbatch --nodes=1  benchmarks/distributed_supercell_erf_benchmark.sh
#   NGPUS=4  sbatch --nodes=1  benchmarks/distributed_supercell_erf_benchmark.sh
#   NGPUS=8  sbatch --nodes=2  benchmarks/distributed_supercell_erf_benchmark.sh
#   NGPUS=16 sbatch --nodes=4  benchmarks/distributed_supercell_erf_benchmark.sh
#   NGPUS=32 sbatch --nodes=8  benchmarks/distributed_supercell_erf_benchmark.sh
#   NGPUS=64 sbatch --nodes=16 benchmarks/distributed_supercell_erf_benchmark.sh
#
# Perlmutter has 4 A100-80GB GPUs per node.

module load julia/1.12.1
module load nccl/2.29.2-cu13

export MPICH_GPU_SUPPORT_ENABLED=1
export JULIA_CUDA_MEMORY_POOL=none
export LD_LIBRARY_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"
export LD_PRELOAD="/usr/lib64/libstdc++.so.6:${CRAY_MPICH_ROOTDIR}/gtl/lib/libmpi_gtl_cuda.so"

NGPUS="${NGPUS:-1}"
FLOAT_TYPE="${FLOAT_TYPE:-Float32}"
export NT="${NT:-100}"
export NX_PER_GPU="${NX_PER_GPU:-50}"
export NY_PER_GPU="${NY_PER_GPU:-400}"
export PARTITION_X_ONLY="${PARTITION_X_ONLY:-1}"

NCCL_FLAG=""
[ "${USE_NCCL:-0}" = "1" ] && NCCL_FLAG="--nccl"

srun --ntasks="${NGPUS}" --gpus="${NGPUS}" --gpu-bind=none \
    julia --project=. benchmarks/distributed_supercell_erf_benchmark.jl --float-type "$FLOAT_TYPE" $NCCL_FLAG
