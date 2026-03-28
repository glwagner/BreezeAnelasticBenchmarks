#!/bin/bash
#SBATCH --job-name=supercell-weak
#SBATCH --account=m5176_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=distributed_supercell_benchmark-%j.out
#SBATCH --error=distributed_supercell_benchmark-%j.err

# WENO5 weak scaling: 400×400×80 per GPU, halo=(5,5,5), x-partition.
#
# Usage (pass --nodes and NGPUS on sbatch command line):
#   NGPUS=1  sbatch --nodes=1  benchmarks/distributed_supercell_benchmark.sh
#   NGPUS=4  sbatch --nodes=1  benchmarks/distributed_supercell_benchmark.sh
#   NGPUS=8  sbatch --nodes=2  benchmarks/distributed_supercell_benchmark.sh
#   NGPUS=16 sbatch --nodes=4  benchmarks/distributed_supercell_benchmark.sh
#   NGPUS=32 sbatch --nodes=8  benchmarks/distributed_supercell_benchmark.sh
#   NGPUS=64 sbatch --nodes=16 benchmarks/distributed_supercell_benchmark.sh
#
# Perlmutter has 4 A100-80GB GPUs per node.

module load julia/1.12.1
module load nccl/2.29.2-cu13
export LD_PRELOAD=/usr/lib64/libstdc++.so.6

export MPICH_GPU_SUPPORT_ENABLED=1
export JULIA_CUDA_MEMORY_POOL=none

NGPUS="${NGPUS:-1}"
FLOAT_TYPE="${FLOAT_TYPE:-Float32}"
export NT="${NT:-100}"

NCCL_FLAG=""
[ "${USE_NCCL:-0}" = "1" ] && NCCL_FLAG="--nccl"

srun --ntasks="${NGPUS}" --gpus="${NGPUS}" --gpu-bind=none \
    julia --project=. benchmarks/distributed_supercell_benchmark.jl --float-type "$FLOAT_TYPE" $NCCL_FLAG
