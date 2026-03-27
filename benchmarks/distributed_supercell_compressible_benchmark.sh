#!/bin/bash
#SBATCH --job-name=supercell-comp
#SBATCH --account=m5176_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=distributed_supercell_compressible_benchmark-%j.out
#SBATCH --error=distributed_supercell_compressible_benchmark-%j.err

# Compressible dynamics weak scaling: fully explicit, Centered(2),
# ScalarDiffusivity, no Poisson pressure solve.
# 200×200×80 per GPU, x-only partition.
#
# Usage (pass --nodes and NGPUS on sbatch command line):
#   NGPUS=1  sbatch --nodes=1  benchmarks/distributed_supercell_compressible_benchmark.sh
#   NGPUS=4  sbatch --nodes=1  benchmarks/distributed_supercell_compressible_benchmark.sh
#   NGPUS=8  sbatch --nodes=2  benchmarks/distributed_supercell_compressible_benchmark.sh
#
# Perlmutter has 4 A100-80GB GPUs per node.

module load julia/1.12.1

export MPICH_GPU_SUPPORT_ENABLED=1
export JULIA_CUDA_MEMORY_POOL=none

NGPUS="${NGPUS:-1}"
FLOAT_TYPE="${FLOAT_TYPE:-Float32}"
export NT="${NT:-10}"
export NX_PER_GPU="${NX_PER_GPU:-200}"
export NY_PER_GPU="${NY_PER_GPU:-200}"

srun --ntasks="${NGPUS}" --gpus="${NGPUS}" --gpu-bind=none \
    julia --project=. benchmarks/distributed_supercell_compressible_benchmark.jl --float-type "$FLOAT_TYPE" \
    2>/dev/null
