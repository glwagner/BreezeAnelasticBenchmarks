#!/bin/bash
#SBATCH --job-name=warmup
#SBATCH --account=m5176_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=2
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=warmup-%j.out
#SBATCH --error=warmup-%j.err

# Warmup job: run all three benchmark configs on 8 GPUs (2 nodes) with NT=1
# to cache JIT compilation. Subsequent jobs will start much faster.

module load julia/1.12.1

export MPICH_GPU_SUPPORT_ENABLED=1
export JULIA_CUDA_MEMORY_POOL=none

echo "=== Warmup: compressible 50x400x80 ==="
NT=1 NX_PER_GPU=50 NY_PER_GPU=400 \
srun --ntasks=8 --gpus=8 --gpu-bind=none \
    julia --project=. benchmarks/distributed_supercell_compressible_benchmark.jl

echo "=== Warmup: ERF 50x400x80 ==="
NT=1 NX_PER_GPU=50 NY_PER_GPU=400 PARTITION_X_ONLY=1 \
srun --ntasks=8 --gpus=8 --gpu-bind=none \
    julia --project=. benchmarks/distributed_supercell_erf_benchmark.jl

echo "=== Warmup: WENO 400x400x80 ==="
NT=1 NX_PER_GPU=400 NY_PER_GPU=400 \
srun --ntasks=8 --gpus=8 --gpu-bind=none \
    julia --project=. benchmarks/distributed_supercell_benchmark.jl

echo "=== Warmup complete ==="
