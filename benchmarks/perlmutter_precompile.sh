#!/bin/bash
#SBATCH --job-name=precompile
#SBATCH --account=m5176_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=precompile-%j.out
#SBATCH --error=precompile-%j.err

# Distributed precompile: caches all 3 configs × 3 architectures
# (GPU, MPI Distributed, NCCL Distributed) on 2 GPUs.

module load julia/1.12.1
module load nccl/2.29.2-cu13

export MPICH_GPU_SUPPORT_ENABLED=1
export JULIA_CUDA_MEMORY_POOL=none
export LD_PRELOAD="/usr/lib64/libstdc++.so.6:${CRAY_MPICH_ROOTDIR}/gtl/lib/libmpi_gtl_cuda.so"
export LD_LIBRARY_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"

srun --ntasks=2 --gpus=2 --gpu-bind=none \
    julia --project=. -e '
        using Logging; disable_logging(Logging.Warn)
        using MPI; MPI.Init()
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        using Pkg; Pkg.precompile()
        rank == 0 && println("Distributed precompile complete (MPI + NCCL × 3 configs)")
    '
