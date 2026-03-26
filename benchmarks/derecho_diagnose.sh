#!/bin/bash
#PBS -A UMIT0049
#PBS -N diag-perf
#PBS -j oe
#PBS -q main
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=64:mpiprocs=1:ngpus=1:gpu_type=a100:mem=384GB

module --force purge
module load ncarenv nvhpc cuda cray-mpich

# These can be overridden via -v on qsub
export MPICH_GPU_SUPPORT_ENABLED=1
export JULIA_MPI_HAS_CUDA=true
export PALS_TRANSFER=false

# JULIA_CUDA_MEMORY_POOL can be set via -v; if not set, Julia/CUDA.jl uses its default (binned)
JULIA_VERSION="${JULIA_VERSION:-1.12}"

cd "${PBS_O_WORKDIR}"

echo "=== Environment ==="
echo "JULIA_VERSION=$JULIA_VERSION"
echo "JULIA_CUDA_MEMORY_POOL=${JULIA_CUDA_MEMORY_POOL:-<not set, using default>}"
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.current.sm,clocks.current.memory --format=csv
echo ""

julia +${JULIA_VERSION} --project=. benchmarks/diagnose_performance.jl
