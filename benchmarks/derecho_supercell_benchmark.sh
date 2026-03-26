#!/bin/bash
#PBS -A UMIT0049
#PBS -N supercell-bench
#PBS -j oe
#PBS -q main
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=64:mpiprocs=1:ngpus=1:gpu_type=a100:mem=384GB

module --force purge
module load ncarenv nvhpc cuda cray-mpich

export MPICH_GPU_SUPPORT_ENABLED=1
export JULIA_MPI_HAS_CUDA=true
export PALS_TRANSFER=false
export JULIA_CUDA_MEMORY_POOL=none

FLOAT_TYPE="${FLOAT_TYPE:-Float32}"

cd "${PBS_O_WORKDIR}"

julia +1.12 --project=. benchmarks/supercell_benchmark.jl --float-type "$FLOAT_TYPE"
