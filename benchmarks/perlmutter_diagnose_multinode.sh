#!/bin/bash
#SBATCH --job-name=no-gtl-test
#SBATCH --account=m5176_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=2
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=no-gtl-test-%j.out
#SBATCH --error=no-gtl-test-%j.err

module load julia/1.12.1
module load nccl/2.29.2-cu13

# NO LD_PRELOAD of GTL — test if multi-node hangs without it
# GPU-aware MPI won't work but we just want to see if the job progresses
export MPICH_GPU_SUPPORT_ENABLED=0
export JULIA_CUDA_MEMORY_POOL=none
export LD_PRELOAD="/usr/lib64/libstdc++.so.6"

srun --ntasks=8 --gpus=8 --gpu-bind=none \
    julia --project=. -e '
        using Logging; disable_logging(Logging.Warn)
        using MPI; MPI.Init()
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        rank == 0 && println("MPI init done, loading packages...")
        using BreezeAnelasticBenchmarks
        using Oceananigans.Units
        rank == 0 && println("Packages loaded, creating model...")
        arch = Distributed(GPU(); partition = Partition(8, 1))
        rank == 0 && println("Arch created, building model...")
        model = setup_supercell_compressible(arch; Nx=400, Ny=400, Lx=168000.0*8, Ly=168000.0)
        rank == 0 && println("Model created, running 1 step...")
        run_benchmark!(model, 1)
        rank == 0 && println("Step done!")
        MPI.Barrier(MPI.COMM_WORLD)
        rank == 0 && println("All ranks finished")
    '
