#!/bin/bash
#SBATCH --job-name=supercell-bench
#SBATCH --account=m5176_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=supercell_benchmark-%j.out
#SBATCH --error=supercell_benchmark-%j.err

module load julia/1.12.1

FLOAT_TYPE="${FLOAT_TYPE:-Float32}"

julia --project=. benchmarks/supercell_benchmark.jl --float-type "$FLOAT_TYPE"
