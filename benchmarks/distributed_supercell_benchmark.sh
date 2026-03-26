#!/bin/bash
#SBATCH --job-name=supercell-weak
#SBATCH --account=m5176_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=distributed_supercell_benchmark-%j.out
#SBATCH --error=distributed_supercell_benchmark-%j.err

# Pass --nodes on the sbatch command line, e.g.:
#   NGPUS=4 sbatch --nodes=1 benchmarks/distributed_supercell_benchmark.sh
#   NGPUS=8 sbatch --nodes=2 benchmarks/distributed_supercell_benchmark.sh
# (Perlmutter has 4 A100 GPUs per node)

module load julia/1.12.1

NGPUS="${NGPUS:-2}"
FLOAT_TYPE="${FLOAT_TYPE:-Float32}"

srun --ntasks="${NGPUS}" --gpus="${NGPUS}" \
    julia --project=. benchmarks/distributed_supercell_benchmark.jl --float-type "$FLOAT_TYPE"
