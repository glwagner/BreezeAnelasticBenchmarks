#!/bin/bash
#
# Weak-scaling supercell benchmark (MPI, one rank per GPU)
#
# Usage:
#   NGPUS=2 sbatch distributed_supercell_benchmark.sh
#   NGPUS=4 sbatch distributed_supercell_benchmark.sh
#
#SBATCH --job-name=supercell-weak
#SBATCH --account=m5176_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=distributed_supercell_benchmark-%j.out
#SBATCH --error=distributed_supercell_benchmark-%j.err

PROJECT_DIR="/global/u1/g/glwagner/BreezeAnelasticBenchmarks"

module load julia/1.12.1
JULIA="${JULIA:-julia}"

NGPUS="${NGPUS:-2}"
FLOAT_TYPE="${FLOAT_TYPE:-Float32}"

echo "=========================================="
echo "Supercell Weak Scaling -- Perlmutter"
echo "=========================================="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         ${SLURM_NODELIST}"
echo "GPUs:         ${NGPUS}"
echo "Float type:   ${FLOAT_TYPE}"
echo "Julia:        $($JULIA --version)"
echo "=========================================="

cd "${PROJECT_DIR}"

srun --ntasks="${NGPUS}" --gpus="${NGPUS}" \
    $JULIA --project=. distributed_supercell_benchmark.jl \
    --float-type "${FLOAT_TYPE}"

echo "=========================================="
echo "Benchmark finished with exit code: $?"
echo "=========================================="
