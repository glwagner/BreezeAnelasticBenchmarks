#!/bin/bash
#
# Supercell benchmark on a single A100 GPU
#
# Usage:
#   sbatch supercell_benchmark.sh
#
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

PROJECT_DIR="/global/u1/g/glwagner/BreezeAnelasticBenchmarks"

module load julia/1.12.1
JULIA="${JULIA:-julia}"

echo "=========================================="
echo "Supercell Benchmark -- Perlmutter (1 GPU)"
echo "=========================================="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         ${SLURM_NODELIST}"
echo "Julia:        $($JULIA --version)"
echo "=========================================="

cd "${PROJECT_DIR}"

FLOAT_TYPE="${FLOAT_TYPE:-Float32}"
echo "Float type:   ${FLOAT_TYPE}"

$JULIA --project=. supercell_benchmark.jl --float-type "${FLOAT_TYPE}"

echo "=========================================="
echo "Benchmark finished with exit code: $?"
echo "=========================================="
