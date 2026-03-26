#!/bin/bash
# Submit ERF-equivalent weak scaling suite on Derecho
# Centered(2), ScalarDiffusivity(ν=200,κ=200), halo=(1,1,1)
# 200×200×80 per GPU, 2D partition
#
# Usage:
#   bash benchmarks/derecho_erf_scaling_suite.sh

SCRIPT=benchmarks/derecho_distributed_supercell_erf_benchmark.sh

echo "Submitting ERF-equivalent weak scaling suite..."

# 1 GPU / 1 node
JOB1=$(qsub -v NGPUS=1  -l select=1:ncpus=64:mpiprocs=1:ngpus=1:gpu_type=a100:mem=384GB $SCRIPT)
echo "1  GPU  (1  node): $JOB1"

# 4 GPUs / 1 node (Partition 2×2)
JOB4=$(qsub -v NGPUS=4  -l select=1:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=384GB $SCRIPT)
echo "4  GPUs (1  node): $JOB4"

# 8 GPUs / 2 nodes (Partition 4×2)
JOB8=$(qsub -v NGPUS=8  -l select=2:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=384GB $SCRIPT)
echo "8  GPUs (2  nodes): $JOB8"

# 16 GPUs / 4 nodes (Partition 4×4)
JOB16=$(qsub -v NGPUS=16 -l select=4:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=384GB $SCRIPT)
echo "16 GPUs (4  nodes): $JOB16"

# 32 GPUs / 8 nodes (Partition 8×4)
JOB32=$(qsub -v NGPUS=32 -l select=8:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=384GB $SCRIPT)
echo "32 GPUs (8  nodes): $JOB32"

# 64 GPUs / 16 nodes (Partition 8×8)
JOB64=$(qsub -v NGPUS=64 -l select=16:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=384GB $SCRIPT)
echo "64 GPUs (16 nodes): $JOB64"

echo ""
echo "Monitor with: qstat -u \$USER"
