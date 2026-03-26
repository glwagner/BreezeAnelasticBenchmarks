#!/bin/bash
# Submit a full weak scaling suite on Derecho: 2, 4, 8, 16 nodes
# (= 8, 16, 32, 64 GPUs with 4 A100s per node)
#
# Each GPU gets 400 x 400 x 80 grid points (168 km x 168 km x 20 km).
# Domain extends in x with number of GPUs via Partition(Ngpus, 1).
#
# Usage:
#   bash benchmarks/derecho_weak_scaling_suite.sh
#
# Requires: performance issues resolved first (expect ~0.6 s per trial)

SCRIPT=benchmarks/derecho_distributed_supercell_benchmark.sh

echo "Submitting weak scaling suite..."

# 1 GPU  / 1 node
JOB1=$(qsub -v NGPUS=1  -l select=1:ncpus=64:mpiprocs=1:ngpus=1:gpu_type=a100:mem=384GB $SCRIPT)
echo "1  GPU  (1  node): $JOB1"

# 2 GPUs / 1 node
JOB2=$(qsub -v NGPUS=2  -l select=1:ncpus=64:mpiprocs=2:ngpus=2:gpu_type=a100:mem=384GB $SCRIPT)
echo "2  GPUs (1  node): $JOB2"

# 4 GPUs / 1 node
JOB4=$(qsub -v NGPUS=4  -l select=1:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=384GB $SCRIPT)
echo "4  GPUs (1  node): $JOB4"

# 8 GPUs / 2 nodes
JOB8=$(qsub -v NGPUS=8  -l select=2:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=384GB $SCRIPT)
echo "8  GPUs (2  nodes): $JOB8"

# 16 GPUs / 4 nodes
JOB16=$(qsub -v NGPUS=16 -l select=4:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=384GB $SCRIPT)
echo "16 GPUs (4  nodes): $JOB16"

# 32 GPUs / 8 nodes
JOB32=$(qsub -v NGPUS=32 -l select=8:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=384GB $SCRIPT)
echo "32 GPUs (8  nodes): $JOB32"

# 64 GPUs / 16 nodes
JOB64=$(qsub -v NGPUS=64 -l select=16:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=384GB $SCRIPT)
echo "64 GPUs (16 nodes): $JOB64"

echo ""
echo "Monitor with: qstat -u \$USER"
