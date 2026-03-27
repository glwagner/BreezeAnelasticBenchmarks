#!/bin/bash
# Submit WENO weak scaling suite on Perlmutter: 1, 2, 4, 8, 16, 32, 64 GPUs
# Each GPU gets 400 x 400 x 80 grid points (168 km x 168 km x 20 km).
# Domain extends in x with number of GPUs via Partition(Ngpus, 1).
#
# Usage:
#   bash benchmarks/perlmutter_weak_scaling_suite.sh
#
# Perlmutter has 4 A100-80GB GPUs per node.

SCRIPT=benchmarks/distributed_supercell_benchmark.sh

echo "Submitting WENO weak scaling suite on Perlmutter..."

# 1 GPU / 1 node
JOB1=$(NGPUS=1  sbatch --nodes=1  --parsable $SCRIPT)
echo "1  GPU  (1  node): $JOB1"

# 2 GPUs / 1 node
JOB2=$(NGPUS=2  sbatch --nodes=1  --parsable $SCRIPT)
echo "2  GPUs (1  node): $JOB2"

# 4 GPUs / 1 node
JOB4=$(NGPUS=4  sbatch --nodes=1  --parsable $SCRIPT)
echo "4  GPUs (1  node): $JOB4"

# 8 GPUs / 2 nodes
JOB8=$(NGPUS=8  sbatch --nodes=2  --parsable $SCRIPT)
echo "8  GPUs (2  nodes): $JOB8"

# 16 GPUs / 4 nodes
JOB16=$(NGPUS=16 sbatch --nodes=4  --parsable $SCRIPT)
echo "16 GPUs (4  nodes): $JOB16"

# 32 GPUs / 8 nodes
JOB32=$(NGPUS=32 sbatch --nodes=8  --parsable $SCRIPT)
echo "32 GPUs (8  nodes): $JOB32"

# 64 GPUs / 16 nodes
JOB64=$(NGPUS=64 sbatch --nodes=16 --parsable $SCRIPT)
echo "64 GPUs (16 nodes): $JOB64"

echo ""
echo "Monitor with: squeue -u \$USER"
