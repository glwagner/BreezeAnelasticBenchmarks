#!/bin/bash
# Submit ERF-equivalent weak scaling suite on Perlmutter
# Centered(2), ScalarDiffusivity(ν=200,κ=200), halo=(1,1,1)
# 50×400×80 per GPU, x-only partition (matching ERF comparison config)
#
# Usage:
#   bash benchmarks/perlmutter_erf_scaling_suite.sh
#
# Perlmutter has 4 A100-80GB GPUs per node.

SCRIPT=benchmarks/distributed_supercell_erf_benchmark.sh

echo "Submitting ERF-equivalent weak scaling suite on Perlmutter..."
echo "Config: 50×400×80 per GPU, x-only partition"

# 1 GPU / 1 node (50×400×80)
JOB1=$(NGPUS=1  sbatch --nodes=1  --parsable $SCRIPT)
echo "1  GPU  (1  node): $JOB1"

# 2 GPUs / 1 node (100×400×80)
JOB2=$(NGPUS=2  sbatch --nodes=1  --parsable $SCRIPT)
echo "2  GPUs (1  node): $JOB2"

# 4 GPUs / 1 node (200×400×80)
JOB4=$(NGPUS=4  sbatch --nodes=1  --parsable $SCRIPT)
echo "4  GPUs (1  node): $JOB4"

# 8 GPUs / 2 nodes (400×400×80)
JOB8=$(NGPUS=8  sbatch --nodes=2  --parsable $SCRIPT)
echo "8  GPUs (2  nodes): $JOB8"

# 16 GPUs / 4 nodes (800×400×80)
JOB16=$(NGPUS=16 sbatch --nodes=4  --parsable $SCRIPT)
echo "16 GPUs (4  nodes): $JOB16"

# 20 GPUs / 5 nodes (1000×400×80)
JOB20=$(NGPUS=20 sbatch --nodes=5  --parsable $SCRIPT)
echo "20 GPUs (5  nodes): $JOB20"

# 40 GPUs / 10 nodes (2000×400×80)
JOB40=$(NGPUS=40 sbatch --nodes=10 --parsable $SCRIPT)
echo "40 GPUs (10 nodes): $JOB40"

echo ""
echo "Monitor with: squeue -u \$USER"
