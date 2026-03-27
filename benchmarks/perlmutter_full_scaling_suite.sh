#!/bin/bash
# Submit full Perlmutter weak scaling suite: WENO, ERF anelastic, and compressible
# at both 50×400×80/GPU and 400×400×80/GPU grid sizes.
# NT=100 steps to match Derecho results.
#
# Usage:
#   bash benchmarks/perlmutter_full_scaling_suite.sh
#
# Perlmutter has 4 A100-80GB GPUs per node.

WENO_SCRIPT=benchmarks/distributed_supercell_benchmark.sh
ERF_SCRIPT=benchmarks/distributed_supercell_erf_benchmark.sh
COMP_SCRIPT=benchmarks/distributed_supercell_compressible_benchmark.sh

echo "===== Perlmutter Full Scaling Suite (NT=100) ====="
echo ""

# ---- 50×400×80 per GPU (ERF comparison grid) ----

echo "--- WENO5 anelastic, 50×400×80/GPU ---"
for spec in "1 1" "2 1" "4 1" "8 2" "16 4" "20 5" "40 10"; do
    read -r NGPUS NODES <<< "$spec"
    JOB=$(NGPUS=$NGPUS NX_PER_GPU=50 NY_PER_GPU=400 sbatch --nodes=$NODES --parsable $WENO_SCRIPT)
    printf "%2d GPUs (%2d nodes): %s\n" $NGPUS $NODES $JOB
done
echo ""

echo "--- ERF anelastic, 50×400×80/GPU ---"
for spec in "1 1" "2 1" "4 1" "8 2" "16 4" "20 5" "40 10"; do
    read -r NGPUS NODES <<< "$spec"
    JOB=$(NGPUS=$NGPUS sbatch --nodes=$NODES --parsable $ERF_SCRIPT)
    printf "%2d GPUs (%2d nodes): %s\n" $NGPUS $NODES $JOB
done
echo ""

echo "--- Compressible, 50×400×80/GPU ---"
for spec in "1 1" "2 1" "4 1" "8 2" "16 4" "20 5" "40 10"; do
    read -r NGPUS NODES <<< "$spec"
    JOB=$(NGPUS=$NGPUS sbatch --nodes=$NODES --parsable $COMP_SCRIPT)
    printf "%2d GPUs (%2d nodes): %s\n" $NGPUS $NODES $JOB
done
echo ""

# ---- 400×400×80 per GPU (full supercell grid) ----

echo "--- WENO5 anelastic, 400×400×80/GPU ---"
for spec in "1 1" "2 1" "4 1" "8 2" "16 4" "20 5"; do
    read -r NGPUS NODES <<< "$spec"
    JOB=$(NGPUS=$NGPUS NX_PER_GPU=400 NY_PER_GPU=400 sbatch --nodes=$NODES --parsable $WENO_SCRIPT)
    printf "%2d GPUs (%2d nodes): %s\n" $NGPUS $NODES $JOB
done
echo ""

echo "--- ERF anelastic, 400×400×80/GPU ---"
for spec in "1 1" "2 1" "4 1" "8 2" "16 4" "20 5"; do
    read -r NGPUS NODES <<< "$spec"
    JOB=$(NGPUS=$NGPUS NX_PER_GPU=400 NY_PER_GPU=400 sbatch --nodes=$NODES --parsable $ERF_SCRIPT)
    printf "%2d GPUs (%2d nodes): %s\n" $NGPUS $NODES $JOB
done
echo ""

echo "--- Compressible, 400×400×80/GPU ---"
for spec in "1 1" "2 1" "4 1" "8 2" "16 4" "20 5"; do
    read -r NGPUS NODES <<< "$spec"
    JOB=$(NGPUS=$NGPUS NX_PER_GPU=400 NY_PER_GPU=400 sbatch --nodes=$NODES --parsable $COMP_SCRIPT)
    printf "%2d GPUs (%2d nodes): %s\n" $NGPUS $NODES $JOB
done

echo ""
echo "Monitor with: squeue -u \$USER"
