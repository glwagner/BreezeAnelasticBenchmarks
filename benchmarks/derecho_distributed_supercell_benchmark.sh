#!/bin/bash
#PBS -A UMIT0049
#PBS -N supercell-weak
#PBS -j oe
#PBS -q main
#PBS -l walltime=01:00:00

# Usage (weak scaling, 1-8 GPUs):
#   NGPUS=1 qsub -v NGPUS=1 -l select=1:ncpus=64:mpiprocs=1:ngpus=1:gpu_type=a100:mem=384GB benchmarks/derecho_distributed_supercell_benchmark.sh
#   NGPUS=2 qsub -v NGPUS=2 -l select=1:ncpus=64:mpiprocs=2:ngpus=2:gpu_type=a100:mem=384GB benchmarks/derecho_distributed_supercell_benchmark.sh
#   NGPUS=4 qsub -v NGPUS=4 -l select=1:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=384GB benchmarks/derecho_distributed_supercell_benchmark.sh
#   NGPUS=8 qsub -v NGPUS=8 -l select=2:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=384GB benchmarks/derecho_distributed_supercell_benchmark.sh
#
# Derecho has 4 A100 GPUs per node, so 8 GPUs requires select=2.

module --force purge
module load ncarenv nvhpc cuda cray-mpich

export MPICH_GPU_SUPPORT_ENABLED=1
export JULIA_MPI_HAS_CUDA=true
export PALS_TRANSFER=false
# CUDA-aware MPI requires JULIA_CUDA_MEMORY_POOL=none to avoid cuIpcGetMemHandle errors
export JULIA_CUDA_MEMORY_POOL=none

NGPUS="${NGPUS:-1}"
FLOAT_TYPE="${FLOAT_TYPE:-Float32}"

cd "${PBS_O_WORKDIR}"

cat > launch.sh << 'EoF_s'
#!/bin/bash
export MPICH_GPU_SUPPORT_ENABLED=1
export LOCAL_RANK=${PMI_LOCAL_RANK}
export GLOBAL_RANK=${PMI_RANK}
export CUDA_VISIBLE_DEVICES=$(expr ${LOCAL_RANK} % 4)
echo "Global Rank ${GLOBAL_RANK} / Local Rank ${LOCAL_RANK} / CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} / $(hostname)"
exec $*
EoF_s
chmod +x launch.sh

mpiexec -n "${NGPUS}" -ppn $(( NGPUS < 4 ? NGPUS : 4 )) \
    ./launch.sh julia +1.12 --project=. \
    benchmarks/distributed_supercell_benchmark.jl --float-type "$FLOAT_TYPE"
