#!/bin/bash
#PBS -A UMIT0049
#PBS -N transpose-bench
#PBS -j oe
#PBS -q main
#PBS -l walltime=00:30:00

module --force purge
module load ncarenv nvhpc cuda cray-mpich

export MPICH_GPU_SUPPORT_ENABLED=1
export JULIA_MPI_HAS_CUDA=true
export PALS_TRANSFER=false
export JULIA_CUDA_MEMORY_POOL=none

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

mpiexec ./launch.sh julia +1.12 --project=. \
    benchmarks/benchmark_transpose_strategies.jl
