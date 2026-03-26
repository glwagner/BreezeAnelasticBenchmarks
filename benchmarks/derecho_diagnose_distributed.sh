#!/bin/bash
#PBS -A UMIT0049
#PBS -N diag-dist
#PBS -j oe
#PBS -q main
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=64:mpiprocs=1:ngpus=1:gpu_type=a100:mem=384GB

module --force purge
module load ncarenv nvhpc cuda cray-mpich

export MPICH_GPU_SUPPORT_ENABLED=1
export JULIA_MPI_HAS_CUDA=true
export PALS_TRANSFER=false

# JULIA_CUDA_MEMORY_POOL can be set via -v; if not set, uses default
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

echo "JULIA_CUDA_MEMORY_POOL=${JULIA_CUDA_MEMORY_POOL:-<not set>}"

mpiexec -n 1 -ppn 1 \
    ./launch.sh julia +1.12 --project=. \
    benchmarks/distributed_supercell_benchmark.jl --float-type Float32
