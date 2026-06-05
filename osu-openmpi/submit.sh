#!/bin/bash -e

#SBATCH --job-name          osu_gh200
#SBATCH --partition         gpu_gh200_144gb
#SBATCH --account           gpu_kir.prj
#SBATCH --nodes             2
##SBATCH --ntasks-per-node   2
#SBATCH --cpus-per-task     72
#SBATCH --gpus-per-node     2
#SBATCH --time              00:20:00
#SBATCH --output            slog/%j.out

export PATH=/apps/kir/eb/hpc-utils/aarch64:$PATH
export PYTHONNOUSERSITE=1
module use /apps/eb/el9/2025a/aarch64/modules/all/

module purge
module load OpenMPI/5.0.7-GCC-14.2.0

export OMP_NUM_THREADS=1
export UCX_TLS=rc_mlx5,dc_mlx5,self,sm
export UCX_RNDV_SCHEME=get_zcopy
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1

BIN=/gpfs3/well/kir-scratch/sansom/mat611/GH200_benchmarking/osu-micro-benchmarks-7.5.2/install/libexec/osu-micro-benchmarks/mpi

srun --gpu-bind=closest $BIN/collective/osu_allreduce -d cuda
srun --gpu-bind=closest -n 2 -N 2 $BIN/pt2pt/osu_bw -d cuda -i 1000   # inter-node, fabric
srun --gpu-bind=closest -n 2 -N 1 $BIN/pt2pt/osu_bibw -d cuda -i 1000   # intra-node, NV18
