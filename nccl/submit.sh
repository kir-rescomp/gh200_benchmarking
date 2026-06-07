#!/bin/bash

#SBATCH --job-name        nccl_gh200
#SBATCH --partition       gpu_gh200_144gb
#SBATCH --account         gpu_kir.prj
#SBATCH --nodes           4
#SBATCH --ntasks-per-node 2
#SBATCH --cpus-per-task   1
#SBATCH --gpus-per-node   2
#SBATCH --time            10:00:00
#SBATCH --output          slog/%j.out

export PATH=/apps/kir/eb/hpc-utils/aarch64:$PATH
export PYTHONNOUSERSITE=1
module use /apps/eb/el9/2025a/aarch64/modules/all/

module purge
module load OpenMPI/5.0.7-GCC-14.2.0 NCCL/2.27.7-GCCcore-14.2.0-CUDA-12.8.0

export UCX_TLS=rc_mlx5,dc_mlx5,self,sm
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1

BIN=/gpfs3/well/kir-scratch/sansom/mat611/GH200_benchmarking/nccl-tests/build

# -b 1G -e 128G -f 2 sweeps from 1GB to 128GB doubling each step — 8 message sizes.
# Each gets 100 iterations with 10 warmup. 128GB is close to the per-GPU HBM limit (143GB)
# so it'll push the memory hard. If 128GB OOMs drop -e to 64G.

# Below is to squeeze more out of the IB-peak
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_NET_GDR_LEVEL=5


srun --gpu-bind=closest $BIN/all_reduce_perf \
  -b 1G \
  -e 128G \
  -f 2 \
  -n 1000 \
  -w 10 \
  -d float \
  -o sum
