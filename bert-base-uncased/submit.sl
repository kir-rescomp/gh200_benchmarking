#!/bin/bash -e

#SBATCH --job-name          Torch_Distributed
#SBATCH --account           gpu_kir.prj
#SBATCH --partition         gpu_gh200_144gb
#SBATCH --mem               400G
#SBATCH --nodes             1
#SBATCH --gres              gpu:2
#SBATCH --ntasks-per-node   2
#SBATCH --cpus-per-task     36
#SBATCH --time              02:10:00
#SBATCH --output            slog/%j.out

module purge
export PATH=/well/kir/scratch/sansom/mat611/Github/gh200_benchmarking/uv:$PATH
source /well/kir/scratch/sansom/mat611/Github/gh200_benchmarking/uv/.venv/bin/activate

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29600

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "GPUs: $SLURM_GPUS_ON_NODE"

torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py

echo "Job finished at $(date)"
