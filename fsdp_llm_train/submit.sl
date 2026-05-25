#!/bin/bash

#SBATCH --job-name          gh200_test1_fsdp
#SBATCH --account           gpu_kir.prj
#SBATCH --partition         gpu_gh200_144gb
#SBATCH --nodes             1
#SBATCH --ntasks-per-node   2
#SBATCH --gpus-per-task     1
#SBATCH --cpus-per-task     36
#SBATCH --mem               400G
#SBATCH --time              03:00:00
#SBATCH --output            slog/%j.out

module purge
module load CUDA/12.6.0

source /well/kir/scratch/sansom/mat611/Github/gh200_benchmarking/uv/.venv/bin/activate

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29600
export NCCL_DEBUG=INFO

# Collect GPU stats in background throughout the job
nvidia-smi \
    --query-gpu=timestamp,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv,nounits \
    -l 5 \
    -f gpu-stats-${SLURM_JOB_ID}.csv &

echo "Job started  : $(date)"
echo "Node         : $(hostname)"
echo "GPUs on node : $SLURM_GPUS_ON_NODE"

torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    fsdp_llm_train.py

echo "Job finished : $(date)"
