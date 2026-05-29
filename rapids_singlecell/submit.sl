#!/bin/bash

#SBATCH --job-name          gh200_test1_fsdp
#SBATCH --account           gpu_kir.prj
#SBATCH --partition         gpu_gh200_144gb
#SBATCH --nodes             1
#SBATCH --ntasks-per-node   2
#SBATCH --gpus-per-task     1
#SBATCH --cpus-per-task     12
#SBATCH --mem               400G
#SBATCH --time              04:30:00
#SBATCH --output            slog/%j.out


export PATH=/apps/kir/eb/hpc-utils/aarch64:$PATH
module use /users/sansom/mat611/easybuild/kir-test/neoverse_v2/modules/all
module load Miniforge3/26.3.2-2-aarch64
source activate rapids-sc


# Collect GPU stats in background throughout the job
nvidia-smi \
    --query-gpu=timestamp,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv,nounits \
    -l 5 >> gpu-stats-${SLURM_JOB_ID}.csv &
NVIDIA_SMI_PID=$!

# Kill it cleanly on exit (normal or error)
trap "kill $NVIDIA_SMI_PID 2>/dev/null" EXIT

python rsc_benchmark.py
