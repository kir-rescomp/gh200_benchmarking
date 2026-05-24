#!/bin/bash -e

#SBATCH --job-name          gh200_hbm
#SBATCH --account           gpu_kir.prj
#SBATCH --partition         gpu_gh200_144gb
#SBATCH --mem               72G
#SBATCH --nodes             1
#SBATCH --gres              gpu:1
#SBATCH --ntasks-per-node   1
#SBATCH --cpus-per-task     16
#SBATCH --time              01:00:00
#SBATCH --output            slog/%j.out

module purge
export PATH=/well/kir/scratch/sansom/mat611/Github/gh200_benchmarking/uv:$PATH
source /well/kir/scratch/sansom/mat611/Github/gh200_benchmarking/uv/.venv/bin/activate

python hbm_gemm_sweep.py

