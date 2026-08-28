#!/bin/bash

#SBATCH --job-name      cellbender-TT
#SBATCH --account       gpu_kir.prj
#SBATCH --partition     gpu_kir
#SBATCH --ntasks        1
#SBATCH --cpus-per-task 8
#SBATCH --mem           64G
#SBATCH --gres          gpu:1
#SBATCH --time          01:00:00
#SBATCH --array         1-216%1
#SBATCH --output        slog/%x-%A_%a.out

set -euo pipefail

# activate custom built environment for aarch64
source /gpfs3/well/kir/projects/mirror/gh200_environments/cellbender-py312/activate_cellb_env.sh

# Define the manifest ENV variable 
MANIFEST=${MANIFEST:-samples_TT.txt}

[[ -f "$MANIFEST" ]] || { echo "Manifest not found: $MANIFEST" >&2; exit 1; }

DIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
[[ -n "$DIR" ]] || { echo "No manifest entry at line ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; }

SAMPLE=$(basename "$DIR")
INPUT="$DIR/raw_feature_bc_matrix.h5"
OUTPUT="$DIR/rbg_output.h5"

if [[ -f "$OUTPUT" ]]; then
    echo "SKIP ${SAMPLE} - output already present"
    exit 0
fi

mkdir -p "$DIR/slog" # We have to create a secondary slog directory per-sample to hold the std.out
cd "$DIR"            # keeps CellBender's ckpt.tar.gz with the sample


cellbender remove-background \
    --cuda \
    --input  "$INPUT" \
    --output "$OUTPUT" \
    --checkpoint-mins 15 \
  > "$DIR/slog/${SAMPLE}.out" 2>&1

echo "DONE ${SAMPLE} $(date -Is)"
