#!/usr/bin/env bash

# Run a previously selected boundary weight on all five folds.
# Usage: snellius/submit_swin_unetr_boundary_cv.sh <BOUNDARY_WEIGHT> [RUN_TAG] [FOLDS]

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <BOUNDARY_WEIGHT> [RUN_TAG] [FOLDS]" >&2
    exit 2
fi

PROJECT="${PROJECT:-$HOME/projects/ai4mi_project}"
if [[ ! -d "$PROJECT/.git" && -d "$HOME/ai4mi_project/.git" ]]; then
    PROJECT="$HOME/ai4mi_project"
fi
BOUNDARY_WEIGHT="$1"
RUN_TAG="${2:-boundary_cv}"
FOLDS="${3:-0 1 2 3 4}"

cd "$PROJECT"
mkdir -p logs
job_file="logs/swin_boundary_${RUN_TAG}.jobs"
: > "$job_file"

for fold in $FOLDS; do
    job=$(sbatch --parsable \
        --job-name="swin_bnd_f${fold}" \
        --export=ALL,FOLD="$fold",BOUNDARY_WEIGHT="$BOUNDARY_WEIGHT",RUN_TAG="$RUN_TAG" \
        snellius/train_swin_unetr_boundary.sbatch)
    echo "weight=$BOUNDARY_WEIGHT fold=$fold job=$job" | tee -a "$job_file"
done

echo "Saved job IDs to: $job_file"
echo "Monitor with: squeue -u $USER"
