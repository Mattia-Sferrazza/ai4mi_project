#!/usr/bin/env bash

# Screen small boundary weights on fold 0 only.  Select the weight using the
# pre-defined rule in the documentation before running all five folds.
# Usage: snellius/submit_swin_unetr_boundary_screen.sh [RUN_TAG] [WEIGHTS]

set -euo pipefail

PROJECT="${PROJECT:-$HOME/projects/ai4mi_project}"
if [[ ! -d "$PROJECT/.git" && -d "$HOME/ai4mi_project/.git" ]]; then
    PROJECT="$HOME/ai4mi_project"
fi
RUN_TAG="${1:-boundary_screen}"
WEIGHTS="${2:-0.005 0.01 0.02}"

cd "$PROJECT"
mkdir -p logs
job_file="logs/swin_boundary_${RUN_TAG}.jobs"
: > "$job_file"

for weight in $WEIGHTS; do
    tag="${weight//./p}"
    job=$(sbatch --parsable \
        --job-name="swin_bnd_${tag}" \
        --export=ALL,FOLD=0,BOUNDARY_WEIGHT="$weight",RUN_TAG="$RUN_TAG" \
        snellius/train_swin_unetr_boundary.sbatch)
    echo "weight=$weight fold=0 job=$job" | tee -a "$job_file"
done

echo "Saved job IDs to: $job_file"
echo "Monitor with: squeue -u $USER"
