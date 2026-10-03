#!/usr/bin/env bash

set -euo pipefail

DATASET="${1:-SEGTHOR_FULL}"
RUN_TAG="${2:-augmentation_screen_seed0}"
SEED="${3:-0}"
EPOCHS="${4:-25}"
PROJECT="${PROJECT:-$HOME/projects/ai4mi_project}"

case "$DATASET" in
    SEGTHOR_FULL|SEGTHOR_FULL_CLAHE) ;;
    *) echo "Unsupported dataset: $DATASET" >&2; exit 2 ;;
esac
if ! [[ "$SEED" =~ ^[0-9]+$ && "$EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
    echo "SEED must be non-negative and EPOCHS must be positive." >&2
    exit 2
fi

cd "$PROJECT"
mkdir -p logs
job_file="logs/augmentation_${RUN_TAG}.jobs"
if [[ -e "$job_file" ]]; then
    echo "Job record already exists; choose a new RUN_TAG: $job_file" >&2
    exit 1
fi

for augmentation in geometric intensity combined; do
    job_id=$(sbatch --parsable \
        --job-name="aug_${augmentation}_$SEED" \
        --export=ALL,DATASET="$DATASET",AUGMENTATION="$augmentation",SEED="$SEED",EPOCHS="$EPOCHS",RUN_TAG="$RUN_TAG" \
        snellius/train_preprocessing_experiment.sbatch)
    printf 'job=%s dataset=%s augmentation=%s seed=%s epochs=%s\n' \
        "$job_id" "$DATASET" "$augmentation" "$SEED" "$EPOCHS" \
        | tee -a "$job_file"
done

job_ids=$(awk '{for (i=1; i<=NF; i++) if ($i ~ /^job=/) {split($i,a,"="); printf "%s%s", sep, a[2]; sep=","}}' "$job_file")
echo "Saved job IDs to: $job_file"
echo "Monitor with: squeue -j $job_ids"
echo "Results root: /scratch-shared/$USER/ai4mi_project/results/preprocessing/$RUN_TAG"
