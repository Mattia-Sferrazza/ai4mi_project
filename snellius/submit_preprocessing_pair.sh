#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -lt 5 || "$#" -gt 7 ]]; then
    echo "Usage: $0 DATASET_A AUG_A DATASET_B AUG_B RUN_TAG [SEEDS] [EPOCHS]" >&2
    echo "Example: $0 SEGTHOR_FULL none SEGTHOR_FULL_CLAHE none clahe_screen 0 25" >&2
    exit 2
fi

DATASET_A="$1"
AUG_A="$2"
DATASET_B="$3"
AUG_B="$4"
RUN_TAG="$5"
SEEDS="${6:-0}"
EPOCHS="${7:-25}"
PROJECT="${PROJECT:-$HOME/projects/ai4mi_project}"

cd "$PROJECT"
mkdir -p logs

job_file="logs/preprocessing_${RUN_TAG}.jobs"
if [[ -e "$job_file" ]]; then
    echo "Job record already exists; choose a new RUN_TAG: $job_file" >&2
    exit 1
fi

IFS=',' read -r -a seed_values <<< "$SEEDS"
for seed in "${seed_values[@]}"; do
    if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
        echo "Invalid seed: $seed" >&2
        exit 2
    fi

    job_a=$(sbatch --parsable \
        --job-name="prep_a_${seed}" \
        --export=ALL,DATASET="$DATASET_A",AUGMENTATION="$AUG_A",SEED="$seed",EPOCHS="$EPOCHS",RUN_TAG="$RUN_TAG" \
        snellius/train_preprocessing_experiment.sbatch)
    job_b=$(sbatch --parsable \
        --job-name="prep_b_${seed}" \
        --export=ALL,DATASET="$DATASET_B",AUGMENTATION="$AUG_B",SEED="$seed",EPOCHS="$EPOCHS",RUN_TAG="$RUN_TAG" \
        snellius/train_preprocessing_experiment.sbatch)

    printf 'seed=%s job_a=%s dataset_a=%s augmentation_a=%s job_b=%s dataset_b=%s augmentation_b=%s\n' \
        "$seed" "$job_a" "$DATASET_A" "$AUG_A" "$job_b" "$DATASET_B" "$AUG_B" \
        | tee -a "$job_file"
done

job_ids=$(awk '{for (i=1; i<=NF; i++) if ($i ~ /^job_[ab]=/) {split($i,a,"="); printf "%s%s", sep, a[2]; sep=","}}' "$job_file")
echo "Saved job IDs to: $job_file"
echo "Monitor with: squeue -j $job_ids"
echo "Results root: /scratch-shared/$USER/ai4mi_project/results/preprocessing/$RUN_TAG"
