#!/usr/bin/env bash
#SBATCH --job-name=segthor_prep_exp
#SBATCH --account=gpuuva084
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=110G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

: "${DATASET:?Set DATASET to SEGTHOR_FULL or SEGTHOR_FULL_CLAHE}"
: "${AUGMENTATION:?Set AUGMENTATION to none or combined}"
: "${RUN_TAG:?Set a unique RUN_TAG}"

SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-25}"
PROJECT="${PROJECT:-$HOME/projects/ai4mi_project}"
SCRATCH_PROJECT="${SCRATCH_PROJECT:-/scratch-shared/$USER/ai4mi_project}"

case "$DATASET" in
    SEGTHOR_FULL|SEGTHOR_FULL_CLAHE) ;;
    *) echo "Unsupported DATASET=$DATASET" >&2; exit 2 ;;
esac
case "$AUGMENTATION" in
    none|combined) ;;
    *) echo "Unsupported AUGMENTATION=$AUGMENTATION" >&2; exit 2 ;;
esac
if ! [[ "$SEED" =~ ^[0-9]+$ && "$EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
    echo "SEED must be non-negative and EPOCHS must be positive." >&2
    exit 2
fi

DATASET_DIR="$SCRATCH_PROJECT/data/$DATASET"
DATA_LINK="$PROJECT/data/$DATASET"
RUN_DIR="$SCRATCH_PROJECT/results/preprocessing/$RUN_TAG/$DATASET/$AUGMENTATION/seed$SEED"

cd "$PROJECT"

module purge
module load 2025
module load Anaconda3/2025.06-1
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_SILENT=true

echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Git commit: $(git rev-parse HEAD)"
echo "Dataset: $DATASET"
echo "Augmentation: $AUGMENTATION"
echo "Training seed: $SEED"
echo "Epochs: $EPOCHS"
echo "Run directory: $RUN_DIR"
python --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("PyTorch CUDA:", torch.cuda.get_device_name(0))'

python -m unittest discover -s tests -v

if [[ ! -d "$DATASET_DIR/img" || ! -d "$DATASET_DIR/gt" ]]; then
    echo "Processed dataset is missing: $DATASET_DIR" >&2
    exit 1
fi
python -m preprocessing.validate_processed_segthor "$DATASET_DIR"

mkdir -p "$PROJECT/data"
if [[ -L "$DATA_LINK" ]]; then
    if [[ "$(readlink -f "$DATA_LINK")" != "$(readlink -f "$DATASET_DIR")" ]]; then
        echo "$DATA_LINK points to the wrong dataset." >&2
        exit 1
    fi
elif [[ -e "$DATA_LINK" ]]; then
    echo "$DATA_LINK exists and is not the expected scratch symlink." >&2
    exit 1
else
    ln -s "$DATASET_DIR" "$DATA_LINK"
fi

if [[ -e "$RUN_DIR" ]]; then
    echo "Run directory already exists; refusing to overwrite: $RUN_DIR" >&2
    exit 1
fi
mkdir -p "$RUN_DIR/wandb"
export WANDB_DIR="$RUN_DIR/wandb"

python -u main.py \
    --epochs "$EPOCHS" \
    --dataset "$DATASET" \
    --mode full \
    --loss ce_dice \
    --augmentation "$AUGMENTATION" \
    --seed "$SEED" \
    --dest "$RUN_DIR" \
    --gpu

python -m plotting.plot \
    --metric_file "$RUN_DIR/loss_tra.npy" \
    --dest "$RUN_DIR/loss_train.png" --headless
python -m plotting.plot \
    --metric_file "$RUN_DIR/loss_val.npy" \
    --dest "$RUN_DIR/loss_val.png" --headless
python -m plotting.plot \
    --metric_file "$RUN_DIR/dice3d_val.npy" \
    --dest "$RUN_DIR/dice3d_val.png" --headless

python -m evaluation.summarize_segthor_run --run-dir "$RUN_DIR" \
    | tee "$RUN_DIR/summary.txt"
python -m evaluation.evaluate_saved_predictions \
    --pred-dir "$RUN_DIR/best_epoch/val" \
    --gt-dir "$DATASET_DIR/gt" \
    --split-file "$DATASET_DIR/split.json" \
    --subset val \
    --spacing-file "$DATASET_DIR/spacing.pkl" \
    --csv "$RUN_DIR/best_epoch_metrics.csv" \
    | tee "$RUN_DIR/best_epoch_metrics.txt"

echo "Finished: $(date)"
echo "Results: $RUN_DIR"
