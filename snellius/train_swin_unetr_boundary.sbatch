#!/usr/bin/env bash
#SBATCH --job-name=swin_boundary
#SBATCH --account=gpuuva084
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Trains one SwinUNETR fold with the original DiceCE objective plus a small
# Hausdorff distance-transform auxiliary term.  All other settings are shared
# with train_swin_unetr.sbatch.

set -euo pipefail

: "${FOLD:?Submit with FOLD=<fold index 0-4>}"
BOUNDARY_WEIGHT="${BOUNDARY_WEIGHT:-0.01}"
RUN_TAG="${RUN_TAG:-boundary}"
EPOCHS="${EPOCHS:-500}"
VAL_INTERVAL="${VAL_INTERVAL:-10}"
PATIENCE="${PATIENCE:-10}"

PROJECT="${PROJECT:-$HOME/projects/ai4mi_project}"
if [[ ! -d "$PROJECT/.git" && -d "$HOME/ai4mi_project/.git" ]]; then
    PROJECT="$HOME/ai4mi_project"
fi
BUNDLE="$PROJECT/models/segmentation_template_monai"
DATA_ZIP="${DATA_ZIP:-$PROJECT/data/segthor_train_full.zip}"
FOLDS_FILE="${FOLDS_FILE:-$PROJECT/data/SEGTHOR_FULL_PREPROCESSED_split_cv/folds.json}"
PRETRAINED="$BUNDLE/models/btcv_swin_unetr.pt"
DATA_DIR="${TMPDIR:?TMPDIR is not set}/segthor_train_full"
WEIGHT_TAG="${BOUNDARY_WEIGHT//./p}"
RESULTS_DIR="$PROJECT/results/segthor_swin_unetr_boundary/$RUN_TAG/lambda_$WEIGHT_TAG/fold_$FOLD"

cd "$PROJECT"

module purge
module load 2025
module load Anaconda3/2025.06-1
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$BUNDLE"

echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Fold: $FOLD"
echo "Boundary weight: $BOUNDARY_WEIGHT"
echo "Epochs: $EPOCHS, validation interval: $VAL_INTERVAL, patience: $PATIENCE"
echo "Results directory: $RESULTS_DIR"
echo "Git commit: $(git rev-parse HEAD)"
python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c 'import torch, monai, ignite, einops, fire; from monai.losses import HausdorffDTLoss; assert torch.cuda.is_available(), "CUDA is unavailable"; print("PyTorch CUDA:", torch.cuda.get_device_name(0), "| MONAI", monai.__version__)'
python -c 'import sys; w=float(sys.argv[1]); assert w >= 0, "boundary weight must be non-negative"' "$BOUNDARY_WEIGHT"

if [[ ! -f "$PRETRAINED" ]]; then
    echo "Pretrained weights missing: $PRETRAINED" >&2
    echo "Fetch them with: python -m monai.bundle download_large_files --bundle_path $BUNDLE" >&2
    exit 1
fi
if [[ ! -f "$DATA_ZIP" ]]; then
    echo "Dataset archive missing: $DATA_ZIP" >&2
    exit 1
fi
if [[ ! -f "$FOLDS_FILE" ]]; then
    echo "Cross-validation split missing: $FOLDS_FILE" >&2
    exit 1
fi
if [[ -e "$RESULTS_DIR" ]]; then
    echo "Directory already exists; refusing to overwrite: $RESULTS_DIR" >&2
    exit 1
fi
mkdir -p "$RESULTS_DIR" "$DATA_DIR"

unzip -q "$DATA_ZIP" -d "$DATA_DIR"
volume_count=$(find "$DATA_DIR/train" -name 'GT.nii.gz' | wc -l)
echo "Volumes: $volume_count in $DATA_DIR/train"
if [[ "$volume_count" -ne 40 ]]; then
    echo "Expected 40 labelled volumes in $DATA_ZIP" >&2
    exit 1
fi

bundle_args=(
    --meta_file "$BUNDLE/configs/metadata.json"
    --logging_file "$BUNDLE/configs/logging.conf"
    --bundle_root "$BUNDLE"
    --data_root "$DATA_DIR/train"
    --folds_file "$FOLDS_FILE"
    --fold "$FOLD"
)

python -u -m monai.bundle run \
    --config_file "['$BUNDLE/configs/common.yaml','$BUNDLE/configs/train.yaml','$BUNDLE/configs/train_boundary.yaml']" \
    "${bundle_args[@]}" \
    --output_dir "$RESULTS_DIR" \
    --pretrained_path "$PRETRAINED" \
    --boundary_weight "$BOUNDARY_WEIGHT" \
    --num_epochs "$EPOCHS" \
    --val_interval "$VAL_INTERVAL" \
    --early_stop_patience "$PATIENCE" \
    --num_workers $((SLURM_CPUS_PER_TASK - 2))

python -u -m monai.bundle run \
    --config_file "['$BUNDLE/configs/common.yaml','$BUNDLE/configs/test.yaml']" \
    "${bundle_args[@]}" \
    --ckpt_path "$RESULTS_DIR/model.pt" \
    --output_dir "$RESULTS_DIR/eval" \
    --save_pred True \
    | tee "$RESULTS_DIR/eval_summary.txt"

# Evaluate the saved, inverse-transformed predictions in each patient's
# original physical space. This produces the values needed to decide whether
# the boundary term actually improves HD95 rather than merely training loss.
python -u -m evaluation.evaluate_nifti_predictions \
    --prediction-dir "$RESULTS_DIR/eval" \
    --data-root "$DATA_DIR/train" \
    --folds-file "$FOLDS_FILE" \
    --fold "$FOLD" \
    --output-dir "$RESULTS_DIR/eval"

cat > "$RESULTS_DIR/experiment.txt" <<EOF
objective=DiceCE + $BOUNDARY_WEIGHT * log(1 + HausdorffDT)
boundary_weight=$BOUNDARY_WEIGHT
boundary_alpha=2.0
fold=$FOLD
git_commit=$(git rev-parse HEAD)
EOF

echo "Finished: $(date)"
