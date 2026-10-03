#!/usr/bin/env bash
#SBATCH --job-name=segthor_preprocess
#SBATCH --account=gpuuva084
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT="${PROJECT:-$HOME/projects/ai4mi_project}"
SCRATCH_PROJECT="${SCRATCH_PROJECT:-/scratch-shared/$USER/ai4mi_project}"
SOURCE_ZIP="$SCRATCH_PROJECT/source/segthor_train_full.zip"
RAW_DATA="$SCRATCH_PROJECT/data/segthor_train_full"
BASE_DATA="$SCRATCH_PROJECT/data/SEGTHOR_FULL"
CLAHE_DATA="$SCRATCH_PROJECT/data/SEGTHOR_FULL_CLAHE"
QC_DIR="$SCRATCH_PROJECT/results/preprocessing/qc"
BUILD_ROOT="$SCRATCH_PROJECT/data/.preprocess-build-${SLURM_JOB_ID}"

cd "$PROJECT"

module purge
module load 2025
module load Anaconda3/2025.06-1
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg

echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Project: $PROJECT"
echo "Scratch project: $SCRATCH_PROJECT"
echo "Git commit: $(git rev-parse HEAD)"
python --version

python -m unittest discover -s tests -v

patient_count() {
    local root="$1"
    if [[ ! -d "$root/train" ]]; then
        printf '0\n'
        return
    fi
    find "$root/train" -mindepth 1 -maxdepth 1 -type d -name 'Patient_*' | wc -l
}

if [[ "$(patient_count "$RAW_DATA")" -eq 40 ]]; then
    echo "Reusing extracted 40-patient source: $RAW_DATA"
else
    if [[ -e "$RAW_DATA" ]]; then
        echo "Source exists but is not a complete 40-patient dataset: $RAW_DATA" >&2
        exit 1
    fi
    if [[ ! -f "$SOURCE_ZIP" ]]; then
        echo "Missing source archive: $SOURCE_ZIP" >&2
        echo "Upload data/segthor_train_full.zip to that location first." >&2
        exit 1
    fi

    expected_hash=$(awk 'NR == 1 {print $1}' data/segthor_train_full.sha256)
    actual_hash=$(sha256sum "$SOURCE_ZIP" | awk '{print $1}')
    if [[ "$actual_hash" != "$expected_hash" ]]; then
        echo "Archive checksum mismatch." >&2
        echo "Expected: $expected_hash" >&2
        echo "Actual:   $actual_hash" >&2
        exit 1
    fi

    mkdir -p "$BUILD_ROOT/raw"
    unzip -q "$SOURCE_ZIP" -d "$BUILD_ROOT/raw"
    mapfile -t train_dirs < <(find "$BUILD_ROOT/raw" -maxdepth 3 -type d -name train)
    if [[ "${#train_dirs[@]}" -ne 1 ]]; then
        echo "Expected one extracted train directory; found ${#train_dirs[@]}" >&2
        exit 1
    fi
    extracted_root="${train_dirs[0]%/train}"
    if [[ "$(patient_count "$extracted_root")" -ne 40 ]]; then
        echo "Extracted archive does not contain exactly 40 patients." >&2
        exit 1
    fi
    mkdir -p "$(dirname "$RAW_DATA")"
    mv "$extracted_root" "$RAW_DATA"
    echo "Extracted source dataset to: $RAW_DATA"
fi

mkdir -p "$QC_DIR"

if [[ -d "$BASE_DATA" && -d "$CLAHE_DATA" ]]; then
    echo "Both processed datasets already exist; validating them without rebuilding."
elif [[ -e "$BASE_DATA" || -e "$CLAHE_DATA" ]]; then
    echo "Only one final processed dataset exists. Refusing an ambiguous partial rebuild." >&2
    echo "Base:  $BASE_DATA" >&2
    echo "CLAHE: $CLAHE_DATA" >&2
    exit 1
else
    mkdir -p "$BUILD_ROOT"
    base_build="$BUILD_ROOT/SEGTHOR_FULL"
    clahe_build="$BUILD_ROOT/SEGTHOR_FULL_CLAHE"

    python -u -m preprocessing.slice_segthor \
        --source_dir "$RAW_DATA" \
        --dest_dir "$base_build" \
        --shape 384 384 \
        --target-spacing 1.0 1.0 \
        --window -1000 1000 \
        --validation-count 6 \
        --test-count 6 \
        --test-pool-start 21 \
        --split-seed 0 \
        --process "$SLURM_CPUS_PER_TASK"

    python -u -m preprocessing.validate_processed_segthor "$base_build"

    python -u -m preprocessing.slice_segthor \
        --source_dir "$RAW_DATA" \
        --dest_dir "$clahe_build" \
        --shape 384 384 \
        --target-spacing 1.0 1.0 \
        --window -1000 1000 \
        --clahe \
        --clahe-kernel-size 32 \
        --clahe-clip-limit 0.01 \
        --validation-count 6 \
        --test-count 6 \
        --test-pool-start 21 \
        --split-seed 0 \
        --process "$SLURM_CPUS_PER_TASK"

    python -u -m preprocessing.validate_processed_segthor "$clahe_build"
    python -u -m preprocessing.validate_preprocessing_ablation \
        "$base_build" "$clahe_build"

    mv "$base_build" "$BASE_DATA"
    mv "$clahe_build" "$CLAHE_DATA"
    echo "Published validated datasets to their final scratch paths."
fi

python -u -m preprocessing.validate_processed_segthor "$BASE_DATA" \
    | tee "$QC_DIR/segthor_full_validation.json"
python -u -m preprocessing.validate_processed_segthor "$CLAHE_DATA" \
    | tee "$QC_DIR/segthor_full_clahe_validation.json"
python -u -m preprocessing.validate_preprocessing_ablation \
    "$BASE_DATA" "$CLAHE_DATA" \
    | tee "$QC_DIR/clahe_pair_validation.json"

python -u -m preprocessing.preview_augmentations \
    --data-dir "$BASE_DATA" \
    --seed 0 \
    --output "$QC_DIR/augmentation_preview.png"
python -u -m preprocessing.preview_preprocessing_ablation \
    --baseline "$BASE_DATA" \
    --variant "$CLAHE_DATA" \
    --output "$QC_DIR/clahe_preview.png"

ensure_link() {
    local link="$1"
    local target="$2"
    if [[ -L "$link" ]]; then
        if [[ "$(readlink -f "$link")" != "$(readlink -f "$target")" ]]; then
            echo "Existing link points to the wrong dataset: $link" >&2
            exit 1
        fi
    elif [[ -e "$link" ]]; then
        echo "Path exists and is not a symlink; refusing to replace it: $link" >&2
        exit 1
    else
        ln -s "$target" "$link"
    fi
}

mkdir -p "$PROJECT/data"
ensure_link "$PROJECT/data/SEGTHOR_FULL" "$BASE_DATA"
ensure_link "$PROJECT/data/SEGTHOR_FULL_CLAHE" "$CLAHE_DATA"

cp "$BASE_DATA/split.json" "$QC_DIR/split.json"

echo "Fixed split:"
cat "$BASE_DATA/split.json"
echo "QC outputs: $QC_DIR"
echo "Finished: $(date)"
