#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "Usage: $0 RUN_TAG" >&2
    exit 2
fi

RUN_TAG="$1"
RESULT_ROOT="/scratch-shared/$USER/ai4mi_project/results/preprocessing/$RUN_TAG"

if [[ ! -d "$RESULT_ROOT" ]]; then
    echo "Result directory does not exist: $RESULT_ROOT" >&2
    exit 1
fi

while IFS= read -r summary; do
    echo
    echo "===== $summary ====="
    cat "$summary"
    metrics="${summary%/summary.txt}/best_epoch_metrics.txt"
    if [[ -f "$metrics" ]]; then
        cat "$metrics"
    fi
done < <(find "$RESULT_ROOT" -type f -name summary.txt | sort)
