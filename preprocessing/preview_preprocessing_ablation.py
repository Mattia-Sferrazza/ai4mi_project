#!/usr/bin/env python3
"""Create a side-by-side visual QC figure for baseline and CLAHE datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from data_loading.dataset import make_dataset, patient_from_slice
from preprocessing.validate_preprocessing_ablation import validate_pair


def choose_examples(data_dir: Path, count: int) -> list[tuple[Path, Path]]:
    best_by_patient: dict[str, tuple[tuple[int, int], Path, Path]] = {}
    for image_path, label_path in make_dataset(data_dir, "train"):
        if label_path is None:
            continue
        label = np.asarray(Image.open(label_path))
        foreground = set(np.unique(label).astype(int)) - {0}
        score = (len(foreground), int(np.count_nonzero(label)))
        patient = patient_from_slice(image_path)
        if patient not in best_by_patient or score > best_by_patient[patient][0]:
            best_by_patient[patient] = (score, image_path, label_path)
    ranked = sorted(best_by_patient.values(), key=lambda item: item[0], reverse=True)
    if len(ranked) < count:
        raise RuntimeError(f"Only {len(ranked)} patients are available for {count} examples")
    return [(image_path, label_path) for _, image_path, label_path in ranked[:count]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("data/SEGTHOR_FULL"))
    parser.add_argument("--variant", type=Path, default=Path("data/SEGTHOR_FULL_CLAHE"))
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eda/preprocessing_figures/clahe_preview.png"),
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")

    validate_pair(args.baseline, args.variant)
    examples = choose_examples(args.baseline, args.samples)
    figure, axes = plt.subplots(args.samples, 2, figsize=(8, 3.8 * args.samples), squeeze=False)
    colors = matplotlib.colors.ListedColormap(
        ["black", "#4c78a8", "#f2cf5b", "#59a14f", "#e07b39"]
    )
    for row, (baseline_image_path, label_path) in enumerate(examples):
        variant_image_path = args.variant / "img" / baseline_image_path.name
        label = np.rint(np.asarray(Image.open(label_path)).astype(float) / 63).astype(int)
        for column, (title, image_path) in enumerate(
            (("HU window only", baseline_image_path), ("HU window + CLAHE", variant_image_path))
        ):
            axis = axes[row, column]
            axis.imshow(Image.open(image_path), cmap="gray", vmin=0, vmax=255)
            axis.imshow(
                np.ma.masked_where(label == 0, label),
                cmap=colors,
                vmin=0,
                vmax=4,
                alpha=0.35,
            )
            axis.set_title(f"{title}\n{baseline_image_path.stem}")
            axis.axis("off")

    figure.suptitle("Deterministic preprocessing ablation QC", fontsize=15)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved CLAHE comparison to: {args.output}")


if __name__ == "__main__":
    main()
