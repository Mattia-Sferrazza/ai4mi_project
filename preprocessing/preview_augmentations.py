#!/usr/bin/env python3
"""Save a visual QC grid for the proposed online SegTHOR augmentations."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as torch_functional
from PIL import Image

from data_loading.augmentations import SegmentationAugmentation, build_augmentation
from data_loading.dataset import make_dataset


def load_pair(image_path: Path, label_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    image = torch.from_numpy(np.asarray(Image.open(image_path)).copy()).float()[None] / 255.0
    encoded = torch.from_numpy(np.asarray(Image.open(label_path)).copy()).long()
    class_mask = torch.round(encoded.float() / 63.0).long()
    target = torch_functional.one_hot(class_mask, num_classes=5).permute(2, 0, 1).float()
    return image, target


def select_foreground_pair(data_dir: Path) -> tuple[Path, Path]:
    best_pair: tuple[Path, Path] | None = None
    best_score = (-1, -1)
    for image_path, label_path in make_dataset(data_dir, "train"):
        if label_path is None:
            continue
        label = np.asarray(Image.open(label_path))
        foreground_values = set(np.unique(label).astype(int)) - {0}
        score = (len(foreground_values), int(np.count_nonzero(label)))
        if score > best_score:
            best_score = score
            best_pair = (image_path, label_path)
        if len(foreground_values) == 4:
            break
    if best_pair is None or best_score[0] == 0:
        raise RuntimeError("No labelled foreground training slice was found")
    return best_pair


def forced_augmentation(mode: str) -> SegmentationAugmentation | None:
    augmentation = build_augmentation(mode)
    if augmentation is None:
        return None
    return replace(
        augmentation,
        geometric_probability=1.0,
        intensity_probability=1.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/SEGTHOR_FULL"))
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eda/preprocessing_figures/augmentation_preview.png"),
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")

    image_path, label_path = select_foreground_pair(args.data_dir)
    image, target = load_pair(image_path, label_path)
    modes = ("none", "geometric", "intensity", "combined")
    figure, axes = plt.subplots(
        len(modes), args.samples, figsize=(4 * args.samples, 3.8 * len(modes)), squeeze=False
    )
    colors = matplotlib.colors.ListedColormap(
        ["black", "#4c78a8", "#f2cf5b", "#59a14f", "#e07b39"]
    )

    for row, mode in enumerate(modes):
        augmentation = forced_augmentation(mode)
        for column in range(args.samples):
            torch.manual_seed(args.seed + row * 100 + column)
            augmented_image, augmented_target = (
                (image.clone(), target.clone())
                if augmentation is None
                else augmentation(image.clone(), target.clone())
            )
            axis = axes[row, column]
            axis.imshow(augmented_image.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
            mask = augmented_target.argmax(dim=0).numpy()
            axis.imshow(np.ma.masked_where(mask == 0, mask), cmap=colors, vmin=0, vmax=4, alpha=0.45)
            axis.set_title(f"{mode} #{column + 1}")
            axis.axis("off")

    figure.suptitle(f"Online augmentation QC — {image_path.stem}", fontsize=15)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved augmentation preview to: {args.output}")


if __name__ == "__main__":
    main()
