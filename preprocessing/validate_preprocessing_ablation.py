#!/usr/bin/env python3
"""Verify that a preprocessing ablation changes images and nothing else."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


def filenames(directory: Path) -> set[str]:
    return {path.name for path in directory.glob("*.png")}


def validate_pair(baseline: Path, variant: Path) -> dict[str, object]:
    baseline_split = json.loads((baseline / "split.json").read_text(encoding="utf-8"))
    variant_split = json.loads((variant / "split.json").read_text(encoding="utf-8"))
    if baseline_split != variant_split:
        raise AssertionError("Baseline and variant patient splits differ")

    with (baseline / "spacing.pkl").open("rb") as file:
        baseline_spacing = pickle.load(file)
    with (variant / "spacing.pkl").open("rb") as file:
        variant_spacing = pickle.load(file)
    if baseline_spacing != variant_spacing:
        raise AssertionError("Baseline and variant voxel spacings differ")

    baseline_images = filenames(baseline / "img")
    variant_images = filenames(variant / "img")
    baseline_labels = filenames(baseline / "gt")
    variant_labels = filenames(variant / "gt")
    if baseline_images != variant_images or baseline_labels != variant_labels:
        raise AssertionError("Baseline and variant filenames differ")
    if baseline_images != baseline_labels:
        raise AssertionError("Images and labels are not paired")

    changed_images = sum(
        (baseline / "img" / name).read_bytes() != (variant / "img" / name).read_bytes()
        for name in sorted(baseline_images)
    )
    changed_labels = sum(
        (baseline / "gt" / name).read_bytes() != (variant / "gt" / name).read_bytes()
        for name in sorted(baseline_labels)
    )
    if changed_images == 0:
        raise AssertionError("The preprocessing variant did not change any image")
    if changed_labels != 0:
        raise AssertionError(f"The preprocessing variant changed {changed_labels} masks")

    baseline_metadata = json.loads(
        (baseline / "preprocessing.json").read_text(encoding="utf-8")
    )
    variant_metadata = json.loads(
        (variant / "preprocessing.json").read_text(encoding="utf-8")
    )
    baseline_clahe = baseline_metadata.pop("clahe")
    variant_clahe = variant_metadata.pop("clahe")
    if baseline_metadata != variant_metadata:
        raise AssertionError("Non-CLAHE preprocessing metadata differs")
    if baseline_clahe["enabled"] or not variant_clahe["enabled"]:
        raise AssertionError("Expected baseline CLAHE off and variant CLAHE on")

    return {
        "patients": sum(
            len(baseline_split[subset]) for subset in ("train", "val", "test")
        ),
        "slices": len(baseline_images),
        "changed_images": changed_images,
        "changed_labels": changed_labels,
        "same_split": True,
        "same_spacing": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("variant", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_pair(args.baseline, args.variant), indent=2))


if __name__ == "__main__":
    main()
