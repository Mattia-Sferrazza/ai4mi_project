#!/usr/bin/env python3
"""Evaluate full-volume NIfTI predictions against SegTHOR ground truth."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from utils import surface_hd95


ORGANS = {1: "esophagus", 2: "heart", 3: "trachea", 4: "aorta"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds-file", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--patients", nargs="+")
    args = parser.parse_args()
    if args.patients is None and (args.folds_file is None or args.fold is None):
        parser.error("provide --patients or both --folds-file and --fold")
    return args


def validation_patients(args: argparse.Namespace) -> list[str]:
    if args.patients is not None:
        return sorted(args.patients)
    folds = json.loads(args.folds_file.read_text(encoding="utf-8"))["folds"]
    fold = folds[str(args.fold)] if isinstance(folds, dict) else folds[args.fold]
    return sorted(fold["val"])


def find_prediction(root: Path, patient: str) -> Path:
    candidates: list[Path] = []
    for pattern in (f"{patient}_seg.nii.gz", f"{patient}_seg.nii", f"{patient}.nii.gz"):
        candidates.extend(root.rglob(pattern))
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise FileNotFoundError(
            f"Expected one prediction for {patient} in {root}, found {unique}"
        )
    return unique[0]


def safe_ratio(numerator: int, denominator: int, empty_value: float = float("nan")) -> float:
    return float(numerator / denominator) if denominator else empty_value


def evaluate_patient(prediction_path: Path, target_path: Path, patient: str) -> list[dict]:
    prediction_image = nib.load(prediction_path)
    target_image = nib.load(target_path)
    prediction = np.rint(np.asarray(prediction_image.dataobj)).astype(np.uint8)
    target = np.rint(np.asarray(target_image.dataobj)).astype(np.uint8)

    if prediction.shape != target.shape:
        raise ValueError(
            f"{patient}: prediction shape {prediction.shape} != target shape {target.shape}"
        )
    if not np.allclose(prediction_image.affine, target_image.affine, atol=1e-3):
        raise ValueError(f"{patient}: prediction and target affines do not match")
    if not set(np.unique(prediction)).issubset(set(range(5))):
        raise ValueError(f"{patient}: prediction contains labels outside 0..4")
    if not set(np.unique(target)).issubset(set(range(5))):
        raise ValueError(f"{patient}: target contains labels outside 0..4")

    spacing = tuple(float(value) for value in target_image.header.get_zooms()[:3])
    rows = []
    for class_id, organ in ORGANS.items():
        pred_mask = prediction == class_id
        target_mask = target == class_id
        tp = int(np.logical_and(pred_mask, target_mask).sum())
        fp = int(np.logical_and(pred_mask, ~target_mask).sum())
        fn = int(np.logical_and(~pred_mask, target_mask).sum())
        rows.append(
            {
                "patient": patient,
                "class": class_id,
                "organ": organ,
                "dice": safe_ratio(2 * tp, 2 * tp + fp + fn),
                "hd95_mm": surface_hd95(pred_mask, target_mask, spacing),
                # If the organ exists in the target but the model predicts no
                # voxels, precision is treated as zero (a complete miss), not
                # silently omitted as NaN.
                "precision": safe_ratio(
                    tp, tp + fp, 0.0 if target_mask.any() else float("nan")
                ),
                "recall": safe_ratio(tp, tp + fn),
                "prediction_voxels": int(pred_mask.sum()),
                "ground_truth_voxels": int(target_mask.sum()),
            }
        )
    return rows


def format_value(metric: str, value: float) -> str:
    if not np.isfinite(value):
        return str(value)
    return f"{value:.2f}" if metric == "hd95_mm" else f"{value:.4f}"


def write_summary(rows: list[dict], path: Path) -> None:
    metrics = (
        ("dice", "3D Dice"),
        ("hd95_mm", "3D HD95 (mm)"),
        ("precision", "3D voxel precision"),
        ("recall", "3D voxel recall"),
    )
    lines = [
        "# Full-volume validation metrics",
        "",
        "Predictions and labels are compared in the original patient space. HD95 uses the NIfTI voxel spacing and is reported in millimetres.",
    ]
    for metric, title in metrics:
        organ_means = []
        lines += ["", f"## {title}", "", "| Organ | Mean | Median | Worst |", "|---|---:|---:|---:|"]
        for organ in ORGANS.values():
            organ_rows = [row for row in rows if row["organ"] == organ]
            values = np.asarray([row[metric] for row in organ_rows], dtype=float)
            mean = float(np.nanmean(values))
            median = float(np.nanmedian(values))
            worst_index = (
                int(np.nanargmax(values))
                if metric == "hd95_mm"
                else int(np.nanargmin(values))
            )
            worst = float(values[worst_index])
            patient = organ_rows[worst_index]["patient"]
            organ_means.append(mean)
            lines.append(
                f"| {organ.title()} | {format_value(metric, mean)} | "
                f"{format_value(metric, median)} | {format_value(metric, worst)} ({patient}) |"
            )
        lines.append(
            f"\nMean over the four organ means: **{format_value(metric, float(np.mean(organ_means)))}**"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    patients = validation_patients(args)
    rows: list[dict] = []
    for patient in patients:
        prediction = find_prediction(args.prediction_dir, patient)
        target = args.data_root / patient / "GT.nii.gz"
        if not target.is_file():
            raise FileNotFoundError(target)
        rows.extend(evaluate_patient(prediction, target, patient))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "nifti_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = args.output_dir / "nifti_metrics_summary.md"
    write_summary(rows, summary_path)
    print(summary_path.read_text(encoding="utf-8"))
    print(f"Saved metrics to: {csv_path}")


if __name__ == "__main__":
    main()
