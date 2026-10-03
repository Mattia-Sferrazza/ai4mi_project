#!/usr/bin/env python3
"""Measure cohort variation relevant to SegTHOR augmentation choices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from preprocessing.slice_segthor import centre_crop, crop_shape_for_spacing


ORGAN_NAMES = {1: "esophagus", 2: "heart", 3: "trachea", 4: "aorta"}
OUTPUT_SHAPE = (384, 384)
TARGET_SPACING = (1.0, 1.0)
HU_WINDOW_WIDTH = 2000.0


def percentile_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def largest_component(mask: np.ndarray) -> np.ndarray:
    labelled, components = ndimage.label(mask)
    if components == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0
    return labelled == int(np.argmax(sizes))


def body_slice_statistics(image: np.ndarray, spacing: tuple[float, float]) -> dict[str, float] | None:
    # The thoracic wall is connected at -500 HU; fill lung cavities before
    # selecting the largest component so centroid and angle represent the body.
    body = largest_component(ndimage.binary_fill_holes(image > -500))
    coordinates = np.argwhere(body)
    if len(coordinates) < 1000:
        return None

    centre = coordinates.mean(axis=0)
    image_centre = (np.asarray(image.shape, dtype=float) - 1.0) / 2.0
    centred = coordinates - centre
    covariance = np.cov(centred, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    angle_from_column_axis = abs(float(np.degrees(np.arctan2(major_axis[0], major_axis[1]))))
    # A principal axis has no direction and either image axis represents an
    # untilted acquisition. Fold the angle into a 0-45 degree deviation.
    angle_from_column_axis %= 90.0
    angle_from_column_axis = min(angle_from_column_axis, 90.0 - angle_from_column_axis)

    extent = coordinates.max(axis=0) - coordinates.min(axis=0) + 1

    local_median = ndimage.median_filter(image.astype(np.float32), size=3)
    local_range = (
        ndimage.maximum_filter(image, size=3).astype(np.float32)
        - ndimage.minimum_filter(image, size=3).astype(np.float32)
    )
    homogeneous_soft_tissue = body & (image >= -100) & (image <= 100) & (local_range <= 80)
    residual = image.astype(np.float32) - local_median
    noise_sigma = (
        1.4826 * float(np.median(np.abs(residual[homogeneous_soft_tissue])))
        if np.count_nonzero(homogeneous_soft_tissue) >= 100
        else float("nan")
    )

    return {
        "offset_x_mm": float((centre[0] - image_centre[0]) * spacing[0]),
        "offset_y_mm": float((centre[1] - image_centre[1]) * spacing[1]),
        "roll_degrees": angle_from_column_axis,
        "body_x_mm": float(extent[0] * spacing[0]),
        "body_y_mm": float(extent[1] * spacing[1]),
        "noise_sigma_hu": noise_sigma,
    }


def analyse_patient(patient_dir: Path) -> dict[str, object]:
    patient = patient_dir.name
    ct_image = nib.load(str(patient_dir / f"{patient}.nii.gz"))
    gt_image = nib.load(str(patient_dir / "GT.nii.gz"))
    ct = np.asanyarray(ct_image.dataobj)
    gt = np.asanyarray(gt_image.dataobj).astype(np.uint8)
    spacing = tuple(float(value) for value in ct_image.header.get_zooms()[:3])

    crop_shape = crop_shape_for_spacing(ct.shape, spacing, OUTPUT_SHAPE, TARGET_SPACING)
    cropped_ct, _ = centre_crop(ct, crop_shape)
    cropped_gt, _ = centre_crop(gt, crop_shape)
    effective_spacing = (
        spacing[0] * crop_shape[0] / OUTPUT_SHAPE[0],
        spacing[1] * crop_shape[1] / OUTPUT_SHAPE[1],
    )

    foreground_coordinates = np.argwhere(cropped_gt > 0)
    foreground_centre = foreground_coordinates[:, :2].mean(axis=0)
    foreground_min = foreground_coordinates[:, :2].min(axis=0)
    foreground_max = foreground_coordinates[:, :2].max(axis=0)
    crop_centre = (np.asarray(crop_shape, dtype=float) - 1.0) / 2.0
    foreground_extent = (
        foreground_coordinates[:, :2].max(axis=0)
        - foreground_coordinates[:, :2].min(axis=0)
        + 1
    )

    foreground_slices = np.flatnonzero(np.any(cropped_gt > 0, axis=(0, 1)))
    selected_positions = np.linspace(0, len(foreground_slices) - 1, min(12, len(foreground_slices)))
    selected_slices = np.unique(foreground_slices[np.rint(selected_positions).astype(int)])
    body_rows = [
        row
        for z_index in selected_slices
        if (row := body_slice_statistics(cropped_ct[:, :, z_index], spacing[:2])) is not None
    ]

    row: dict[str, object] = {
        "patient": patient,
        "original_spacing_xy_mm": list(spacing[:2]),
        "effective_spacing_xy_mm": list(effective_spacing),
        "foreground_offset_x_mm": float((foreground_centre[0] - crop_centre[0]) * spacing[0]),
        "foreground_offset_y_mm": float((foreground_centre[1] - crop_centre[1]) * spacing[1]),
        "foreground_extent_x_mm": float(foreground_extent[0] * spacing[0]),
        "foreground_extent_y_mm": float(foreground_extent[1] * spacing[1]),
        "foreground_margin_x_low_mm": float(foreground_min[0] * spacing[0]),
        "foreground_margin_x_high_mm": float((crop_shape[0] - 1 - foreground_max[0]) * spacing[0]),
        "foreground_margin_y_low_mm": float(foreground_min[1] * spacing[1]),
        "foreground_margin_y_high_mm": float((crop_shape[1] - 1 - foreground_max[1]) * spacing[1]),
        "minimum_foreground_margin_mm": float(
            min(
                foreground_min[0] * spacing[0],
                (crop_shape[0] - 1 - foreground_max[0]) * spacing[0],
                foreground_min[1] * spacing[1],
                (crop_shape[1] - 1 - foreground_max[1]) * spacing[1],
            )
        ),
        "body_offset_x_mm": float(np.median([item["offset_x_mm"] for item in body_rows])),
        "body_offset_y_mm": float(np.median([item["offset_y_mm"] for item in body_rows])),
        "body_roll_degrees": float(np.median([item["roll_degrees"] for item in body_rows])),
        "body_x_mm": float(np.median([item["body_x_mm"] for item in body_rows])),
        "body_y_mm": float(np.median([item["body_y_mm"] for item in body_rows])),
        "noise_sigma_hu": float(np.nanmedian([item["noise_sigma_hu"] for item in body_rows])),
        "organs": {},
    }
    organ_rows: dict[str, dict[str, float]] = {}
    for label, organ in ORGAN_NAMES.items():
        mask = cropped_gt == label
        coordinates = np.argwhere(mask)
        values = cropped_ct[mask].astype(np.float32)
        centre = coordinates[:, :2].mean(axis=0)
        extent = coordinates[:, :2].max(axis=0) - coordinates[:, :2].min(axis=0) + 1
        organ_rows[organ] = {
            "median_hu": float(np.median(values)),
            "iqr_hu": float(np.percentile(values, 75) - np.percentile(values, 25)),
            "offset_x_mm": float((centre[0] - crop_centre[0]) * spacing[0]),
            "offset_y_mm": float((centre[1] - crop_centre[1]) * spacing[1]),
            "extent_x_mm": float(extent[0] * spacing[0]),
            "extent_y_mm": float(extent[1] * spacing[1]),
        }
    row["organs"] = organ_rows
    return row


def analyse(source: Path) -> dict[str, object]:
    patients = sorted(path for path in (source / "train").glob("Patient_*") if path.is_dir())
    if len(patients) != 40:
        raise RuntimeError(f"Expected 40 patients, found {len(patients)}")

    rows = []
    for index, patient_dir in enumerate(patients, start=1):
        print(f"[{index:02d}/40] {patient_dir.name}", flush=True)
        rows.append(analyse_patient(patient_dir))

    summary: dict[str, object] = {
        "patients": 40,
        "body_offset_x_mm": percentile_summary([float(row["body_offset_x_mm"]) for row in rows]),
        "body_offset_y_mm": percentile_summary([float(row["body_offset_y_mm"]) for row in rows]),
        "absolute_body_offset_x_mm": percentile_summary([abs(float(row["body_offset_x_mm"])) for row in rows]),
        "absolute_body_offset_y_mm": percentile_summary([abs(float(row["body_offset_y_mm"])) for row in rows]),
        "body_roll_degrees": percentile_summary([float(row["body_roll_degrees"]) for row in rows]),
        "body_x_mm": percentile_summary([float(row["body_x_mm"]) for row in rows]),
        "body_y_mm": percentile_summary([float(row["body_y_mm"]) for row in rows]),
        "noise_sigma_hu": percentile_summary([float(row["noise_sigma_hu"]) for row in rows]),
        "foreground_offset_x_mm": percentile_summary([float(row["foreground_offset_x_mm"]) for row in rows]),
        "foreground_offset_y_mm": percentile_summary([float(row["foreground_offset_y_mm"]) for row in rows]),
        "minimum_foreground_margin_mm": percentile_summary(
            [float(row["minimum_foreground_margin_mm"]) for row in rows]
        ),
        "organs": {},
        "current_augmentation_physical_equivalents": {
            "translation_5_percent_mm": 0.05 * OUTPUT_SHAPE[0],
            "brightness_0_05_hu": 0.05 * HU_WINDOW_WIDTH,
            "noise_sigma_0_02_hu": 0.02 * HU_WINDOW_WIDTH,
        },
    }
    organ_summary: dict[str, object] = {}
    for organ in ORGAN_NAMES.values():
        organ_summary[organ] = {
            metric: percentile_summary(
                [float(row["organs"][organ][metric]) for row in rows]  # type: ignore[index]
            )
            for metric in (
                "median_hu",
                "iqr_hu",
                "offset_x_mm",
                "offset_y_mm",
                "extent_x_mm",
                "extent_y_mm",
            )
        }
    summary["organs"] = organ_summary
    return {"summary": summary, "patients": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/segthor_train_full"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eda/preprocessing_figures/augmentation_variation.json"),
    )
    args = parser.parse_args()
    result = analyse(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved detailed results to: {args.output}")


if __name__ == "__main__":
    main()
