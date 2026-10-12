import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np

from evaluation.evaluate_nifti_predictions import evaluate_patient


class NiftiPredictionEvaluationTests(unittest.TestCase):
    def test_perfect_prediction_and_physical_hd95(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            affine = np.diag([1.0, 1.0, 2.5, 1.0])
            target = np.zeros((8, 8, 8), dtype=np.uint8)
            for class_id in range(1, 5):
                target[class_id, 2:4, 2:4] = class_id
            prediction = target.copy()
            target_path = root / "GT.nii.gz"
            prediction_path = root / "Patient_01_seg.nii.gz"
            nib.save(nib.Nifti1Image(target, affine), target_path)
            nib.save(nib.Nifti1Image(prediction, affine), prediction_path)

            rows = evaluate_patient(prediction_path, target_path, "Patient_01")
            self.assertEqual(len(rows), 4)
            for row in rows:
                self.assertEqual(row["dice"], 1.0)
                self.assertEqual(row["hd95_mm"], 0.0)
                self.assertEqual(row["precision"], 1.0)
                self.assertEqual(row["recall"], 1.0)

    def test_mismatched_affine_is_rejected(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            labels = np.ones((3, 3, 3), dtype=np.uint8)
            target_path = root / "GT.nii.gz"
            prediction_path = root / "Patient_01_seg.nii.gz"
            nib.save(nib.Nifti1Image(labels, np.eye(4)), target_path)
            nib.save(nib.Nifti1Image(labels, np.diag([2.0, 1.0, 1.0, 1.0])), prediction_path)
            with self.assertRaises(ValueError):
                evaluate_patient(prediction_path, target_path, "Patient_01")


if __name__ == "__main__":
    unittest.main()
