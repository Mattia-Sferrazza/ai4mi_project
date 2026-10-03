import json
import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image

from data_loading.augmentations import SegmentationAugmentation, build_augmentation
from data_loading.dataset import make_dataset
from data_loading.reproducibility import seed_everything
from preprocessing.validate_preprocessing_ablation import validate_pair


class SplitAwareDatasetTests(unittest.TestCase):
    def test_manifest_filters_train_validation_and_labelled_test(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "img").mkdir()
            (root / "gt").mkdir()
            patients = ["Patient_01", "Patient_02", "Patient_03"]
            for patient in patients:
                array = np.zeros((4, 4), dtype=np.uint8)
                Image.fromarray(array).save(root / "img" / f"{patient}_0000.png")
                Image.fromarray(array).save(root / "gt" / f"{patient}_0000.png")
            (root / "split.json").write_text(
                json.dumps({
                    "train": ["Patient_01"],
                    "val": ["Patient_02"],
                    "test": ["Patient_03"],
                }),
                encoding="utf-8",
            )

            train = make_dataset(root, "train")
            validation = make_dataset(root, "val")
            test = make_dataset(root, "test")

            self.assertEqual(train[0][0].stem, "Patient_01_0000")
            self.assertEqual(validation[0][0].stem, "Patient_02_0000")
            self.assertEqual(test[0][0].stem, "Patient_03_0000")
            self.assertIsNotNone(test[0][1])

    def test_clahe_ablation_validator_requires_identical_split_and_masks(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "baseline"
            variant = root / "variant"
            split = {"train": ["Patient_01"], "val": [], "test": []}
            spacing = {"Patient_01": (1.0, 1.0, 2.0)}
            metadata = {"output_shape": [4, 4], "patients": [], "clahe": {"enabled": False}}
            for directory, value, clahe in ((baseline, 10, False), (variant, 20, True)):
                (directory / "img").mkdir(parents=True)
                (directory / "gt").mkdir()
                Image.fromarray(np.full((4, 4), value, dtype=np.uint8)).save(
                    directory / "img" / "Patient_01_0000.png"
                )
                Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(
                    directory / "gt" / "Patient_01_0000.png"
                )
                (directory / "split.json").write_text(json.dumps(split), encoding="utf-8")
                with (directory / "spacing.pkl").open("wb") as output:
                    pickle.dump(spacing, output)
                current_metadata = dict(metadata)
                current_metadata["clahe"] = {"enabled": clahe}
                (directory / "preprocessing.json").write_text(
                    json.dumps(current_metadata), encoding="utf-8"
                )

            result = validate_pair(baseline, variant)
            self.assertEqual(result["changed_images"], 1)
            self.assertEqual(result["changed_labels"], 0)


class SegmentationAugmentationTests(unittest.TestCase):
    def setUp(self):
        self.image = torch.zeros((1, 32, 32), dtype=torch.float32)
        self.image[:, 10:22, 12:20] = 0.75
        class_mask = torch.zeros((32, 32), dtype=torch.int64)
        class_mask[10:22, 12:20] = 2
        self.target = torch.nn.functional.one_hot(class_mask, num_classes=5).permute(2, 0, 1).float()

    def test_none_mode_disables_augmentation(self):
        self.assertIsNone(build_augmentation("none"))

    def test_default_strengths_match_cohort_based_limits(self):
        transform = SegmentationAugmentation()

        self.assertEqual(transform.max_rotation_degrees, 5.0)
        self.assertEqual(transform.max_translation_fraction, 0.03)
        self.assertEqual(transform.scale_range, (0.9, 1.1))
        self.assertEqual(transform.brightness_delta, 0.02)
        self.assertEqual(transform.contrast_range, (0.95, 1.05))
        self.assertEqual(transform.gamma_range, (0.95, 1.05))
        self.assertEqual(transform.max_noise_sigma, 0.01)

    def test_geometric_transform_preserves_valid_one_hot_mask(self):
        transform = SegmentationAugmentation(
            geometric=True,
            intensity=False,
            geometric_probability=1.0,
        )
        torch.manual_seed(3)
        image, target = transform(self.image, self.target)

        self.assertEqual(image.shape, self.image.shape)
        self.assertEqual(target.shape, self.target.shape)
        self.assertTrue(torch.all(target.sum(dim=0) == 1))
        self.assertTrue(set(torch.unique(target).tolist()).issubset({0.0, 1.0}))

    def test_intensity_transform_does_not_change_mask(self):
        transform = SegmentationAugmentation(
            geometric=False,
            intensity=True,
            intensity_probability=1.0,
        )
        torch.manual_seed(4)
        image, target = transform(self.image, self.target)

        self.assertTrue(torch.equal(target, self.target))
        self.assertFalse(torch.equal(image, self.image))
        self.assertGreaterEqual(float(image.min()), 0.0)
        self.assertLessEqual(float(image.max()), 1.0)

    def test_same_run_seed_reproduces_online_augmentation(self):
        transform = SegmentationAugmentation(
            geometric=True,
            intensity=True,
            geometric_probability=1.0,
            intensity_probability=1.0,
        )
        seed_everything(2)
        first_image, first_target = transform(self.image, self.target)
        seed_everything(2)
        second_image, second_target = transform(self.image, self.target)

        self.assertTrue(torch.equal(first_image, second_image))
        self.assertTrue(torch.equal(first_target, second_target))


if __name__ == "__main__":
    unittest.main()
