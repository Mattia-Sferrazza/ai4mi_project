import unittest

import torch

from losses import CrossEntropy, CrossEntropyDiceLoss, DiceLoss
from utils import PatientVolumeDice, class2one_hot, dice_coef, patient_id_from_stem


class LossTests(unittest.TestCase):
    def setUp(self):
        target_class = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.int64)
        self.target = class2one_hot(target_class, K=2)
        self.perfect = self.target.float()
        self.wrong = 1.0 - self.perfect

    def test_losses_prefer_perfect_prediction(self):
        for loss in (
            CrossEntropy(idk=[0, 1]),
            DiceLoss(idk=[0, 1]),
            CrossEntropyDiceLoss(idk=[0, 1]),
        ):
            with self.subTest(loss=loss.__class__.__name__):
                self.assertLess(loss(self.perfect, self.target).item(),
                                loss(self.wrong, self.target).item())

    def test_combined_loss_is_sum_of_components(self):
        ce = CrossEntropy(idk=[0, 1])
        dice = DiceLoss(idk=[0, 1])
        combined = CrossEntropyDiceLoss(idk=[0, 1])
        expected = ce(self.perfect, self.target) + dice(self.perfect, self.target)
        self.assertTrue(torch.allclose(combined(self.perfect, self.target), expected))


class PatientVolumeDiceTests(unittest.TestCase):
    def test_patient_id_parsing(self):
        self.assertEqual(patient_id_from_stem("Patient_19_0123"), "Patient_19")
        self.assertEqual(patient_id_from_stem("toy"), "toy")

    def test_volume_dice_does_not_reward_empty_slices(self):
        # Slice 0 contains class 1 in the target but the prediction misses it.
        target_0 = class2one_hot(torch.tensor([[[1, 1], [0, 0]]]), K=2)
        pred_0 = class2one_hot(torch.zeros((1, 2, 2), dtype=torch.int64), K=2)

        # Slice 1 is empty for class 1 in both target and prediction.  The old
        # slice-average metric awards a perfect class-1 Dice to this slice.
        target_1 = class2one_hot(torch.zeros((1, 2, 2), dtype=torch.int64), K=2)
        pred_1 = target_1.clone()

        slice_average = torch.cat((dice_coef(pred_0, target_0),
                                   dice_coef(pred_1, target_1)))[:, 1].mean()
        self.assertAlmostEqual(slice_average.item(), 0.5, places=6)

        metric = PatientVolumeDice(classes=2)
        metric.update(pred_0, target_0, ["Patient_01_0000"])
        metric.update(pred_1, target_1, ["Patient_01_0001"])
        patients, volume_dice = metric.compute()

        self.assertEqual(patients, ["Patient_01"])
        self.assertAlmostEqual(volume_dice[0, 1].item(), 0.0, places=6)

    def test_perfect_patient_volume_scores_one(self):
        labels = torch.tensor([
            [[0, 1], [1, 0]],
            [[1, 1], [0, 0]],
        ], dtype=torch.int64)
        one_hot = class2one_hot(labels, K=2)
        metric = PatientVolumeDice(classes=2)
        metric.update(one_hot, one_hot, ["Patient_01_0000", "Patient_01_0001"])
        _, volume_dice = metric.compute()
        self.assertTrue(torch.allclose(volume_dice, torch.ones_like(volume_dice)))

    def test_empty_patient_class_is_nan(self):
        labels = class2one_hot(torch.zeros((1, 2, 2), dtype=torch.int64), K=2)
        metric = PatientVolumeDice(classes=2)
        metric.update(labels, labels, ["Patient_01_0000"])
        _, volume_dice = metric.compute()
        self.assertTrue(torch.isnan(volume_dice[0, 1]))


if __name__ == "__main__":
    unittest.main()
