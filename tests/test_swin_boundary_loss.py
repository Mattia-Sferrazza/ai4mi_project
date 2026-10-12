import sys
import unittest
from pathlib import Path

import torch


BUNDLE = Path(__file__).resolve().parents[1] / "models" / "segmentation_template_monai"
sys.path.insert(0, str(BUNDLE))

try:
    from monai.losses import DiceCELoss
    from scripts.boundary_loss import DiceCEHausdorffLoss
except ImportError:
    DiceCELoss = None
    DiceCEHausdorffLoss = None


@unittest.skipUnless(DiceCEHausdorffLoss is not None, "MONAI is not installed")
class DiceCEHausdorffLossTests(unittest.TestCase):
    @staticmethod
    def sample() -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(0)
        logits = torch.randn(1, 5, 8, 8, 8, requires_grad=True)
        target = torch.zeros(1, 1, 8, 8, 8, dtype=torch.long)
        target[:, :, 1:3, 1:3, 1:3] = 1
        target[:, :, 3:6, 3:6, 3:6] = 2
        target[:, :, 1:3, 5:7, 2:5] = 3
        target[:, :, 5:7, 1:3, 4:7] = 4
        return logits, target

    def test_zero_weight_is_original_dice_ce(self):
        logits, target = self.sample()
        original = DiceCELoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
            batch=True,
        )
        combined = DiceCEHausdorffLoss(boundary_weight=0.0)
        self.assertTrue(torch.equal(combined(logits, target), original(logits, target)))

    def test_total_is_region_plus_weighted_boundary(self):
        logits, target = self.sample()
        weight = 0.02
        loss = DiceCEHausdorffLoss(boundary_weight=weight)
        expected = loss.region_loss(logits, target) + weight * loss.boundary_term(logits, target)
        self.assertTrue(torch.allclose(loss(logits, target), expected))

    def test_loss_has_finite_gradient(self):
        logits, target = self.sample()
        value = DiceCEHausdorffLoss(boundary_weight=0.01)(logits, target)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_invalid_hyperparameters_are_rejected(self):
        with self.assertRaises(ValueError):
            DiceCEHausdorffLoss(boundary_weight=-0.01)
        with self.assertRaises(ValueError):
            DiceCEHausdorffLoss(boundary_alpha=0.0)


if __name__ == "__main__":
    unittest.main()
