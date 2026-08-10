import unittest

import numpy as np
import torch

from multisource_doa.data.simulator import steering_vector
from multisource_doa.config import ExperimentConfig
from multisource_doa.data.simulator import generate_two_source_sample
from multisource_doa.models.pc_nss import MultiScalePCNSS
from multisource_doa.training.engine import collate_samples
from multisource_doa.training.losses import (
    aggregate_scale_weights,
    compose_total_loss,
    dominance_loss,
    normalized_lag_smooth_l1,
    peak_separation_loss,
    pcnss_loss,
    resolution_score,
    scale_distillation_loss,
)
from multisource_doa.training.teacher import build_scale_teacher


def _covariance(angles_deg):
    steering = steering_vector(np.asarray(angles_deg), sensor_count=8)
    matrix = steering @ steering.conj().T + 0.05 * np.eye(8)
    return torch.from_numpy(matrix.astype(np.complex64)).unsqueeze(0)


class ResolutionAwareLossTest(unittest.TestCase):
    def test_optional_scale_target_changes_only_scale_branch(self):
        sample = generate_two_source_sample(
            ExperimentConfig(), split_seed=902, index=0,
            rho=1.0, snr_db=5.0, snapshot_count=20,
            center_deg=0.0, separation_deg=3.0,
        )
        batch = collate_samples([sample])
        output = MultiScalePCNSS()(
            batch.raw_lags_ri, batch.fbss_lags_ri, batch.valid_mask,
            batch.effective_counts, batch.quality_features,
        )
        teacher = build_scale_teacher(batch.fbss_covariances, batch.true_angles_deg)
        fallback = pcnss_loss(
            output, teacher, batch.target_lags_ri, batch.true_angles_deg,
            batch.valid_mask, batch.effective_counts, epoch=10,
        )
        explicit = pcnss_loss(
            output, teacher, batch.target_lags_ri, batch.true_angles_deg,
            batch.valid_mask, batch.effective_counts, epoch=10,
            scale_distillation_target=teacher.scale_probabilities,
        )
        other_index = (teacher.scale_probabilities.argmax(dim=-1).item() + 1) % 4
        hard = torch.zeros_like(teacher.scale_probabilities)
        hard[:, other_index] = 1.0
        changed = pcnss_loss(
            output, teacher, batch.target_lags_ri, batch.true_angles_deg,
            batch.valid_mask, batch.effective_counts, epoch=10,
            scale_distillation_target=hard,
        )

        torch.testing.assert_close(fallback.scale, explicit.scale)
        for field in ("lag", "residual", "peak", "dominance"):
            torch.testing.assert_close(getattr(fallback, field), getattr(changed, field))
        self.assertNotAlmostEqual(fallback.scale.item(), changed.scale.item())

    def test_scale_target_rejects_invalid_probability_rows(self):
        sample = generate_two_source_sample(
            ExperimentConfig(), split_seed=903, index=0,
            rho=1.0, snr_db=5.0, snapshot_count=20,
        )
        batch = collate_samples([sample])
        output = MultiScalePCNSS()(
            batch.raw_lags_ri, batch.fbss_lags_ri, batch.valid_mask,
            batch.effective_counts, batch.quality_features,
        )
        teacher = build_scale_teacher(batch.fbss_covariances, batch.true_angles_deg)
        for target in (
            torch.ones(1, 3), torch.tensor([[1.0, -1.0, 1.0, 0.0]]),
            torch.tensor([[0.1, 0.1, 0.1, 0.1]]),
            torch.tensor([[float("nan"), 0.0, 0.0, 1.0]]),
        ):
            with self.subTest(target=target), self.assertRaises(ValueError):
                pcnss_loss(
                    output, teacher, batch.target_lags_ri, batch.true_angles_deg,
                    batch.valid_mask, batch.effective_counts, epoch=10,
                    scale_distillation_target=target,
                )

    def test_scale_kl_penalizes_wrong_collapsed_scale(self):
        teacher = torch.tensor([[0.05, 0.10, 0.80, 0.05]])
        aligned = torch.tensor([[0.05, 0.10, 0.80, 0.05]])
        collapsed = torch.tensor([[0.90, 0.04, 0.03, 0.03]])

        aligned_loss = scale_distillation_loss(teacher, aligned)
        collapsed_loss = scale_distillation_loss(teacher, collapsed)

        self.assertLess(aligned_loss.item(), collapsed_loss.item())

    def test_normalized_lag_loss_is_invariant_to_common_amplitude_scale(self):
        prediction = torch.tensor([[[2.0, 0.0], [0.5, -0.1], [0.2, 0.3]]])
        target = torch.tensor([[[2.0, 0.0], [0.4, -0.1], [0.1, 0.2]]])

        base = normalized_lag_smooth_l1(prediction, target)
        scaled = normalized_lag_smooth_l1(10.0 * prediction, 10.0 * target)

        torch.testing.assert_close(base, scaled)

    def test_two_source_covariance_has_lower_peak_loss_than_midpoint_collapse(self):
        true_angles = torch.tensor([[-5.0, 5.0]])
        correct = _covariance([-5.0, 5.0])
        collapsed = _covariance([0.0, 30.0])

        correct_loss = peak_separation_loss(correct, true_angles, margin=0.05)
        collapsed_loss = peak_separation_loss(collapsed, true_angles, margin=0.05)

        self.assertLess(correct_loss.item(), collapsed_loss.item())

    def test_dominance_decreases_when_prediction_beats_best_fixed_scale(self):
        best = torch.tensor([0.4])

        better = dominance_loss(best, torch.tensor([0.6]), tau=0.1)
        worse = dominance_loss(best, torch.tensor([0.1]), tau=0.1)

        self.assertLess(better.item(), worse.item())

    def test_epoch_ten_adds_peak_and_dominance_with_frozen_weights(self):
        values = {
            "lag": torch.tensor(1.0),
            "scale": torch.tensor(2.0),
            "residual": torch.tensor(3.0),
            "peak": torch.tensor(4.0),
            "dominance": torch.tensor(5.0),
        }

        stage_one = compose_total_loss(epoch=9, **values)
        stage_two = compose_total_loss(epoch=10, **values)

        self.assertAlmostEqual(
            stage_one.item(),
            1.0 + 0.5 * 2.0 + 0.01 * 3.0,
            places=6,
        )
        self.assertAlmostEqual(
            stage_two.item(),
            1.0 + 0.5 * 2.0 + 0.01 * 3.0 + 4.0 + 0.5 * 5.0,
            places=6,
        )

    def test_scale_and_resolution_losses_backpropagate(self):
        logits = torch.zeros(1, 4, 8, requires_grad=True)
        mask = torch.tensor(
            [[[lag < size for lag in range(8)] for size in (4, 5, 6, 7)]]
        )
        counts = mask.to(torch.float32)
        weights = torch.softmax(logits.masked_fill(~mask, -1e9), dim=1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        distribution = aggregate_scale_weights(weights, mask, counts)
        teacher = torch.tensor([[0.1, 0.2, 0.6, 0.1]])
        covariance = _covariance([-5.0, 5.0]).requires_grad_()
        true_angles = torch.tensor([[-5.0, 5.0]])
        score = resolution_score(covariance, true_angles)
        loss = (
            scale_distillation_loss(teacher, distribution)
            + peak_separation_loss(covariance, true_angles)
            + dominance_loss(torch.tensor([0.5]), score)
        )

        loss.backward()

        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.isfinite(covariance.grad).all())


if __name__ == "__main__":
    unittest.main()
