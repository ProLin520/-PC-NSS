import unittest

import torch

from multisource_doa.config import ExperimentConfig, SplitName
from multisource_doa.data.simulator import generate_two_source_sample
from multisource_doa.models.pc_nss import MultiScalePCNSS
from multisource_doa.training.engine import (
    collate_samples,
    train_one_epoch,
    validate_model,
)


def _samples(count=4):
    config = ExperimentConfig()
    return [
        generate_two_source_sample(
            config,
            split_seed=1200,
            index=index,
            rho=1.0,
            snr_db=5.0,
            snapshot_count=20,
        )
        for index in range(count)
    ]


class TrainingEngineTest(unittest.TestCase):
    def test_collate_builds_all_model_supervision_and_audit_tensors(self):
        batch = collate_samples(_samples())

        self.assertEqual(batch.raw_lags_ri.shape, (4, 8, 2))
        self.assertEqual(batch.fbss_lags_ri.shape, (4, 4, 8, 2))
        self.assertEqual(batch.valid_mask.shape, (4, 4, 8))
        self.assertEqual(batch.target_lags_ri.shape, (4, 8, 2))
        self.assertEqual(batch.true_angles_deg.shape, (4, 2))
        self.assertEqual(set(batch.fbss_covariances), {4, 5, 6, 7})
        self.assertEqual(batch.sample_seeds, (1200, 1201, 1202, 1203))

    def test_one_epoch_updates_model_and_returns_finite_diagnostics(self):
        torch.manual_seed(2026)
        model = MultiScalePCNSS()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = collate_samples(_samples())
        before = next(model.parameters()).detach().clone()

        metrics = train_one_epoch(
            model,
            [batch],
            optimizer,
            epoch=0,
            device=torch.device("cpu"),
            split=SplitName.TRAIN,
        )

        self.assertFalse(torch.equal(before, next(model.parameters()).detach()))
        for value in metrics.values():
            if isinstance(value, float):
                self.assertTrue(torch.isfinite(torch.tensor(value)))
        self.assertEqual(metrics["weighted_peak"], 0.0)
        self.assertIn("scale_weight_entropy", metrics)
        self.assertIn("projection_change_fro", metrics)
        self.assertIn("best_minus_predicted_resolution", metrics)
        self.assertIn("signal_subspace_angle_deg", metrics)

    def test_validation_scores_four_samples_without_checkpoint_side_effects(self):
        model = MultiScalePCNSS()
        batch = collate_samples(_samples())

        result = validate_model(
            model,
            [batch],
            device=torch.device("cpu"),
            split=SplitName.VALIDATION,
        )

        self.assertEqual(result.metrics["sample_count"], 4)
        self.assertEqual(len(result.scores), 4)

    def test_training_and_validation_reject_locked_test(self):
        model = MultiScalePCNSS()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = collate_samples(_samples())

        with self.assertRaises(PermissionError):
            train_one_epoch(
                model,
                [batch],
                optimizer,
                epoch=0,
                device=torch.device("cpu"),
                split=SplitName.LOCKED_TEST,
            )
        with self.assertRaises(PermissionError):
            validate_model(
                model,
                [batch],
                device=torch.device("cpu"),
                split=SplitName.LOCKED_TEST,
            )


if __name__ == "__main__":
    unittest.main()
