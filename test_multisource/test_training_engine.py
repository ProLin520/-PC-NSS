import unittest

import torch

from multisource_doa.config import ExperimentConfig, SplitName
from multisource_doa.data.simulator import generate_two_source_sample
from multisource_doa.models.pc_nss import MultiScalePCNSS
from multisource_doa.training.engine import (
    _batch_scale_target,
    collate_samples,
    train_one_epoch,
    validate_model,
)
from multisource_doa.training.teacher import build_scale_teacher


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
    def test_cached_targets_follow_batch_seed_order_and_missing_seed_fails(self):
        batch = collate_samples(_samples())
        lookup = {
            seed: tuple(float(index == position) for index in range(4))
            for position, seed in enumerate(reversed(batch.sample_seeds))
        }
        target = _batch_scale_target(batch, lookup, torch.device("cpu"))
        self.assertEqual(target.shape, (4, 4))
        self.assertEqual(tuple(target[0].tolist()), lookup[batch.sample_seeds[0]])
        with self.assertRaises(KeyError):
            _batch_scale_target(batch, {}, torch.device("cpu"))

    def test_explicit_physical_targets_match_fallback_training_step(self):
        batch = collate_samples(_samples())
        teacher = build_scale_teacher(batch.fbss_covariances, batch.true_angles_deg)
        lookup = {
            seed: tuple(float(value) for value in teacher.scale_probabilities[index])
            for index, seed in enumerate(batch.sample_seeds)
        }
        torch.manual_seed(2026)
        fallback_model = MultiScalePCNSS()
        explicit_model = MultiScalePCNSS()
        explicit_model.load_state_dict(fallback_model.state_dict())
        fallback_optimizer = torch.optim.Adam(fallback_model.parameters(), lr=1e-3)
        explicit_optimizer = torch.optim.Adam(explicit_model.parameters(), lr=1e-3)

        fallback = train_one_epoch(
            fallback_model, [batch], fallback_optimizer, epoch=0,
            device=torch.device("cpu"), split=SplitName.TRAIN,
        )
        explicit = train_one_epoch(
            explicit_model, [batch], explicit_optimizer, epoch=0,
            device=torch.device("cpu"), split=SplitName.TRAIN,
            scale_targets_by_seed=lookup,
        )

        self.assertEqual(fallback.keys(), explicit.keys())
        for key in fallback:
            self.assertAlmostEqual(fallback[key], explicit[key], places=6)
        for left, right in zip(
            fallback_model.parameters(), explicit_model.parameters(), strict=True
        ):
            torch.testing.assert_close(left, right, atol=1e-7, rtol=1e-6)

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
