import inspect
import unittest

import numpy as np
import torch

from multisource_doa.config import ExperimentConfig
from multisource_doa.data.simulator import generate_two_source_sample
from multisource_doa.models.pc_nss import MultiScalePCNSS, masked_softmax
from multisource_doa.physics.lags import build_multiscale_views


def _model_inputs(batch_size=2):
    config = ExperimentConfig()
    rows = []
    for index in range(batch_size):
        sample = generate_two_source_sample(
            config,
            split_seed=880,
            index=index,
            rho=1.0,
            snr_db=5.0,
            snapshot_count=20,
        )
        rows.append(
            build_multiscale_views(
                sample.snapshots,
                subarray_sizes=config.physics.fbss_subarray_sizes,
                output_size=config.array.sensor_count,
                source_count=config.data.source_count,
            )
        )
    raw = torch.from_numpy(
        np.stack(
            [np.stack([item.raw_lags.real, item.raw_lags.imag], axis=-1) for item in rows]
        ).astype(np.float32)
    )
    fbss = torch.from_numpy(
        np.stack(
            [
                np.stack([item.fbss_lags.real, item.fbss_lags.imag], axis=-1)
                for item in rows
            ]
        ).astype(np.float32)
    )
    mask = torch.from_numpy(np.stack([item.valid_mask for item in rows]))
    counts = torch.from_numpy(
        np.stack([item.effective_counts for item in rows]).astype(np.float32)
    )
    quality = torch.from_numpy(
        np.stack([item.quality_features for item in rows]).astype(np.float32)
    )
    return raw, fbss, mask, counts, quality


class PCNSSModelTest(unittest.TestCase):
    def test_masked_softmax_zeroes_invalid_views_and_handles_all_masked_lag(self):
        logits = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        mask = torch.tensor([[[True, False], [True, False], [False, False]]])

        weights = masked_softmax(logits, mask, dim=1)

        torch.testing.assert_close(weights[:, :, 0].sum(dim=1), torch.ones(1))
        torch.testing.assert_close(weights[:, :, 1], torch.zeros(1, 3))
        self.assertEqual(weights[0, 2, 0].item(), 0.0)

    def test_forward_respects_shapes_bounds_masks_and_structure(self):
        model = MultiScalePCNSS()
        raw, fbss, mask, counts, quality = _model_inputs()

        output = model(raw, fbss, mask, counts, quality)

        self.assertEqual(output.scale_weights.shape, (2, 4, 8))
        self.assertEqual(output.anchor_lags_ri.shape, (2, 8, 2))
        self.assertEqual(output.lag_residual_ri.shape, (2, 8, 2))
        self.assertEqual(output.covariance.shape, (2, 8, 8))
        self.assertTrue(torch.equal(output.scale_weights[~mask], torch.zeros_like(output.scale_weights[~mask])))
        weight_sums = output.scale_weights.sum(dim=1)
        torch.testing.assert_close(weight_sums[:, :7], torch.ones(2, 7), atol=1e-6, rtol=0.0)
        torch.testing.assert_close(weight_sums[:, 7], torch.zeros(2), atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            output.anchor_lags_ri[:, 7],
            output.normalized_raw_lags_ri[:, 7],
            atol=1e-6,
            rtol=0.0,
        )
        residual_magnitude = torch.linalg.vector_norm(output.lag_residual_ri, dim=-1)
        self.assertLessEqual(residual_magnitude.max().item(), 0.10 + 1e-6)
        self.assertGreaterEqual(output.diagonal_loading.min().item(), 0.0)
        self.assertLessEqual(output.diagonal_loading.max().item(), 0.05 + 1e-7)
        torch.testing.assert_close(output.covariance, output.covariance.mH, atol=1e-5, rtol=0.0)
        torch.testing.assert_close(
            output.covariance.diagonal(dim1=-2, dim2=-1).real.sum(-1),
            torch.full((2,), 8.0),
            atol=1e-5,
            rtol=0.0,
        )
        self.assertGreaterEqual(torch.linalg.eigvalsh(output.covariance).min().item(), 1e-6 - 1e-5)

    def test_parameter_budget_and_forward_signature_exclude_labels(self):
        model = MultiScalePCNSS()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        signature = inspect.signature(model.forward)

        self.assertGreaterEqual(parameter_count, 30_000)
        self.assertLessEqual(parameter_count, 80_000)
        forbidden = {"angles", "snr", "rho", "snapshot_count", "domain", "labels"}
        self.assertTrue(forbidden.isdisjoint(signature.parameters))

    def test_every_trainable_parameter_receives_finite_gradient(self):
        model = MultiScalePCNSS()
        raw, fbss, mask, counts, quality = _model_inputs()

        output = model(raw, fbss, mask, counts, quality)
        loss = output.covariance.abs().square().mean() + output.scale_weights.square().mean()
        loss.backward()

        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)


if __name__ == "__main__":
    unittest.main()
