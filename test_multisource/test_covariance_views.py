import unittest

import numpy as np

from multisource_doa.config import ExperimentConfig
from multisource_doa.data.simulator import generate_two_source_sample
from multisource_doa.physics.covariance import sample_covariance
from multisource_doa.physics.lags import (
    build_multiscale_views,
    covariance_to_lags,
)
from multisource_doa.physics.spatial_smoothing import fbss_covariance


class CovarianceViewTest(unittest.TestCase):
    def test_sample_covariance_matches_definition(self):
        snapshots = np.array(
            [
                [1.0 + 1.0j, 2.0 - 1.0j, -0.5 + 0.25j],
                [0.5 - 0.5j, -1.0 + 2.0j, 1.5 + 0.0j],
            ],
            dtype=np.complex128,
        )

        expected = snapshots @ snapshots.conj().T / snapshots.shape[1]
        np.testing.assert_allclose(sample_covariance(snapshots), expected)

    def test_fbss_matches_explicit_reference(self):
        rng = np.random.default_rng(41)
        snapshots = rng.standard_normal((8, 13)) + 1j * rng.standard_normal((8, 13))
        subarray_size = 5
        subarray_count = snapshots.shape[0] - subarray_size + 1
        forward = np.zeros((subarray_size, subarray_size), dtype=np.complex128)
        for start in range(subarray_count):
            subarray = snapshots[start : start + subarray_size]
            forward += subarray @ subarray.conj().T / snapshots.shape[1]
        forward /= subarray_count
        reversal = np.fliplr(np.eye(subarray_size, dtype=np.complex128))
        expected = 0.5 * (forward + reversal @ forward.conj() @ reversal)

        actual = fbss_covariance(snapshots, subarray_size)

        np.testing.assert_allclose(actual, expected, atol=1e-12)
        np.testing.assert_allclose(actual, actual.conj().T, atol=1e-12)

    def test_first_column_lag_convention_and_padding(self):
        covariance = np.array(
            [
                [10.0, 1.0 - 2.0j, 3.0 + 1.0j],
                [1.0 + 2.0j, 20.0, 5.0 - 4.0j],
                [3.0 - 1.0j, 5.0 + 4.0j, 30.0],
            ],
            dtype=np.complex128,
        )

        lags, mask, diagonal_counts = covariance_to_lags(covariance, output_size=5)

        np.testing.assert_allclose(
            lags[:3],
            np.array([20.0, 3.0 + 3.0j, 3.0 - 1.0j]),
        )
        np.testing.assert_array_equal(mask, np.array([True, True, True, False, False]))
        np.testing.assert_array_equal(diagonal_counts, np.array([3.0, 2.0, 1.0, 0.0, 0.0]))
        np.testing.assert_array_equal(lags[3:], np.zeros(2, dtype=np.complex128))

    def test_positive_phase_sample_builds_four_fbss_views(self):
        config = ExperimentConfig()
        sample = generate_two_source_sample(
            config,
            split_seed=300,
            index=2,
            rho=1.0,
            snr_db=5.0,
            snapshot_count=20,
            center_deg=0.0,
            separation_deg=4.0,
        )

        views = build_multiscale_views(
            sample.snapshots,
            subarray_sizes=config.physics.fbss_subarray_sizes,
            output_size=config.array.sensor_count,
            source_count=config.data.source_count,
        )

        self.assertEqual(views.raw_covariance.shape, (8, 8))
        self.assertEqual(views.raw_lags.shape, (8,))
        self.assertEqual(set(views.fbss_covariances), {4, 5, 6, 7})
        self.assertEqual(views.fbss_lags.shape, (4, 8))
        self.assertEqual(views.valid_mask.shape, (4, 8))
        self.assertEqual(views.effective_counts.shape, (4, 8))
        self.assertEqual(views.quality_features.shape, (4, 6))
        np.testing.assert_array_equal(views.valid_mask[0], np.arange(8) < 4)
        np.testing.assert_array_equal(views.valid_mask[-1], np.arange(8) < 7)
        self.assertTrue(np.isfinite(views.quality_features).all())
        self.assertFalse(8 in views.fbss_covariances)


if __name__ == "__main__":
    unittest.main()
