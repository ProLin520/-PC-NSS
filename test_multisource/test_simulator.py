import unittest

import numpy as np

from multisource_doa.config import ExperimentConfig
from multisource_doa.data.simulator import (
    generate_two_source_sample,
    steering_vector,
)


class CoherentSourceSimulatorTest(unittest.TestCase):
    def setUp(self):
        self.config = ExperimentConfig()

    def test_positive_phase_steering_convention(self):
        steering = steering_vector(
            np.array([30.0]), sensor_count=3, spacing_wavelengths=0.5
        )

        np.testing.assert_allclose(steering[0], np.array([1.0 + 0.0j]))
        np.testing.assert_allclose(steering[1], np.array([1.0j]), atol=1e-12)
        np.testing.assert_allclose(steering[2], np.array([-1.0 + 0.0j]), atol=1e-12)

    def test_sample_is_deterministic_and_fully_coherent(self):
        sample_a = generate_two_source_sample(
            self.config,
            split_seed=1234,
            index=7,
            rho=1.0,
            snr_db=5.0,
            snapshot_count=20,
        )
        sample_b = generate_two_source_sample(
            self.config,
            split_seed=1234,
            index=7,
            rho=1.0,
            snr_db=5.0,
            snapshot_count=20,
        )

        np.testing.assert_array_equal(sample_a.snapshots, sample_b.snapshots)
        np.testing.assert_array_equal(sample_a.angles_deg, sample_b.angles_deg)
        self.assertAlmostEqual(abs(sample_a.source_correlation), 1.0, places=12)
        self.assertEqual(sample_a.sample_seed, 1241)
        self.assertTrue(np.all(np.diff(sample_a.angles_deg) > 0.0))
        self.assertEqual(sample_a.snapshots.shape, (8, 20))

    def test_target_covariance_removes_coherent_cross_terms(self):
        sample = generate_two_source_sample(
            self.config,
            split_seed=20,
            index=3,
            rho=1.0,
            snr_db=0.0,
            snapshot_count=50,
        )

        np.testing.assert_allclose(
            sample.target_covariance,
            sample.target_covariance.conj().T,
            atol=1e-12,
        )
        self.assertGreater(np.linalg.eigvalsh(sample.target_covariance).min(), 0.0)
        np.testing.assert_allclose(
            sample.target_covariance,
            sample.steering_matrix
            @ np.diag(sample.source_powers)
            @ sample.steering_matrix.conj().T
            + sample.noise_power * np.eye(8),
            atol=1e-12,
        )

    def test_rho_point_nine_has_expected_empirical_correlation(self):
        correlations = []
        for index in range(256):
            sample = generate_two_source_sample(
                self.config,
                split_seed=9000,
                index=index,
                rho=0.9,
                snr_db=5.0,
                snapshot_count=50,
            )
            correlations.append(abs(sample.source_correlation))

        self.assertGreater(np.mean(correlations), 0.80)
        self.assertLess(np.mean(correlations), 0.98)


if __name__ == "__main__":
    unittest.main()
