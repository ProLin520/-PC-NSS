import unittest

import numpy as np

from multisource_doa.data.simulator import steering_vector
from multisource_doa.physics.root_music import estimate_root_music


class RootMusicTest(unittest.TestCase):
    def test_positive_phase_theoretical_covariance_recovers_two_angles(self):
        true_angles = np.array([-8.0, 7.0])
        steering = steering_vector(true_angles, sensor_count=8)
        covariance = (
            steering @ np.diag([1.0, 0.8]) @ steering.conj().T
            + 1e-3 * np.eye(8)
        )

        result = estimate_root_music(
            covariance,
            source_count=2,
            spacing_wavelengths=0.5,
            angle_limits_deg=(-60.0, 60.0),
        )

        self.assertTrue(result.success, result.failure_reason)
        np.testing.assert_allclose(result.angles_deg, true_angles, atol=0.05)
        self.assertEqual(result.selected_roots.shape, (2,))
        self.assertGreater(result.minimum_root_separation, 0.0)

    def test_nonfinite_covariance_returns_explicit_failure(self):
        covariance = np.eye(8, dtype=np.complex128)
        covariance[0, 0] = np.nan

        result = estimate_root_music(covariance, source_count=2)

        self.assertFalse(result.success)
        self.assertEqual(result.failure_reason, "nonfinite_covariance")
        self.assertEqual(result.angles_deg.size, 0)

    def test_invalid_source_count_returns_explicit_failure(self):
        result = estimate_root_music(np.eye(3), source_count=3)

        self.assertFalse(result.success)
        self.assertEqual(result.failure_reason, "invalid_source_count")

    def test_rankless_identity_does_not_fabricate_two_angles(self):
        result = estimate_root_music(np.eye(8), source_count=2)

        self.assertFalse(result.success)
        self.assertIn(
            result.failure_reason,
            {"insufficient_distinct_roots", "rankless_covariance"},
        )


if __name__ == "__main__":
    unittest.main()
