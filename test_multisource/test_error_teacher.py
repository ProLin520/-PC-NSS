import math
import unittest

import numpy as np

from multisource_doa.baselines.classical import DOAEstimate
from multisource_doa.config import ExperimentConfig
from multisource_doa.data.simulator import generate_two_source_sample
from multisource_doa.training.error_teacher import (
    build_error_teacher_row,
    teacher_probabilities_from_rmspe,
)


def _estimate(size, angles, success=True, reason=None):
    return DOAEstimate(
        algorithm=f"fbss_root_music_L{size}",
        angles_deg=np.asarray(angles, dtype=np.float64),
        success=success,
        failure_reason=reason,
        runtime_seconds=0.0,
        metadata={"subarray_size": size},
    )


class ErrorTeacherProbabilityTest(unittest.TestCase):
    def test_unique_best_is_one_hot_and_ties_share_mass(self):
        self.assertEqual(
            teacher_probabilities_from_rmspe({4: 2.0, 5: 1.0, 6: 3.0, 7: 4.0}),
            (0.0, 1.0, 0.0, 0.0),
        )
        self.assertEqual(
            teacher_probabilities_from_rmspe(
                {4: 1.0, 5: 1.0 + 5e-7, 6: 2.0, 7: 60.0}
            ),
            (0.5, 0.5, 0.0, 0.0),
        )

    def test_all_failed_is_uniform_and_invalid_values_are_rejected(self):
        self.assertEqual(
            teacher_probabilities_from_rmspe(
                {4: 60.0, 5: 60.0, 6: 60.0, 7: 60.0}
            ),
            (0.25, 0.25, 0.25, 0.25),
        )
        for values in (
            {4: 1.0, 5: math.nan, 6: 2.0, 7: 3.0},
            {4: 1.0, 5: -1.0, 6: 2.0, 7: 3.0},
            {4: 1.0, 5: 2.0, 6: 3.0},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                teacher_probabilities_from_rmspe(values)

    def test_row_uses_failure_aware_matching_and_keeps_failed_scale(self):
        sample = generate_two_source_sample(
            ExperimentConfig(),
            split_seed=901,
            index=0,
            rho=1.0,
            snr_db=5.0,
            snapshot_count=20,
            center_deg=0.0,
            separation_deg=4.0,
        )
        estimates = {
            4: _estimate(4, sample.angles_deg[::-1]),
            5: _estimate(5, [], False, "no_valid_roots"),
            6: _estimate(6, sample.angles_deg + 1.0),
            7: _estimate(7, sample.angles_deg + 2.0),
        }

        row = build_error_teacher_row(
            sample, sample_index=0, estimates_by_scale=estimates
        )

        self.assertAlmostEqual(row["sample_rmspe_deg_L4"], 0.0)
        self.assertEqual(row["sample_rmspe_deg_L5"], 60.0)
        self.assertEqual(row["teacher_probabilities"], (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(row["failure_reason_L5"], "no_valid_roots")
        self.assertIsNone(row["estimated_angle_1_deg_L5"])
        self.assertEqual(row["sample_index"], 0)


if __name__ == "__main__":
    unittest.main()
