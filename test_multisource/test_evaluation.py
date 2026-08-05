import math
import unittest

import numpy as np

from multisource_doa.evaluation.matching import hungarian_match, is_resolved
from multisource_doa.evaluation.metrics import (
    aggregate_metrics,
    paired_comparison,
    score_doa_sample,
)


class FailureAwareEvaluationTest(unittest.TestCase):
    def test_hungarian_matching_is_permutation_invariant(self):
        match = hungarian_match([-10.0, 12.0], [12.0, -10.0])

        self.assertTrue(match.success)
        np.testing.assert_allclose(match.absolute_errors_deg, [0.0, 0.0])
        np.testing.assert_allclose(match.estimated_angles_deg, [-10.0, 12.0])

    def test_missing_angle_receives_sixty_degree_penalty(self):
        match = hungarian_match([-10.0, 10.0], [10.0])

        self.assertFalse(match.success)
        np.testing.assert_allclose(match.absolute_errors_deg, [60.0, 0.0])
        self.assertEqual(match.failure_reason, "missing_angle")

    def test_nonfinite_and_duplicate_estimates_are_failures(self):
        nonfinite = hungarian_match([-2.0, 2.0], [np.nan, np.inf])
        duplicate = hungarian_match([-2.0, 2.0], [0.0, 0.0])

        np.testing.assert_allclose(nonfinite.absolute_errors_deg, [60.0, 60.0])
        np.testing.assert_allclose(duplicate.absolute_errors_deg, [60.0, 60.0])
        self.assertEqual(nonfinite.failure_reason, "nonfinite_estimate")
        self.assertEqual(duplicate.failure_reason, "duplicate_estimate")

    def test_resolution_requires_angle_accuracy_and_half_true_separation(self):
        resolved = hungarian_match([-2.0, 2.0], [-1.5, 1.5])
        collapsed = hungarian_match([-2.0, 2.0], [-0.75, 0.75])

        self.assertTrue(is_resolved(resolved, [-2.0, 2.0]))
        self.assertFalse(is_resolved(collapsed, [-2.0, 2.0]))

    def test_aggregate_rmspe_keeps_failed_samples_in_denominator(self):
        scores = [
            score_doa_sample(0, [-5.0, 5.0], [-5.0, 5.0]),
            score_doa_sample(1, [-5.0, 5.0], [np.nan, np.inf]),
        ]

        metrics = aggregate_metrics(scores)

        self.assertAlmostEqual(metrics["failure_aware_rmspe_deg"], math.sqrt(1800.0))
        self.assertEqual(metrics["failure_count"], 1)
        self.assertAlmostEqual(metrics["resolution_rate"], 0.5)

    def test_paired_comparison_reports_overall_and_stratum_counts(self):
        reference = [
            score_doa_sample(
                1,
                [-2.0, 2.0],
                [-1.0, 3.0],
                strata={"separation_deg": 4.0},
            ),
            score_doa_sample(
                2,
                [-4.0, 4.0],
                [-4.0, 4.0],
                strata={"separation_deg": 8.0},
            ),
        ]
        candidate = [
            score_doa_sample(
                1,
                [-2.0, 2.0],
                [-1.5, 2.5],
                strata={"separation_deg": 4.0},
            ),
            score_doa_sample(
                2,
                [-4.0, 4.0],
                [-3.0, 5.0],
                strata={"separation_deg": 8.0},
            ),
        ]

        comparison = paired_comparison(reference, candidate)

        self.assertEqual(comparison["overall"], {"win": 1, "tie": 0, "loss": 1})
        self.assertEqual(
            comparison["by_separation_deg"]["4.0"],
            {"win": 1, "tie": 0, "loss": 0},
        )


if __name__ == "__main__":
    unittest.main()
