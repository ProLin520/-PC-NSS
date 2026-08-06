import math
import unittest

import numpy as np

import multisource_doa.evaluation.metrics as metrics_module
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
            comparison["by_separation_deg"]["[4,6)"],
            {"win": 1, "tie": 0, "loss": 0},
        )

    def test_paired_comparison_bins_continuous_strata_at_frozen_boundaries(self):
        separations = (2.0, 3.999, 4.0, 5.999, 6.0, 7.999, 8.0, 10.0)
        snr_values = (-5.0, -0.001, 0.0, 4.999, 5.0, 9.999, 10.0, 5.0)
        reference = []
        candidate = []
        for sample_id, (separation, snr_db) in enumerate(
            zip(separations, snr_values)
        ):
            strata = {
                "separation_deg": separation,
                "snr_db": snr_db,
                "snapshot_count": 20 if sample_id % 2 == 0 else 50,
                "rho": 1.0 if sample_id % 2 == 0 else 0.9,
            }
            score = score_doa_sample(
                sample_id,
                [-0.5 * separation, 0.5 * separation],
                [-0.5 * separation, 0.5 * separation],
                strata=strata,
            )
            reference.append(score)
            candidate.append(score)

        comparison = paired_comparison(reference, candidate)

        self.assertEqual(
            set(comparison["by_separation_deg"]),
            {"[2,4)", "[4,6)", "[6,8)", "[8,10]"},
        )
        self.assertEqual(
            [
                comparison["by_separation_deg"][label]["tie"]
                for label in ("[2,4)", "[4,6)", "[6,8)", "[8,10]")
            ],
            [2, 2, 2, 2],
        )
        self.assertEqual(
            set(comparison["by_snr_db"]),
            {"[-5,0)", "[0,5)", "[5,10]"},
        )
        self.assertEqual(set(comparison["by_snapshot_count"]), {"20", "50"})
        self.assertEqual(set(comparison["by_rho"]), {"0.9", "1.0"})

    def test_paired_comparison_rejects_out_of_range_bins_and_duplicate_ids(self):
        out_of_range = score_doa_sample(
            1,
            [-5.5, 5.5],
            [-5.5, 5.5],
            strata={"separation_deg": 11.0},
        )
        with self.assertRaisesRegex(ValueError, "separation_deg"):
            paired_comparison([out_of_range], [out_of_range])

        duplicate = score_doa_sample(1, [-1.5, 1.5], [-1.5, 1.5])
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            paired_comparison([duplicate, duplicate], [duplicate, duplicate])

    def test_near_separation_audit_decomposes_resolution_and_tail_errors(self):
        truth = [-1.5, 1.5]
        near_scores = [
            score_doa_sample(1, truth, [-1.5, 1.5], strata={"separation_deg": 3.0}),
            score_doa_sample(2, truth, [-0.7, 0.7], strata={"separation_deg": 3.0}),
            score_doa_sample(3, truth, [20.0, 24.0], strata={"separation_deg": 3.0}),
            score_doa_sample(4, truth, [40.0, 44.0], strata={"separation_deg": 3.0}),
            score_doa_sample(5, truth, [70.0, 74.0], strata={"separation_deg": 3.0}),
            score_doa_sample(
                6,
                [-2.5, 2.5],
                [70.0, 74.0],
                strata={"separation_deg": 5.0},
            ),
        ]

        audit = metrics_module.aggregate_near_separation_audit(near_scores)

        self.assertEqual(audit["separation_bin"], "[2,4)")
        self.assertEqual(audit["sample_count"], 5)
        self.assertEqual(
            audit["both_angle_errors_within_1_deg"],
            {"count": 2, "rate": 0.4},
        )
        self.assertEqual(
            audit["estimated_separation_at_least_half_true"],
            {"count": 4, "rate": 0.8},
        )
        self.assertEqual(audit["resolved"], {"count": 1, "rate": 0.2})
        self.assertEqual(audit["sample_rmspe_gt_10_deg"]["count"], 3)
        self.assertEqual(audit["sample_rmspe_gt_30_deg"]["count"], 2)
        self.assertEqual(audit["sample_rmspe_gt_60_deg"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
