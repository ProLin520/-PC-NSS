import json
import tempfile
import unittest
from pathlib import Path

from multisource_doa.diagnostics.teacher_ranking import (
    TeacherRankingResult,
    component_ranking_diagnostics,
)
from multisource_doa.diagnostics.teacher_ranking_reporting import (
    build_teacher_component_summary,
    build_teacher_oracle_confusion,
    build_teacher_ranking_decision,
    build_teacher_ranking_stratified_summary,
    build_teacher_ranking_summary,
    write_teacher_ranking_report,
)


def _sample_row(
    seed: int,
    *,
    rho: float = 1.0,
    snr_db: float = 5.0,
    snapshot_count: int = 20,
    cohort: str = "resolved",
    signal=(4.0, 3.0, 2.0, 1.0),
    rmspe=(1.0, 2.0, 3.0, 4.0),
):
    components = component_ranking_diagnostics(
        q_true_1=(0.0, 0.0, 0.0, 0.0),
        q_true_2=(0.0, 0.0, 0.0, 0.0),
        q_midpoint=signal,
        rmspe_deg=rmspe,
    )
    row = {
        "split": "validation",
        "sample_seed": seed,
        "rho": rho,
        "snr_db": snr_db,
        "snapshot_count": snapshot_count,
        "threshold_cohort": cohort,
        "q_midpoint_range": components["q_midpoint_range"],
        "q_midpoint_std": components["q_midpoint_std"],
        "negative_truth_mean_range": components["negative_truth_mean_range"],
        "negative_truth_mean_std": components["negative_truth_mean_std"],
        "current_score_range": components["current_score_range"],
        "current_score_std": components["current_score_std"],
        "cancellation_ratio": components["cancellation_ratio"],
        "cancellation_denominator_zero": components[
            "cancellation_denominator_zero"
        ],
    }
    for signal_name, metrics in components["signals"].items():
        for metric_name, value in metrics.items():
            row[f"{signal_name}_{metric_name}"] = value
    return row


class TeacherRankingReportingTest(unittest.TestCase):
    def test_summary_keeps_null_correlations_and_aggregates_pairwise_counts(self):
        rows = [
            _sample_row(1),
            _sample_row(2, signal=(1.0, 1.0, 1.0, 1.0)),
        ]

        summary = build_teacher_ranking_summary(rows)

        current = summary["signals"]["current_score"]
        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(current["spearman_rho"]["defined_count"], 1)
        self.assertEqual(current["spearman_rho"]["null_count"], 1)
        self.assertEqual(current["pairwise_comparable_count"], 12)
        self.assertAlmostEqual(current["pairwise_concordance_rate"], 0.5)
        self.assertEqual(current["top1_oracle_agreement_count"], 2)

    def test_strata_have_17_bins_per_signal_and_account_for_every_sample(self):
        rows = [
            _sample_row(1, rho=0.8, snr_db=-5.0, snapshot_count=8),
            _sample_row(
                2,
                rho=1.0,
                snr_db=10.0,
                snapshot_count=50,
                cohort="far_miss_gt_2",
            ),
        ]

        strata = build_teacher_ranking_stratified_summary(rows)

        self.assertEqual(len(strata), 51)
        for signal in ("current_score", "q_midpoint", "negative_truth_mean"):
            selected = [row for row in strata if row["signal"] == signal]
            self.assertEqual(len(selected), 17)
            for dimension in ("rho", "snr_db", "snapshot_count", "threshold_cohort"):
                self.assertEqual(
                    sum(
                        row["sample_count"]
                        for row in selected
                        if row["dimension"] == dimension
                    ),
                    2,
                )
        self.assertTrue(any(row["sample_count"] == 0 for row in strata))

    def test_confusion_fractionally_allocates_tied_oracle_and_sums_to_samples(self):
        rows = [
            _sample_row(1, rmspe=(1.0, 1.0 + 5e-7, 3.0, 4.0)),
            _sample_row(2, signal=(1.0, 2.0, 4.0, 3.0), rmspe=(4, 3, 1, 2)),
        ]

        confusion = build_teacher_oracle_confusion(rows)

        self.assertEqual(len(confusion), 16)
        self.assertAlmostEqual(sum(row["weighted_count"] for row in confusion), 2.0)
        self.assertTrue(all(row["oracle_tie_sample_count"] == 1 for row in confusion))
        cell = next(
            row
            for row in confusion
            if row["teacher_top1_scale"] == 4 and row["oracle_scale"] == 5
        )
        self.assertAlmostEqual(cell["weighted_count"], 0.5)

    def test_component_summary_reports_null_cancellation_count(self):
        row = _sample_row(1)
        row["cancellation_ratio"] = None
        row["cancellation_denominator_zero"] = True

        summary = build_teacher_component_summary([row])

        self.assertEqual(summary["cancellation_ratio"]["defined_count"], 0)
        self.assertEqual(summary["cancellation_ratio"]["null_count"], 1)
        self.assertEqual(summary["cancellation_denominator_zero_count"], 1)

    def test_frozen_decision_selects_calibration_component_or_invalid(self):
        def signal(pairwise, top1, top2, regret=0.5, kendall=0.3):
            return {
                "pairwise_concordance_rate": pairwise,
                "top1_oracle_agreement_rate": top1,
                "top2_oracle_coverage_rate": top2,
                "top1_regret_deg": {"median": regret},
                "kendall_tau_b": {"median": kendall},
            }

        supportive = [
            {
                "dimension": dimension,
                "bin": str(index),
                "signal": signal_name,
                "sample_count": 1,
                "pairwise_concordance_rate": pairwise,
                "top2_oracle_coverage_rate": top2,
            }
            for signal_name, pairwise, top2 in (
                ("current_score", 0.55, 0.65),
                ("q_midpoint", 0.60, 0.70),
                ("negative_truth_mean", 0.10, 0.10),
            )
            for dimension in ("rho", "snr_db", "snapshot_count")
            for index in range(2)
        ]
        summary = {
            "signals": {
                "current_score": signal(0.60, 0.40, 0.70),
                "q_midpoint": signal(0.65, 0.30, 0.70),
                "negative_truth_mean": signal(0.10, 0.10, 0.10),
            }
        }
        calibration = build_teacher_ranking_decision(
            summary,
            supportive,
            task15_margin_over_tau_median=0.099,
            engineering_integrity=True,
        )
        self.assertEqual(calibration["mechanism_conclusion"], "calibration_only")
        self.assertFalse(calibration["training_authorized"])

        summary["signals"]["current_score"] = signal(0.59, 0.39, 0.69)
        component = build_teacher_ranking_decision(
            summary,
            supportive,
            task15_margin_over_tau_median=0.099,
            engineering_integrity=True,
        )
        self.assertEqual(component["mechanism_conclusion"], "component_cancellation")
        self.assertEqual(component["candidate_components"], ["q_midpoint"])

        summary["signals"]["q_midpoint"] = signal(0.639, 0.30, 0.70)
        invalid = build_teacher_ranking_decision(
            summary,
            supportive,
            task15_margin_over_tau_median=0.099,
            engineering_integrity=True,
        )
        self.assertEqual(invalid["mechanism_conclusion"], "ranking_invalid")
        self.assertFalse(invalid["training_authorized"])

    def test_null_correlations_fail_gates_without_aborting_the_report(self):
        null_signal = {
            "pairwise_concordance_rate": None,
            "top1_oracle_agreement_rate": 0.0,
            "top2_oracle_coverage_rate": 0.0,
            "top1_regret_deg": {"median": 2.0},
            "kendall_tau_b": {"median": None},
        }
        summary = {"signals": {name: dict(null_signal) for name in (
            "current_score", "q_midpoint", "negative_truth_mean"
        )}}

        decision = build_teacher_ranking_decision(
            summary,
            [],
            task15_margin_over_tau_median=0.01,
            engineering_integrity=True,
        )

        self.assertEqual(decision["mechanism_conclusion"], "ranking_invalid")
        self.assertFalse(decision["calibration_gates"]["median_kendall_tau_b"])

    def test_writer_creates_exactly_eight_schema_v1_files_and_refuses_overwrite(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name) / "report"
        result = TeacherRankingResult(sample_rows=(_sample_row(1),))
        source_manifest = {
            "sample_count": 1,
            "device": "cpu",
            "batch_size": 128,
            "no_model_forward": True,
            "teacher_modified": False,
            "training_performed": False,
        }

        write_teacher_ranking_report(
            result,
            output,
            diagnostic_config={"stage": "smoke"},
            source_manifest=source_manifest,
            task15_margin_over_tau_median=0.01,
            engineering_integrity=True,
        )

        self.assertEqual(
            {path.name for path in output.iterdir()},
            {
                "diagnostic_config.json",
                "source_manifest.json",
                "teacher_ranking_sample_diagnostics.csv",
                "teacher_ranking_summary.json",
                "teacher_component_summary.json",
                "teacher_ranking_stratified_summary.csv",
                "teacher_oracle_confusion.csv",
                "decision.json",
            },
        )
        manifest = json.loads((output / "source_manifest.json").read_text())
        self.assertEqual(manifest["teacher_ranking_schema_version"], 1)
        with self.assertRaises(FileExistsError):
            write_teacher_ranking_report(
                result,
                output,
                diagnostic_config={"stage": "smoke"},
                source_manifest=source_manifest,
                task15_margin_over_tau_median=0.01,
                engineering_integrity=True,
            )


if __name__ == "__main__":
    unittest.main()
