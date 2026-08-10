import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from multisource_doa.diagnostics.teacher_confidence import TeacherDiagnosticResult
from multisource_doa.diagnostics.teacher_reporting import (
    build_teacher_decision,
    build_teacher_stratified_summary,
    build_teacher_summary,
    write_teacher_diagnostic_report,
)


class TeacherReportingTest(unittest.TestCase):
    def _row(
        self,
        sample_seed: int,
        *,
        rho: float,
        snr_db: float,
        snapshot_count: int,
        cohort: str,
        entropy_current: float = 0.96,
        entropy_counterfactual: float = 0.88,
        pmax_current: float = 0.30,
        pmax_counterfactual: float = 0.40,
        agreement: bool = True,
        regret: float = 0.5,
    ) -> dict[str, object]:
        return {
            "sample_seed": sample_seed,
            "true_angle_1_deg": -1.5,
            "true_angle_2_deg": 1.5,
            "rho": rho,
            "snr_db": snr_db,
            "snapshot_count": snapshot_count,
            "separation_deg": 3.0,
            "threshold_cohort": cohort,
            **{f"teacher_score_L{size}": 0.5 - 0.1 * index for index, size in enumerate((4, 5, 6, 7))},
            **{f"teacher_p_current_L{size}": value for size, value in zip((4, 5, 6, 7), (0.4, 0.3, 0.2, 0.1), strict=True)},
            **{f"teacher_p_counterfactual_L{size}": value for size, value in zip((4, 5, 6, 7), (0.55, 0.25, 0.15, 0.05), strict=True)},
            **{f"student_p_L{size}": 0.25 for size in (4, 5, 6, 7)},
            "teacher_entropy_current": entropy_current,
            "teacher_entropy_counterfactual": entropy_counterfactual,
            "teacher_max_probability_current": pmax_current,
            "teacher_max_probability_counterfactual": pmax_counterfactual,
            "teacher_dominant_scale": 4,
            "teacher_dominant_scale_current": 4,
            "teacher_dominant_scale_counterfactual": 4,
            "student_entropy_normalized": 1.0,
            "student_max_probability": 0.25,
            "student_dominant_scale": 4,
            "teacher_score_margin": 0.1,
            "teacher_score_margin_over_tau": 1.0,
            "teacher_student_kl": 0.1,
            "teacher_student_js": 0.02,
            "oracle_best_scales": (4,),
            "teacher_oracle_agreement": agreement,
            "teacher_regret_deg": regret,
            **{f"fbss_L{size}_sample_rmspe_deg": 0.5 + index for index, size in enumerate((4, 5, 6, 7))},
        }

    def _rows(self) -> list[dict[str, object]]:
        return [
            self._row(1, rho=0.8, snr_db=-2.0, snapshot_count=8, cohort="resolved"),
            self._row(2, rho=0.9, snr_db=2.0, snapshot_count=20, cohort="near_miss_1_1p25"),
            self._row(3, rho=0.99, snr_db=7.0, snapshot_count=50, cohort="near_miss_1p25_1p5"),
            self._row(4, rho=1.0, snr_db=10.0, snapshot_count=20, cohort="far_miss_gt_2"),
        ]

    def test_summary_reports_fixed_distributions_agreement_and_dominance(self):
        summary = build_teacher_summary(self._rows())

        self.assertEqual(summary["sample_count"], 4)
        self.assertEqual(summary["metrics"]["teacher_entropy_current"]["count"], 4)
        self.assertAlmostEqual(
            summary["metrics"]["teacher_entropy_current"]["median"], 0.96
        )
        self.assertEqual(summary["teacher_oracle_agreement_count"], 4)
        self.assertAlmostEqual(summary["teacher_oracle_agreement_rate"], 1.0)
        self.assertEqual(summary["dominant_scale_counts"]["teacher_current"]["4"], 4)
        self.assertEqual(summary["dominant_scale_counts"]["student"]["7"], 0)

    def test_strata_include_all_fixed_bins_and_each_dimension_sums_to_total(self):
        strata = build_teacher_stratified_summary(self._rows())
        expected = {"rho": 4, "snr_db": 3, "snapshot_count": 3, "threshold_cohort": 7}

        self.assertEqual(len(strata), 17)
        for dimension, bin_count in expected.items():
            selected = [row for row in strata if row["dimension"] == dimension]
            self.assertEqual(len(selected), bin_count)
            self.assertEqual(sum(row["sample_count"] for row in selected), 4)
        empty = next(
            row
            for row in strata
            if row["dimension"] == "threshold_cohort"
            and row["bin"] == "estimation_failure"
        )
        self.assertEqual(empty["sample_count"], 0)
        self.assertIsNone(empty["teacher_entropy_current_median"])

    def test_decision_requires_every_frozen_gate(self):
        rows = self._rows()
        summary = build_teacher_summary(rows)
        strata = build_teacher_stratified_summary(rows)
        passing = build_teacher_decision(summary, strata)
        self.assertTrue(passing["allow_tau_preregistration"])
        self.assertFalse(passing["training_authorized"])

        mutations = (
            ("entropy", "teacher_entropy_current", "median", 0.89),
            ("entropy_drop", "teacher_entropy_counterfactual", "median", 0.92),
            ("pmax_rise", "teacher_max_probability_counterfactual", "median", 0.34),
            ("regret", "teacher_regret_deg", "median", 1.01),
        )
        for name, metric, statistic, value in mutations:
            changed = json.loads(json.dumps(summary))
            changed["metrics"][metric][statistic] = value
            with self.subTest(gate=name):
                self.assertFalse(
                    build_teacher_decision(changed, strata)["allow_tau_preregistration"]
                )
        changed = json.loads(json.dumps(summary))
        changed["teacher_oracle_agreement_rate"] = 0.39
        self.assertFalse(build_teacher_decision(changed, strata)["allow_tau_preregistration"])

    def test_decision_requires_two_supporting_dimensions_and_integrity(self):
        summary = build_teacher_summary(self._rows())
        strata = build_teacher_stratified_summary(self._rows())
        unsupported = [dict(row) for row in strata]
        for row in unsupported:
            if row["dimension"] in ("rho", "snr_db") and row["sample_count"]:
                row["teacher_entropy_drop_median"] = 0.01
        self.assertFalse(
            build_teacher_decision(summary, unsupported)["allow_tau_preregistration"]
        )
        changed = json.loads(json.dumps(summary))
        changed["engineering_integrity"] = False
        self.assertFalse(build_teacher_decision(changed, strata)["allow_tau_preregistration"])

    def test_writer_creates_exact_schema_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "teacher-report"
            result = TeacherDiagnosticResult(sample_rows=tuple(self._rows()))
            report = write_teacher_diagnostic_report(
                result,
                output,
                diagnostic_config={"stage": "diagnose_validation_teacher", "device": "cpu"},
                source_manifest={
                    "sample_count": 4,
                    "no_model_forward": True,
                    "training_performed": False,
                },
            )

            self.assertEqual(report, output)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "diagnostic_config.json",
                    "source_manifest.json",
                    "teacher_sample_diagnostics.csv",
                    "teacher_summary.json",
                    "teacher_stratified_summary.csv",
                    "decision.json",
                },
            )
            manifest = json.loads((output / "source_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["teacher_diagnostic_schema_version"], 1)
            with (output / "teacher_sample_diagnostics.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                written_rows = list(csv.DictReader(handle))
            self.assertEqual(len(written_rows), 4)
            self.assertEqual(json.loads(written_rows[0]["oracle_best_scales"]), [4])

            with self.assertRaises(FileExistsError):
                write_teacher_diagnostic_report(
                    result,
                    output,
                    diagnostic_config={"stage": "diagnose_validation_teacher"},
                    source_manifest={"sample_count": 4},
                )

    def test_nonfinite_value_is_rejected_before_output_creation(self):
        rows = self._rows()
        rows[0]["teacher_student_kl"] = math.nan
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "teacher-report"
            with self.assertRaisesRegex(ValueError, "finite"):
                write_teacher_diagnostic_report(
                    TeacherDiagnosticResult(sample_rows=tuple(rows)),
                    output,
                    diagnostic_config={"stage": "diagnose_validation_teacher"},
                    source_manifest={"sample_count": 4},
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
