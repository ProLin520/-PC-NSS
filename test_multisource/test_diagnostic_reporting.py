import csv
import json
import tempfile
import unittest
from pathlib import Path

from multisource_doa.diagnostics.near_resolution import NearDiagnosticResult
from multisource_doa.diagnostics.reporting import (
    RHO_VALUES,
    SNAPSHOT_VALUES,
    THRESHOLD_COHORTS,
    build_mechanism_summary,
    build_stratified_summary,
    write_near_diagnostic_report,
)


class DiagnosticReportingTest(unittest.TestCase):
    @staticmethod
    def _row(
        *,
        rho: float = 0.8,
        snr_db: float = -2.5,
        snapshot_count: int = 8,
        threshold_cohort: str = "resolved",
        sample_seed: int = 1,
    ) -> dict:
        return {
            "sample_seed": sample_seed,
            "rho": rho,
            "snr_db": snr_db,
            "snapshot_count": snapshot_count,
            "threshold_cohort": threshold_cohort,
            "absolute_error_1_deg": 0.4,
            "absolute_error_2_deg": 0.6,
            "sample_rmspe_deg": 0.5,
            "success": True,
            "estimated_separation_at_least_half_true": True,
            "l7_absolute_error_1_deg": 0.5,
            "l7_absolute_error_2_deg": 0.7,
            "l7_sample_rmspe_deg": 0.6,
            "l7_success": True,
            "l7_estimated_separation_at_least_half_true": True,
            "p_L4": 0.4,
            "p_L5": 0.3,
            "p_L6": 0.2,
            "p_L7": 0.1,
            "scale_entropy_normalized": 0.75,
            "dominant_scale": 4,
            "residual_magnitude_p50": 0.01,
            "residual_magnitude_p95": 0.02,
            "residual_magnitude_max": 0.03,
            "saturated_lag_rate": 0.25,
            "train_projection_change": 0.04,
            "eval_projection_change": 0.05,
            "total_projection_change": 0.06,
            "dykstra_converged": False,
        }

    def _covered_rows(self) -> list[dict]:
        rows = []
        seed = 1
        for rho in RHO_VALUES:
            for snr_db in (-2.5, 2.5, 7.5):
                for snapshot_count in SNAPSHOT_VALUES:
                    for cohort in THRESHOLD_COHORTS:
                        rows.append(
                            self._row(
                                rho=rho,
                                snr_db=snr_db,
                                snapshot_count=snapshot_count,
                                threshold_cohort=cohort,
                                sample_seed=seed,
                            )
                        )
                        seed += 1
        return rows

    def test_fixed_strata_account_for_every_sample(self):
        rows = self._covered_rows()

        summary = build_stratified_summary(rows)

        for dimension in ("rho", "snr_db", "snapshot_count", "threshold_cohort"):
            self.assertEqual(
                sum(row["sample_count"] for row in summary if row["dimension"] == dimension),
                len(rows),
            )

    def test_fixed_strata_reject_out_of_protocol_values(self):
        for field, value in (("snr_db", 10.1), ("rho", 0.95), ("snapshot_count", 16)):
            row = self._row()
            row[field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    build_stratified_summary([row])

    def test_mechanism_summary_keeps_nonconverged_rows(self):
        rows = [
            self._row(threshold_cohort="resolved"),
            self._row(threshold_cohort="resolved", sample_seed=2),
        ]
        rows[1]["scale_entropy_normalized"] = 0.25
        rows[1]["dominant_scale"] = 7

        summary = build_mechanism_summary(rows)

        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["cohorts"]["resolved"]["sample_count"], 2)
        self.assertEqual(
            summary["cohorts"]["resolved"]["dominant_scale_counts"],
            {"4": 1, "5": 0, "6": 0, "7": 1},
        )
        self.assertEqual(
            summary["cohorts"]["resolved"]["metrics"]["scale_entropy_normalized"]["count"],
            2,
        )

    def test_report_refuses_overwrite_and_aggregates_the_sample_csv(self):
        rows = self._covered_rows()
        rows[0]["scale_entropy_normalized"] = 0.5
        rows[0]["saturated_lag_rate"] = 0.75
        result = NearDiagnosticResult(sample_rows=tuple(rows))

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "diagnostic_report"
            write_near_diagnostic_report(
                result,
                output,
                diagnostic_config={"batch_size": 4},
                source_manifest={"source": "frozen-validation"},
            )

            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "diagnostic_config.json",
                    "source_manifest.json",
                    "near_sample_diagnostics.csv",
                    "threshold_summary.json",
                    "stratified_summary.csv",
                    "mechanism_summary.json",
                },
            )
            with self.assertRaises(FileExistsError):
                write_near_diagnostic_report(
                    result,
                    output,
                    diagnostic_config={},
                    source_manifest={},
                )

            with (output / "near_sample_diagnostics.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                sample_rows = list(csv.DictReader(handle))
            threshold_summary = json.loads(
                (output / "threshold_summary.json").read_text(encoding="utf-8")
            )
            with (output / "stratified_summary.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                stratified_rows = list(csv.DictReader(handle))

        max_error_passes = sum(
            max(
                float(row["absolute_error_1_deg"]),
                float(row["absolute_error_2_deg"]),
            )
            <= 1.0
            for row in sample_rows
        )
        resolved_rows = [
            row for row in sample_rows if row["threshold_cohort"] == "resolved"
        ]
        entropy_mean = sum(
            float(row["scale_entropy_normalized"]) for row in resolved_rows
        ) / len(resolved_rows)
        saturation_rate = sum(
            float(row["saturated_lag_rate"]) for row in resolved_rows
        ) / len(resolved_rows)
        overall = next(
            row
            for row in stratified_rows
            if row["dimension"] == "threshold_cohort" and row["bin"] == "resolved"
        )

        self.assertEqual(
            threshold_summary["max_error_le_1p00_deg"]["count"], max_error_passes
        )
        self.assertAlmostEqual(float(overall["scale_entropy_normalized_mean"]), entropy_mean)
        self.assertAlmostEqual(float(overall["saturated_lag_rate"]), saturation_rate)
        self.assertEqual(
            threshold_summary["algorithms"]["pcnss_root_music"]
            ["max_error_le_1p00_deg"]["count"],
            max_error_passes,
        )
        self.assertEqual(
            threshold_summary["algorithms"]["fbss_root_music_L7"]
            ["max_error_le_1p00_deg"]["count"],
            len(sample_rows),
        )
        self.assertEqual(
            threshold_summary["paired_comparison"]["metric"],
            "sample_rmspe_deg",
        )
        self.assertEqual(
            threshold_summary["paired_comparison"]["candidate_algorithm"],
            "pcnss_root_music",
        )
        self.assertEqual(
            threshold_summary["paired_comparison"]["reference_algorithm"],
            "fbss_root_music_L7",
        )
        self.assertEqual(
            threshold_summary["paired_comparison"]["win"]["count"],
            len(sample_rows),
        )


if __name__ == "__main__":
    unittest.main()
