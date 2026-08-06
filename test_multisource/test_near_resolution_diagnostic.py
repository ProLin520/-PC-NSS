import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from multisource_doa.diagnostics.near_resolution import (
    EXPECTED_EVALUATOR_CODE_SHA,
    build_threshold_summary,
    classify_threshold_cohort,
    load_near_audit,
)


class NearResolutionThresholdTest(unittest.TestCase):
    def test_one_degree_boundaries_are_mutually_exclusive(self):
        self.assertEqual(classify_threshold_cohort(True, True, 1.0), "resolved")
        self.assertEqual(
            classify_threshold_cohort(True, True, 1.000001),
            "near_miss_1_1p25",
        )
        self.assertEqual(classify_threshold_cohort(True, True, 1.25), "near_miss_1_1p25")
        self.assertEqual(
            classify_threshold_cohort(True, True, 1.5),
            "near_miss_1p25_1p5",
        )
        self.assertEqual(classify_threshold_cohort(True, True, 2.0), "near_miss_1p5_2")
        self.assertEqual(classify_threshold_cohort(True, True, 2.000001), "far_miss_gt_2")
        self.assertEqual(classify_threshold_cohort(True, False, 0.5), "separation_failure")
        self.assertEqual(classify_threshold_cohort(False, False, 60.0), "estimation_failure")

    def test_threshold_summary_uses_maximum_matched_angle_error(self):
        rows = [
            {"algorithm": "pcnss_root_music", "absolute_error_1_deg": 0.4,
             "absolute_error_2_deg": 0.75},
            {"algorithm": "pcnss_root_music", "absolute_error_1_deg": 0.8,
             "absolute_error_2_deg": 1.25},
        ]
        summary = build_threshold_summary(rows, "pcnss_root_music")
        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["max_error_le_0p75_deg"]["count"], 1)
        self.assertEqual(summary["max_error_le_1p25_deg"]["count"], 2)


class NearAuditLoadTest(unittest.TestCase):
    def _write_report(
        self,
        directory: Path,
        *,
        code_sha: str = EXPECTED_EVALUATOR_CODE_SHA,
        checkpoint_sha: str | None = None,
        pcnss_rows: list[dict] | None = None,
        fbss_rows: list[dict] | None = None,
        stage: str = "evaluate_validation",
        run_split: str = "validation",
        summary_split: str = "validation",
        schema_version: int = 2,
    ) -> Path:
        checkpoint = directory / "frozen_checkpoint.pt"
        checkpoint.write_bytes(b"frozen checkpoint")
        (directory / "run_config.json").write_text(
            json.dumps({"stage": stage, "split": run_split}), encoding="utf-8"
        )
        (directory / "summary.json").write_text(
            json.dumps({"split": summary_split, "report_schema_version": schema_version}),
            encoding="utf-8",
        )
        (directory / "source_manifest.json").write_text(
            json.dumps(
                {
                    "code_sha": code_sha,
                    "checkpoint_sha": checkpoint_sha
                    or hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                }
            ),
            encoding="utf-8",
        )
        rows = (pcnss_rows or [self._row("pcnss_root_music")]) + (
            fbss_rows or [self._row("fbss_root_music_L7")]
        )
        with (directory / "predictions.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return checkpoint

    @staticmethod
    def _row(algorithm: str, **overrides: object) -> dict:
        row = {
            "split": "validation",
            "sample_seed": 7001,
            "algorithm": algorithm,
            "true_angle_1_deg": -1.5,
            "true_angle_2_deg": 1.5,
            "absolute_error_1_deg": 0.5,
            "absolute_error_2_deg": 0.75,
            "success": True,
            "estimated_separation_at_least_half_true": True,
            "rho": 1.0,
            "snr_db": 5.0,
            "snapshot_count": 20,
            "separation_deg": 3.0,
        }
        row.update(overrides)
        return row

    def test_loads_valid_near_pair_and_records_input_hashes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            report = Path(temporary_directory)
            checkpoint = self._write_report(report)

            selection = load_near_audit(report, checkpoint, expected_near_count=1)

        self.assertEqual(len(selection.labels), 1)
        self.assertEqual(selection.labels[0].sample_seed, 7001)
        self.assertEqual(selection.labels[0].threshold_cohort, "resolved")
        self.assertEqual(set(selection.input_sha256), {
            "run_config.json", "summary.json", "source_manifest.json", "predictions.csv"
        })

    def test_rejects_code_sha_and_checkpoint_sha_mismatches(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            report = Path(temporary_directory)
            checkpoint = self._write_report(report, code_sha="wrong")
            with self.assertRaisesRegex(ValueError, "code SHA"):
                load_near_audit(report, checkpoint, expected_near_count=1)

            checkpoint = self._write_report(
                report, checkpoint_sha="not the checkpoint hash"
            )
            with self.assertRaisesRegex(ValueError, "checkpoint SHA"):
                load_near_audit(report, checkpoint, expected_near_count=1)

    def test_rejects_duplicate_seed_pair_set_and_metadata_mismatches(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            report = Path(temporary_directory)
            duplicate_rows = [
                self._row("pcnss_root_music"),
                self._row("pcnss_root_music", absolute_error_1_deg=0.6),
            ]
            checkpoint = self._write_report(report, pcnss_rows=duplicate_rows)
            with self.assertRaisesRegex(ValueError, "duplicate sample_seed"):
                load_near_audit(report, checkpoint, expected_near_count=1)

            checkpoint = self._write_report(
                report,
                fbss_rows=[self._row("fbss_root_music_L7", sample_seed=7002)],
            )
            with self.assertRaisesRegex(ValueError, "sample_seed sets"):
                load_near_audit(report, checkpoint, expected_near_count=1)

            checkpoint = self._write_report(
                report,
                fbss_rows=[self._row("fbss_root_music_L7", rho=0.9)],
            )
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                load_near_audit(report, checkpoint, expected_near_count=1)

    def test_rejects_non_near_rows_and_unexpected_near_count(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            report = Path(temporary_directory)
            checkpoint = self._write_report(
                report,
                pcnss_rows=[self._row("pcnss_root_music", separation_deg=4.0)],
                fbss_rows=[self._row("fbss_root_music_L7", separation_deg=4.0)],
            )
            with self.assertRaisesRegex(ValueError, "expected_near_count"):
                load_near_audit(report, checkpoint, expected_near_count=1)

    def test_rejects_non_validation_prediction_rows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            report = Path(temporary_directory)
            checkpoint = self._write_report(
                report,
                pcnss_rows=[self._row("pcnss_root_music", split="development")],
                fbss_rows=[self._row("fbss_root_music_L7", split="development")],
            )

            with self.assertRaisesRegex(ValueError, "prediction rows must be validation"):
                load_near_audit(report, checkpoint, expected_near_count=1)

    def test_labels_are_sorted_by_sample_seed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            report = Path(temporary_directory)
            checkpoint = self._write_report(
                report,
                pcnss_rows=[
                    self._row("pcnss_root_music", sample_seed=7002),
                    self._row("pcnss_root_music", sample_seed=7001),
                ],
                fbss_rows=[
                    self._row("fbss_root_music_L7", sample_seed=7002),
                    self._row("fbss_root_music_L7", sample_seed=7001),
                ],
            )

            selection = load_near_audit(report, checkpoint, expected_near_count=2)

        self.assertEqual([label.sample_seed for label in selection.labels], [7001, 7002])


if __name__ == "__main__":
    unittest.main()
