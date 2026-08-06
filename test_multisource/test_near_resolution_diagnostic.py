import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from multisource_doa.config import ExperimentConfig
from multisource_doa.data.simulator import generate_two_source_sample
from multisource_doa.diagnostics import near_resolution
from multisource_doa.diagnostics.near_resolution import (
    EXPECTED_EVALUATOR_CODE_SHA,
    NearAuditLabel,
    build_threshold_summary,
    classify_threshold_cohort,
    load_near_audit,
)
from multisource_doa.models.pc_nss import MultiScalePCNSS
from multisource_doa.physics.projection import ProjectionResult
from multisource_doa.training.losses import aggregate_scale_weights


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


class NearResolutionMechanismTest(unittest.TestCase):
    def test_scale_entropy_normalization(self):
        weights = torch.tensor(
            [
                [
                    [0.25, 0.0, 0.0],
                    [0.25, 1.0, 0.0],
                    [0.25, 0.0, 0.0],
                    [0.25, 0.0, 1.0],
                ]
            ]
        )
        valid_mask = torch.tensor(
            [
                [
                    [True, False, True],
                    [True, True, True],
                    [True, False, True],
                    [True, False, True],
                ]
            ]
        )
        effective_counts = torch.tensor(
            [
                [
                    [4.0, 0.0, 0.0],
                    [2.0, 0.5, 0.0],
                    [4.0, 0.0, 0.0],
                    [2.0, 0.0, 0.5],
                ]
            ]
        )

        metrics = near_resolution.scale_weight_diagnostics(
            weights, valid_mask, effective_counts
        )
        expected = aggregate_scale_weights(weights, valid_mask, effective_counts)

        self.assertAlmostEqual(metrics[0]["scale_entropy_normalized"], 1.0, places=6)
        self.assertIsNone(metrics[0]["lag_entropy_normalized"][1])
        self.assertEqual(metrics[0]["dominant_scale"], 4)
        np.testing.assert_allclose(
            [metrics[0][f"p_L{size}"] for size in (4, 5, 6, 7)],
            expected[0].numpy(),
        )

    def test_residual_saturation_boundary(self):
        residual = torch.tensor([[[0.094999, 0.0], [0.095, 0.0], [0.10, 0.0]]])

        metrics = near_resolution.residual_diagnostics(residual, residual_limit=0.10)

        self.assertEqual(metrics[0]["saturated_lag_count"], 2)
        self.assertAlmostEqual(metrics[0]["saturated_lag_rate"], 2 / 3)

    def test_projection_changes_keep_nonconverged_rows(self):
        candidate = np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.complex128)
        train_projected = np.asarray([[[2.0, 0.0], [0.0, 2.0]]], dtype=np.complex128)
        final = np.asarray([[4.0, 0.0], [0.0, 4.0]], dtype=np.complex128)

        metrics = near_resolution.projection_diagnostics(
            candidate,
            train_projected,
            projection_fn=lambda _: ProjectionResult(
                matrix=final,
                converged=False,
                iterations=9,
                hermitian_error=0.1,
                toeplitz_error=0.2,
                trace_error=0.3,
                min_eigenvalue=-0.4,
            ),
        )

        self.assertEqual(len(metrics), 1)
        self.assertAlmostEqual(metrics[0]["train_projection_change"], 1.0)
        self.assertAlmostEqual(metrics[0]["eval_projection_change"], 1.0)
        self.assertAlmostEqual(metrics[0]["total_projection_change"], 3.0)
        self.assertFalse(metrics[0]["dykstra_converged"])

    def test_diagnose_near_samples_preserves_order_and_joins_labels(self):
        config = ExperimentConfig()
        samples = [
            generate_two_source_sample(
                config,
                split_seed=901,
                index=index,
                rho=1.0,
                snr_db=5.0,
                snapshot_count=20,
                center_deg=float(index),
                separation_deg=3.0,
            )
            for index in range(4)
        ]
        labels_by_seed = {
            sample.sample_seed: NearAuditLabel(
                sample_seed=sample.sample_seed,
                rho=sample.rho,
                snr_db=sample.snr_db,
                snapshot_count=sample.snapshot_count,
                separation_deg=3.0,
                pcnss_row={
                    "absolute_error_1_deg": 0.5,
                    "absolute_error_2_deg": 0.75,
                    "success": True,
                },
                fbss_l7_row={},
                threshold_cohort="resolved",
            )
            for sample in samples
        }
        model = MultiScalePCNSS()
        model.train()

        result = near_resolution.diagnose_near_samples(
            samples,
            labels_by_seed,
            model,
            device=torch.device("cpu"),
            batch_size=2,
        )

        self.assertEqual(len(result.sample_rows), 4)
        self.assertEqual(
            [row["sample_seed"] for row in result.sample_rows],
            [sample.sample_seed for sample in samples],
        )
        self.assertEqual({row["sample_seed"] for row in result.sample_rows}, set(labels_by_seed))
        self.assertFalse(model.training)
        self.assertTrue(all(row["threshold_cohort"] == "resolved" for row in result.sample_rows))

    def test_diagnose_near_samples_uses_frozen_residual_limit(self):
        config = ExperimentConfig()
        sample = generate_two_source_sample(
            config,
            split_seed=950,
            index=0,
            rho=1.0,
            snr_db=5.0,
            snapshot_count=20,
            center_deg=0.0,
            separation_deg=3.0,
        )
        label = NearAuditLabel(
            sample_seed=sample.sample_seed,
            rho=sample.rho,
            snr_db=sample.snr_db,
            snapshot_count=sample.snapshot_count,
            separation_deg=3.0,
            pcnss_row={},
            fbss_l7_row={},
            threshold_cohort="resolved",
        )
        model = MultiScalePCNSS(residual_fraction=0.08)
        with torch.no_grad():
            model.residual_head[-1].weight.zero_()
            model.residual_head[-1].bias.copy_(torch.tensor([10.0, 0.0]))

        result = near_resolution.diagnose_near_samples(
            [sample],
            {sample.sample_seed: label},
            model,
            device=torch.device("cpu"),
            batch_size=1,
        )

        self.assertEqual(result.sample_rows[0]["saturated_lag_count"], 0)


if __name__ == "__main__":
    unittest.main()
