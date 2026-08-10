import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from multisource_doa.config import ExperimentConfig
from multisource_doa.data.simulator import generate_two_source_sample
from multisource_doa.diagnostics.teacher_ranking import (
    TeacherRankingLabel,
    component_ranking_diagnostics,
    diagnose_teacher_ranking_samples,
    load_teacher_ranking_inputs,
    rank_signal_against_rmspe,
)
from multisource_doa.training.engine import collate_samples
from multisource_doa.training.teacher import build_scale_teacher


class TeacherRankingMetricTest(unittest.TestCase):
    def test_perfect_and_reversed_rankings_have_expected_correlations(self):
        rmspe = (1.0, 2.0, 3.0, 4.0)

        perfect = rank_signal_against_rmspe((4.0, 3.0, 2.0, 1.0), rmspe)
        reversed_result = rank_signal_against_rmspe((1.0, 2.0, 3.0, 4.0), rmspe)

        self.assertAlmostEqual(perfect["spearman_rho"], 1.0)
        self.assertAlmostEqual(perfect["kendall_tau_b"], 1.0)
        self.assertEqual(perfect["concordant_pair_count"], 6)
        self.assertAlmostEqual(perfect["pairwise_concordance_rate"], 1.0)
        self.assertAlmostEqual(reversed_result["spearman_rho"], -1.0)
        self.assertAlmostEqual(reversed_result["kendall_tau_b"], -1.0)
        self.assertEqual(reversed_result["discordant_pair_count"], 6)

    def test_constant_signal_keeps_samples_and_counts_teacher_ties_as_misses(self):
        result = rank_signal_against_rmspe(
            (1.0, 1.0, 1.0, 1.0),
            (1.0, 2.0, 3.0, 4.0),
        )

        self.assertIsNone(result["spearman_rho"])
        self.assertIsNone(result["kendall_tau_b"])
        self.assertEqual(result["teacher_tie_pair_count"], 6)
        self.assertEqual(result["oracle_tie_pair_count"], 0)
        self.assertAlmostEqual(result["pairwise_concordance_rate"], 0.0)
        self.assertEqual(result["top1_scale"], 4)
        self.assertEqual(result["top2_scales"], (4, 5))

    def test_oracle_tie_is_preserved_and_excluded_only_from_pairwise_denominator(self):
        result = rank_signal_against_rmspe(
            (0.1, 0.4, 0.3, 0.2),
            (1.0, 1.0 + 5e-7, 3.0, 4.0),
        )

        self.assertEqual(result["oracle_best_scales"], (4, 5))
        self.assertEqual(result["oracle_tie_pair_count"], 1)
        self.assertEqual(
            result["pairwise_comparable_count"],
            result["concordant_pair_count"]
            + result["discordant_pair_count"]
            + result["teacher_tie_pair_count"],
        )
        self.assertEqual(result["top1_scale"], 5)
        self.assertTrue(result["top1_oracle_agreement"])
        self.assertTrue(result["top2_oracle_coverage"])
        self.assertAlmostEqual(result["top1_regret_deg"], 5e-7)

    def test_top2_coverage_uses_fixed_score_then_scale_order(self):
        result = rank_signal_against_rmspe(
            (0.5, 0.5, 0.4, 0.3),
            (5.0, 4.0, 1.0, 3.0),
        )

        self.assertEqual(result["top1_scale"], 4)
        self.assertEqual(result["top2_scales"], (4, 5))
        self.assertFalse(result["top1_oracle_agreement"])
        self.assertFalse(result["top2_oracle_coverage"])
        self.assertEqual(result["exact_signal_tie_pair_count"], 1)
        self.assertAlmostEqual(result["top1_regret_deg"], 4.0)

    def test_component_diagnostics_expose_cancellation_and_three_signals(self):
        result = component_ranking_diagnostics(
            q_true_1=(0.1, 0.2, 0.3, 0.4),
            q_true_2=(0.1, 0.2, 0.3, 0.4),
            q_midpoint=(0.4, 0.3, 0.2, 0.1),
            rmspe_deg=(1.0, 2.0, 3.0, 4.0),
        )

        self.assertEqual(result["q_truth_mean"], (0.1, 0.2, 0.3, 0.4))
        self.assertEqual(result["negative_truth_mean"], (-0.1, -0.2, -0.3, -0.4))
        for actual, expected in zip(
            result["current_score"], (0.3, 0.1, -0.1, -0.3)
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(result["current_score_range"], 0.6)
        self.assertAlmostEqual(result["cancellation_ratio"], 1.0)
        self.assertAlmostEqual(
            result["signals"]["current_score"]["spearman_rho"], 1.0
        )

    def test_zero_component_ranges_return_null_cancellation_ratio(self):
        result = component_ranking_diagnostics(
            q_true_1=(0.2, 0.2, 0.2, 0.2),
            q_true_2=(0.2, 0.2, 0.2, 0.2),
            q_midpoint=(0.3, 0.3, 0.3, 0.3),
            rmspe_deg=(1.0, 2.0, 3.0, 4.0),
        )

        self.assertIsNone(result["cancellation_ratio"])
        self.assertEqual(result["cancellation_denominator_zero"], True)

    def test_nonfinite_and_wrong_length_vectors_are_rejected(self):
        for signal, rmspe, message in (
            ((1.0, 2.0, 3.0), (1.0, 2.0, 3.0), "four"),
            ((1.0, 2.0, 3.0, float("nan")), (1.0, 2.0, 3.0, 4.0), "finite"),
            ((1.0, 2.0, 3.0, 4.0), (1.0, 2.0, 3.0, float("inf")), "finite"),
        ):
            with self.subTest(signal=signal, rmspe=rmspe):
                with self.assertRaisesRegex(ValueError, message):
                    rank_signal_against_rmspe(signal, rmspe)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class TeacherRankingInputTest(unittest.TestCase):
    def _write_task15(self, mutation: str | None = None):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        task15 = root / "task15"
        audit = root / "audit"
        task14 = root / "task14"
        task15.mkdir()
        audit.mkdir()
        task14.mkdir()
        upstream_paths = {
            "audit/run_config.json": audit / "run_config.json",
            "audit/summary.json": audit / "summary.json",
            "audit/source_manifest.json": audit / "source_manifest.json",
            "audit/predictions.csv": audit / "predictions.csv",
            "task14/source_manifest.json": task14 / "source_manifest.json",
            "task14/near_sample_diagnostics.csv": task14 / "near_sample_diagnostics.csv",
        }
        for name, path in upstream_paths.items():
            path.write_text(name + "\n", encoding="utf-8")
        input_hashes = {name: _sha256(path) for name, path in upstream_paths.items()}
        if mutation == "upstream_hash":
            input_hashes["audit/predictions.csv"] = "wrong"
        config = {
            "stage": "diagnose_validation_teacher",
            "split": "validation",
            "report_directory": str(audit),
            "task14_directory": str(task14),
            "device": "cpu",
            "batch_size": 128,
        }
        (task15 / "diagnostic_config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        manifest = {
            "teacher_diagnostic_schema_version": (
                2 if mutation == "schema" else 1
            ),
            "sample_count": 1,
            "device": "cpu",
            "batch_size": 128,
            "no_model_forward": True,
            "training_performed": False,
            "validation_split_seed": 202708040,
            "input_sha256": input_hashes,
        }
        (task15 / "source_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        score_values = (-0.1, -0.2, -0.3, -0.4)
        probability_values = (0.4, 0.3, 0.2, 0.1)
        if mutation == "probability":
            probability_values = (0.4, 0.3, 0.2, 0.2)
        row = {
            "sample_seed": 202708040,
            "true_angle_1_deg": -1.5,
            "true_angle_2_deg": 1.5,
            "rho": 1.0,
            "snr_db": 5.0,
            "snapshot_count": 20,
            "separation_deg": 3.0,
            "threshold_cohort": "resolved",
            **{
                f"teacher_score_L{size}": value
                for size, value in zip((4, 5, 6, 7), score_values, strict=True)
            },
            **{
                f"teacher_p_current_L{size}": value
                for size, value in zip(
                    (4, 5, 6, 7), probability_values, strict=True
                )
            },
            **{
                f"teacher_p_counterfactual_L{size}": value
                for size, value in zip(
                    (4, 5, 6, 7), (0.7, 0.2, 0.08, 0.02), strict=True
                )
            },
            **{
                f"student_p_L{size}": value
                for size, value in zip(
                    (4, 5, 6, 7), (0.25, 0.25, 0.25, 0.25), strict=True
                )
            },
            **{
                f"fbss_L{size}_sample_rmspe_deg": value
                for size, value in zip(
                    (4, 5, 6, 7), (1.0, 2.0, 3.0, 4.0), strict=True
                )
            },
        }
        if mutation == "nonfinite":
            row["teacher_score_L7"] = "nan"
        with (task15 / "teacher_sample_diagnostics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        (task15 / "teacher_summary.json").write_text("{}", encoding="utf-8")
        (task15 / "teacher_stratified_summary.csv").write_text(
            "dimension,bin,sample_count\n", encoding="utf-8"
        )
        (task15 / "decision.json").write_text("{}", encoding="utf-8")
        return task15, temporary

    def test_authenticates_task15_six_files_and_upstream_hashes(self):
        task15, temporary = self._write_task15()
        self.addCleanup(temporary.cleanup)

        inputs = load_teacher_ranking_inputs(task15, expected_count=1)

        self.assertEqual(tuple(inputs.labels_by_seed), (202708040,))
        self.assertEqual(len(inputs.task15_sha256), 6)
        self.assertEqual(inputs.validation_split_seed, 202708040)

    def test_rejects_schema_upstream_hash_probability_and_nonfinite_values(self):
        for mutation, message in (
            ("schema", "schema"),
            ("upstream_hash", "upstream"),
            ("probability", "probabilities"),
            ("nonfinite", "finite"),
        ):
            with self.subTest(mutation=mutation):
                task15, temporary = self._write_task15(mutation)
                self.addCleanup(temporary.cleanup)
                with self.assertRaisesRegex(ValueError, message):
                    load_teacher_ranking_inputs(task15, expected_count=1)


class TeacherRankingPhysicalReconstructionTest(unittest.TestCase):
    def _sample_and_label(self):
        sample = generate_two_source_sample(
            ExperimentConfig(),
            split_seed=901,
            index=0,
            rho=1.0,
            snr_db=5.0,
            snapshot_count=20,
            center_deg=0.0,
            separation_deg=3.0,
        )
        batch = collate_samples([sample])
        scores = build_scale_teacher(
            batch.fbss_covariances, batch.true_angles_deg
        ).scale_scores[0]
        label = TeacherRankingLabel(
            sample_seed=sample.sample_seed,
            true_angles_deg=tuple(float(value) for value in sample.angles_deg),
            rho=float(sample.rho),
            snr_db=float(sample.snr_db),
            snapshot_count=int(sample.snapshot_count),
            separation_deg=float(np.diff(sample.angles_deg)[0]),
            threshold_cohort="resolved",
            task15_scores=tuple(float(value) for value in scores),
            fixed_rmspe_deg={4: 1.0, 5: 2.0, 6: 3.0, 7: 4.0},
        )
        return sample, label

    def test_reconstructs_components_without_model_and_matches_task15_scores(self):
        sample, label = self._sample_and_label()

        result = diagnose_teacher_ranking_samples(
            [sample], {sample.sample_seed: label}, batch_size=128
        )

        row = result.sample_rows[0]
        self.assertEqual(row["sample_seed"], sample.sample_seed)
        for size in (4, 5, 6, 7):
            self.assertAlmostEqual(
                row[f"current_score_L{size}"],
                label.task15_scores[(4, 5, 6, 7).index(size)],
                delta=1e-7,
            )
            self.assertAlmostEqual(
                row[f"current_score_L{size}"],
                row[f"q_midpoint_L{size}"] - row[f"q_truth_mean_L{size}"],
                delta=1e-7,
            )

    def test_rejects_task15_score_mismatch_and_metadata_mismatch(self):
        sample, label = self._sample_and_label()
        mismatch_score = TeacherRankingLabel(
            **{**label.__dict__, "task15_scores": (1.0, 1.0, 1.0, 1.0)}
        )
        with self.assertRaisesRegex(ValueError, "Task 15 score mismatch"):
            diagnose_teacher_ranking_samples(
                [sample], {sample.sample_seed: mismatch_score}
            )
        mismatch_metadata = TeacherRankingLabel(
            **{**label.__dict__, "rho": 0.9}
        )
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            diagnose_teacher_ranking_samples(
                [sample], {sample.sample_seed: mismatch_metadata}
            )


if __name__ == "__main__":
    unittest.main()
