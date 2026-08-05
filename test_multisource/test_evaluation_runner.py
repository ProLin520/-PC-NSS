import json
import tempfile
import unittest
from pathlib import Path

import torch

from multisource_doa.baselines.registry import (
    ExternalBaselineStatus,
    build_baseline_registry,
)
from multisource_doa.config import ExperimentConfig, SplitName
from multisource_doa.data.simulator import generate_two_source_sample
from multisource_doa.evaluation.reporting import write_evaluation_report
from multisource_doa.evaluation.runner import evaluate_samples
from multisource_doa.models.pc_nss import MultiScalePCNSS


def _samples():
    config = ExperimentConfig()
    return [
        generate_two_source_sample(
            config,
            split_seed=4500,
            index=index,
            rho=1.0,
            snr_db=5.0,
            snapshot_count=20,
        )
        for index in range(4)
    ]


class EvaluationRunnerTest(unittest.TestCase):
    def test_registry_marks_external_deep_baselines_as_not_integrated(self):
        registry = build_baseline_registry()

        self.assertEqual(registry["subspacenet"].status, ExternalBaselineStatus.NOT_INTEGRATED)
        self.assertEqual(registry["da_music"].status, ExternalBaselineStatus.NOT_INTEGRATED)
        self.assertEqual(registry["deepmusic"].status, ExternalBaselineStatus.NOT_INTEGRATED)

    def test_validation_runs_all_first_stage_estimators_and_selects_global_l(self):
        result = evaluate_samples(
            _samples(),
            MultiScalePCNSS(),
            split=SplitName.VALIDATION,
            device=torch.device("cpu"),
        )

        expected = {
            "music",
            "root_music",
            "esprit",
            "pcnss_root_music",
            *{f"sps_root_music_L{size}" for size in (4, 5, 6, 7)},
            *{f"fbss_root_music_L{size}" for size in (4, 5, 6, 7)},
        }
        self.assertEqual(set(result.summaries), expected)
        self.assertIn(result.best_fixed_fbss_scale, (4, 5, 6, 7))
        self.assertEqual(len(result.predictions), 4 * len(expected))
        self.assertFalse(any("oracle" in row["algorithm"] for row in result.predictions))

    def test_development_does_not_select_a_new_best_scale(self):
        result = evaluate_samples(
            _samples(),
            MultiScalePCNSS(),
            split=SplitName.DEVELOPMENT,
            device=torch.device("cpu"),
        )

        self.assertIsNone(result.best_fixed_fbss_scale)

    def test_locked_test_is_rejected(self):
        with self.assertRaises(PermissionError):
            evaluate_samples(
                _samples(),
                MultiScalePCNSS(),
                split=SplitName.LOCKED_TEST,
                device=torch.device("cpu"),
            )

    def test_report_writes_fixed_schema_without_claiming_research_acceptance(self):
        result = evaluate_samples(
            _samples(),
            MultiScalePCNSS(),
            split=SplitName.VALIDATION,
            device=torch.device("cpu"),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report"
            write_evaluation_report(
                result,
                output,
                run_config={"dry_run": True},
                source_manifest={"split": "validation"},
                code_sha="abc123",
                checkpoint_sha="not-a-formal-checkpoint",
            )

            expected_files = {
                "run_config.json",
                "source_manifest.json",
                "predictions.csv",
                "summary.json",
                "paired_comparisons.csv",
                "failure_reasons.csv",
                "runtime_summary.json",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected_files)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["framework_validation"])
            self.assertEqual(summary["research_acceptance"], "not_run")
            with self.assertRaises(FileExistsError):
                write_evaluation_report(
                    result,
                    output,
                    run_config={"dry_run": True},
                    source_manifest={"split": "validation"},
                    code_sha="abc123",
                    checkpoint_sha="not-a-formal-checkpoint",
                )


if __name__ == "__main__":
    unittest.main()
