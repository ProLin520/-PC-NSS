import inspect
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch

from multisource_doa.baselines.registry import (
    ExternalBaselineStatus,
    build_baseline_registry,
)
from multisource_doa.config import ExperimentConfig, SplitName
from multisource_doa.data.simulator import generate_two_source_sample
from multisource_doa.evaluation.reporting import write_evaluation_report
import multisource_doa.evaluation.runner as runner_module
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
        self.assertEqual(result.runtime_seconds["pcnss_inference_batch_size"], 128)
        self.assertIn("both_angle_errors_within_1_deg", result.predictions[0])
        self.assertIn(
            "estimated_separation_at_least_half_true",
            result.predictions[0],
        )

    def test_evaluator_defaults_to_checkpoint_validation_batch_size(self):
        parameter = inspect.signature(evaluate_samples).parameters[
            "inference_batch_size"
        ]

        self.assertEqual(parameter.default, 128)

    def test_neural_covariances_are_consistent_across_batch_sizes(self):
        samples = _samples()
        model = MultiScalePCNSS().eval()

        batch_one, _ = runner_module._infer_pcnss_covariances(
            samples,
            model,
            torch.device("cpu"),
            batch_size=1,
        )
        batch_four, _ = runner_module._infer_pcnss_covariances(
            samples,
            model,
            torch.device("cpu"),
            batch_size=4,
        )

        np.testing.assert_allclose(
            batch_one,
            batch_four,
            rtol=1e-5,
            atol=2e-6,
        )

    def test_cuda_timing_synchronizes_but_cpu_path_does_not(self):
        with mock.patch.object(torch.cuda, "synchronize") as synchronize:
            runner_module._synchronize_if_cuda(torch.device("cpu"))
            synchronize.assert_not_called()

            cuda_device = torch.device("cuda:0")
            runner_module._synchronize_if_cuda(cuda_device)
            synchronize.assert_called_once_with(cuda_device)

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
            self.assertEqual(summary["report_schema_version"], 2)
            self.assertTrue(summary["framework_validation"])
            self.assertEqual(summary["research_acceptance"], "not_run")
            self.assertEqual(summary["near_separation_audit"]["separation_bin"], "[2,4)")
            self.assertEqual(
                set(summary["near_separation_audit"]["algorithms"]),
                set(result.summaries),
            )
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
