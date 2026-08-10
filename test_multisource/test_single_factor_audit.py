import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from multisource_doa.config import ExperimentConfig, SplitName
from multisource_doa.data.dataset import PCNSSDataset
from multisource_doa.data.manifest import build_split_manifest
from multisource_doa.models.pc_nss import MultiScalePCNSS
from multisource_doa.training.error_teacher import build_error_teacher_row
from multisource_doa.training.single_factor_audit import (
    audit_single_factor_inputs,
)
from multisource_doa.training.single_factor_reporting import (
    write_single_factor_audit_report,
)
from multisource_doa.training.teacher_cache import write_teacher_cache


def _write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class SingleFactorAuditTest(unittest.TestCase):
    def _inputs(self, root, *, environment=True):
        root = Path(root)
        config = ExperimentConfig()
        training = root / "training"
        validation = root / "validation"
        task16 = root / "task16"
        cache = root / "cache"
        for path in (training, validation, task16):
            path.mkdir()
        _write_json(training / "train_manifest.json", build_split_manifest(config, SplitName.TRAIN))
        _write_json(training / "validation_manifest.json", build_split_manifest(config, SplitName.VALIDATION))
        (training / "metrics.csv").write_text("epoch,selected\n35,True\n", encoding="utf-8")
        model = MultiScalePCNSS()
        metadata = {
            "device": "cpu", "batch_size": 128, "shuffle": True,
            "total_epochs": 50, "learning_rate": 1e-3,
            "physical_path_regression_version": 1,
        } if environment else {}
        checkpoint = training / "best.pt"
        torch.save({
            "model_state_dict": model.state_dict(), "epoch": 35,
            "selection_metric_name": "failure_aware_rmspe_deg",
            "selection_metric_value": 7.264,
            "experiment_config": asdict(config), "model_seed": 2026,
            "data_split_seed": config.split.seeds[SplitName.VALIDATION],
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "code_sha": "baseline", "training_metadata": metadata,
        }, checkpoint)
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        _write_json(training / "best.pt.sha256.json", {
            "checkpoint": "best.pt", "checkpoint_sha256": checkpoint_sha,
            "selection_metric_name": "failure_aware_rmspe_deg",
            "selection_metric_value": 7.264, "epoch": 35,
        })
        _write_json(validation / "run_config.json", {"stage": "evaluate_validation"})
        _write_json(validation / "source_manifest.json", {
            **build_split_manifest(config, SplitName.VALIDATION),
            "checkpoint_sha": checkpoint_sha,
        })
        _write_json(validation / "summary.json", {"report_schema_version": 2, "split": "validation"})
        for name, content in (
            ("predictions.csv", "split,sample_seed,algorithm\n"),
            ("paired_comparisons.csv", "comparison,group,bin,win,tie,loss\n"),
            ("failure_reasons.csv", "algorithm,failure_reason,count\n"),
            ("runtime_summary.json", "{}\n"),
        ):
            (validation / name).write_text(content, encoding="utf-8")
        task16_names = (
            "diagnostic_config.json", "source_manifest.json",
            "teacher_ranking_summary.json", "teacher_component_summary.json",
            "decision.json",
        )
        for name in task16_names:
            _write_json(task16 / name, {})
        for name in (
            "teacher_ranking_sample_diagnostics.csv",
            "teacher_ranking_stratified_summary.csv", "teacher_oracle_confusion.csv",
        ):
            (task16 / name).write_text("header\n", encoding="utf-8")
        _write_json(task16 / "source_manifest.json", {"teacher_ranking_schema_version": 1})
        _write_json(task16 / "decision.json", {
            "mechanism_conclusion": "ranking_invalid", "training_authorized": False,
        })
        dataset = PCNSSDataset(SplitName.TRAIN, config)
        rows = [build_error_teacher_row(dataset[i], sample_index=i) for i in range(4)]
        write_teacher_cache(
            rows, cache, experiment_config=config,
            run_config={"stage": "smoke", "device": "cpu", "batch_size": 128},
            code_sha="current", source_sha256={"source.py": "a" * 64},
            expected_count=4,
        )
        return {
            "baseline_training_directory": training,
            "baseline_validation_directory": validation,
            "task16_directory": task16,
            "teacher_cache_directory": cache,
            "experiment_config": config,
            "expected_cache_count": 4,
        }

    def test_valid_identity_allows_reuse_and_task16_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            inputs = self._inputs(temporary)
            result = audit_single_factor_inputs(**inputs)
            self.assertTrue(result.baseline_reuse_allowed)
            self.assertTrue(all(result.gates.values()))
            _write_json(Path(inputs["task16_directory"]) / "decision.json", {
                "mechanism_conclusion": "calibration_only", "training_authorized": False,
            })
            with self.assertRaises(ValueError):
                audit_single_factor_inputs(**inputs)

    def test_unproven_environment_requires_physical_control_rerun(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = audit_single_factor_inputs(**self._inputs(temporary, environment=False))
            self.assertFalse(result.baseline_reuse_allowed)
            self.assertFalse(result.gates["same_training_environment"])
            self.assertEqual(result.evidence["required_action"], "rerun_physical_control")

    def test_writer_emits_exact_three_files_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            inputs = self._inputs(temporary)
            result = audit_single_factor_inputs(**inputs)
            output = Path(temporary) / "audit"
            write_single_factor_audit_report(
                result, output, run_config={"stage": "audit_single_factor"}
            )
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"audit_config.json", "source_manifest.json", "single_factor_audit.json"},
            )
            with self.assertRaises(FileExistsError):
                write_single_factor_audit_report(result, output, run_config={})


if __name__ == "__main__":
    unittest.main()
