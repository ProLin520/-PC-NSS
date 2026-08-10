import csv
import json
import tempfile
import unittest
from pathlib import Path

from multisource_doa.evaluation.teacher_experiment import audit_teacher_experiment
from multisource_doa.evaluation.teacher_experiment_reporting import (
    write_teacher_experiment_report,
)


REPORT_FILES = (
    "run_config.json", "source_manifest.json", "predictions.csv", "summary.json",
    "paired_comparisons.csv", "failure_reasons.csv", "runtime_summary.json",
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_report(root: Path, *, candidate: bool, mutation: str | None = None):
    root.mkdir()
    metadata = {
        "teacher_mode": "failure_aware_error" if candidate else "physical",
        "scale_distillation_target_source": (
            "train_only_failure_aware_rmspe" if candidate else "physical_music_score"
        ),
        "dominance_target_source": "physical_music_score",
        "teacher_cache_sha256": "a" * 64 if candidate else None,
        "single_factor_audit_sha256": "b" * 64 if candidate else None,
    }
    _write_json(root / "run_config.json", {"split": "validation", "model_seed": 2026})
    _write_json(root / "source_manifest.json", {
        "split": "validation", "size": 8, "sample_seed_start": 100,
        "sample_seed_end": 107, "generator_version": "test-v1",
        "experiment_config": {"frozen": True}, "checkpoint_sha": ("d" if candidate else "c") * 64,
        "training_metadata": metadata,
    })
    rows = []
    for algorithm in ("pcnss_root_music", "fbss_root_music_L7"):
        for index in range(8):
            near = index < 4
            resolved = (index < (3 if candidate else 2)) if algorithm == "pcnss_root_music" and near else True
            if algorithm == "fbss_root_music_L7" and near:
                resolved = index < 3
            error = 0.5 if candidate and algorithm == "pcnss_root_music" else 0.6
            if mutation == "rmspe" and candidate and algorithm == "pcnss_root_music" and not near:
                error = 5.0
            if mutation == "near_original" and candidate and algorithm == "pcnss_root_music" and near:
                resolved = index < 2
            if mutation == "near_l7" and candidate and algorithm == "pcnss_root_music" and near:
                resolved = index < 2
            if mutation == "overall_resolution" and candidate and algorithm == "pcnss_root_music" and not near:
                resolved = False
            success = not (mutation == "failures" and candidate and algorithm == "pcnss_root_music" and index == 7)
            if not success:
                error = 60.0
            rows.append({
                "split": "validation", "sample_seed": 100 + index, "algorithm": algorithm,
                "true_angle_1_deg": -2.0, "true_angle_2_deg": 1.0 if near else 5.0,
                "estimated_angle_1_deg": -2.0 + error, "estimated_angle_2_deg": (1.0 if near else 5.0) + error,
                "absolute_error_1_deg": error, "absolute_error_2_deg": error,
                "sample_rmspe_deg": error, "success": success,
                "both_angle_errors_within_1_deg": error <= 1.0,
                "estimated_separation_at_least_half_true": resolved, "resolved": resolved,
                "failure_reason": "synthetic" if not success else "", "rho": 1.0,
                "snr_db": 5.0, "snapshot_count": 20,
                "separation_deg": 3.0 if near else 7.0, "runtime_seconds": 0.0,
            })
    with (root / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    _write_json(root / "summary.json", {
        "report_schema_version": 2, "split": "validation", "best_fixed_fbss_scale": 7,
        "algorithms": {}, "near_separation_audit": {},
    })
    (root / "paired_comparisons.csv").write_text("comparison,group,bin,win,tie,loss\n", encoding="utf-8")
    (root / "failure_reasons.csv").write_text("algorithm,failure_reason,count\n", encoding="utf-8")
    _write_json(root / "runtime_summary.json", {})


class TeacherExperimentAuditTest(unittest.TestCase):
    def test_six_gates_are_conjunctive_and_writer_emits_five_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); baseline = root / "a"; candidate = root / "b"
            _write_report(baseline, candidate=False); _write_report(candidate, candidate=True)
            result = audit_teacher_experiment(baseline, candidate, expected_sample_count=8)
            self.assertTrue(all(result.decision["gates"].values()))
            self.assertEqual(result.decision["conclusion"], "seed2026_gate_passed")
            output = root / "audit"
            write_teacher_experiment_report(result, output, run_config={"stage": "smoke"})
            self.assertEqual({p.name for p in output.iterdir()}, {
                "experiment_audit_config.json", "source_manifest.json",
                "paired_and_transitions.csv", "stratified_summary.csv", "decision.json",
            })
            with self.assertRaises(FileExistsError):
                write_teacher_experiment_report(result, output, run_config={})

    def test_each_performance_gate_can_stop_the_experiment(self):
        expected = {
            "near_original": "near_resolution_improves_over_original",
            "near_l7": "near_resolution_not_below_fbss_L7",
            "rmspe": "overall_rmspe_not_worse",
            "overall_resolution": "overall_resolution_not_worse",
            "failures": "failure_count_not_worse",
        }
        for mutation, gate in expected.items():
            with self.subTest(gate=gate), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); baseline = root / "a"; candidate = root / "b"
                _write_report(baseline, candidate=False)
                _write_report(candidate, candidate=True, mutation=mutation)
                result = audit_teacher_experiment(baseline, candidate, expected_sample_count=8)
                self.assertFalse(result.decision["gates"][gate])
                self.assertEqual(result.decision["conclusion"], "experiment_failed")
                self.assertFalse(result.decision["development_authorized"])

    def test_protocol_identity_is_a_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); baseline = root / "a"; candidate = root / "b"
            _write_report(baseline, candidate=False); _write_report(candidate, candidate=True)
            manifest = json.loads((candidate / "source_manifest.json").read_text())
            manifest["generator_version"] = "changed"
            _write_json(candidate / "source_manifest.json", manifest)
            result = audit_teacher_experiment(baseline, candidate, expected_sample_count=8)
            self.assertFalse(result.decision["gates"]["protocol_identity"])


if __name__ == "__main__":
    unittest.main()
