import json
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from multisource_doa.config import ExperimentConfig, SplitName
from multisource_doa.data.dataset import PCNSSDataset
from multisource_doa.data.simulator import generate_two_source_sample
from multisource_doa.diagnostics.near_resolution import (
    NearAuditLabel,
    diagnose_near_samples,
)
from multisource_doa.diagnostics.reporting import write_near_diagnostic_report
from multisource_doa.models.pc_nss import MultiScalePCNSS
from multisource_doa.training.error_teacher import build_error_teacher_row
from multisource_doa.training.single_factor_audit import SingleFactorAuditResult
from multisource_doa.training.single_factor_reporting import (
    write_single_factor_audit_report,
)
from multisource_doa.training.teacher_cache import (
    load_teacher_cache,
    write_teacher_cache,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = PROJECT_ROOT / "scripts" / "run_multiscale_pcnss.py"
DIAGNOSTIC_SCRIPT = PROJECT_ROOT / "scripts" / "diagnose_pcnss_near_resolution.py"
TEACHER_DIAGNOSTIC_SCRIPT = (
    PROJECT_ROOT / "scripts" / "diagnose_pcnss_teacher_confidence.py"
)
TEACHER_RANKING_SCRIPT = (
    PROJECT_ROOT / "scripts" / "diagnose_pcnss_teacher_ranking.py"
)
ERROR_TEACHER_CACHE_SCRIPT = (
    PROJECT_ROOT / "scripts" / "build_pcnss_failure_aware_teacher_cache.py"
)
SINGLE_FACTOR_AUDIT_SCRIPT = (
    PROJECT_ROOT / "scripts" / "audit_pcnss_teacher_single_factor.py"
)


class EntrypointTest(unittest.TestCase):
    def test_near_resolution_diagnostic_defaults_and_guards_are_safe(self):
        namespace = runpy.run_path(str(DIAGNOSTIC_SCRIPT))
        config = namespace["RUN_CONFIG"]

        self.assertEqual(config["stage"], "dry_run")
        self.assertTrue(config["dry_run"])
        self.assertEqual(config["split"], "validation")
        self.assertEqual(config["device"], "cpu")
        self.assertEqual(config["batch_size"], 128)
        self.assertFalse(config["allow_locked_test"])
        self.assertFalse(config["overwrite"])
        self.assertNotIn("development", namespace["STAGES"])
        self.assertNotIn("locked_test", namespace["STAGES"])

        with self.assertRaisesRegex(ValueError, "dry_run"):
            namespace["run_diagnostic"](dict(config, stage="diagnose_validation_near"))
        with self.assertRaisesRegex(PermissionError, "locked_test"):
            namespace["run_diagnostic"](
                dict(
                    config,
                    stage="diagnose_validation_near",
                    dry_run=False,
                    allow_locked_test=True,
                )
            )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                namespace["run_diagnostic"](
                    dict(
                        config,
                        stage="diagnose_validation_near",
                        dry_run=False,
                        output_root=str(output),
                    )
                )

    def test_near_resolution_dry_run_creates_no_output(self):
        namespace = runpy.run_path(str(DIAGNOSTIC_SCRIPT))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused"
            result = namespace["run_dry_run"](
                dict(namespace["RUN_CONFIG"], output_root=str(output))
            )

            self.assertEqual(result["stage"], "dry_run")
            self.assertFalse(result["locked_test_access"])
            self.assertFalse(result["output_created"])
            self.assertFalse(output.exists())

    def test_near_resolution_config_accepts_only_known_keys(self):
        namespace = runpy.run_path(str(DIAGNOSTIC_SCRIPT))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"batch_size": 4}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "batch_size.*128"):
                namespace["load_config"](path)

            path.write_text(json.dumps({"unexpected": True}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown config keys"):
                namespace["load_config"](path)

    def test_near_resolution_direct_entrypoints_reject_non_frozen_batch_size(self):
        namespace = runpy.run_path(str(DIAGNOSTIC_SCRIPT))
        invalid = dict(namespace["RUN_CONFIG"], batch_size=4)

        with self.assertRaisesRegex(ValueError, "batch_size.*128"):
            namespace["run_dry_run"](invalid)
        with self.assertRaisesRegex(ValueError, "batch_size.*128"):
            namespace["run_diagnostic"](
                dict(invalid, stage="diagnose_validation_near", dry_run=False)
            )

    def test_near_resolution_cpu_smoke_writes_six_files_from_train_samples(self):
        config = ExperimentConfig()
        samples = [
            generate_two_source_sample(
                config,
                split_seed=config.split.seeds[SplitName.TRAIN],
                index=index,
                rho=1.0,
                snr_db=5.0,
                snapshot_count=20,
                center_deg=float(index),
                separation_deg=3.0,
            )
            for index in range(4)
        ]
        labels = {
            sample.sample_seed: NearAuditLabel(
                sample_seed=sample.sample_seed,
                rho=sample.rho,
                snr_db=sample.snr_db,
                snapshot_count=sample.snapshot_count,
                separation_deg=3.0,
                pcnss_row={
                    "absolute_error_1_deg": 0.5,
                    "absolute_error_2_deg": 0.75,
                    "sample_rmspe_deg": 0.6373774391990981,
                    "success": True,
                    "estimated_separation_at_least_half_true": True,
                },
                fbss_l7_row={
                    "absolute_error_1_deg": 0.6,
                    "absolute_error_2_deg": 0.8,
                    "sample_rmspe_deg": 0.7071067811865476,
                    "success": True,
                    "estimated_separation_at_least_half_true": True,
                },
                threshold_cohort="resolved",
            )
            for sample in samples
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "smoke.pt"
            torch.save({"model_state_dict": MultiScalePCNSS().state_dict()}, checkpoint)
            result = diagnose_near_samples(
                samples,
                labels,
                MultiScalePCNSS(),
                device=torch.device("cpu"),
                batch_size=2,
            )
            output = write_near_diagnostic_report(
                result,
                root / "diagnostic",
                diagnostic_config={"checkpoint_path": str(checkpoint), "batch_size": 2},
                source_manifest={"source": "temporary-train-smoke"},
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
            self.assertEqual(
                {row["sample_seed"] for row in result.sample_rows},
                {sample.sample_seed for sample in samples},
            )
            self.assertFalse(any(
                "development" in field or "locked" in field
                for row in result.sample_rows
                for field in row
            ))

    def test_run_config_is_safe_for_parameterless_pycharm_execution(self):
        namespace = runpy.run_path(str(RUN_SCRIPT))
        config = namespace["RUN_CONFIG"]

        self.assertEqual(config["stage"], "dry_run")
        self.assertTrue(config["dry_run"])
        self.assertEqual(config["sample_count"], 4)
        self.assertFalse(config["allow_locked_test"])
        self.assertFalse(config["overwrite"])
        self.assertEqual(config["evaluation_batch_size"], 128)
        self.assertEqual(config["teacher_mode"], "physical")
        self.assertEqual(config["teacher_cache_path"], "")
        self.assertEqual(config["single_factor_audit_path"], "")
        self.assertNotIn("evaluate_locked_test", namespace["STAGES"])

    def test_failure_aware_training_requires_inputs_before_model_creation(self):
        namespace = runpy.run_path(str(RUN_SCRIPT))
        values = dict(
            namespace["RUN_CONFIG"],
            stage="train",
            dry_run=False,
            teacher_mode="failure_aware_error",
            teacher_cache_path="",
            single_factor_audit_path="",
            output_root="unused",
        )
        with mock.patch.object(
            namespace["MultiScalePCNSS"], "__init__", side_effect=AssertionError
        ):
            with self.assertRaises(ValueError):
                namespace["run_stage"](values)

    def test_physical_teacher_rejects_ambiguous_cache_configuration(self):
        namespace = runpy.run_path(str(RUN_SCRIPT))
        with self.assertRaises(ValueError):
            namespace["_load_teacher_training_context"](
                {
                    **namespace["RUN_CONFIG"],
                    "teacher_cache_path": "unexpected",
                },
                ExperimentConfig(),
                expected_count=4,
            )

    def test_failure_aware_smoke_consumes_authenticated_four_row_cache(self):
        namespace = runpy.run_path(str(RUN_SCRIPT))
        config = ExperimentConfig()
        dataset = PCNSSDataset(SplitName.TRAIN, config)
        rows = [
            build_error_teacher_row(dataset[index], sample_index=index)
            for index in range(4)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_directory = root / "cache"
            write_teacher_cache(
                rows,
                cache_directory,
                experiment_config=config,
                run_config={"device": "cpu", "batch_size": 128},
                code_sha="smoke",
                source_sha256={"source.py": "a" * 64},
                expected_count=4,
            )
            cache = load_teacher_cache(cache_directory, config, expected_count=4)
            audit_directory = root / "audit"
            write_single_factor_audit_report(
                SingleFactorAuditResult(
                    baseline_reuse_allowed=True,
                    gates={"synthetic_smoke": True},
                    evidence={"required_action": "reuse_baseline"},
                    source_sha256={"teacher_cache": dict(cache.file_sha256)},
                ),
                audit_directory,
                run_config={"stage": "smoke"},
            )
            result = namespace["run_smoke"](
                {
                    **namespace["RUN_CONFIG"],
                    "stage": "smoke_train",
                    "teacher_mode": "failure_aware_error",
                    "teacher_cache_path": str(cache_directory),
                    "single_factor_audit_path": str(audit_directory),
                    "output_root": str(root / "training_smoke"),
                }
            )
            self.assertEqual(
                result["training_metadata"]["scale_distillation_target_source"],
                "train_only_failure_aware_rmspe",
            )
            self.assertFalse(result["formal_checkpoint_written"])

    def test_stage_parser_rejects_combined_stage_string(self):
        namespace = runpy.run_path(str(RUN_SCRIPT))

        with self.assertRaisesRegex(ValueError, "每次只运行一个阶段"):
            namespace["validate_stage"]("train evaluate_development")

    def test_dry_run_checks_one_sample_without_creating_output(self):
        namespace = runpy.run_path(str(RUN_SCRIPT))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused"
            values = dict(namespace["RUN_CONFIG"], output_root=str(output))

            result = namespace["run_dry_run"](values)

            self.assertEqual(result["stage"], "dry_run")
            self.assertEqual(result["parameter_count"], 46_916)
            self.assertTrue(result["physical_chain_finite"])
            self.assertFalse(result["locked_test_access"])
            self.assertFalse(output.exists())

    def test_four_sample_smoke_writes_isolated_audit_without_formal_checkpoint(self):
        namespace = runpy.run_path(str(RUN_SCRIPT))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke"
            values = dict(namespace["RUN_CONFIG"], output_root=str(output))

            result = namespace["run_smoke"](values)

            self.assertEqual(result["sample_count"], 4)
            self.assertEqual(result["epoch_count"], 1)
            self.assertTrue((output / "smoke_summary.json").is_file())
            self.assertFalse((output / "best.pt").exists())
            self.assertFalse((output / "locked_test_manifest.json").exists())


class TeacherDiagnosticEntrypointTest(unittest.TestCase):
    def test_default_dry_run_is_cpu_only_and_creates_nothing(self):
        namespace = runpy.run_path(str(TEACHER_DIAGNOSTIC_SCRIPT))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused"
            values = dict(namespace["RUN_CONFIG"], output_root=str(output))

            result = namespace["run_stage"](values)

            self.assertEqual(result["stage"], "dry_run")
            self.assertEqual(result["device"], "cpu")
            self.assertEqual(result["batch_size"], 128)
            self.assertFalse(result["locked_test_access"])
            self.assertFalse(result["output_created"])
            self.assertTrue(result["no_model_forward"])
            self.assertFalse(result["training_performed"])
            self.assertFalse(output.exists())

    def test_direct_paths_reject_unsafe_runtime_values(self):
        namespace = runpy.run_path(str(TEACHER_DIAGNOSTIC_SCRIPT))
        base = dict(namespace["RUN_CONFIG"])
        cases = (
            ({"device": "cuda"}, ValueError, "CPU"),
            ({"batch_size": 4}, ValueError, "128"),
            ({"split": "development"}, PermissionError, "validation"),
            ({"allow_locked_test": True}, PermissionError, "locked_test"),
            ({"tau_current": 0.2}, ValueError, "tau_current"),
            ({"tau_counterfactual": 0.1}, ValueError, "tau_counterfactual"),
            ({"overwrite": True}, ValueError, "overwrite"),
        )
        for update, error_type, message in cases:
            with self.subTest(update=update):
                with self.assertRaisesRegex(error_type, message):
                    namespace["run_dry_run"]({**base, **update})

    def test_config_rejects_unknown_keys_and_non_frozen_batch(self):
        namespace = runpy.run_path(str(TEACHER_DIAGNOSTIC_SCRIPT))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"unexpected": True}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown config keys"):
                namespace["load_config"](path)

            path.write_text(json.dumps({"batch_size": 4}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "128"):
                namespace["load_config"](path)

    def test_formal_path_rejects_dry_run_and_existing_output_before_reads(self):
        namespace = runpy.run_path(str(TEACHER_DIAGNOSTIC_SCRIPT))
        base = dict(namespace["RUN_CONFIG"], stage="diagnose_validation_teacher")
        with self.assertRaisesRegex(ValueError, "dry_run"):
            namespace["run_diagnostic"](base)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                namespace["run_diagnostic"](
                    {**base, "dry_run": False, "output_root": str(output)}
                )

    def test_four_sample_smoke_uses_train_data_and_writes_six_files(self):
        namespace = runpy.run_path(str(TEACHER_DIAGNOSTIC_SCRIPT))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke"
            result = namespace["run_smoke"](
                {
                    **namespace["RUN_CONFIG"],
                    "stage": "smoke",
                    "sample_count": 4,
                    "output_root": str(output),
                }
            )

            self.assertEqual(result["sample_count"], 4)
            self.assertFalse(result["training_performed"])
            self.assertTrue(result["no_model_forward"])
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
            manifest = json.loads(
                (output / "source_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["no_model_forward"])
            self.assertFalse(manifest["training_performed"])
            self.assertNotIn("checkpoint_path", manifest)


class TeacherRankingEntrypointTest(unittest.TestCase):
    def test_default_dry_run_is_cpu_only_and_creates_nothing(self):
        namespace = runpy.run_path(str(TEACHER_RANKING_SCRIPT))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused"
            result = namespace["run_stage"](
                {**namespace["RUN_CONFIG"], "output_root": str(output)}
            )

            self.assertEqual(result["stage"], "dry_run")
            self.assertEqual(result["device"], "cpu")
            self.assertEqual(result["batch_size"], 128)
            self.assertTrue(result["no_model_forward"])
            self.assertFalse(result["teacher_modified"])
            self.assertFalse(result["training_performed"])
            self.assertFalse(output.exists())

    def test_unsafe_direct_values_and_unknown_json_keys_are_rejected(self):
        namespace = runpy.run_path(str(TEACHER_RANKING_SCRIPT))
        base = dict(namespace["RUN_CONFIG"])
        for update, error_type, message in (
            ({"device": "cuda"}, ValueError, "CPU"),
            ({"batch_size": 4}, ValueError, "128"),
            ({"split": "development"}, PermissionError, "validation"),
            ({"allow_locked_test": True}, PermissionError, "locked_test"),
            ({"overwrite": True}, ValueError, "overwrite"),
        ):
            with self.subTest(update=update):
                with self.assertRaisesRegex(error_type, message):
                    namespace["run_dry_run"]({**base, **update})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"unexpected": True}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown config keys"):
                namespace["load_config"](path)

    def test_formal_path_rejects_dry_run_and_existing_output_before_reads(self):
        namespace = runpy.run_path(str(TEACHER_RANKING_SCRIPT))
        base = dict(
            namespace["RUN_CONFIG"], stage="diagnose_validation_teacher_ranking"
        )
        with self.assertRaisesRegex(ValueError, "dry_run"):
            namespace["run_diagnostic"](base)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                namespace["run_diagnostic"](
                    {**base, "dry_run": False, "output_root": str(output)}
                )

    def test_four_sample_smoke_uses_train_data_and_writes_eight_files(self):
        namespace = runpy.run_path(str(TEACHER_RANKING_SCRIPT))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke"
            result = namespace["run_smoke"](
                {
                    **namespace["RUN_CONFIG"],
                    "stage": "smoke",
                    "sample_count": 4,
                    "output_root": str(output),
                }
            )

            self.assertEqual(result["sample_count"], 4)
            self.assertTrue(result["no_model_forward"])
            self.assertFalse(result["teacher_modified"])
            self.assertFalse(result["training_performed"])
            self.assertEqual(len(list(output.iterdir())), 8)
            manifest = json.loads(
                (output / "source_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["source"], "temporary-train-smoke")
            self.assertNotIn("checkpoint_path", manifest)


class ErrorTeacherCacheEntrypointTest(unittest.TestCase):
    def test_default_is_cpu_train_dry_run_and_creates_nothing(self):
        namespace = runpy.run_path(str(ERROR_TEACHER_CACHE_SCRIPT))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused"
            result = namespace["run_stage"](
                {**namespace["RUN_CONFIG"], "output_root": str(output)}
            )
            self.assertEqual(result["stage"], "dry_run")
            self.assertFalse(result["output_created"])
            self.assertTrue(result["train_only"])
            self.assertFalse(result["training_performed"])
            self.assertFalse(output.exists())

    def test_formal_guards_reject_unsafe_values_and_unknown_keys(self):
        namespace = runpy.run_path(str(ERROR_TEACHER_CACHE_SCRIPT))
        base = dict(
            namespace["RUN_CONFIG"],
            stage="build_train_teacher_cache",
            dry_run=False,
            sample_count=40_000,
        )
        for update, error_type in (
            ({"split": "validation"}, PermissionError),
            ({"device": "cuda"}, ValueError),
            ({"sample_count": 39_999}, ValueError),
            ({"batch_size": 64}, ValueError),
            ({"overwrite": True}, ValueError),
            ({"allow_locked_test": True}, PermissionError),
        ):
            with self.subTest(update=update), self.assertRaises(error_type):
                namespace["_validate_safe_config"]({**base, **update}, formal=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"unknown": True}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown config keys"):
                namespace["load_config"](path)

    def test_four_sample_smoke_writes_authenticated_three_file_cache(self):
        namespace = runpy.run_path(str(ERROR_TEACHER_CACHE_SCRIPT))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cache"
            result = namespace["run_stage"](
                {
                    **namespace["RUN_CONFIG"],
                    "stage": "smoke",
                    "sample_count": 4,
                    "output_root": str(output),
                }
            )
            loaded = load_teacher_cache(
                result["cache"], ExperimentConfig(), expected_count=4
            )
            self.assertEqual(len(loaded.labels_by_seed), 4)
            self.assertEqual(len(list(output.iterdir())), 3)


class SingleFactorAuditEntrypointTest(unittest.TestCase):
    def test_default_reads_nothing_and_creates_nothing(self):
        namespace = runpy.run_path(str(SINGLE_FACTOR_AUDIT_SCRIPT))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused"
            result = namespace["run_stage"](
                {**namespace["RUN_CONFIG"], "output_root": str(output)}
            )
            self.assertEqual(result["stage"], "dry_run")
            self.assertFalse(result["output_created"])
            self.assertFalse(result["training_performed"])
            self.assertFalse(output.exists())

    def test_formal_requires_all_sources_and_rejects_unsafe_values(self):
        namespace = runpy.run_path(str(SINGLE_FACTOR_AUDIT_SCRIPT))
        base = dict(
            namespace["RUN_CONFIG"], stage="audit_single_factor", dry_run=False
        )
        with self.assertRaises(ValueError):
            namespace["run_stage"](base)
        for update, error_type in (
            ({"device": "cuda"}, ValueError),
            ({"allow_locked_test": True}, PermissionError),
            ({"overwrite": True}, ValueError),
        ):
            with self.subTest(update=update), self.assertRaises(error_type):
                namespace["_validate_safe_config"]({**base, **update})

    def test_smoke_writes_three_file_nontraining_report(self):
        namespace = runpy.run_path(str(SINGLE_FACTOR_AUDIT_SCRIPT))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke"
            result = namespace["run_stage"](
                {
                    **namespace["RUN_CONFIG"],
                    "stage": "smoke",
                    "output_root": str(output),
                }
            )
            self.assertEqual(len(list(output.iterdir())), 3)
            self.assertFalse(result["training_performed"])


if __name__ == "__main__":
    unittest.main()
