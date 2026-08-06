import json
import runpy
import tempfile
import unittest
from pathlib import Path

import torch

from multisource_doa.config import ExperimentConfig, SplitName
from multisource_doa.data.simulator import generate_two_source_sample
from multisource_doa.diagnostics.near_resolution import (
    NearAuditLabel,
    diagnose_near_samples,
)
from multisource_doa.diagnostics.reporting import write_near_diagnostic_report
from multisource_doa.models.pc_nss import MultiScalePCNSS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = PROJECT_ROOT / "scripts" / "run_multiscale_pcnss.py"
DIAGNOSTIC_SCRIPT = PROJECT_ROOT / "scripts" / "diagnose_pcnss_near_resolution.py"


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
            self.assertEqual(namespace["load_config"](path)["batch_size"], 4)

            path.write_text(json.dumps({"unexpected": True}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown config keys"):
                namespace["load_config"](path)

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
                    "success": True,
                },
                fbss_l7_row={
                    "absolute_error_1_deg": 0.6,
                    "absolute_error_2_deg": 0.8,
                    "success": True,
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
        self.assertNotIn("evaluate_locked_test", namespace["STAGES"])

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


if __name__ == "__main__":
    unittest.main()
