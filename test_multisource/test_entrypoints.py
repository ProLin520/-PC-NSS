import runpy
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = PROJECT_ROOT / "scripts" / "run_multiscale_pcnss.py"


class EntrypointTest(unittest.TestCase):
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
