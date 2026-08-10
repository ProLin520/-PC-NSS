import json
import tempfile
import unittest
from pathlib import Path

import torch

from multisource_doa.config import ExperimentConfig, SplitName
from multisource_doa.models.pc_nss import MultiScalePCNSS
from multisource_doa.training.artifacts import (
    CheckpointManager,
    prepare_run_directory,
)


class ArtifactAuditTest(unittest.TestCase):
    def test_prepare_run_directory_refuses_existing_target(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "run"
            target.mkdir()

            with self.assertRaises(FileExistsError):
                prepare_run_directory(target, refuse_overwrite=True)

    def test_checkpoint_updates_only_for_strictly_better_validation_rmspe(self):
        with tempfile.TemporaryDirectory() as directory:
            model = MultiScalePCNSS()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            manager = CheckpointManager(Path(directory))
            common = {
                "model": model,
                "optimizer": optimizer,
                "experiment_config": ExperimentConfig(),
                "model_seed": 2026,
                "data_split_seed": 202_708_040,
                "code_sha": "abc123",
                "split": SplitName.VALIDATION,
            }

            self.assertTrue(manager.update(metric_value=2.0, epoch=0, **common))
            self.assertFalse(manager.update(metric_value=2.0, epoch=1, **common))
            self.assertFalse(manager.update(metric_value=3.0, epoch=2, **common))
            self.assertTrue(manager.update(metric_value=1.5, epoch=3, **common))

            checkpoint = Path(directory) / "best.pt"
            sidecar = Path(directory) / "best.pt.sha256.json"
            self.assertTrue(checkpoint.is_file())
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(payload["selection_metric_name"], "failure_aware_rmspe_deg")
            self.assertEqual(payload["selection_metric_value"], 1.5)
            self.assertEqual(payload["epoch"], 3)
            audit = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(len(audit["checkpoint_sha256"]), 64)

    def test_checkpoint_selection_rejects_non_validation_split(self):
        with tempfile.TemporaryDirectory() as directory:
            model = MultiScalePCNSS()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            manager = CheckpointManager(Path(directory))

            with self.assertRaises(PermissionError):
                manager.update(
                    metric_value=1.0,
                    epoch=0,
                    model=model,
                    optimizer=optimizer,
                    experiment_config=ExperimentConfig(),
                    model_seed=2026,
                    data_split_seed=0,
                    code_sha="abc123",
                    split=SplitName.LOCKED_TEST,
                )

    def test_checkpoint_optionally_records_teacher_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            model = MultiScalePCNSS()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            metadata = {
                "teacher_mode": "failure_aware_error",
                "scale_distillation_target_source": "train_only_failure_aware_rmspe",
                "dominance_target_source": "physical_music_score",
                "teacher_cache_sha256": "a" * 64,
                "single_factor_audit_sha256": "b" * 64,
            }
            manager = CheckpointManager(Path(directory))
            manager.update(
                metric_value=1.0,
                epoch=0,
                model=model,
                optimizer=optimizer,
                experiment_config=ExperimentConfig(),
                model_seed=2026,
                data_split_seed=202_708_040,
                code_sha="abc123",
                split=SplitName.VALIDATION,
                training_metadata=metadata,
            )
            payload = torch.load(
                Path(directory) / "best.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(payload["training_metadata"], metadata)


if __name__ == "__main__":
    unittest.main()
