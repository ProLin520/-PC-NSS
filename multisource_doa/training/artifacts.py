"""Refuse-overwrite run directories and SHA-audited checkpoints."""

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from multisource_doa.config import ExperimentConfig, SplitName


def prepare_run_directory(
    path: str | Path,
    *,
    refuse_overwrite: bool = True,
) -> Path:
    destination = Path(path)
    if destination.exists() and refuse_overwrite:
        raise FileExistsError(f"refusing to overwrite output directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CheckpointManager:
    def __init__(
        self,
        output_directory: str | Path,
        selection_metric_name: str = "failure_aware_rmspe_deg",
    ):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.selection_metric_name = selection_metric_name
        self.best_metric = math.inf

    def update(
        self,
        *,
        metric_value: float,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        experiment_config: ExperimentConfig,
        model_seed: int,
        data_split_seed: int,
        code_sha: str,
        split: SplitName,
        training_metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        if SplitName(split) is not SplitName.VALIDATION:
            raise PermissionError("checkpoint selection is restricted to validation")
        if not math.isfinite(metric_value):
            raise ValueError("checkpoint metric must be finite")
        if metric_value >= self.best_metric:
            return False
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "selection_metric_name": self.selection_metric_name,
            "selection_metric_value": float(metric_value),
            "experiment_config": asdict(experiment_config),
            "model_seed": int(model_seed),
            "data_split_seed": int(data_split_seed),
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "code_sha": str(code_sha),
        }
        if training_metadata is not None:
            payload["training_metadata"] = dict(training_metadata)
        destination = self.output_directory / "best.pt"
        temporary = self.output_directory / "best.pt.tmp"
        torch.save(payload, temporary)
        temporary.replace(destination)
        checkpoint_sha = _sha256(destination)
        sidecar = self.output_directory / "best.pt.sha256.json"
        sidecar.write_text(
            json.dumps(
                {
                    "checkpoint": destination.name,
                    "checkpoint_sha256": checkpoint_sha,
                    "selection_metric_name": self.selection_metric_name,
                    "selection_metric_value": float(metric_value),
                    "epoch": int(epoch),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        self.best_metric = float(metric_value)
        return True
