"""Read-only identity audit for the Task 17 single-factor experiment."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from multisource_doa.config import ExperimentConfig, SplitName
from multisource_doa.data.manifest import build_split_manifest
from multisource_doa.training.teacher_cache import load_teacher_cache, sha256_file


TASK16_FILES = (
    "diagnostic_config.json",
    "source_manifest.json",
    "teacher_ranking_sample_diagnostics.csv",
    "teacher_ranking_summary.json",
    "teacher_component_summary.json",
    "teacher_ranking_stratified_summary.csv",
    "teacher_oracle_confusion.csv",
    "decision.json",
)
BASELINE_TRAINING_FILES = (
    "train_manifest.json",
    "validation_manifest.json",
    "metrics.csv",
    "best.pt",
    "best.pt.sha256.json",
)
EVALUATION_REPORT_FILES = (
    "run_config.json",
    "source_manifest.json",
    "predictions.csv",
    "summary.json",
    "paired_comparisons.csv",
    "failure_reasons.csv",
    "runtime_summary.json",
)
EXPECTED_PARAMETER_COUNT = 46_916
PHYSICAL_PATH_REGRESSION_VERSION = 1


@dataclass(frozen=True)
class SingleFactorAuditResult:
    baseline_reuse_allowed: bool
    gates: Mapping[str, bool]
    evidence: Mapping[str, Any]
    source_sha256: Mapping[str, Mapping[str, str]]


def _required_paths(directory: str | Path, filenames: tuple[str, ...]) -> dict[str, Path]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"source directory not found: {root}")
    paths = {name: root / name for name in filenames}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"source files are missing: {missing}")
    return paths


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in paths.items()}


def _without_training_metadata(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "training_metadata"}


def audit_single_factor_inputs(
    *,
    baseline_training_directory: str | Path,
    baseline_validation_directory: str | Path,
    task16_directory: str | Path,
    teacher_cache_directory: str | Path,
    experiment_config: ExperimentConfig,
    expected_cache_count: int,
) -> SingleFactorAuditResult:
    training_paths = _required_paths(
        baseline_training_directory, BASELINE_TRAINING_FILES
    )
    validation_paths = _required_paths(
        baseline_validation_directory, EVALUATION_REPORT_FILES
    )
    task16_paths = _required_paths(task16_directory, TASK16_FILES)
    task16_manifest = _json(task16_paths["source_manifest.json"])
    decision = _json(task16_paths["decision.json"])
    if task16_manifest.get("teacher_ranking_schema_version") != 1:
        raise ValueError("Task 16 schema mismatch")
    if (
        decision.get("mechanism_conclusion") != "ranking_invalid"
        or decision.get("training_authorized") is not False
    ):
        raise ValueError("Task 16 must conclude ranking_invalid without training authority")

    cache = load_teacher_cache(
        teacher_cache_directory,
        experiment_config,
        expected_count=expected_cache_count,
        regenerate_metadata=True,
    )
    checkpoint = training_paths["best.pt"]
    checkpoint_sha = sha256_file(checkpoint)
    sidecar = _json(training_paths["best.pt.sha256.json"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    train_manifest = _json(training_paths["train_manifest.json"])
    validation_manifest = _json(training_paths["validation_manifest.json"])
    report_manifest = _json(validation_paths["source_manifest.json"])
    report_summary = _json(validation_paths["summary.json"])
    metadata = payload.get("training_metadata")
    environment_expected = {
        "device": "cpu",
        "batch_size": 128,
        "shuffle": True,
        "total_epochs": 50,
        "learning_rate": 1e-3,
        "physical_path_regression_version": PHYSICAL_PATH_REGRESSION_VERSION,
    }
    current_physical_metadata = {
        **environment_expected,
        "teacher_mode": "physical",
        "scale_distillation_target_source": "physical_music_score",
        "dominance_target_source": "physical_music_score",
        "teacher_cache_sha256": None,
        "single_factor_audit_sha256": None,
        "teacher_label_counts": None,
    }
    manifest_metadata = (
        train_manifest.get("training_metadata"),
        validation_manifest.get("training_metadata"),
        report_manifest.get("training_metadata"),
    )
    gates = {
        "task16_ranking_invalid": True,
        "same_data_identity": (
            _without_training_metadata(train_manifest)
            == build_split_manifest(experiment_config, SplitName.TRAIN)
            and _without_training_metadata(validation_manifest)
            == build_split_manifest(experiment_config, SplitName.VALIDATION)
        ),
        "same_manifest_training_metadata": (
            manifest_metadata == (None, None, None)
            or manifest_metadata == (metadata, metadata, metadata)
        ),
        "same_model_seed": payload.get("model_seed") == 2026,
        "same_experiment_config": payload.get("experiment_config")
        == asdict(experiment_config),
        "same_parameter_count": payload.get("parameter_count")
        == EXPECTED_PARAMETER_COUNT,
        "same_checkpoint_rule": payload.get("selection_metric_name")
        == "failure_aware_rmspe_deg",
        "same_checkpoint_identity": (
            sidecar.get("checkpoint_sha256") == checkpoint_sha
            and report_manifest.get("checkpoint_sha") == checkpoint_sha
        ),
        "validation_report_identity": (
            report_summary.get("report_schema_version") == 2
            and report_summary.get("split") == SplitName.VALIDATION.value
        ),
        "teacher_cache_valid": cache.manifest.get("train_only") is True,
        "same_training_environment": metadata
        in (environment_expected, current_physical_metadata),
    }
    allowed = all(gates.values())
    source_sha = {
        "baseline_training": _hashes(training_paths),
        "baseline_validation": _hashes(validation_paths),
        "task16": _hashes(task16_paths),
        "teacher_cache": dict(cache.file_sha256),
    }
    return SingleFactorAuditResult(
        baseline_reuse_allowed=allowed,
        gates=gates,
        evidence={
            "required_action": (
                "reuse_baseline" if allowed else "rerun_physical_control"
            ),
            "baseline_code_sha": payload.get("code_sha"),
            "baseline_epoch": payload.get("epoch"),
            "baseline_selection_metric": payload.get("selection_metric_value"),
            "teacher_cache_csv_sha256": cache.manifest.get("csv_sha256"),
            "task16_mechanism_conclusion": decision.get("mechanism_conclusion"),
            "training_authorized": False,
        },
        source_sha256=source_sha,
    )
