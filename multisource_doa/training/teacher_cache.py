"""Immutable three-file cache for train-only angular-error teacher labels."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from multisource_doa.config import ExperimentConfig, SplitName
from multisource_doa.data.dataset import PCNSSDataset
from multisource_doa.data.manifest import build_split_manifest
from multisource_doa.training.error_teacher import (
    ERROR_TIE_TOLERANCE_DEG,
    SCALE_SIZES,
    teacher_probabilities_from_rmspe,
)


TEACHER_CACHE_SCHEMA_VERSION = 1
CACHE_FILENAMES = (
    "teacher_cache_config.json",
    "teacher_cache_manifest.json",
    "train_teacher_labels.csv",
)


@dataclass(frozen=True)
class TeacherCache:
    labels_by_seed: Mapping[int, tuple[float, float, float, float]]
    manifest: Mapping[str, Any]
    file_sha256: Mapping[str, str]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path.name}")
    return payload


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_row(row: Mapping[str, Any], index: int, train_seed: int) -> None:
    if int(row.get("sample_index", -1)) != index:
        raise ValueError("teacher cache sample_index must be continuous")
    if int(row.get("sample_seed", -1)) != train_seed + index:
        raise ValueError("teacher cache sample_seed must follow the train split")
    rmspe = {
        size: _finite(row.get(f"sample_rmspe_deg_L{size}"), f"RMSPE L{size}")
        for size in SCALE_SIZES
    }
    for value in rmspe.values():
        if value < 0.0:
            raise ValueError("teacher cache RMSPE must be non-negative")
    expected = teacher_probabilities_from_rmspe(rmspe)
    probabilities = tuple(
        _finite(value, "teacher probability")
        for value in row.get("teacher_probabilities", ())
    )
    if len(probabilities) != len(SCALE_SIZES) or not np.allclose(
        probabilities, expected, atol=1e-12, rtol=0.0
    ):
        raise ValueError("teacher probabilities do not match fixed-scale RMSPE")
    for size in SCALE_SIZES:
        if not isinstance(row.get(f"success_L{size}"), bool):
            raise ValueError(f"success_L{size} must be boolean")
        if not isinstance(row.get(f"failure_reason_L{size}"), str):
            raise ValueError(f"failure_reason_L{size} must be a string")
        for angle_index in (1, 2):
            _finite(
                row.get(f"absolute_error_{angle_index}_deg_L{size}"),
                f"absolute error L{size}",
            )
            estimate = row.get(f"estimated_angle_{angle_index}_deg_L{size}")
            if estimate is not None:
                _finite(estimate, f"estimated angle L{size}")


def _flatten_row(row: Mapping[str, Any]) -> dict[str, Any]:
    flattened = {
        key: value
        for key, value in row.items()
        if key not in ("teacher_probabilities", "best_scales")
    }
    probabilities = row["teacher_probabilities"]
    for size, probability in zip(SCALE_SIZES, probabilities, strict=True):
        flattened[f"teacher_p_L{size}"] = float(probability)
    flattened["best_scales"] = json.dumps(list(row["best_scales"]))
    return flattened


def _label_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    dominant = Counter()
    tied = 0
    failed = 0
    for row in rows:
        winners = tuple(row["best_scales"])
        if len(winners) == 1:
            dominant[str(winners[0])] += 1
        if bool(row["has_tied_best"]):
            tied += 1
        if bool(row["all_scales_failed"]):
            failed += 1
    return {
        "unique_best_by_scale": {
            str(size): int(dominant.get(str(size), 0)) for size in SCALE_SIZES
        },
        "tied_best_count": tied,
        "all_scales_failed_count": failed,
    }


def write_teacher_cache(
    rows: Sequence[Mapping[str, Any]],
    output_directory: str | Path,
    *,
    experiment_config: ExperimentConfig,
    run_config: Mapping[str, Any],
    code_sha: str,
    source_sha256: Mapping[str, str],
    expected_count: int,
) -> Path:
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    items = [dict(row) for row in rows]
    if len(items) != expected_count:
        raise ValueError("teacher cache sample count mismatch")
    train_seed = int(experiment_config.split.seeds[SplitName.TRAIN])
    for index, row in enumerate(items):
        _validate_row(row, index, train_seed)
    if not code_sha:
        raise ValueError("code_sha is required")
    for name, digest in source_sha256.items():
        if not name or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("source SHA-256 entries must be lowercase hexadecimal")

    flattened = [_flatten_row(row) for row in items]
    fieldnames = list(flattened[0]) if flattened else []
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / "teacher_cache_config.json"
    csv_path = output / "train_teacher_labels.csv"
    manifest_path = output / "teacher_cache_manifest.json"
    _write_json(config_path, dict(run_config))
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(flattened)
    split_manifest = build_split_manifest(experiment_config, SplitName.TRAIN)
    manifest = {
        "teacher_cache_schema_version": TEACHER_CACHE_SCHEMA_VERSION,
        "algorithm": "failure_aware_fixed_scale_rmspe_hard_teacher",
        "split": SplitName.TRAIN.value,
        "sample_count": len(items),
        "sample_seed_start": train_seed,
        "sample_seed_end": train_seed + len(items) - 1,
        "batch_size": int(run_config.get("batch_size", 128)),
        "device": str(run_config.get("device", "cpu")),
        "code_sha": str(code_sha),
        "source_sha256": dict(source_sha256),
        "config_sha256": sha256_file(config_path),
        "csv_sha256": sha256_file(csv_path),
        "split_manifest": split_manifest,
        "failure_penalty_deg": 60.0,
        "tie_tolerance_deg": ERROR_TIE_TOLERANCE_DEG,
        "scale_sizes": list(SCALE_SIZES),
        "label_counts": _label_counts(items),
        "train_only": True,
        "no_model_forward": True,
        "training_performed": False,
        "validation_accessed": False,
        "development_accessed": False,
        "locked_test_accessed": False,
    }
    _write_json(manifest_path, manifest)
    return output


def _parse_bool(value: str, name: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{name} must be True or False")


def _parse_row(raw: Mapping[str, str]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample_index": int(raw["sample_index"]),
        "sample_seed": int(raw["sample_seed"]),
        "true_angle_1_deg": _finite(raw["true_angle_1_deg"], "true angle"),
        "true_angle_2_deg": _finite(raw["true_angle_2_deg"], "true angle"),
        "separation_deg": _finite(raw["separation_deg"], "separation"),
        "rho": _finite(raw["rho"], "rho"),
        "snr_db": _finite(raw["snr_db"], "snr_db"),
        "snapshot_count": int(raw["snapshot_count"]),
        "teacher_probabilities": tuple(
            _finite(raw[f"teacher_p_L{size}"], "teacher probability")
            for size in SCALE_SIZES
        ),
        "best_scales": tuple(int(value) for value in json.loads(raw["best_scales"])),
        "has_tied_best": _parse_bool(raw["has_tied_best"], "has_tied_best"),
        "all_scales_failed": _parse_bool(
            raw["all_scales_failed"], "all_scales_failed"
        ),
    }
    for size in SCALE_SIZES:
        row[f"success_L{size}"] = _parse_bool(raw[f"success_L{size}"], "success")
        row[f"failure_reason_L{size}"] = raw[f"failure_reason_L{size}"]
        for angle_index in (1, 2):
            estimate = raw[f"estimated_angle_{angle_index}_deg_L{size}"]
            row[f"estimated_angle_{angle_index}_deg_L{size}"] = (
                None if estimate == "" else _finite(estimate, "estimated angle")
            )
            row[f"absolute_error_{angle_index}_deg_L{size}"] = _finite(
                raw[f"absolute_error_{angle_index}_deg_L{size}"], "absolute error"
            )
        row[f"sample_rmspe_deg_L{size}"] = _finite(
            raw[f"sample_rmspe_deg_L{size}"], "sample RMSPE"
        )
    return row


def _validate_metadata(row: Mapping[str, Any], sample: Any) -> None:
    expected = {
        "true_angle_1_deg": float(sample.angles_deg[0]),
        "true_angle_2_deg": float(sample.angles_deg[1]),
        "separation_deg": float(abs(np.diff(sample.angles_deg)[0])),
        "rho": float(sample.rho),
        "snr_db": float(sample.snr_db),
        "snapshot_count": int(sample.snapshot_count),
    }
    for key, value in expected.items():
        actual = row[key]
        if isinstance(value, int):
            matches = actual == value
        else:
            matches = bool(np.isclose(actual, value, atol=1e-7, rtol=1e-6))
        if not matches:
            raise ValueError(f"teacher cache metadata mismatch: {key}")


def load_teacher_cache(
    directory: str | Path,
    experiment_config: ExperimentConfig,
    *,
    expected_count: int,
    regenerate_metadata: bool = True,
    expected_source_sha256: Mapping[str, str] | None = None,
) -> TeacherCache:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"teacher cache directory not found: {root}")
    if {path.name for path in root.iterdir()} != set(CACHE_FILENAMES):
        raise ValueError("teacher cache must contain exactly three files")
    paths = {name: root / name for name in CACHE_FILENAMES}
    manifest = _read_json(paths["teacher_cache_manifest.json"])
    if manifest.get("teacher_cache_schema_version") != TEACHER_CACHE_SCHEMA_VERSION:
        raise ValueError("teacher cache schema mismatch")
    if manifest.get("split") != SplitName.TRAIN.value or manifest.get("train_only") is not True:
        raise PermissionError("teacher cache must be train-only")
    if manifest.get("sample_count") != expected_count:
        raise ValueError("teacher cache sample count mismatch")
    if manifest.get("split_manifest") != build_split_manifest(
        experiment_config, SplitName.TRAIN
    ):
        raise ValueError("teacher cache experiment config mismatch")
    if manifest.get("config_sha256") != sha256_file(paths["teacher_cache_config.json"]):
        raise ValueError("teacher cache config SHA mismatch")
    if manifest.get("csv_sha256") != sha256_file(paths["train_teacher_labels.csv"]):
        raise ValueError("teacher cache CSV SHA mismatch")
    if expected_source_sha256 is not None and manifest.get("source_sha256") != dict(
        expected_source_sha256
    ):
        raise ValueError("teacher cache source SHA mismatch")
    for key, expected in (
        ("training_performed", False),
        ("validation_accessed", False),
        ("development_accessed", False),
        ("locked_test_accessed", False),
    ):
        if manifest.get(key) is not expected:
            raise ValueError(f"teacher cache {key} mismatch")

    with paths["train_teacher_labels.csv"].open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        raw_rows = list(csv.DictReader(handle))
    if len(raw_rows) != expected_count:
        raise ValueError("teacher cache CSV row count mismatch")
    rows = [_parse_row(raw) for raw in raw_rows]
    train_seed = int(experiment_config.split.seeds[SplitName.TRAIN])
    dataset = PCNSSDataset(SplitName.TRAIN, experiment_config)
    labels: dict[int, tuple[float, float, float, float]] = {}
    for index, row in enumerate(rows):
        _validate_row(row, index, train_seed)
        seed = int(row["sample_seed"])
        if seed in labels:
            raise ValueError("duplicate sample_seed in teacher cache")
        if regenerate_metadata:
            _validate_metadata(row, dataset[index])
        labels[seed] = tuple(row["teacher_probabilities"])
    if manifest.get("label_counts") != _label_counts(rows):
        raise ValueError("teacher cache label counts mismatch")
    return TeacherCache(
        labels_by_seed=labels,
        manifest=manifest,
        file_sha256={name: sha256_file(path) for name, path in paths.items()},
    )
