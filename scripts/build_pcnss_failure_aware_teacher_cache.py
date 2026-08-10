"""Build the frozen train-only failure-aware scale-teacher cache."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multisource_doa.config import ExperimentConfig, SplitName
from multisource_doa.data.dataset import PCNSSDataset
from multisource_doa.training.error_teacher import build_error_teacher_row
from multisource_doa.training.teacher_cache import sha256_file, write_teacher_cache


RUN_CONFIG = {
    "stage": "dry_run",
    "dry_run": True,
    "split": "train",
    "output_root": "outputs/pcnss_failure_aware_teacher_cache",
    "device": "cpu",
    "batch_size": 128,
    "sample_count": 1,
    "allow_locked_test": False,
    "overwrite": False,
}
STAGES = ("dry_run", "smoke", "build_train_teacher_cache")
FROZEN_BATCH_SIZE = 128
FORMAL_SAMPLE_COUNT = 40_000
SOURCE_PATHS = (
    "multisource_doa/baselines/classical.py",
    "multisource_doa/evaluation/matching.py",
    "multisource_doa/evaluation/metrics.py",
    "multisource_doa/training/error_teacher.py",
    "multisource_doa/training/teacher_cache.py",
)


def validate_stage(stage: Any) -> str:
    if not isinstance(stage, str) or stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    return stage


def _validate_safe_config(values: dict[str, Any], *, formal: bool) -> None:
    if values.get("split") != SplitName.TRAIN.value:
        raise PermissionError("teacher cache accepts train split only")
    if values.get("device") != "cpu":
        raise ValueError("teacher cache generation is CPU only")
    if values.get("batch_size") != FROZEN_BATCH_SIZE:
        raise ValueError(f"batch_size must be fixed at {FROZEN_BATCH_SIZE}")
    if values.get("allow_locked_test") is not False:
        raise PermissionError("locked_test access is forbidden")
    if values.get("overwrite") is not False:
        raise ValueError("overwrite must remain false")
    if formal and values.get("sample_count") != FORMAL_SAMPLE_COUNT:
        raise ValueError(f"formal sample_count must be {FORMAL_SAMPLE_COUNT}")


def _code_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_sha256() -> dict[str, str]:
    return {path: sha256_file(PROJECT_ROOT / path) for path in SOURCE_PATHS}


def _rows(sample_count: int) -> list[dict[str, Any]]:
    config = ExperimentConfig()
    dataset = PCNSSDataset(SplitName.TRAIN, config)
    return [
        build_error_teacher_row(dataset[index], sample_index=index)
        for index in range(sample_count)
    ]


def run_dry_run(values: dict[str, Any]) -> dict[str, Any]:
    _validate_safe_config(values, formal=False)
    row = _rows(1)[0]
    return {
        "stage": "dry_run",
        "sample_count": 1,
        "sample_seed": row["sample_seed"],
        "teacher_probabilities": row["teacher_probabilities"],
        "train_only": True,
        "no_model_forward": True,
        "training_performed": False,
        "locked_test_access": False,
        "output_created": False,
    }


def run_smoke(values: dict[str, Any]) -> dict[str, Any]:
    _validate_safe_config(values, formal=False)
    if values.get("sample_count") != 4:
        raise ValueError("teacher cache smoke sample_count must be fixed at 4")
    output = Path(str(values["output_root"]))
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    report = write_teacher_cache(
        _rows(4),
        output,
        experiment_config=ExperimentConfig(),
        run_config=values,
        code_sha=_code_sha(),
        source_sha256=_source_sha256(),
        expected_count=4,
    )
    return {
        "stage": "smoke",
        "sample_count": 4,
        "cache": str(report),
        "train_only": True,
        "no_model_forward": True,
        "training_performed": False,
    }


def run_formal_cache(values: dict[str, Any]) -> dict[str, Any]:
    _validate_safe_config(values, formal=True)
    if values.get("dry_run") is not False:
        raise ValueError("正式 cache 前必须把 dry_run 改为 False")
    output = Path(str(values["output_root"]))
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    report = write_teacher_cache(
        _rows(FORMAL_SAMPLE_COUNT),
        output,
        experiment_config=ExperimentConfig(),
        run_config=values,
        code_sha=_code_sha(),
        source_sha256=_source_sha256(),
        expected_count=FORMAL_SAMPLE_COUNT,
    )
    return {
        "stage": "build_train_teacher_cache",
        "sample_count": FORMAL_SAMPLE_COUNT,
        "cache": str(report),
        "train_only": True,
        "no_model_forward": True,
        "training_performed": False,
    }


def run_stage(values: dict[str, Any]) -> dict[str, Any]:
    stage = validate_stage(values.get("stage"))
    if stage == "dry_run":
        return run_dry_run(values)
    if stage == "smoke":
        return run_smoke(values)
    return run_formal_cache(values)


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return dict(RUN_CONFIG)
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("config JSON must contain an object")
    unknown = set(loaded) - set(RUN_CONFIG)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    values = {**RUN_CONFIG, **loaded}
    stage = validate_stage(values.get("stage"))
    _validate_safe_config(values, formal=stage == "build_train_teacher_cache")
    return values


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="path to a complete or partial JSON config")
    arguments = parser.parse_args(argv)
    result = run_stage(load_config(arguments.config))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
