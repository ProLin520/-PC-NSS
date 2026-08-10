"""Audit Task 17 inputs before either single-factor training arm may run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multisource_doa.config import ExperimentConfig
from multisource_doa.training.single_factor_audit import (
    SingleFactorAuditResult,
    audit_single_factor_inputs,
)
from multisource_doa.training.single_factor_reporting import (
    write_single_factor_audit_report,
)


RUN_CONFIG = {
    "stage": "dry_run",
    "dry_run": True,
    "baseline_training_directory": "",
    "baseline_validation_directory": "",
    "task16_directory": "",
    "teacher_cache_directory": "",
    "output_root": "outputs/pcnss_teacher_single_factor_audit",
    "device": "cpu",
    "allow_locked_test": False,
    "overwrite": False,
}
STAGES = ("dry_run", "smoke", "audit_single_factor")
FORMAL_CACHE_COUNT = 40_000


def validate_stage(stage: Any) -> str:
    if not isinstance(stage, str) or stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    return stage


def _validate_safe_config(values: dict[str, Any]) -> None:
    if values.get("device") != "cpu":
        raise ValueError("single-factor audit is CPU only")
    if values.get("allow_locked_test") is not False:
        raise PermissionError("locked_test access is forbidden")
    if values.get("overwrite") is not False:
        raise ValueError("overwrite must remain false")


def run_dry_run(values: dict[str, Any]) -> dict[str, Any]:
    _validate_safe_config(values)
    return {
        "stage": "dry_run",
        "output_created": False,
        "no_model_forward": True,
        "training_performed": False,
        "locked_test_accessed": False,
    }


def run_smoke(values: dict[str, Any]) -> dict[str, Any]:
    _validate_safe_config(values)
    synthetic = SingleFactorAuditResult(
        baseline_reuse_allowed=True,
        gates={"synthetic_integrity": True},
        evidence={
            "required_action": "reuse_baseline",
            "training_authorized": False,
        },
        source_sha256={"synthetic": {"source": "a" * 64}},
    )
    output = write_single_factor_audit_report(
        synthetic,
        values["output_root"],
        run_config=values,
    )
    return {
        "stage": "smoke",
        "report": str(output),
        "baseline_reuse_allowed": True,
        "no_model_forward": True,
        "training_performed": False,
    }


def run_formal_audit(values: dict[str, Any]) -> dict[str, Any]:
    _validate_safe_config(values)
    if values.get("dry_run") is not False:
        raise ValueError("正式审计前必须把 dry_run 改为 False")
    source_keys = (
        "baseline_training_directory",
        "baseline_validation_directory",
        "task16_directory",
        "teacher_cache_directory",
    )
    missing = [key for key in source_keys if not str(values.get(key, "")).strip()]
    if missing:
        raise ValueError(f"formal audit requires source directories: {missing}")
    output = Path(str(values["output_root"]))
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    result = audit_single_factor_inputs(
        baseline_training_directory=values["baseline_training_directory"],
        baseline_validation_directory=values["baseline_validation_directory"],
        task16_directory=values["task16_directory"],
        teacher_cache_directory=values["teacher_cache_directory"],
        experiment_config=ExperimentConfig(),
        expected_cache_count=FORMAL_CACHE_COUNT,
    )
    report = write_single_factor_audit_report(
        result,
        output,
        run_config=values,
    )
    return {
        "stage": "audit_single_factor",
        "report": str(report),
        "baseline_reuse_allowed": result.baseline_reuse_allowed,
        "required_action": result.evidence["required_action"],
        "no_model_forward": True,
        "training_performed": False,
    }


def run_stage(values: dict[str, Any]) -> dict[str, Any]:
    stage = validate_stage(values.get("stage"))
    if stage == "dry_run":
        return run_dry_run(values)
    if stage == "smoke":
        return run_smoke(values)
    return run_formal_audit(values)


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
    validate_stage(values.get("stage"))
    _validate_safe_config(values)
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
