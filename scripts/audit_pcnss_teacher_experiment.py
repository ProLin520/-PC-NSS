"""Audit the frozen seed-2026 Task 17 validation comparison without inference."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multisource_doa.config import ExperimentConfig
from multisource_doa.evaluation.teacher_experiment import (
    TeacherExperimentResult,
    audit_teacher_experiment,
)
from multisource_doa.evaluation.teacher_experiment_reporting import (
    write_teacher_experiment_report,
)
from multisource_doa.training.single_factor_reporting import (
    SINGLE_FACTOR_AUDIT_SCHEMA_VERSION,
)
from multisource_doa.training.teacher_cache import load_teacher_cache, sha256_file


RUN_CONFIG = {
    "stage": "dry_run",
    "dry_run": True,
    "baseline_validation_directory": "",
    "candidate_validation_directory": "",
    "single_factor_audit_directory": "",
    "teacher_cache_directory": "",
    "output_root": "outputs/pcnss_teacher_experiment_audit",
    "split": "validation",
    "device": "cpu",
    "allow_locked_test": False,
    "overwrite": False,
}
STAGES = ("dry_run", "smoke", "audit_validation_teacher_experiment")


def _validate(values: dict[str, Any]) -> None:
    if values.get("split") != "validation":
        raise PermissionError("experiment audit accepts validation only")
    if values.get("device") != "cpu":
        raise ValueError("experiment audit is CPU only")
    if values.get("allow_locked_test") is not False:
        raise PermissionError("locked_test access is forbidden")
    if values.get("overwrite") is not False:
        raise ValueError("overwrite must remain false")


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def run_dry_run(values: dict[str, Any]) -> dict[str, Any]:
    _validate(values)
    return {"stage": "dry_run", "output_created": False, "no_model_forward": True, "evaluation_performed": False, "training_performed": False, "locked_test_accessed": False}


def run_smoke(values: dict[str, Any]) -> dict[str, Any]:
    _validate(values)
    gates = {
        "near_resolution_improves_over_original": True,
        "near_resolution_not_below_fbss_L7": True,
        "overall_rmspe_not_worse": True,
        "overall_resolution_not_worse": True,
        "failure_count_not_worse": True,
        "protocol_identity": True,
    }
    result = TeacherExperimentResult(
        decision={"gates": gates, "conclusion": "seed2026_gate_passed", "development_authorized": False, "multi_seed_authorized": False, "locked_test_authorized": False, "required_action": "request_development_approval"},
        paired_rows=({"row_type": "paired", "sample_seed": 1, "outcome": "win", "transition": "A0_B1"},),
        transition_rows=({"row_type": "transition", "transition": "A0_B1", "count": 1},),
        stratified_rows=({"stratum": "separation_deg", "bin": "[2,4)", "count": 1},),
        source_sha256={"synthetic": {"source": "a" * 64}},
    )
    output = write_teacher_experiment_report(result, values["output_root"], run_config=values)
    return {"stage": "smoke", "report": str(output), "no_model_forward": True, "evaluation_performed": False, "training_performed": False}


def _authenticate_external_identity(values: dict[str, Any]) -> dict[str, dict[str, str]]:
    cache = load_teacher_cache(values["teacher_cache_directory"], ExperimentConfig(), expected_count=40_000, regenerate_metadata=True)
    audit = Path(values["single_factor_audit_directory"])
    expected = {"audit_config.json", "source_manifest.json", "single_factor_audit.json"}
    if not audit.is_dir() or {path.name for path in audit.iterdir()} != expected:
        raise ValueError("single-factor audit must contain exactly three files")
    manifest = _json(audit / "source_manifest.json")
    decision = _json(audit / "single_factor_audit.json")
    if manifest.get("single_factor_audit_schema_version") != SINGLE_FACTOR_AUDIT_SCHEMA_VERSION or decision.get("baseline_reuse_allowed") is not True:
        raise PermissionError("passing single-factor audit is required")
    if manifest.get("source_sha256", {}).get("teacher_cache") != dict(cache.file_sha256):
        raise ValueError("single-factor audit cache identity mismatch")
    candidate_manifest = _json(Path(values["candidate_validation_directory"]) / "source_manifest.json")
    metadata = candidate_manifest.get("training_metadata", {})
    if metadata.get("teacher_cache_sha256") != cache.file_sha256["teacher_cache_manifest.json"]:
        raise ValueError("candidate cache SHA mismatch")
    if metadata.get("single_factor_audit_sha256") != sha256_file(audit / "single_factor_audit.json"):
        raise ValueError("candidate single-factor audit SHA mismatch")
    return {
        "teacher_cache": dict(cache.file_sha256),
        "single_factor_audit": {
            name: sha256_file(audit / name) for name in sorted(expected)
        },
    }


def run_formal(values: dict[str, Any]) -> dict[str, Any]:
    _validate(values)
    if values.get("dry_run") is not False:
        raise ValueError("正式结果审计前必须把 dry_run 改为 False")
    required = ("baseline_validation_directory", "candidate_validation_directory", "single_factor_audit_directory", "teacher_cache_directory")
    missing = [key for key in required if not str(values.get(key, "")).strip()]
    if missing:
        raise ValueError(f"formal audit requires inputs: {missing}")
    output = Path(values["output_root"])
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    external_sha = _authenticate_external_identity(values)
    result = audit_teacher_experiment(values["baseline_validation_directory"], values["candidate_validation_directory"], expected_sample_count=5_000)
    result = replace(
        result,
        source_sha256={**result.source_sha256, **external_sha},
    )
    report = write_teacher_experiment_report(result, output, run_config=values)
    return {"stage": "audit_validation_teacher_experiment", "report": str(report), "conclusion": result.decision["conclusion"], "development_authorized": False, "no_model_forward": True, "evaluation_performed": False, "training_performed": False}


def run_stage(values: dict[str, Any]) -> dict[str, Any]:
    stage = values.get("stage")
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    if stage == "dry_run": return run_dry_run(values)
    if stage == "smoke": return run_smoke(values)
    return run_formal(values)


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None: return dict(RUN_CONFIG)
    loaded = _json(Path(path))
    unknown = set(loaded) - set(RUN_CONFIG)
    if unknown: raise ValueError(f"unknown config keys: {sorted(unknown)}")
    values = {**RUN_CONFIG, **loaded}; _validate(values); return values


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config")
    args = parser.parse_args(argv); result = run_stage(load_config(args.config))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True)); return result


if __name__ == "__main__": main()
