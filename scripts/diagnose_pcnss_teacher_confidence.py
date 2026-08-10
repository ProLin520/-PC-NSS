"""Safe CPU-only entrypoint for frozen PC-NSS teacher-confidence diagnosis."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multisource_doa.config import ExperimentConfig, SplitName
from multisource_doa.data.dataset import PCNSSDataset
from multisource_doa.data.simulator import generate_two_source_sample
from multisource_doa.diagnostics.teacher_confidence import (
    COUNTERFACTUAL_TAU,
    CURRENT_TAU,
    TeacherAuthorityLabel,
    diagnose_teacher_samples,
    load_teacher_diagnostic_inputs,
    validate_regenerated_metadata,
)
from multisource_doa.diagnostics.teacher_reporting import (
    write_teacher_diagnostic_report,
)


RUN_CONFIG = {
    "stage": "dry_run",
    "dry_run": True,
    "split": "validation",
    "report_directory": "",
    "task14_directory": "",
    "output_root": "outputs/pcnss_teacher_confidence_diagnostic",
    "device": "cpu",
    "batch_size": 128,
    "sample_count": 4,
    "expected_source_count": 5000,
    "expected_near_count": 1270,
    "tau_current": 0.10,
    "tau_counterfactual": 0.05,
    "allow_locked_test": False,
    "overwrite": False,
}
STAGES = ("dry_run", "smoke", "diagnose_validation_teacher")
FROZEN_BATCH_SIZE = 128
FROZEN_SOURCE_COUNT = 5000
FROZEN_NEAR_COUNT = 1270


def validate_stage(stage: Any) -> str:
    if not isinstance(stage, str) or stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    return stage


def _validate_safe_config(values: dict[str, Any], *, formal: bool) -> None:
    if values.get("device") != "cpu":
        raise ValueError("teacher diagnosis is CPU only")
    if values.get("batch_size") != FROZEN_BATCH_SIZE:
        raise ValueError(f"batch_size must be fixed at {FROZEN_BATCH_SIZE}")
    if values.get("split") != "validation":
        raise PermissionError("teacher diagnosis accepts validation only")
    if values.get("allow_locked_test") is not False:
        raise PermissionError("locked_test access is forbidden")
    if values.get("overwrite") is not False:
        raise ValueError("overwrite must remain false")
    if not _same_float(values.get("tau_current"), CURRENT_TAU):
        raise ValueError(f"tau_current must be fixed at {CURRENT_TAU}")
    if not _same_float(values.get("tau_counterfactual"), COUNTERFACTUAL_TAU):
        raise ValueError(
            f"tau_counterfactual must be fixed at {COUNTERFACTUAL_TAU}"
        )
    if formal:
        if values.get("expected_source_count") != FROZEN_SOURCE_COUNT:
            raise ValueError(
                f"expected_source_count must be fixed at {FROZEN_SOURCE_COUNT}"
            )
        if values.get("expected_near_count") != FROZEN_NEAR_COUNT:
            raise ValueError(
                f"expected_near_count must be fixed at {FROZEN_NEAR_COUNT}"
            )


def _same_float(value: Any, expected: float) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isclose(number, expected, rel_tol=0.0, abs_tol=1e-12)


def _refuse_existing_output(values: dict[str, Any]) -> Path:
    output = Path(str(values["output_root"]))
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    return output


def _code_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def run_dry_run(values: dict[str, Any]) -> dict[str, Any]:
    """Validate safe defaults without reading artifacts or creating output."""

    _validate_safe_config(values, formal=False)
    return {
        "stage": "dry_run",
        "locked_test_access": False,
        "output_created": False,
        "device": "cpu",
        "batch_size": FROZEN_BATCH_SIZE,
        "no_model_forward": True,
        "training_performed": False,
    }


def run_smoke(values: dict[str, Any]) -> dict[str, Any]:
    """Run four train samples with synthetic authority and no neural model."""

    _validate_safe_config(values, formal=False)
    sample_count = values.get("sample_count")
    if sample_count != 4:
        raise ValueError("teacher smoke sample_count must be fixed at 4")
    output = _refuse_existing_output(values)
    config = ExperimentConfig()
    split_seed = config.split.seeds[SplitName.TRAIN]
    samples = [
        generate_two_source_sample(
            config,
            split_seed=split_seed,
            index=index,
            rho=1.0,
            snr_db=5.0,
            snapshot_count=20,
            center_deg=float(index),
            separation_deg=3.0,
        )
        for index in range(sample_count)
    ]
    labels = {
        sample.sample_seed: TeacherAuthorityLabel(
            sample_seed=sample.sample_seed,
            true_angles_deg=tuple(float(value) for value in sample.angles_deg),
            rho=float(sample.rho),
            snr_db=float(sample.snr_db),
            snapshot_count=int(sample.snapshot_count),
            separation_deg=float(abs(np.diff(sample.angles_deg)[0])),
            threshold_cohort="far_miss_gt_2",
            student_probabilities=(0.25, 0.25, 0.25, 0.25),
            fixed_rmspe_deg={4: 0.5, 5: 1.0, 6: 1.5, 7: 2.0},
        )
        for sample in samples
    }
    result = diagnose_teacher_samples(
        samples,
        labels,
        batch_size=FROZEN_BATCH_SIZE,
        tau_current=CURRENT_TAU,
        tau_counterfactual=COUNTERFACTUAL_TAU,
    )
    report = write_teacher_diagnostic_report(
        result,
        output,
        diagnostic_config=values,
        source_manifest={
            "diagnostic_code_sha": _code_sha(),
            "source": "temporary-train-smoke",
            "split_seed": split_seed,
            "sample_count": sample_count,
            "device": "cpu",
            "batch_size": FROZEN_BATCH_SIZE,
            "tau_current": CURRENT_TAU,
            "tau_counterfactual": COUNTERFACTUAL_TAU,
            "no_model_forward": True,
            "training_performed": False,
        },
    )
    return {
        "stage": "smoke",
        "sample_count": sample_count,
        "report": str(report),
        "no_model_forward": True,
        "training_performed": False,
    }


def run_diagnostic(values: dict[str, Any]) -> dict[str, Any]:
    """Run the one approved CPU-only diagnosis on frozen validation inputs."""

    _validate_safe_config(values, formal=True)
    if values.get("dry_run") is not False:
        raise ValueError("正式诊断前必须把 dry_run 改为 False")
    output = _refuse_existing_output(values)
    if not values.get("report_directory") or not values.get("task14_directory"):
        raise ValueError("formal diagnosis requires both authenticated input directories")
    inputs = load_teacher_diagnostic_inputs(
        values["report_directory"],
        values["task14_directory"],
        expected_source_count=FROZEN_SOURCE_COUNT,
        expected_near_count=FROZEN_NEAR_COUNT,
    )
    config = ExperimentConfig()
    dataset = PCNSSDataset(SplitName.VALIDATION, config)
    split_seed = config.split.seeds[SplitName.VALIDATION]
    samples = []
    for sample_seed, label in inputs.labels_by_seed.items():
        index = sample_seed - split_seed
        if not 0 <= index < len(dataset):
            raise ValueError("sample_seed maps outside validation")
        sample = dataset[index]
        validate_regenerated_metadata(sample, label)
        samples.append(sample)
    result = diagnose_teacher_samples(
        samples,
        inputs.labels_by_seed,
        batch_size=FROZEN_BATCH_SIZE,
        tau_current=CURRENT_TAU,
        tau_counterfactual=COUNTERFACTUAL_TAU,
    )
    report = write_teacher_diagnostic_report(
        result,
        output,
        diagnostic_config=values,
        source_manifest={
            "diagnostic_code_sha": _code_sha(),
            "input_sha256": inputs.input_sha256,
            "audit_source_manifest": inputs.source_manifest["audit"],
            "task14_source_manifest": inputs.source_manifest["task14"],
            "validation_split_seed": split_seed,
            "sample_count": len(samples),
            "device": "cpu",
            "batch_size": FROZEN_BATCH_SIZE,
            "tau_current": CURRENT_TAU,
            "tau_counterfactual": COUNTERFACTUAL_TAU,
            "no_model_forward": True,
            "training_performed": False,
        },
    )
    return {
        "stage": "diagnose_validation_teacher",
        "sample_count": len(samples),
        "report": str(report),
        "no_model_forward": True,
        "training_performed": False,
    }


def run_stage(values: dict[str, Any]) -> dict[str, Any]:
    stage = validate_stage(values.get("stage"))
    if stage == "dry_run":
        return run_dry_run(values)
    if stage == "smoke":
        return run_smoke(values)
    return run_diagnostic(values)


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return dict(RUN_CONFIG)
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("config JSON must contain an object")
    unknown_keys = set(loaded) - set(RUN_CONFIG)
    if unknown_keys:
        raise ValueError(f"unknown config keys: {sorted(unknown_keys)}")
    values = {**RUN_CONFIG, **loaded}
    stage = validate_stage(values.get("stage"))
    _validate_safe_config(values, formal=stage == "diagnose_validation_teacher")
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
