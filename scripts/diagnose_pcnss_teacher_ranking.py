"""Safe CPU-only entrypoint for the frozen Task 16 teacher ranking diagnosis."""

from __future__ import annotations

import argparse
import json
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
from multisource_doa.diagnostics.teacher_ranking import (
    TeacherRankingLabel,
    diagnose_teacher_ranking_samples,
    load_teacher_ranking_inputs,
)
from multisource_doa.diagnostics.teacher_ranking_reporting import (
    write_teacher_ranking_report,
)
from multisource_doa.training.engine import collate_samples
from multisource_doa.training.teacher import build_scale_teacher


RUN_CONFIG = {
    "stage": "dry_run",
    "dry_run": True,
    "split": "validation",
    "task15_directory": "",
    "output_root": "outputs/pcnss_teacher_ranking_diagnostic",
    "device": "cpu",
    "batch_size": 128,
    "sample_count": 4,
    "expected_count": 1270,
    "allow_locked_test": False,
    "overwrite": False,
}
STAGES = ("dry_run", "smoke", "diagnose_validation_teacher_ranking")
FROZEN_BATCH_SIZE = 128
FROZEN_SAMPLE_COUNT = 1270


def validate_stage(stage: Any) -> str:
    if not isinstance(stage, str) or stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    return stage


def _validate_safe_config(values: dict[str, Any], *, formal: bool) -> None:
    if values.get("device") != "cpu":
        raise ValueError("teacher ranking diagnosis is CPU only")
    if values.get("batch_size") != FROZEN_BATCH_SIZE:
        raise ValueError(f"batch_size must be fixed at {FROZEN_BATCH_SIZE}")
    if values.get("split") != "validation":
        raise PermissionError("teacher ranking diagnosis accepts validation only")
    if values.get("allow_locked_test") is not False:
        raise PermissionError("locked_test access is forbidden")
    if values.get("overwrite") is not False:
        raise ValueError("overwrite must remain false")
    if formal and values.get("expected_count") != FROZEN_SAMPLE_COUNT:
        raise ValueError(f"expected_count must be fixed at {FROZEN_SAMPLE_COUNT}")


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
    """Validate frozen defaults without reading sources or creating output."""

    _validate_safe_config(values, formal=False)
    return {
        "stage": "dry_run",
        "locked_test_access": False,
        "output_created": False,
        "device": "cpu",
        "batch_size": FROZEN_BATCH_SIZE,
        "no_model_forward": True,
        "teacher_modified": False,
        "training_performed": False,
    }


def run_smoke(values: dict[str, Any]) -> dict[str, Any]:
    """Exercise the physical ranking path on four train samples only."""

    _validate_safe_config(values, formal=False)
    if values.get("sample_count") != 4:
        raise ValueError("teacher ranking smoke sample_count must be fixed at 4")
    output = _refuse_existing_output(values)
    config = ExperimentConfig()
    split_seed = config.split.seeds[SplitName.TRAIN]
    samples = [
        generate_two_source_sample(
            config,
            split_seed=split_seed,
            index=index,
            rho=(0.8, 0.9, 0.99, 1.0)[index],
            snr_db=(-2.0, 2.0, 7.0, 10.0)[index],
            snapshot_count=(8, 20, 50, 20)[index],
            center_deg=float(index),
            separation_deg=3.0,
        )
        for index in range(4)
    ]
    batch = collate_samples(samples)
    scores = build_scale_teacher(
        batch.fbss_covariances, batch.true_angles_deg
    ).scale_scores
    labels = {
        sample.sample_seed: TeacherRankingLabel(
            sample_seed=sample.sample_seed,
            true_angles_deg=tuple(float(value) for value in sample.angles_deg),
            rho=float(sample.rho),
            snr_db=float(sample.snr_db),
            snapshot_count=int(sample.snapshot_count),
            separation_deg=float(abs(np.diff(sample.angles_deg)[0])),
            threshold_cohort=(
                "resolved" if index == 0 else "far_miss_gt_2"
            ),
            task15_scores=tuple(float(value) for value in scores[index]),
            fixed_rmspe_deg={
                size: float(index + scale_index + 1)
                for scale_index, size in enumerate((4, 5, 6, 7))
            },
        )
        for index, sample in enumerate(samples)
    }
    result = diagnose_teacher_ranking_samples(
        samples, labels, batch_size=FROZEN_BATCH_SIZE
    )
    report = write_teacher_ranking_report(
        result,
        output,
        diagnostic_config=values,
        source_manifest={
            "diagnostic_code_sha": _code_sha(),
            "source": "temporary-train-smoke",
            "split_seed": split_seed,
            "sample_count": 4,
            "device": "cpu",
            "batch_size": FROZEN_BATCH_SIZE,
            "no_model_forward": True,
            "teacher_modified": False,
            "training_performed": False,
        },
        task15_margin_over_tau_median=0.01,
        engineering_integrity=True,
    )
    return {
        "stage": "smoke",
        "sample_count": 4,
        "report": str(report),
        "no_model_forward": True,
        "teacher_modified": False,
        "training_performed": False,
    }


def run_diagnostic(values: dict[str, Any]) -> dict[str, Any]:
    """Run the one approved 1270-sample validation ranking diagnosis."""

    _validate_safe_config(values, formal=True)
    if values.get("dry_run") is not False:
        raise ValueError("正式诊断前必须把 dry_run 改为 False")
    output = _refuse_existing_output(values)
    if not values.get("task15_directory"):
        raise ValueError("formal diagnosis requires an authenticated Task 15 directory")
    inputs = load_teacher_ranking_inputs(
        values["task15_directory"], expected_count=FROZEN_SAMPLE_COUNT
    )
    config = ExperimentConfig()
    if inputs.validation_split_seed != config.split.seeds[SplitName.VALIDATION]:
        raise ValueError("Task 15 validation split seed mismatch")
    dataset = PCNSSDataset(SplitName.VALIDATION, config)
    samples = []
    for sample_seed in inputs.labels_by_seed:
        index = sample_seed - inputs.validation_split_seed
        if not 0 <= index < len(dataset):
            raise ValueError("sample_seed maps outside validation")
        samples.append(dataset[index])
    result = diagnose_teacher_ranking_samples(
        samples,
        inputs.labels_by_seed,
        batch_size=FROZEN_BATCH_SIZE,
    )
    margin, integrity = _task15_gate_context(inputs.task15_summary)
    report = write_teacher_ranking_report(
        result,
        output,
        diagnostic_config=values,
        source_manifest={
            "diagnostic_code_sha": _code_sha(),
            "task15_input_sha256": inputs.task15_sha256,
            "task15_upstream_sha256": inputs.upstream_sha256,
            "task15_diagnostic_code_sha": inputs.task15_manifest.get(
                "diagnostic_code_sha"
            ),
            "validation_split_seed": inputs.validation_split_seed,
            "sample_count": len(samples),
            "device": "cpu",
            "batch_size": FROZEN_BATCH_SIZE,
            "no_model_forward": True,
            "teacher_modified": False,
            "training_performed": False,
        },
        task15_margin_over_tau_median=margin,
        engineering_integrity=integrity,
    )
    return {
        "stage": "diagnose_validation_teacher_ranking",
        "sample_count": len(samples),
        "report": str(report),
        "no_model_forward": True,
        "teacher_modified": False,
        "training_performed": False,
    }


def _task15_gate_context(summary: dict[str, Any]) -> tuple[float, bool]:
    try:
        margin = float(summary["metrics"]["teacher_score_margin_over_tau"]["median"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Task 15 summary is missing median margin/tau") from error
    if not np.isfinite(margin):
        raise ValueError("Task 15 median margin/tau must be finite")
    integrity = summary.get("engineering_integrity") is True
    return margin, integrity


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
    _validate_safe_config(
        values, formal=stage == "diagnose_validation_teacher_ranking"
    )
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
