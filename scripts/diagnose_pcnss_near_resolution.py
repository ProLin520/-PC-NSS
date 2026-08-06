"""Safe, read-only entrypoint for frozen PC-NSS near-resolution diagnostics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multisource_doa.config import ExperimentConfig, SplitName
from multisource_doa.data.dataset import PCNSSDataset
from multisource_doa.data.simulator import DOASample
from multisource_doa.diagnostics.near_resolution import (
    NearAuditLabel,
    diagnose_near_samples,
    load_near_audit,
)
from multisource_doa.diagnostics.reporting import write_near_diagnostic_report
from multisource_doa.models.pc_nss import MultiScalePCNSS


RUN_CONFIG = {
    "stage": "dry_run",
    "dry_run": True,
    "split": "validation",
    "report_directory": "",
    "checkpoint_path": "",
    "output_root": "outputs/pcnss_near_resolution_diagnostic",
    "device": "cpu",
    "batch_size": 128,
    "expected_near_count": 1270,
    "allow_locked_test": False,
    "overwrite": False,
}

STAGES = ("dry_run", "diagnose_validation_near")


def validate_stage(stage: str) -> str:
    if not isinstance(stage, str) or stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    return stage


def _device(values: dict[str, Any]) -> torch.device:
    requested = str(values.get("device", "cpu"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _code_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validate_regenerated_metadata(sample: DOASample, label: NearAuditLabel) -> None:
    source = label.pcnss_row
    actual_separation = float(abs(np.diff(sample.angles_deg)[0]))
    angle_matches = np.allclose(
        sample.angles_deg,
        (source["true_angle_1_deg"], source["true_angle_2_deg"]),
        rtol=0.0,
        atol=1e-9,
    )
    separation_matches = np.isclose(
        actual_separation,
        label.separation_deg,
        rtol=0.0,
        atol=1e-9,
    )
    if (
        not angle_matches
        or not separation_matches
        or sample.rho != label.rho
        or sample.snr_db != label.snr_db
        or sample.snapshot_count != label.snapshot_count
    ):
        raise ValueError(f"metadata mismatch for sample_seed {sample.sample_seed}")


def _refuse_existing_output(values: dict[str, Any]) -> None:
    output = Path(str(values["output_root"]))
    if output.exists() and not bool(values.get("overwrite", False)):
        raise FileExistsError(f"refusing to overwrite output directory: {output}")


def run_dry_run(values: dict[str, Any]) -> dict[str, Any]:
    """Validate the safe invocation path without reading artifacts or writing output."""

    if values.get("allow_locked_test", False):
        raise PermissionError("dry-run cannot enable locked_test")
    if values.get("split") != "validation":
        raise PermissionError("near-resolution diagnostic accepts validation only")
    return {
        "stage": "dry_run",
        "locked_test_access": False,
        "output_created": False,
        "device": str(_device(values)),
        "batch_size": int(values["batch_size"]),
    }


def run_diagnostic(values: dict[str, Any]) -> dict[str, Any]:
    """Diagnose only audit-selected validation samples from a frozen checkpoint."""

    if values.get("dry_run", True):
        raise ValueError("正式诊断前必须把 dry_run 改为 False")
    if values.get("split") != "validation":
        raise PermissionError("near-resolution diagnostic accepts validation only")
    if values.get("allow_locked_test", False):
        raise PermissionError("locked_test access is forbidden")
    _refuse_existing_output(values)
    selection = load_near_audit(
        values["report_directory"],
        values["checkpoint_path"],
        expected_near_count=int(values["expected_near_count"]),
    )
    config = ExperimentConfig()
    dataset = PCNSSDataset(SplitName.VALIDATION, config)
    split_seed = config.split.seeds[SplitName.VALIDATION]
    samples: list[DOASample] = []
    labels_by_seed: dict[int, NearAuditLabel] = {}
    for label in selection.labels:
        index = label.sample_seed - split_seed
        if not 0 <= index < len(dataset):
            raise ValueError("sample_seed maps outside validation")
        sample = dataset[index]
        _validate_regenerated_metadata(sample, label)
        samples.append(sample)
        labels_by_seed[label.sample_seed] = label
    device = _device(values)
    model = MultiScalePCNSS().to(device)
    payload = torch.load(
        values["checkpoint_path"],
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(payload["model_state_dict"])
    result = diagnose_near_samples(
        samples,
        labels_by_seed,
        model,
        device=device,
        batch_size=int(values["batch_size"]),
    )
    report = write_near_diagnostic_report(
        result,
        values["output_root"],
        diagnostic_config=values,
        source_manifest={
            "diagnostic_code_sha": _code_sha(),
            "checkpoint_sha": selection.source_manifest["checkpoint_sha"],
            "evaluator_code_sha": selection.source_manifest["code_sha"],
            "source_report_directory": str(values["report_directory"]),
            "input_sha256": selection.input_sha256,
            "validation_split_seed": split_seed,
            "sample_count": len(selection.labels),
            "batch_size": int(values["batch_size"]),
            "device": str(device),
            "residual_limit": 0.10,
            "residual_saturation_threshold": 0.095,
        },
        refuse_overwrite=not bool(values.get("overwrite", False)),
    )
    return {"stage": values["stage"], "sample_count": len(samples), "report": str(report)}


def run_stage(values: dict[str, Any]) -> dict[str, Any]:
    stage = validate_stage(values.get("stage"))
    if stage == "dry_run":
        return run_dry_run(values)
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
    return {**RUN_CONFIG, **loaded}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="path to a complete or partial JSON config")
    arguments = parser.parse_args(argv)
    result = run_stage(load_config(arguments.config))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
