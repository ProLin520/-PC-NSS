"""PyCharm-friendly entrypoint for the locked Multi-Scale PC-NSS workflow."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multisource_doa.config import ExperimentConfig, SplitName
from multisource_doa.data.dataset import PCNSSDataset
from multisource_doa.data.manifest import build_split_manifest, write_split_manifest
from multisource_doa.data.simulator import generate_two_source_sample
from multisource_doa.evaluation.reporting import write_evaluation_report
from multisource_doa.evaluation.runner import evaluate_samples
from multisource_doa.models.pc_nss import MultiScalePCNSS
from multisource_doa.physics.lags import build_multiscale_views
from multisource_doa.training.artifacts import CheckpointManager, prepare_run_directory
from multisource_doa.training.engine import collate_samples, train_one_epoch, validate_model


RUN_CONFIG = {
    "stage": "dry_run",
    "dry_run": True,
    "model_seed": 2026,
    "split": "train",
    "sample_count": 4,
    "output_root": "outputs/multiscale_pcnss_snap20",
    "allow_locked_test": False,
    "overwrite": False,
    "device": "cpu",
    "checkpoint_path": "",
    "selected_best_fbss_scale": None,
}

STAGES = (
    "dry_run",
    "smoke_train",
    "train",
    "evaluate_validation",
    "evaluate_development",
)


def validate_stage(stage: str) -> str:
    if not isinstance(stage, str) or stage not in STAGES:
        raise ValueError(
            f"stage must be one of {STAGES}; 每次只运行一个阶段，不能组合多个名称"
        )
    return stage


def _device(values: dict) -> torch.device:
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


def _checkpoint_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_inputs_from_view(view):
    raw = torch.from_numpy(
        np.stack([view.raw_lags.real, view.raw_lags.imag], axis=-1)
        .astype(np.float32)[None]
    )
    fbss = torch.from_numpy(
        np.stack([view.fbss_lags.real, view.fbss_lags.imag], axis=-1)
        .astype(np.float32)[None]
    )
    mask = torch.from_numpy(view.valid_mask[None])
    counts = torch.from_numpy(view.effective_counts.astype(np.float32)[None])
    quality = torch.from_numpy(view.quality_features.astype(np.float32)[None])
    return raw, fbss, mask, counts, quality


def run_dry_run(values: dict) -> dict:
    if values.get("allow_locked_test", False):
        raise PermissionError("dry-run cannot enable locked_test")
    config = ExperimentConfig()
    torch.manual_seed(int(values["model_seed"]))
    sample = generate_two_source_sample(
        config,
        split_seed=config.split.seeds[SplitName.TRAIN],
        index=0,
        rho=1.0,
        snr_db=5.0,
        snapshot_count=20,
    )
    view = build_multiscale_views(sample.snapshots)
    model = MultiScalePCNSS().to(_device(values))
    inputs = [tensor.to(_device(values)) for tensor in _model_inputs_from_view(view)]
    with torch.no_grad():
        output = model(*inputs)
    return {
        "stage": "dry_run",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "physical_chain_finite": bool(torch.isfinite(output.covariance).all()),
        "minimum_eigenvalue": float(
            torch.linalg.eigvalsh(output.covariance).amin().cpu()
        ),
        "locked_test_access": False,
        "output_created": False,
        "interpreter": sys.executable,
    }


def run_smoke(values: dict) -> dict:
    if int(values.get("sample_count", 4)) != 4:
        raise ValueError("smoke_train is fixed to exactly 4 samples")
    if values.get("allow_locked_test", False):
        raise PermissionError("smoke_train cannot enable locked_test")
    output = prepare_run_directory(
        values["output_root"],
        refuse_overwrite=not bool(values.get("overwrite", False)),
    )
    config = ExperimentConfig()
    torch.manual_seed(int(values["model_seed"]))
    samples = [
        generate_two_source_sample(
            config,
            split_seed=config.split.seeds[SplitName.TRAIN],
            index=index,
            rho=1.0,
            snr_db=5.0,
            snapshot_count=20,
        )
        for index in range(4)
    ]
    batch = collate_samples(samples)
    device = _device(values)
    model = MultiScalePCNSS().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_metrics = train_one_epoch(
        model,
        [batch],
        optimizer,
        epoch=0,
        device=device,
        split=SplitName.TRAIN,
    )
    validation = validate_model(
        model,
        [batch],
        device=device,
        split=SplitName.VALIDATION,
    )
    summary = {
        "stage": "smoke_train",
        "sample_count": 4,
        "epoch_count": 1,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_metrics": train_metrics,
        "validation_metrics": validation.metrics,
        "locked_test_access": False,
        "formal_checkpoint_written": False,
    }
    (output / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _data_loader(dataset, batch_size: int, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_samples,
    )


def run_formal_train(values: dict) -> dict:
    if values.get("dry_run", True):
        raise ValueError("正式训练前必须把 RUN_CONFIG['dry_run'] 改为 False")
    if values.get("allow_locked_test", False):
        raise PermissionError("formal training cannot enable locked_test")
    config = ExperimentConfig()
    output = prepare_run_directory(
        values["output_root"],
        refuse_overwrite=not bool(values.get("overwrite", False)),
    )
    write_split_manifest(output / "train_manifest.json", config, SplitName.TRAIN)
    write_split_manifest(
        output / "validation_manifest.json", config, SplitName.VALIDATION
    )
    torch.manual_seed(int(values["model_seed"]))
    device = _device(values)
    model = MultiScalePCNSS().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.training.learning_rate
    )
    train_loader = _data_loader(
        PCNSSDataset(SplitName.TRAIN, config),
        config.training.batch_size,
        shuffle=True,
    )
    validation_loader = _data_loader(
        PCNSSDataset(SplitName.VALIDATION, config),
        config.training.batch_size,
        shuffle=False,
    )
    checkpoints = CheckpointManager(output)
    metric_rows = []
    for epoch in range(config.training.total_epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            epoch=epoch,
            device=device,
            split=SplitName.TRAIN,
        )
        validation = validate_model(
            model,
            validation_loader,
            device=device,
            split=SplitName.VALIDATION,
        )
        selected = checkpoints.update(
            metric_value=validation.metrics["failure_aware_rmspe_deg"],
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            experiment_config=config,
            model_seed=int(values["model_seed"]),
            data_split_seed=config.split.seeds[SplitName.VALIDATION],
            code_sha=_code_sha(),
            split=SplitName.VALIDATION,
        )
        metric_rows.append(
            {
                "epoch": epoch,
                "selected": selected,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{
                    f"validation_{key}": value
                    for key, value in validation.metrics.items()
                    if not isinstance(value, dict)
                },
            }
        )
        with (output / "metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
            writer.writeheader()
            writer.writerows(metric_rows)
    return {
        "stage": "train",
        "epochs": config.training.total_epochs,
        "output_root": str(output),
        "best_validation_rmspe_deg": checkpoints.best_metric,
    }


def run_formal_evaluation(values: dict, split: SplitName) -> dict:
    if values.get("dry_run", True):
        raise ValueError("正式评价前必须把 RUN_CONFIG['dry_run'] 改为 False")
    if split is SplitName.LOCKED_TEST or values.get("allow_locked_test", False):
        raise PermissionError("locked_test has no entrypoint in the foundation stage")
    config = ExperimentConfig()
    checkpoint = Path(values.get("checkpoint_path") or Path(values["output_root"]) / "best.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    device = _device(values)
    model = MultiScalePCNSS().to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    dataset = PCNSSDataset(split, config)
    samples = [dataset[index] for index in range(len(dataset))]
    result = evaluate_samples(
        samples,
        model,
        split=split,
        device=device,
        selected_best_fbss_scale=values.get("selected_best_fbss_scale"),
    )
    report_name = (
        "validation_report" if split is SplitName.VALIDATION else "development_report"
    )
    report = write_evaluation_report(
        result,
        Path(values["output_root"]) / report_name,
        run_config=values,
        source_manifest=build_split_manifest(config, split),
        code_sha=_code_sha(),
        checkpoint_sha=_checkpoint_sha(checkpoint),
    )
    return {
        "stage": f"evaluate_{split.value}",
        "report": str(report),
        "best_fixed_fbss_scale": result.best_fixed_fbss_scale,
    }


def run_stage(values: dict) -> dict:
    stage = validate_stage(values["stage"])
    if stage == "dry_run":
        return run_dry_run(values)
    if stage == "smoke_train":
        return run_smoke(values)
    if stage == "train":
        return run_formal_train(values)
    if stage == "evaluate_validation":
        return run_formal_evaluation(values, SplitName.VALIDATION)
    return run_formal_evaluation(values, SplitName.DEVELOPMENT)


def main() -> None:
    print(json.dumps(run_stage(RUN_CONFIG), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
