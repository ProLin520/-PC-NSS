"""Small, auditable PC-NSS training and validation engine."""

from dataclasses import dataclass, fields
from typing import Iterable, Mapping

import numpy as np
import torch

from multisource_doa.config import SplitName
from multisource_doa.data.simulator import DOASample
from multisource_doa.evaluation.metrics import (
    SampleScore,
    aggregate_metrics,
    score_doa_sample,
)
from multisource_doa.models.pc_nss import MultiScalePCNSS, lags_to_toeplitz
from multisource_doa.physics.lags import build_multiscale_views, covariance_to_lags
from multisource_doa.physics.projection import dykstra_structured_projection
from multisource_doa.physics.root_music import estimate_root_music
from multisource_doa.training.losses import pcnss_loss
from multisource_doa.training.teacher import build_scale_teacher


@dataclass(frozen=True)
class PCNSSBatch:
    raw_lags_ri: torch.Tensor
    fbss_lags_ri: torch.Tensor
    valid_mask: torch.Tensor
    effective_counts: torch.Tensor
    quality_features: torch.Tensor
    target_lags_ri: torch.Tensor
    true_angles_deg: torch.Tensor
    fbss_covariances: dict[int, torch.Tensor]
    sample_seeds: tuple[int, ...]
    rhos: tuple[float, ...]
    snr_db: tuple[float, ...]
    snapshot_counts: tuple[int, ...]


@dataclass(frozen=True)
class ValidationResult:
    metrics: dict
    scores: tuple[SampleScore, ...]


def _complex_to_ri(values: np.ndarray) -> np.ndarray:
    return np.stack([values.real, values.imag], axis=-1).astype(np.float32)


def collate_samples(samples: list[DOASample]) -> PCNSSBatch:
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    view_rows = [build_multiscale_views(sample.snapshots) for sample in samples]
    target_rows = [
        covariance_to_lags(sample.target_covariance, output_size=8)[0]
        for sample in samples
    ]
    fbss_covariances = {
        size: torch.from_numpy(
            np.stack([row.fbss_covariances[size] for row in view_rows]).astype(
                np.complex64
            )
        )
        for size in (4, 5, 6, 7)
    }
    return PCNSSBatch(
        raw_lags_ri=torch.from_numpy(
            np.stack([_complex_to_ri(row.raw_lags) for row in view_rows])
        ),
        fbss_lags_ri=torch.from_numpy(
            np.stack([_complex_to_ri(row.fbss_lags) for row in view_rows])
        ),
        valid_mask=torch.from_numpy(np.stack([row.valid_mask for row in view_rows])),
        effective_counts=torch.from_numpy(
            np.stack([row.effective_counts for row in view_rows]).astype(np.float32)
        ),
        quality_features=torch.from_numpy(
            np.stack([row.quality_features for row in view_rows]).astype(np.float32)
        ),
        target_lags_ri=torch.from_numpy(
            np.stack([_complex_to_ri(row) for row in target_rows])
        ),
        true_angles_deg=torch.from_numpy(
            np.stack([sample.angles_deg for sample in samples]).astype(np.float32)
        ),
        fbss_covariances=fbss_covariances,
        sample_seeds=tuple(sample.sample_seed for sample in samples),
        rhos=tuple(sample.rho for sample in samples),
        snr_db=tuple(sample.snr_db for sample in samples),
        snapshot_counts=tuple(sample.snapshot_count for sample in samples),
    )


def _to_device(batch: PCNSSBatch, device: torch.device) -> PCNSSBatch:
    values = {}
    for item in fields(batch):
        value = getattr(batch, item.name)
        if isinstance(value, torch.Tensor):
            values[item.name] = value.to(device)
        elif isinstance(value, dict):
            values[item.name] = {
                key: tensor.to(device) for key, tensor in value.items()
            }
        else:
            values[item.name] = value
    return PCNSSBatch(**values)


ScaleTargetLookup = Mapping[int, tuple[float, float, float, float]]


def _batch_scale_target(
    batch: PCNSSBatch,
    lookup: ScaleTargetLookup,
    device: torch.device,
) -> torch.Tensor:
    missing = [seed for seed in batch.sample_seeds if seed not in lookup]
    if missing:
        raise KeyError(f"teacher cache missing sample seeds: {missing[:4]}")
    return torch.tensor(
        [lookup[seed] for seed in batch.sample_seeds],
        dtype=torch.float32,
        device=device,
    )


def _batch_diagnostics(output, breakdown, batch: PCNSSBatch) -> dict[str, float]:
    distribution = breakdown.scale_distribution.detach()
    entropy = -(
        distribution.clamp_min(1e-8) * distribution.clamp_min(1e-8).log()
    ).sum(dim=-1).mean()
    residual_magnitude = torch.linalg.vector_norm(
        output.lag_residual_ri.detach(), dim=-1
    ).reshape(-1)
    candidate_norm = torch.linalg.matrix_norm(
        output.candidate_covariance.detach(), ord="fro"
    ).clamp_min(1e-8)
    projection_change = (
        torch.linalg.matrix_norm(
            output.covariance.detach() - output.candidate_covariance.detach(),
            ord="fro",
        )
        / candidate_norm
    ).mean()
    covariance = output.covariance.detach()
    target_covariance = lags_to_toeplitz(batch.target_lags_ri).detach()
    _, predicted_vectors = torch.linalg.eigh(covariance)
    _, target_vectors = torch.linalg.eigh(target_covariance)
    predicted_signal = predicted_vectors[..., -2:]
    target_signal = target_vectors[..., -2:]
    cosines = torch.linalg.svdvals(predicted_signal.mH @ target_signal).clamp(0.0, 1.0)
    subspace_angle = torch.rad2deg(torch.acos(cosines)).mean()
    hermitian_error = (
        torch.linalg.matrix_norm(covariance - covariance.mH, ord="fro")
        / torch.linalg.matrix_norm(covariance, ord="fro").clamp_min(1e-8)
    ).mean()
    metrics = {
        "total": breakdown.total.detach().item(),
        "lag": breakdown.lag.detach().item(),
        "scale": breakdown.scale.detach().item(),
        "residual": breakdown.residual.detach().item(),
        "peak": breakdown.peak.detach().item(),
        "dominance": breakdown.dominance.detach().item(),
        "weighted_peak": breakdown.weighted_peak.detach().item(),
        "weighted_dominance": breakdown.weighted_dominance.detach().item(),
        "scale_weight_entropy": entropy.item(),
        "residual_magnitude_p50": torch.quantile(residual_magnitude, 0.50).item(),
        "residual_magnitude_p95": torch.quantile(residual_magnitude, 0.95).item(),
        "loading_p50": torch.quantile(output.diagonal_loading.detach(), 0.50).item(),
        "loading_p95": torch.quantile(output.diagonal_loading.detach(), 0.95).item(),
        "projection_change_fro": projection_change.item(),
        "hermitian_error": hermitian_error.item(),
        "trace_error": (
            covariance.diagonal(dim1=-2, dim2=-1).real.sum(-1) - 8.0
        ).abs().mean().item(),
        "minimum_eigenvalue": torch.linalg.eigvalsh(covariance).amin().item(),
        "best_minus_predicted_resolution": (
            breakdown.best_fixed_resolution_score.detach()
            - breakdown.predicted_resolution_score.detach()
        ).mean().item(),
        "signal_subspace_angle_deg": subspace_angle.item(),
    }
    for view_index, size in enumerate((4, 5, 6, 7)):
        metrics[f"scale_weight_mean_L{size}"] = distribution[:, view_index].mean().item()
        for lag in range(8):
            metrics[f"scale_weight_L{size}_lag{lag}"] = (
                output.scale_weights.detach()[:, view_index, lag].mean().item()
            )
    return metrics


def train_one_epoch(
    model: MultiScalePCNSS,
    batches: Iterable[PCNSSBatch],
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    device: torch.device,
    split: SplitName = SplitName.TRAIN,
    scale_targets_by_seed: ScaleTargetLookup | None = None,
) -> dict[str, float]:
    if SplitName(split) is not SplitName.TRAIN:
        raise PermissionError("training updates are restricted to the train split")
    model.train()
    accumulated: dict[str, float] = {}
    batch_count = 0
    for cpu_batch in batches:
        batch = _to_device(cpu_batch, device)
        teacher = build_scale_teacher(
            batch.fbss_covariances,
            batch.true_angles_deg,
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch.raw_lags_ri,
            batch.fbss_lags_ri,
            batch.valid_mask,
            batch.effective_counts,
            batch.quality_features,
        )
        scale_target = (
            None
            if scale_targets_by_seed is None
            else _batch_scale_target(cpu_batch, scale_targets_by_seed, device)
        )
        breakdown = pcnss_loss(
            output,
            teacher,
            batch.target_lags_ri,
            batch.true_angles_deg,
            batch.valid_mask,
            batch.effective_counts,
            epoch=epoch,
            scale_distillation_target=scale_target,
        )
        if not torch.isfinite(breakdown.total):
            raise FloatingPointError("non-finite PC-NSS training loss")
        breakdown.total.backward()
        optimizer.step()
        diagnostics = _batch_diagnostics(output, breakdown, batch)
        for key, value in diagnostics.items():
            accumulated[key] = accumulated.get(key, 0.0) + value
        batch_count += 1
    if batch_count == 0:
        raise ValueError("train_one_epoch requires at least one batch")
    return {key: value / batch_count for key, value in accumulated.items()}


def validate_model(
    model: MultiScalePCNSS,
    batches: Iterable[PCNSSBatch],
    *,
    device: torch.device,
    split: SplitName = SplitName.VALIDATION,
) -> ValidationResult:
    split_name = SplitName(split)
    if split_name is SplitName.LOCKED_TEST:
        raise PermissionError("locked_test validation requires separate approval")
    if split_name not in (SplitName.VALIDATION, SplitName.DEVELOPMENT):
        raise PermissionError("model evaluation is restricted to validation/development")
    model.eval()
    scores: list[SampleScore] = []
    with torch.no_grad():
        for cpu_batch in batches:
            batch = _to_device(cpu_batch, device)
            output = model(
                batch.raw_lags_ri,
                batch.fbss_lags_ri,
                batch.valid_mask,
                batch.effective_counts,
                batch.quality_features,
            )
            covariances = output.covariance.detach().cpu().numpy()
            truths = batch.true_angles_deg.detach().cpu().numpy()
            for index, covariance in enumerate(covariances):
                projection = dykstra_structured_projection(covariance)
                if projection.converged:
                    estimate = estimate_root_music(projection.matrix, source_count=2)
                    estimated_angles = estimate.angles_deg
                    success = estimate.success
                    failure_reason = estimate.failure_reason
                else:
                    estimated_angles = np.empty(0, dtype=np.float64)
                    success = False
                    failure_reason = "projection_not_converged"
                separation = float(abs(truths[index, 1] - truths[index, 0]))
                scores.append(
                    score_doa_sample(
                        cpu_batch.sample_seeds[index],
                        truths[index],
                        estimated_angles,
                        estimate_success=success,
                        failure_reason=failure_reason,
                        strata={
                            "separation_deg": separation,
                            "snr_db": cpu_batch.snr_db[index],
                            "snapshot_count": cpu_batch.snapshot_counts[index],
                            "rho": cpu_batch.rhos[index],
                        },
                    )
                )
    if not scores:
        raise ValueError("validate_model requires at least one sample")
    return ValidationResult(metrics=aggregate_metrics(scores), scores=tuple(scores))
