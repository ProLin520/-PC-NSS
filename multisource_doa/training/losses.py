"""Locked two-stage PC-NSS training losses."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from multisource_doa.models.pc_nss import PCNSSForward
from multisource_doa.training.teacher import (
    ScaleTeacher,
    normalized_music_denominator,
)


@dataclass(frozen=True)
class PCNSSLossBreakdown:
    total: torch.Tensor
    lag: torch.Tensor
    scale: torch.Tensor
    residual: torch.Tensor
    peak: torch.Tensor
    dominance: torch.Tensor
    weighted_lag: torch.Tensor
    weighted_scale: torch.Tensor
    weighted_residual: torch.Tensor
    weighted_peak: torch.Tensor
    weighted_dominance: torch.Tensor
    scale_distribution: torch.Tensor
    predicted_resolution_score: torch.Tensor
    best_fixed_resolution_score: torch.Tensor


def aggregate_scale_weights(
    scale_weights: torch.Tensor,
    valid_mask: torch.Tensor,
    effective_counts: torch.Tensor,
) -> torch.Tensor:
    if not (
        scale_weights.shape == valid_mask.shape == effective_counts.shape
    ):
        raise ValueError("scale weights, mask and counts must share shape")
    reliability = effective_counts.to(scale_weights.dtype) * valid_mask.to(
        scale_weights.dtype
    )
    mass = (scale_weights * reliability).sum(dim=-1)
    return mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def scale_distillation_loss(
    teacher_probabilities: torch.Tensor,
    predicted_distribution: torch.Tensor,
) -> torch.Tensor:
    epsilon = 1e-8
    teacher = teacher_probabilities.clamp_min(epsilon)
    predicted = predicted_distribution.clamp_min(epsilon)
    return (teacher * (teacher.log() - predicted.log())).sum(dim=-1).mean()


def _normalize_lags(lags_ri: torch.Tensor) -> torch.Tensor:
    scale = lags_ri[:, 0, 0].abs().clamp_min(1e-8)
    return lags_ri / scale[:, None, None]


def normalized_lag_smooth_l1(
    predicted_lags_ri: torch.Tensor,
    target_lags_ri: torch.Tensor,
) -> torch.Tensor:
    if predicted_lags_ri.shape != target_lags_ri.shape:
        raise ValueError("predicted and target lags must share shape")
    return F.smooth_l1_loss(
        _normalize_lags(predicted_lags_ri),
        _normalize_lags(target_lags_ri),
    )


def resolution_score(
    covariance: torch.Tensor,
    true_angles_deg: torch.Tensor,
) -> torch.Tensor:
    midpoint = true_angles_deg.mean(dim=-1, keepdim=True)
    queries = torch.cat([true_angles_deg, midpoint], dim=-1)
    q_values = normalized_music_denominator(covariance, queries, source_count=2)
    return q_values[:, 2] - 0.5 * q_values[:, :2].sum(dim=-1)


def peak_separation_loss(
    covariance: torch.Tensor,
    true_angles_deg: torch.Tensor,
    *,
    margin: float = 0.05,
    guard_offset_deg: float = 0.5,
    angle_limits_deg: tuple[float, float] = (-60.0, 60.0),
) -> torch.Tensor:
    first = true_angles_deg[:, 0]
    second = true_angles_deg[:, 1]
    midpoint = 0.5 * (first + second)
    guards = torch.stack(
        [first - guard_offset_deg, midpoint, second + guard_offset_deg],
        dim=-1,
    ).clamp(*angle_limits_deg)
    q_true = normalized_music_denominator(covariance, true_angles_deg, source_count=2)
    q_guard = normalized_music_denominator(covariance, guards, source_count=2)
    true_floor = q_true.max(dim=-1).values
    separation = torch.relu(margin + true_floor[:, None] - q_guard).mean(dim=-1)
    return (0.5 * q_true.sum(dim=-1) + separation).mean()


def dominance_loss(
    best_fixed_score: torch.Tensor,
    predicted_score: torch.Tensor,
    *,
    tau: float = 0.1,
) -> torch.Tensor:
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    return (tau * F.softplus((best_fixed_score - predicted_score) / tau)).mean()


def residual_regularization(output: PCNSSForward) -> torch.Tensor:
    residual = output.lag_residual_ri.square().sum(dim=-1).mean()
    loading = output.diagonal_loading.square().mean()
    return residual + loading


def compose_total_loss(
    *,
    epoch: int,
    lag: torch.Tensor,
    scale: torch.Tensor,
    residual: torch.Tensor,
    peak: torch.Tensor,
    dominance: torch.Tensor,
) -> torch.Tensor:
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    total = 1.0 * lag + 0.5 * scale + 0.01 * residual
    if epoch >= 10:
        total = total + 1.0 * peak + 0.5 * dominance
    return total


def pcnss_loss(
    output: PCNSSForward,
    teacher: ScaleTeacher,
    target_lags_ri: torch.Tensor,
    true_angles_deg: torch.Tensor,
    valid_mask: torch.Tensor,
    effective_counts: torch.Tensor,
    *,
    epoch: int,
) -> PCNSSLossBreakdown:
    distribution = aggregate_scale_weights(
        output.scale_weights,
        valid_mask,
        effective_counts,
    )
    lag = normalized_lag_smooth_l1(output.corrected_lags_ri, target_lags_ri)
    scale = scale_distillation_loss(
        teacher.scale_probabilities.to(distribution.device),
        distribution,
    )
    residual = residual_regularization(output)
    peak = peak_separation_loss(output.covariance, true_angles_deg)
    predicted_score = resolution_score(output.covariance, true_angles_deg)
    best_score = teacher.scale_scores.to(predicted_score.device).max(dim=-1).values
    dominance = dominance_loss(best_score, predicted_score)
    weighted_lag = lag
    weighted_scale = 0.5 * scale
    weighted_residual = 0.01 * residual
    weighted_peak = peak if epoch >= 10 else torch.zeros_like(peak)
    weighted_dominance = 0.5 * dominance if epoch >= 10 else torch.zeros_like(dominance)
    total = compose_total_loss(
        epoch=epoch,
        lag=lag,
        scale=scale,
        residual=residual,
        peak=peak,
        dominance=dominance,
    )
    return PCNSSLossBreakdown(
        total=total,
        lag=lag,
        scale=scale,
        residual=residual,
        peak=peak,
        dominance=dominance,
        weighted_lag=weighted_lag,
        weighted_scale=weighted_scale,
        weighted_residual=weighted_residual,
        weighted_peak=weighted_peak,
        weighted_dominance=weighted_dominance,
        scale_distribution=distribution,
        predicted_resolution_score=predicted_score,
        best_fixed_resolution_score=best_score,
    )
