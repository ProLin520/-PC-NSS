"""Bounded multi-scale PC-NSS lag fusion network."""

from dataclasses import dataclass

import torch
from torch import nn

from multisource_doa.physics.projection import structured_projection_torch


@dataclass(frozen=True)
class PCNSSForward:
    scale_weights: torch.Tensor
    anchor_lags_ri: torch.Tensor
    normalized_raw_lags_ri: torch.Tensor
    lag_residual_ri: torch.Tensor
    corrected_lags_ri: torch.Tensor
    diagonal_loading: torch.Tensor
    candidate_covariance: torch.Tensor
    covariance: torch.Tensor


def masked_softmax(
    logits: torch.Tensor,
    mask: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """Softmax over valid entries; an all-masked slice remains exactly zero."""

    if logits.shape != mask.shape:
        raise ValueError("logits and mask must have identical shapes")
    boolean_mask = mask.to(dtype=torch.bool)
    masked_logits = logits.masked_fill(
        ~boolean_mask,
        torch.finfo(logits.dtype).min,
    )
    weights = torch.softmax(masked_logits, dim=dim)
    weights = torch.where(boolean_mask, weights, torch.zeros_like(weights))
    denominator = weights.sum(dim=dim, keepdim=True)
    return torch.where(
        denominator > 0.0,
        weights / denominator.clamp_min(1e-12),
        torch.zeros_like(weights),
    )


def bounded_complex_vector(
    raw_ri: torch.Tensor,
    max_magnitude: float,
) -> torch.Tensor:
    norm = torch.linalg.vector_norm(raw_ri, dim=-1, keepdim=True)
    direction = raw_ri / norm.clamp_min(1e-12)
    magnitude = torch.tanh(norm) * max_magnitude
    return direction * magnitude


def lags_to_toeplitz(lags_ri: torch.Tensor) -> torch.Tensor:
    if lags_ri.ndim != 3 or lags_ri.shape[-1] != 2:
        raise ValueError("lags_ri must have shape [batch, lag, real_imag]")
    lags = torch.complex(lags_ri[..., 0], lags_ri[..., 1])
    zero_lag = torch.complex(
        lags[:, 0].real,
        torch.zeros_like(lags[:, 0].real),
    )
    lags = torch.cat([zero_lag[:, None], lags[:, 1:]], dim=1)
    size = lags.shape[1]
    covariance = torch.zeros(
        lags.shape[0],
        size,
        size,
        dtype=lags.dtype,
        device=lags.device,
    )
    for lag in range(size):
        rows = torch.arange(lag, size, device=lags.device)
        columns = rows - lag
        covariance[:, rows, columns] = lags[:, lag : lag + 1]
        covariance[:, columns, rows] = lags[:, lag : lag + 1].conj()
    return covariance


class MultiScalePCNSS(nn.Module):
    """Fuse four FBSS views without receiving angle or scenario labels."""

    def __init__(
        self,
        sensor_count: int = 8,
        subarray_sizes: tuple[int, ...] = (4, 5, 6, 7),
        quality_dim: int = 6,
        residual_fraction: float = 0.10,
        loading_fraction: float = 0.05,
        projection_iterations: int = 4,
        eigenvalue_floor: float = 1e-6,
    ):
        super().__init__()
        if sensor_count != 8 or subarray_sizes != (4, 5, 6, 7):
            raise ValueError("the first-round model fixes N=8 and L=(4,5,6,7)")
        self.sensor_count = sensor_count
        self.subarray_sizes = subarray_sizes
        self.residual_fraction = residual_fraction
        self.loading_fraction = loading_fraction
        self.projection_iterations = projection_iterations
        self.eigenvalue_floor = eigenvalue_floor
        self.cell_encoder = nn.Sequential(
            nn.Linear(7, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )
        self.quality_encoder = nn.Sequential(
            nn.Linear(quality_dim, 24),
            nn.GELU(),
            nn.Linear(24, 24),
            nn.GELU(),
        )
        self.lag_embedding = nn.Embedding(sensor_count, 16)
        self.logit_head = nn.Sequential(
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Linear(96, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(108, 96),
            nn.GELU(),
            nn.Linear(96, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )
        self.loading_head = nn.Sequential(
            nn.Linear(88, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.register_buffer(
            "scale_ratios",
            torch.tensor(subarray_sizes, dtype=torch.float32) / sensor_count,
        )

    def _validate_inputs(
        self,
        raw_lags_ri: torch.Tensor,
        fbss_lags_ri: torch.Tensor,
        valid_mask: torch.Tensor,
        effective_counts: torch.Tensor,
        quality_features: torch.Tensor,
    ) -> None:
        batch_size = raw_lags_ri.shape[0]
        expected_raw = (batch_size, self.sensor_count, 2)
        expected_views = (batch_size, len(self.subarray_sizes), self.sensor_count)
        if tuple(raw_lags_ri.shape) != expected_raw:
            raise ValueError(f"raw_lags_ri must have shape {expected_raw}")
        if tuple(fbss_lags_ri.shape) != expected_views + (2,):
            raise ValueError("fbss_lags_ri has an invalid shape")
        if tuple(valid_mask.shape) != expected_views:
            raise ValueError("valid_mask has an invalid shape")
        if tuple(effective_counts.shape) != expected_views:
            raise ValueError("effective_counts has an invalid shape")
        if tuple(quality_features.shape) != (
            batch_size,
            len(self.subarray_sizes),
            6,
        ):
            raise ValueError("quality_features has an invalid shape")

    def forward(
        self,
        raw_lags_ri: torch.Tensor,
        fbss_lags_ri: torch.Tensor,
        valid_mask: torch.Tensor,
        effective_counts: torch.Tensor,
        quality_features: torch.Tensor,
    ) -> PCNSSForward:
        self._validate_inputs(
            raw_lags_ri,
            fbss_lags_ri,
            valid_mask,
            effective_counts,
            quality_features,
        )
        batch_size = raw_lags_ri.shape[0]
        view_count = len(self.subarray_sizes)
        mask = valid_mask.to(dtype=torch.bool)
        trace_scale = raw_lags_ri[:, 0, 0].abs().clamp_min(1e-6)
        normalized_raw = raw_lags_ri / trace_scale[:, None, None]
        normalized_fbss = fbss_lags_ri / trace_scale[:, None, None, None]
        normalized_counts = effective_counts / effective_counts.amax(
            dim=(1, 2),
            keepdim=True,
        ).clamp_min(1.0)

        raw_expanded = normalized_raw[:, None].expand(-1, view_count, -1, -1)
        scale_ratio = self.scale_ratios.to(dtype=raw_lags_ri.dtype)[None, :, None, None]
        scale_ratio = scale_ratio.expand(batch_size, -1, self.sensor_count, -1)
        cell_features = torch.cat(
            [
                normalized_fbss,
                raw_expanded,
                mask[..., None].to(raw_lags_ri.dtype),
                normalized_counts[..., None],
                scale_ratio,
            ],
            dim=-1,
        )
        cell_encoded = self.cell_encoder(cell_features)
        stable_quality = torch.sign(quality_features) * torch.log1p(
            quality_features.abs()
        )
        quality_encoded = self.quality_encoder(stable_quality)

        mask_float = mask[..., None].to(cell_encoded.dtype)
        cell_context = (cell_encoded * mask_float).sum(dim=(1, 2)) / mask_float.sum(
            dim=(1, 2)
        ).clamp_min(1.0)
        quality_context = quality_encoded.mean(dim=1)
        global_context = torch.cat([cell_context, quality_context], dim=-1)

        lag_ids = torch.arange(self.sensor_count, device=raw_lags_ri.device)
        lag_encoded = self.lag_embedding(lag_ids)
        lag_grid = lag_encoded[None, None].expand(batch_size, view_count, -1, -1)
        quality_grid = quality_encoded[:, :, None].expand(-1, -1, self.sensor_count, -1)
        global_grid = global_context[:, None, None].expand(
            -1,
            view_count,
            self.sensor_count,
            -1,
        )
        logits = self.logit_head(
            torch.cat(
                [cell_encoded, quality_grid, lag_grid, global_grid],
                dim=-1,
            )
        ).squeeze(-1)
        weights = masked_softmax(logits, mask, dim=1)

        weighted = (weights[..., None] * normalized_fbss).sum(dim=1)
        has_fbss = mask.any(dim=1)
        anchor = torch.where(has_fbss[..., None], weighted, normalized_raw)
        residual_input = torch.cat(
            [
                anchor,
                normalized_raw,
                lag_encoded[None].expand(batch_size, -1, -1),
                global_context[:, None].expand(-1, self.sensor_count, -1),
            ],
            dim=-1,
        )
        residual = bounded_complex_vector(
            self.residual_head(residual_input),
            self.residual_fraction,
        )
        residual = residual.clone()
        residual[:, 0, 1] = 0.0
        corrected_lags = anchor + residual
        corrected_lags = corrected_lags.clone()
        corrected_lags[:, 0, 1] = 0.0

        loading = torch.sigmoid(self.loading_head(global_context)).squeeze(-1)
        loading = loading * self.loading_fraction
        candidate = lags_to_toeplitz(corrected_lags)
        identity = torch.eye(
            self.sensor_count,
            dtype=candidate.dtype,
            device=candidate.device,
        )
        candidate = candidate + loading[:, None, None] * identity
        covariance = structured_projection_torch(
            candidate,
            target_trace=float(self.sensor_count),
            iterations=self.projection_iterations,
            eigenvalue_floor=self.eigenvalue_floor,
        )
        return PCNSSForward(
            scale_weights=weights,
            anchor_lags_ri=anchor,
            normalized_raw_lags_ri=normalized_raw,
            lag_residual_ri=residual,
            corrected_lags_ri=corrected_lags,
            diagonal_loading=loading,
            candidate_covariance=candidate,
            covariance=covariance,
        )
