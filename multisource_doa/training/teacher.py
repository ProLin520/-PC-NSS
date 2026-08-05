"""Fixed-FBSS resolution teacher for the four locked subarray scales."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ScaleTeacher:
    scale_scores: torch.Tensor
    scale_probabilities: torch.Tensor
    midpoint_deg: torch.Tensor
    subarray_sizes: tuple[int, ...]


def steering_vector_torch(
    angles_deg: torch.Tensor,
    sensor_count: int,
    spacing_wavelengths: float = 0.5,
    complex_dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    real_dtype = torch.float64 if complex_dtype == torch.complex128 else torch.float32
    angles = torch.as_tensor(angles_deg, dtype=real_dtype, device=angles_deg.device)
    sensors = torch.arange(sensor_count, dtype=real_dtype, device=angles.device)
    phase = (
        2.0
        * torch.pi
        * spacing_wavelengths
        * sensors[None, :, None]
        * torch.sin(torch.deg2rad(angles))[:, None, :]
    )
    return torch.exp(torch.complex(torch.zeros_like(phase), phase)).to(complex_dtype)


def normalized_music_denominator(
    covariance: torch.Tensor,
    angles_deg: torch.Tensor,
    source_count: int = 2,
    spacing_wavelengths: float = 0.5,
) -> torch.Tensor:
    """Return a^H Pn a / ||a||^2 for each batched angle."""

    matrix = covariance
    squeeze_batch = matrix.ndim == 2
    if squeeze_batch:
        matrix = matrix.unsqueeze(0)
    if matrix.ndim != 3 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("covariance must have shape [batch, sensor, sensor]")
    sensor_count = matrix.shape[-1]
    if not 0 < source_count < sensor_count:
        raise ValueError("source_count must lie in [1, sensor_count)")
    angles = torch.as_tensor(angles_deg, device=matrix.device)
    if angles.ndim == 1:
        angles = angles.unsqueeze(0).expand(matrix.shape[0], -1)
    if angles.ndim != 2 or angles.shape[0] != matrix.shape[0]:
        raise ValueError("angles_deg must have shape [batch, angle]")
    hermitian = 0.5 * (matrix + matrix.mH)
    _, eigenvectors = torch.linalg.eigh(hermitian)
    noise = eigenvectors[..., : sensor_count - source_count]
    steering = steering_vector_torch(
        angles,
        sensor_count,
        spacing_wavelengths,
        matrix.dtype,
    )
    response = noise.mH @ steering
    numerator = response.abs().square().sum(dim=-2)
    denominator = steering.abs().square().sum(dim=-2).clamp_min(1e-8)
    result = (numerator / denominator).real.clamp(0.0, 1.0)
    return result.squeeze(0) if squeeze_batch else result


def scale_probabilities_from_scores(
    scale_scores: torch.Tensor,
    tau_scale: float = 0.1,
) -> torch.Tensor:
    if tau_scale <= 0.0:
        raise ValueError("tau_scale must be positive")
    return torch.softmax(scale_scores / tau_scale, dim=-1)


def build_scale_teacher(
    fbss_covariances: dict[int, torch.Tensor],
    true_angles_deg: torch.Tensor,
    *,
    tau_scale: float = 0.1,
    source_count: int = 2,
    subarray_sizes: tuple[int, ...] = (4, 5, 6, 7),
) -> ScaleTeacher:
    """Compare fixed physical views without retaining an autograd graph."""

    if set(fbss_covariances) != set(subarray_sizes):
        raise ValueError("fbss_covariances must contain scales (4,5,6,7)")
    angles = torch.as_tensor(true_angles_deg)
    if angles.ndim != 2 or angles.shape[-1] != 2:
        raise ValueError("true_angles_deg must have shape [batch, 2]")
    midpoint = angles.mean(dim=-1)
    query_angles = torch.cat([angles, midpoint[:, None]], dim=-1)
    score_rows = []
    with torch.no_grad():
        for subarray_size in subarray_sizes:
            covariance = fbss_covariances[subarray_size].detach()
            q_values = normalized_music_denominator(
                covariance,
                query_angles.to(covariance.device),
                source_count=source_count,
            )
            score_rows.append(q_values[:, 2] - 0.5 * q_values[:, :2].sum(dim=-1))
        scores = torch.stack(score_rows, dim=-1)
        probabilities = scale_probabilities_from_scores(scores, tau_scale)
    return ScaleTeacher(
        scale_scores=scores,
        scale_probabilities=probabilities,
        midpoint_deg=midpoint.detach(),
        subarray_sizes=subarray_sizes,
    )
