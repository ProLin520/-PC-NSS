"""Fixed positive-phase Root-MUSIC implementation with explicit failures."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RootMusicResult:
    angles_deg: np.ndarray
    success: bool
    failure_reason: str | None
    selected_roots: np.ndarray
    candidate_count: int
    minimum_root_separation: float


def _failure(reason: str, candidate_count: int = 0) -> RootMusicResult:
    return RootMusicResult(
        angles_deg=np.empty(0, dtype=np.float64),
        success=False,
        failure_reason=reason,
        selected_roots=np.empty(0, dtype=np.complex128),
        candidate_count=int(candidate_count),
        minimum_root_separation=0.0,
    )


def _noise_polynomial(noise_projection: np.ndarray) -> np.ndarray:
    size = noise_projection.shape[0]
    ascending = np.asarray(
        [
            np.diag(noise_projection, k=offset).sum()
            for offset in range(-(size - 1), size)
        ],
        dtype=np.complex128,
    )
    scale = np.max(np.abs(ascending))
    if not np.isfinite(scale) or scale <= 0.0:
        return np.empty(0, dtype=np.complex128)
    normalized = ascending / scale
    first = 0
    last = normalized.size
    while first < last and abs(normalized[first]) < 1e-12:
        first += 1
    while last > first and abs(normalized[last - 1]) < 1e-12:
        last -= 1
    return normalized[first:last][::-1]


def estimate_root_music(
    covariance: np.ndarray,
    source_count: int = 2,
    spacing_wavelengths: float = 0.5,
    angle_limits_deg: tuple[float, float] = (-60.0, 60.0),
    duplicate_tolerance_deg: float = 0.05,
) -> RootMusicResult:
    """Estimate known-count ULA DOAs without any learned or spectral fallback."""

    matrix = np.asarray(covariance, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return _failure("nonsquare_covariance")
    if not np.isfinite(matrix).all():
        return _failure("nonfinite_covariance")
    sensor_count = matrix.shape[0]
    if source_count <= 0 or source_count >= sensor_count:
        return _failure("invalid_source_count")
    if spacing_wavelengths <= 0.0:
        return _failure("invalid_spacing")

    hermitian = 0.5 * (matrix + matrix.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    eigen_scale = max(float(np.max(np.abs(eigenvalues))), 1e-12)
    if float(np.ptp(eigenvalues)) <= 1e-10 * eigen_scale:
        return _failure("rankless_covariance")
    noise_subspace = eigenvectors[:, : sensor_count - source_count]
    noise_projection = noise_subspace @ noise_subspace.conj().T
    polynomial = _noise_polynomial(noise_projection)
    if polynomial.size <= 1:
        return _failure("invalid_root_polynomial")
    roots = np.roots(polynomial)
    candidates = roots[
        np.isfinite(roots)
        & (np.abs(roots) > 1e-12)
        & (np.abs(roots) <= 1.0 + 1e-6)
    ]
    candidates = candidates[np.argsort(np.abs(np.abs(candidates) - 1.0))]

    selected_angles: list[float] = []
    selected_roots: list[complex] = []
    lower, upper = angle_limits_deg
    for root in candidates:
        sine_value = float(np.angle(root) / (2.0 * np.pi * spacing_wavelengths))
        if not -1.0 <= sine_value <= 1.0:
            continue
        angle = float(np.rad2deg(np.arcsin(sine_value)))
        if not lower <= angle <= upper:
            continue
        if any(abs(angle - existing) <= duplicate_tolerance_deg for existing in selected_angles):
            continue
        selected_angles.append(angle)
        selected_roots.append(complex(root))
        if len(selected_angles) == source_count:
            break
    if len(selected_angles) != source_count:
        return _failure("insufficient_distinct_roots", candidate_count=candidates.size)

    order = np.argsort(selected_angles)
    angles = np.asarray(selected_angles, dtype=np.float64)[order]
    chosen = np.asarray(selected_roots, dtype=np.complex128)[order]
    minimum_separation = float(np.min(np.diff(angles))) if source_count > 1 else np.inf
    if minimum_separation <= duplicate_tolerance_deg:
        return _failure("duplicate_roots", candidate_count=candidates.size)
    return RootMusicResult(
        angles_deg=angles,
        success=True,
        failure_reason=None,
        selected_roots=chosen,
        candidate_count=int(candidates.size),
        minimum_root_separation=minimum_separation,
    )
