"""Traditional estimators behind a common failure-aware interface."""

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

import numpy as np
from scipy.signal import find_peaks

from multisource_doa.data.simulator import steering_vector
from multisource_doa.physics.root_music import estimate_root_music
from multisource_doa.physics.spatial_smoothing import fbss_covariance, sps_covariance


@dataclass(frozen=True)
class DOAEstimate:
    algorithm: str
    angles_deg: np.ndarray
    success: bool
    failure_reason: str | None
    runtime_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


def _timed(algorithm: str, operation: Callable[[], tuple]) -> DOAEstimate:
    started = perf_counter()
    angles, success, failure_reason, metadata = operation()
    runtime = perf_counter() - started
    return DOAEstimate(
        algorithm=algorithm,
        angles_deg=np.asarray(angles, dtype=np.float64),
        success=bool(success),
        failure_reason=failure_reason,
        runtime_seconds=float(runtime),
        metadata=dict(metadata),
    )


def root_music_raw(
    covariance: np.ndarray,
    source_count: int = 2,
    spacing_wavelengths: float = 0.5,
    angle_limits_deg: tuple[float, float] = (-60.0, 60.0),
) -> DOAEstimate:
    def operation():
        result = estimate_root_music(
            covariance,
            source_count=source_count,
            spacing_wavelengths=spacing_wavelengths,
            angle_limits_deg=angle_limits_deg,
        )
        return (
            result.angles_deg,
            result.success,
            result.failure_reason,
            {
                "candidate_count": result.candidate_count,
                "minimum_root_separation": result.minimum_root_separation,
            },
        )

    return _timed("root_music", operation)


def music_scan(
    covariance: np.ndarray,
    source_count: int = 2,
    spacing_wavelengths: float = 0.5,
    angle_limits_deg: tuple[float, float] = (-60.0, 60.0),
    grid_step_deg: float = 0.05,
) -> DOAEstimate:
    def operation():
        matrix = np.asarray(covariance, dtype=np.complex128)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            return np.empty(0), False, "nonsquare_covariance", {}
        if not np.isfinite(matrix).all():
            return np.empty(0), False, "nonfinite_covariance", {}
        size = matrix.shape[0]
        if source_count <= 0 or source_count >= size:
            return np.empty(0), False, "invalid_source_count", {}
        values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
        if np.ptp(values) <= 1e-10 * max(float(np.max(np.abs(values))), 1e-12):
            return np.empty(0), False, "rankless_covariance", {}
        noise = vectors[:, : size - source_count]
        projection = noise @ noise.conj().T
        grid = np.arange(
            angle_limits_deg[0],
            angle_limits_deg[1] + 0.5 * grid_step_deg,
            grid_step_deg,
        )
        steering = steering_vector(grid, size, spacing_wavelengths)
        denominator = np.einsum(
            "ma,mn,na->a",
            steering.conj(),
            projection,
            steering,
        ).real
        spectrum = 1.0 / np.clip(denominator, 1e-12, None)
        peaks, _ = find_peaks(spectrum, distance=max(1, round(0.1 / grid_step_deg)))
        if peaks.size < source_count:
            return np.empty(0), False, "insufficient_peaks", {"peak_count": int(peaks.size)}
        chosen = peaks[np.argsort(spectrum[peaks])[-source_count:]]
        angles = np.sort(grid[chosen])
        return angles, True, None, {"peak_count": int(peaks.size)}

    return _timed("music", operation)


def esprit(
    covariance: np.ndarray,
    source_count: int = 2,
    spacing_wavelengths: float = 0.5,
    angle_limits_deg: tuple[float, float] = (-60.0, 60.0),
) -> DOAEstimate:
    def operation():
        matrix = np.asarray(covariance, dtype=np.complex128)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            return np.empty(0), False, "nonsquare_covariance", {}
        if not np.isfinite(matrix).all():
            return np.empty(0), False, "nonfinite_covariance", {}
        size = matrix.shape[0]
        if source_count <= 0 or source_count >= size:
            return np.empty(0), False, "invalid_source_count", {}
        values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
        if np.ptp(values) <= 1e-10 * max(float(np.max(np.abs(values))), 1e-12):
            return np.empty(0), False, "rankless_covariance", {}
        signal = vectors[:, -source_count:]
        rotation = np.linalg.pinv(signal[:-1]) @ signal[1:]
        roots = np.linalg.eigvals(rotation)
        sine_values = np.angle(roots) / (2.0 * np.pi * spacing_wavelengths)
        if not np.isfinite(sine_values).all() or np.any(np.abs(sine_values) > 1.0):
            return np.empty(0), False, "invalid_esprit_roots", {}
        angles = np.sort(np.rad2deg(np.arcsin(sine_values)).real)
        lower, upper = angle_limits_deg
        if np.any(angles < lower) or np.any(angles > upper):
            return np.empty(0), False, "angle_out_of_bounds", {}
        if np.min(np.diff(angles)) <= 0.05:
            return np.empty(0), False, "duplicate_roots", {}
        return angles, True, None, {}

    return _timed("esprit", operation)


def _smoothed_root_music(
    snapshots: np.ndarray,
    subarray_size: int,
    source_count: int,
    use_forward_backward: bool,
) -> DOAEstimate:
    prefix = "fbss" if use_forward_backward else "sps"
    algorithm = f"{prefix}_root_music_L{subarray_size}"

    def operation():
        covariance = (
            fbss_covariance(snapshots, subarray_size)
            if use_forward_backward
            else sps_covariance(snapshots, subarray_size)
        )
        result = estimate_root_music(covariance, source_count=source_count)
        return (
            result.angles_deg,
            result.success,
            result.failure_reason,
            {"subarray_size": subarray_size, "candidate_count": result.candidate_count},
        )

    return _timed(algorithm, operation)


def sps_root_music(
    snapshots: np.ndarray,
    subarray_size: int,
    source_count: int = 2,
) -> DOAEstimate:
    return _smoothed_root_music(
        snapshots,
        subarray_size,
        source_count,
        use_forward_backward=False,
    )


def fbss_root_music(
    snapshots: np.ndarray,
    subarray_size: int,
    source_count: int = 2,
) -> DOAEstimate:
    return _smoothed_root_music(
        snapshots,
        subarray_size,
        source_count,
        use_forward_backward=True,
    )


def evaluate_fixed_scale_family(
    snapshots: np.ndarray,
    subarray_sizes: tuple[int, ...] = (4, 5, 6, 7),
    source_count: int = 2,
) -> dict[str, DOAEstimate]:
    estimates: dict[str, DOAEstimate] = {}
    for subarray_size in subarray_sizes:
        for estimator in (sps_root_music, fbss_root_music):
            estimate = estimator(snapshots, subarray_size, source_count)
            estimates[estimate.algorithm] = estimate
    return estimates
