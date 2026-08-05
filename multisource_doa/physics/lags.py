"""Convert raw and FBSS covariance views into padded first-column lags."""

from dataclasses import dataclass

import numpy as np

from multisource_doa.physics.covariance import sample_covariance
from multisource_doa.physics.spatial_smoothing import (
    fbss_covariance,
    subarray_covariances,
)


@dataclass(frozen=True)
class MultiScaleViews:
    raw_covariance: np.ndarray
    raw_lags: np.ndarray
    fbss_covariances: dict[int, np.ndarray]
    fbss_lags: np.ndarray
    valid_mask: np.ndarray
    effective_counts: np.ndarray
    quality_features: np.ndarray


def covariance_to_lags(
    covariance: np.ndarray,
    output_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average lower diagonals so lag k follows the covariance first column."""

    matrix = np.asarray(covariance, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be square")
    if output_size < matrix.shape[0]:
        raise ValueError("output_size cannot be smaller than covariance size")
    size = matrix.shape[0]
    lags = np.zeros(output_size, dtype=np.complex128)
    mask = np.zeros(output_size, dtype=bool)
    diagonal_counts = np.zeros(output_size, dtype=np.float64)
    for lag in range(size):
        diagonal = np.diag(matrix, k=-lag)
        lags[lag] = diagonal.mean()
        mask[lag] = True
        diagonal_counts[lag] = diagonal.size
    return lags, mask, diagonal_counts


def _quality_features(
    snapshots: np.ndarray,
    subarray_size: int,
    raw_covariance: np.ndarray,
    source_count: int,
) -> np.ndarray:
    epsilon = 1e-8
    subarray_views = subarray_covariances(snapshots, subarray_size)
    forward = subarray_views.mean(axis=0)
    backward = fbss_covariance(snapshots, subarray_size)
    eigenvalues = np.linalg.eigvalsh(backward).real
    clipped = np.clip(eigenvalues, epsilon, None)
    raw_mean_power = np.trace(raw_covariance).real / raw_covariance.shape[0]
    mean_power = np.trace(backward).real / subarray_size
    normalized_trace = mean_power / max(raw_mean_power, epsilon)
    log_condition = np.log1p(clipped[-1] / clipped[0])
    noise_count = max(1, subarray_size - source_count)
    noise_floor = clipped[:noise_count].mean()
    signal_noise_ratio = clipped[-source_count] / max(noise_floor, epsilon)
    forward_norm = max(np.linalg.norm(forward, ord="fro"), epsilon)
    fb_change = np.linalg.norm(backward - forward, ord="fro") / forward_norm
    dispersion = np.mean(
        [
            np.linalg.norm(view - forward, ord="fro") / forward_norm
            for view in subarray_views
        ]
    )
    normalized_subarray_count = subarray_views.shape[0] / snapshots.shape[0]
    quality = np.asarray(
        [
            normalized_trace,
            log_condition,
            signal_noise_ratio,
            fb_change,
            dispersion,
            normalized_subarray_count,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(quality).all():
        raise FloatingPointError("non-finite multiscale quality feature")
    return quality


def build_multiscale_views(
    snapshots: np.ndarray,
    subarray_sizes: tuple[int, ...] = (4, 5, 6, 7),
    output_size: int = 8,
    source_count: int = 2,
) -> MultiScaleViews:
    """Build raw full-aperture anchor and four FBSS lag views."""

    array = np.asarray(snapshots, dtype=np.complex128)
    if output_size != array.shape[0]:
        raise ValueError("output_size must equal the full sensor count")
    if tuple(sorted(set(subarray_sizes))) != tuple(subarray_sizes):
        raise ValueError("subarray_sizes must be unique and increasing")
    raw_covariance = sample_covariance(array)
    raw_lags, _, _ = covariance_to_lags(raw_covariance, output_size)
    covariance_views: dict[int, np.ndarray] = {}
    lag_rows = []
    masks = []
    effective_counts = []
    qualities = []
    snapshot_count = array.shape[1]
    sensor_count = array.shape[0]
    for subarray_size in subarray_sizes:
        if source_count >= subarray_size:
            raise ValueError("source_count must be smaller than every subarray")
        covariance = fbss_covariance(array, subarray_size)
        covariance_views[subarray_size] = covariance
        lags, mask, diagonal_counts = covariance_to_lags(
            covariance,
            output_size,
        )
        subarray_count = sensor_count - subarray_size + 1
        lag_rows.append(lags)
        masks.append(mask)
        effective_counts.append(
            diagonal_counts * subarray_count * snapshot_count
        )
        qualities.append(
            _quality_features(
                array,
                subarray_size,
                raw_covariance,
                source_count,
            )
        )
    return MultiScaleViews(
        raw_covariance=raw_covariance,
        raw_lags=raw_lags,
        fbss_covariances=covariance_views,
        fbss_lags=np.stack(lag_rows, axis=0),
        valid_mask=np.stack(masks, axis=0),
        effective_counts=np.stack(effective_counts, axis=0),
        quality_features=np.stack(qualities, axis=0),
    )
