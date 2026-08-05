"""Forward and forward-backward spatial smoothing."""

import numpy as np

from multisource_doa.physics.covariance import sample_covariance


def subarray_covariances(
    snapshots: np.ndarray,
    subarray_size: int,
) -> np.ndarray:
    array = np.asarray(snapshots, dtype=np.complex128)
    if array.ndim != 2:
        raise ValueError("snapshots must have shape [sensor, snapshot]")
    sensor_count = array.shape[0]
    if not 1 < subarray_size <= sensor_count:
        raise ValueError("subarray_size must lie in [2, sensor_count]")
    return np.stack(
        [
            sample_covariance(array[start : start + subarray_size])
            for start in range(sensor_count - subarray_size + 1)
        ],
        axis=0,
    )


def sps_covariance(snapshots: np.ndarray, subarray_size: int) -> np.ndarray:
    """Average all forward overlapping subarray covariances."""

    return subarray_covariances(snapshots, subarray_size).mean(axis=0)


def forward_backward_average(covariance: np.ndarray) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be square")
    reversal = np.fliplr(np.eye(matrix.shape[0], dtype=np.complex128))
    return 0.5 * (matrix + reversal @ matrix.conj() @ reversal)


def fbss_covariance(snapshots: np.ndarray, subarray_size: int) -> np.ndarray:
    """Apply spatial smoothing followed by forward-backward averaging."""

    return forward_backward_average(sps_covariance(snapshots, subarray_size))
