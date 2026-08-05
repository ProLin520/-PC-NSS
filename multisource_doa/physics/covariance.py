"""Sample covariance utilities."""

import numpy as np


def sample_covariance(snapshots: np.ndarray) -> np.ndarray:
    """Return X X^H / T for snapshots arranged as sensors by time."""

    array = np.asarray(snapshots, dtype=np.complex128)
    if array.ndim != 2:
        raise ValueError("snapshots must have shape [sensor, snapshot]")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("snapshots must contain at least one sensor and snapshot")
    if not np.isfinite(array).all():
        raise ValueError("snapshots must be finite")
    return array @ array.conj().T / array.shape[1]
