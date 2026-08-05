"""Hungarian matching and the fixed two-source resolution rule."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


FAILURE_PENALTY_DEG = 60.0
DUPLICATE_TOLERANCE_DEG = 1e-8


@dataclass(frozen=True)
class MatchResult:
    true_angles_deg: np.ndarray
    estimated_angles_deg: np.ndarray
    absolute_errors_deg: np.ndarray
    success: bool
    failure_reason: str | None


def _validate_true_angles(true_angles) -> np.ndarray:
    angles = np.asarray(true_angles, dtype=np.float64).reshape(-1)
    if angles.size != 2 or not np.isfinite(angles).all():
        raise ValueError("true_angles must contain two finite values")
    return angles


def _full_failure(
    true_angles: np.ndarray,
    failure_penalty_deg: float,
    reason: str,
) -> MatchResult:
    return MatchResult(
        true_angles_deg=true_angles,
        estimated_angles_deg=np.full(2, np.nan, dtype=np.float64),
        absolute_errors_deg=np.full(2, failure_penalty_deg, dtype=np.float64),
        success=False,
        failure_reason=reason,
    )


def hungarian_match(
    true_angles,
    estimated_angles,
    failure_penalty_deg: float = FAILURE_PENALTY_DEG,
) -> MatchResult:
    """Match two unordered estimates, preserving penalties for missing values."""

    truth = _validate_true_angles(true_angles)
    supplied = np.asarray(estimated_angles, dtype=np.float64).reshape(-1)
    if failure_penalty_deg <= 0.0:
        raise ValueError("failure_penalty_deg must be positive")
    if supplied.size > 2:
        return _full_failure(truth, failure_penalty_deg, "wrong_estimate_count")
    finite_supplied = supplied[np.isfinite(supplied)]
    if supplied.size == 2 and finite_supplied.size == 2:
        if abs(finite_supplied[0] - finite_supplied[1]) <= DUPLICATE_TOLERANCE_DEG:
            return _full_failure(truth, failure_penalty_deg, "duplicate_estimate")

    estimated = np.full(2, np.nan, dtype=np.float64)
    estimated[: supplied.size] = supplied
    valid = np.isfinite(estimated)
    cost = np.full((2, 2), failure_penalty_deg, dtype=np.float64)
    cost[:, valid] = np.abs(truth[:, None] - estimated[valid][None, :])
    rows, columns = linear_sum_assignment(cost)
    matched = np.full(2, np.nan, dtype=np.float64)
    errors = np.full(2, failure_penalty_deg, dtype=np.float64)
    for row, column in zip(rows, columns):
        if valid[column]:
            matched[row] = estimated[column]
            errors[row] = cost[row, column]

    success = bool(valid.all())
    if success:
        reason = None
    elif supplied.size < 2 or finite_supplied.size == 1:
        reason = "missing_angle"
    else:
        reason = "nonfinite_estimate"
    return MatchResult(
        true_angles_deg=truth,
        estimated_angles_deg=matched,
        absolute_errors_deg=errors,
        success=success,
        failure_reason=reason,
    )


def is_resolved(match: MatchResult, true_angles) -> bool:
    truth = np.sort(_validate_true_angles(true_angles))
    if not match.success or not np.isfinite(match.estimated_angles_deg).all():
        return False
    estimated = np.sort(match.estimated_angles_deg)
    true_separation = float(truth[1] - truth[0])
    estimated_separation = float(estimated[1] - estimated[0])
    return bool(
        np.all(match.absolute_errors_deg <= 1.0)
        and estimated_separation >= 0.5 * true_separation
    )
