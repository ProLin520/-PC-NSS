"""Train-only failure-aware fixed-scale angular-error teacher labels."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from multisource_doa.baselines.classical import (
    DOAEstimate,
    evaluate_fixed_scale_family,
)
from multisource_doa.data.simulator import DOASample
from multisource_doa.evaluation.metrics import score_doa_sample


SCALE_SIZES = (4, 5, 6, 7)
ERROR_TIE_TOLERANCE_DEG = 1e-6


def teacher_probabilities_from_rmspe(
    rmspe_by_scale: Mapping[int, float],
) -> tuple[float, float, float, float]:
    """Return a tie-aware hard distribution over L4-L7."""

    if set(rmspe_by_scale) != set(SCALE_SIZES):
        raise ValueError("rmspe_by_scale must contain L4-L7")
    values = tuple(float(rmspe_by_scale[size]) for size in SCALE_SIZES)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("fixed-scale RMSPE values must be finite and non-negative")
    minimum = min(values)
    winners = tuple(
        index
        for index, value in enumerate(values)
        if value - minimum <= ERROR_TIE_TOLERANCE_DEG
    )
    mass = 1.0 / len(winners)
    return tuple(mass if index in winners else 0.0 for index in range(4))


def build_error_teacher_row(
    sample: DOASample,
    *,
    sample_index: int,
    estimates_by_scale: Mapping[int, DOAEstimate] | None = None,
) -> dict[str, Any]:
    """Score all fixed scales without hiding estimator failures."""

    if isinstance(sample_index, bool) or not isinstance(sample_index, int):
        raise ValueError("sample_index must be an integer")
    estimates = estimates_by_scale
    if estimates is None:
        family = evaluate_fixed_scale_family(
            sample.snapshots,
            subarray_sizes=SCALE_SIZES,
            source_count=2,
        )
        estimates = {
            size: family[f"fbss_root_music_L{size}"] for size in SCALE_SIZES
        }
    if set(estimates) != set(SCALE_SIZES):
        raise ValueError("estimates_by_scale must contain L4-L7")

    row: dict[str, Any] = {
        "sample_index": sample_index,
        "sample_seed": int(sample.sample_seed),
        "true_angle_1_deg": float(sample.angles_deg[0]),
        "true_angle_2_deg": float(sample.angles_deg[1]),
        "separation_deg": float(abs(np.diff(sample.angles_deg)[0])),
        "rho": float(sample.rho),
        "snr_db": float(sample.snr_db),
        "snapshot_count": int(sample.snapshot_count),
    }
    rmspe_by_scale: dict[int, float] = {}
    for size in SCALE_SIZES:
        estimate = estimates[size]
        score = score_doa_sample(
            sample.sample_seed,
            sample.angles_deg,
            estimate.angles_deg,
            estimate_success=estimate.success,
            failure_reason=estimate.failure_reason,
        )
        rmspe_by_scale[size] = score.sample_rmspe_deg
        row[f"success_L{size}"] = score.success
        row[f"failure_reason_L{size}"] = score.failure_reason or ""
        for index in range(2):
            estimate_value = score.match.estimated_angles_deg[index]
            row[f"estimated_angle_{index + 1}_deg_L{size}"] = (
                float(estimate_value) if np.isfinite(estimate_value) else None
            )
            row[f"absolute_error_{index + 1}_deg_L{size}"] = float(
                score.match.absolute_errors_deg[index]
            )
        row[f"sample_rmspe_deg_L{size}"] = score.sample_rmspe_deg

    probabilities = teacher_probabilities_from_rmspe(rmspe_by_scale)
    winners = tuple(
        size
        for size, probability in zip(SCALE_SIZES, probabilities, strict=True)
        if probability > 0.0
    )
    row["teacher_probabilities"] = probabilities
    row["best_scales"] = winners
    row["has_tied_best"] = len(winners) > 1
    row["all_scales_failed"] = all(
        not bool(row[f"success_L{size}"]) for size in SCALE_SIZES
    )
    return row
