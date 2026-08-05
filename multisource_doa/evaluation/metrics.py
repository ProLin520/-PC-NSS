"""Failure-aware aggregate metrics and paired estimator comparisons."""

from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Hashable, Iterable

import numpy as np

from multisource_doa.evaluation.matching import (
    FAILURE_PENALTY_DEG,
    MatchResult,
    hungarian_match,
    is_resolved,
)


@dataclass(frozen=True)
class SampleScore:
    sample_id: Hashable
    match: MatchResult
    resolved: bool
    sample_rmspe_deg: float
    strata: dict[str, Any]

    @property
    def success(self) -> bool:
        return self.match.success

    @property
    def failure_reason(self) -> str | None:
        return self.match.failure_reason


def score_doa_sample(
    sample_id: Hashable,
    true_angles,
    estimated_angles,
    *,
    estimate_success: bool = True,
    failure_reason: str | None = None,
    strata: dict[str, Any] | None = None,
) -> SampleScore:
    match = hungarian_match(true_angles, estimated_angles)
    if not estimate_success:
        match = replace(
            match,
            estimated_angles_deg=np.full(2, np.nan, dtype=np.float64),
            absolute_errors_deg=np.full(2, FAILURE_PENALTY_DEG, dtype=np.float64),
            success=False,
            failure_reason=failure_reason or match.failure_reason or "estimator_failure",
        )
    rmspe = float(np.sqrt(np.mean(np.square(match.absolute_errors_deg))))
    return SampleScore(
        sample_id=sample_id,
        match=match,
        resolved=is_resolved(match, true_angles),
        sample_rmspe_deg=rmspe,
        strata=dict(strata or {}),
    )


def aggregate_metrics(scores: Iterable[SampleScore]) -> dict[str, Any]:
    items = list(scores)
    if not items:
        raise ValueError("at least one SampleScore is required")
    all_errors = np.concatenate(
        [item.match.absolute_errors_deg for item in items],
        axis=0,
    )
    successful_errors = [
        item.match.absolute_errors_deg for item in items if item.success
    ]
    conditional = (
        np.concatenate(successful_errors, axis=0)
        if successful_errors
        else np.empty(0, dtype=np.float64)
    )
    reasons = Counter(
        item.failure_reason for item in items if item.failure_reason is not None
    )
    return {
        "sample_count": len(items),
        "failure_aware_rmspe_deg": float(np.sqrt(np.mean(np.square(all_errors)))),
        "conditional_rmse_deg": (
            float(np.sqrt(np.mean(np.square(conditional))))
            if conditional.size
            else None
        ),
        "mae_deg": float(np.mean(np.abs(all_errors))),
        "p95_abs_error_deg": float(np.percentile(np.abs(all_errors), 95)),
        "p99_abs_error_deg": float(np.percentile(np.abs(all_errors), 99)),
        "max_abs_error_deg": float(np.max(np.abs(all_errors))),
        "resolution_rate": float(np.mean([item.resolved for item in items])),
        "failure_count": int(sum(not item.success for item in items)),
        "failure_reasons": dict(sorted(reasons.items())),
    }


def _empty_counts() -> dict[str, int]:
    return {"win": 0, "tie": 0, "loss": 0}


def _comparison_label(
    reference_rmspe: float,
    candidate_rmspe: float,
    tie_tolerance_deg: float,
) -> str:
    difference = candidate_rmspe - reference_rmspe
    if abs(difference) <= tie_tolerance_deg:
        return "tie"
    return "win" if difference < 0.0 else "loss"


def paired_comparison(
    reference: Iterable[SampleScore],
    candidate: Iterable[SampleScore],
    *,
    tie_tolerance_deg: float = 1e-6,
    strata_names: tuple[str, ...] = (
        "separation_deg",
        "snr_db",
        "snapshot_count",
        "rho",
    ),
) -> dict[str, Any]:
    reference_by_id = {item.sample_id: item for item in reference}
    candidate_by_id = {item.sample_id: item for item in candidate}
    if reference_by_id.keys() != candidate_by_id.keys():
        raise ValueError("paired comparisons require identical sample ids")
    result: dict[str, Any] = {"overall": _empty_counts()}
    for sample_id in reference_by_id:
        left = reference_by_id[sample_id]
        right = candidate_by_id[sample_id]
        label = _comparison_label(
            left.sample_rmspe_deg,
            right.sample_rmspe_deg,
            tie_tolerance_deg,
        )
        result["overall"][label] += 1
        for stratum in strata_names:
            if stratum not in left.strata and stratum not in right.strata:
                continue
            if left.strata.get(stratum) != right.strata.get(stratum):
                raise ValueError(f"paired stratum mismatch for {stratum}")
            group_name = f"by_{stratum}"
            group = result.setdefault(group_name, {})
            value = str(left.strata.get(stratum))
            counts = group.setdefault(value, _empty_counts())
            counts[label] += 1
    return result
