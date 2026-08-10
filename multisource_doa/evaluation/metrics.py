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
    resolution_components,
)


CONTINUOUS_STRATA_BINS = {
    "separation_deg": (
        (2.0, 4.0, "[2,4)"),
        (4.0, 6.0, "[4,6)"),
        (6.0, 8.0, "[6,8)"),
        (8.0, 10.0, "[8,10]"),
    ),
    "snr_db": (
        (-5.0, 0.0, "[-5,0)"),
        (0.0, 5.0, "[0,5)"),
        (5.0, 10.0, "[5,10]"),
    ),
}
NEAR_SEPARATION_BIN = "[2,4)"


@dataclass(frozen=True)
class SampleScore:
    sample_id: Hashable
    match: MatchResult
    resolved: bool
    both_angle_errors_within_1_deg: bool
    estimated_separation_at_least_half_true: bool
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
    components = resolution_components(match, true_angles)
    rmspe = float(np.sqrt(np.mean(np.square(match.absolute_errors_deg))))
    return SampleScore(
        sample_id=sample_id,
        match=match,
        resolved=is_resolved(match, true_angles),
        both_angle_errors_within_1_deg=components[
            "both_angle_errors_within_1_deg"
        ],
        estimated_separation_at_least_half_true=components[
            "estimated_separation_at_least_half_true"
        ],
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


def _continuous_bin_label(stratum: str, raw_value: Any) -> str:
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{stratum} must be a finite number") from error
    if not np.isfinite(value):
        raise ValueError(f"{stratum} must be a finite number")
    bins = CONTINUOUS_STRATA_BINS[stratum]
    for index, (lower, upper, label) in enumerate(bins):
        is_last = index == len(bins) - 1
        if lower <= value < upper or (
            is_last
            and lower <= value
            and (value <= upper or np.isclose(value, upper, atol=1e-9, rtol=0.0))
        ):
            return label
    lower = bins[0][0]
    upper = bins[-1][1]
    raise ValueError(f"{stratum}={value} lies outside [{lower},{upper}]")


def _stratum_label(stratum: str, raw_value: Any) -> str:
    if stratum in CONTINUOUS_STRATA_BINS:
        return _continuous_bin_label(stratum, raw_value)
    return str(raw_value)


def _index_unique_scores(
    scores: Iterable[SampleScore],
    side: str,
) -> dict[Hashable, SampleScore]:
    indexed: dict[Hashable, SampleScore] = {}
    for score in scores:
        if score.sample_id in indexed:
            raise ValueError(f"duplicate sample_id in {side}: {score.sample_id}")
        indexed[score.sample_id] = score
    return indexed


def _count_and_rate(count: int, sample_count: int) -> dict[str, int | float | None]:
    return {
        "count": int(count),
        "rate": float(count / sample_count) if sample_count else None,
    }


def aggregate_near_separation_audit(
    scores: Iterable[SampleScore],
) -> dict[str, Any]:
    """Decompose the frozen resolution rule for separation in [2,4)."""

    items = []
    for score in scores:
        if "separation_deg" not in score.strata:
            raise ValueError("near-separation audit requires separation_deg")
        if _continuous_bin_label(
            "separation_deg", score.strata["separation_deg"]
        ) == NEAR_SEPARATION_BIN:
            items.append(score)
    sample_count = len(items)
    angle_count = sum(item.both_angle_errors_within_1_deg for item in items)
    separation_count = sum(
        item.estimated_separation_at_least_half_true for item in items
    )
    resolved_count = sum(item.resolved for item in items)
    return {
        "separation_bin": NEAR_SEPARATION_BIN,
        "sample_count": sample_count,
        "both_angle_errors_within_1_deg": _count_and_rate(
            angle_count, sample_count
        ),
        "estimated_separation_at_least_half_true": _count_and_rate(
            separation_count, sample_count
        ),
        "resolved": _count_and_rate(resolved_count, sample_count),
        "sample_rmspe_gt_10_deg": _count_and_rate(
            sum(item.sample_rmspe_deg > 10.0 for item in items), sample_count
        ),
        "sample_rmspe_gt_30_deg": _count_and_rate(
            sum(item.sample_rmspe_deg > 30.0 for item in items), sample_count
        ),
        "sample_rmspe_gt_60_deg": _count_and_rate(
            sum(item.sample_rmspe_deg > 60.0 for item in items), sample_count
        ),
    }


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
    reference_by_id = _index_unique_scores(reference, "reference")
    candidate_by_id = _index_unique_scores(candidate, "candidate")
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
            value = _stratum_label(stratum, left.strata.get(stratum))
            counts = group.setdefault(value, _empty_counts())
            counts[label] += 1
    return result
