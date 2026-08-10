"""Permutation-invariant, failure-aware DOA evaluation."""

from .matching import (
    FAILURE_PENALTY_DEG,
    MatchResult,
    hungarian_match,
    is_resolved,
    resolution_components,
)
from .metrics import (
    aggregate_metrics,
    aggregate_near_separation_audit,
    paired_comparison,
    score_doa_sample,
)

__all__ = [
    "FAILURE_PENALTY_DEG",
    "MatchResult",
    "aggregate_metrics",
    "aggregate_near_separation_audit",
    "hungarian_match",
    "is_resolved",
    "paired_comparison",
    "resolution_components",
    "score_doa_sample",
]
