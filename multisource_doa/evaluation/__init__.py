"""Permutation-invariant, failure-aware DOA evaluation."""

from .matching import FAILURE_PENALTY_DEG, MatchResult, hungarian_match, is_resolved
from .metrics import aggregate_metrics, paired_comparison, score_doa_sample

__all__ = [
    "FAILURE_PENALTY_DEG",
    "MatchResult",
    "aggregate_metrics",
    "hungarian_match",
    "is_resolved",
    "paired_comparison",
    "score_doa_sample",
]
