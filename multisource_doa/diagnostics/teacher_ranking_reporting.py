"""Schema-v1 reporting for the frozen Task 16 ranking diagnosis."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

from multisource_doa.diagnostics.teacher_ranking import (
    SCALE_SIZES,
    THRESHOLD_COHORTS,
    TeacherRankingResult,
)


TEACHER_RANKING_SCHEMA_VERSION = 1
SIGNALS = ("current_score", "q_midpoint", "negative_truth_mean")
RHO_VALUES = (0.8, 0.9, 0.99, 1.0)
SNR_BINS = (
    ("[-5,0)", -5.0, 0.0, False),
    ("[0,5)", 0.0, 5.0, False),
    ("[5,10]", 5.0, 10.0, True),
)
SNAPSHOT_VALUES = (8, 20, 50)
PAIR_COUNT_FIELDS = (
    "concordant_pair_count",
    "discordant_pair_count",
    "teacher_tie_pair_count",
    "oracle_tie_pair_count",
    "both_tie_pair_count",
    "pairwise_comparable_count",
    "exact_signal_tie_pair_count",
)
COMPONENT_FIELDS = (
    "q_midpoint_range",
    "q_midpoint_std",
    "negative_truth_mean_range",
    "negative_truth_mean_std",
    "current_score_range",
    "current_score_std",
)


def build_teacher_ranking_summary(
    sample_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = list(sample_rows)
    if not rows:
        raise ValueError("teacher ranking summary requires at least one sample")
    return {
        "sample_count": len(rows),
        "signals": {
            signal: _aggregate_signal(rows, signal) for signal in SIGNALS
        },
    }


def build_teacher_component_summary(
    sample_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = list(sample_rows)
    if not rows:
        raise ValueError("teacher component summary requires at least one sample")
    summary = {
        field: _nullable_distribution([row[field] for row in rows])
        for field in COMPONENT_FIELDS
    }
    summary["cancellation_ratio"] = _nullable_distribution(
        [row["cancellation_ratio"] for row in rows]
    )
    summary["cancellation_denominator_zero_count"] = sum(
        bool(row["cancellation_denominator_zero"]) for row in rows
    )
    return summary


def build_teacher_ranking_stratified_summary(
    sample_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = list(sample_rows)
    if not rows:
        raise ValueError("teacher ranking stratification requires samples")
    dimensions: tuple[
        tuple[str, tuple[str, ...], Callable[[Mapping[str, Any]], str]], ...
    ] = (
        ("rho", tuple(str(value) for value in RHO_VALUES), _rho_bin),
        ("snr_db", tuple(item[0] for item in SNR_BINS), _snr_bin),
        (
            "snapshot_count",
            tuple(str(value) for value in SNAPSHOT_VALUES),
            _snapshot_bin,
        ),
        ("threshold_cohort", THRESHOLD_COHORTS, _cohort_bin),
    )
    output: list[dict[str, Any]] = []
    for signal in SIGNALS:
        for dimension, bin_names, bin_for_row in dimensions:
            dimension_total = 0
            for bin_name in bin_names:
                selected = [row for row in rows if bin_for_row(row) == bin_name]
                aggregate = _aggregate_signal(selected, signal)
                output.append(
                    {
                        "signal": signal,
                        "dimension": dimension,
                        "bin": bin_name,
                        "sample_count": len(selected),
                        **aggregate,
                    }
                )
                dimension_total += len(selected)
            if dimension_total != len(rows):
                raise ValueError(f"{dimension} strata do not account for all samples")
    return output


def build_teacher_oracle_confusion(
    sample_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = list(sample_rows)
    if not rows:
        raise ValueError("teacher oracle confusion requires samples")
    weights = {(teacher, oracle): 0.0 for teacher in SCALE_SIZES for oracle in SCALE_SIZES}
    oracle_tie_count = 0
    for row in rows:
        teacher = int(row["current_score_top1_scale"])
        oracle = tuple(int(value) for value in row["current_score_oracle_best_scales"])
        if teacher not in SCALE_SIZES or not oracle or any(value not in SCALE_SIZES for value in oracle):
            raise ValueError("confusion row contains an invalid scale")
        oracle_tie_count += int(len(oracle) > 1)
        contribution = 1.0 / len(oracle)
        for oracle_scale in oracle:
            weights[(teacher, oracle_scale)] += contribution
    matrix_sum = math.fsum(weights.values())
    if not math.isclose(matrix_sum, len(rows), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("confusion matrix does not preserve sample weight")
    return [
        {
            "teacher_top1_scale": teacher,
            "oracle_scale": oracle,
            "weighted_count": weights[(teacher, oracle)],
            "oracle_tie_sample_count": oracle_tie_count,
            "matrix_weight_sum": matrix_sum,
        }
        for teacher in SCALE_SIZES
        for oracle in SCALE_SIZES
    ]


def build_teacher_ranking_decision(
    summary: Mapping[str, Any],
    stratified_rows: Sequence[Mapping[str, Any]],
    *,
    task15_margin_over_tau_median: float,
    engineering_integrity: bool,
) -> dict[str, Any]:
    signals = summary["signals"]
    dimension_support = {
        signal: {
            dimension: _dimension_support(stratified_rows, signal, dimension)
            for dimension in ("rho", "snr_db", "snapshot_count")
        }
        for signal in SIGNALS
    }
    current = signals["current_score"]
    current_pairwise = _optional_finite(current["pairwise_concordance_rate"])
    calibration_gates = {
        "pairwise_concordance": _at_least(current_pairwise, 0.60),
        "median_kendall_tau_b": _at_least(
            _optional_finite(current["kendall_tau_b"]["median"]), 0.20
        ),
        "top1_oracle_agreement": _at_least(
            _optional_finite(current["top1_oracle_agreement_rate"]), 0.40
        ),
        "top2_oracle_coverage": _at_least(
            _optional_finite(current["top2_oracle_coverage_rate"]), 0.70
        ),
        "median_regret": _at_most(
            _optional_finite(current["top1_regret_deg"]["median"]), 1.0
        ),
        "stratified_support": sum(dimension_support["current_score"].values())
        >= 2,
        "task15_margin_over_tau_low": _finite(task15_margin_over_tau_median)
        < 0.10,
        "engineering_integrity": engineering_integrity is True,
    }
    calibration_only = all(calibration_gates.values())
    component_gates: dict[str, dict[str, bool]] = {}
    candidate_components: list[str] = []
    if not calibration_only:
        for signal in ("q_midpoint", "negative_truth_mean"):
            component = signals[signal]
            component_pairwise = _optional_finite(
                component["pairwise_concordance_rate"]
            )
            gates = {
                "pairwise_concordance": _at_least(component_pairwise, 0.60),
                "pairwise_improvement": (
                    component_pairwise is not None
                    and current_pairwise is not None
                    and component_pairwise - current_pairwise >= 0.05
                ),
                "top2_oracle_coverage": _at_least(
                    _optional_finite(component["top2_oracle_coverage_rate"]),
                    0.70,
                ),
                "stratified_support": sum(dimension_support[signal].values()) >= 2,
            }
            component_gates[signal] = gates
            if all(gates.values()):
                candidate_components.append(signal)
    if calibration_only:
        conclusion = "calibration_only"
        next_direction = "teacher_score_calibration_on_train_split"
    elif candidate_components:
        conclusion = "component_cancellation"
        next_direction = "single_factor_teacher_formula_design_on_train_split"
    else:
        conclusion = "ranking_invalid"
        next_direction = "failure_aware_fixed_scale_angle_rmspe_teacher_on_train_split"
    return {
        "mechanism_conclusion": conclusion,
        "next_direction": next_direction,
        "candidate_components": candidate_components,
        "training_authorized": False,
        "teacher_modified": False,
        "calibration_gates": calibration_gates,
        "component_gates": component_gates,
        "dimension_support": dimension_support,
        "task15_margin_over_tau_median": task15_margin_over_tau_median,
        "engineering_integrity": engineering_integrity,
    }


def write_teacher_ranking_report(
    result: TeacherRankingResult,
    output_directory: str | Path,
    *,
    diagnostic_config: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    task15_margin_over_tau_median: float,
    engineering_integrity: bool,
) -> Path:
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    rows = list(result.sample_rows)
    summary = build_teacher_ranking_summary(rows)
    component_summary = build_teacher_component_summary(rows)
    strata = build_teacher_ranking_stratified_summary(rows)
    confusion = build_teacher_oracle_confusion(rows)
    decision = build_teacher_ranking_decision(
        summary,
        strata,
        task15_margin_over_tau_median=task15_margin_over_tau_median,
        engineering_integrity=engineering_integrity,
    )
    manifest = {
        **source_manifest,
        "teacher_ranking_schema_version": TEACHER_RANKING_SCHEMA_VERSION,
    }
    for field, expected in (
        ("sample_count", len(rows)),
        ("device", "cpu"),
        ("batch_size", 128),
        ("no_model_forward", True),
        ("teacher_modified", False),
        ("training_performed", False),
    ):
        if manifest.get(field) != expected:
            raise ValueError(f"source manifest {field} mismatch")
    for payload in (
        diagnostic_config,
        manifest,
        rows,
        summary,
        component_summary,
        strata,
        confusion,
        decision,
    ):
        _require_json_finite(payload)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "diagnostic_config.json", diagnostic_config)
    _write_json(output / "source_manifest.json", manifest)
    _write_csv(output / "teacher_ranking_sample_diagnostics.csv", rows)
    _write_json(output / "teacher_ranking_summary.json", summary)
    _write_json(output / "teacher_component_summary.json", component_summary)
    _write_csv(output / "teacher_ranking_stratified_summary.csv", strata)
    _write_csv(output / "teacher_oracle_confusion.csv", confusion)
    _write_json(output / "decision.json", decision)
    return output


def _aggregate_signal(
    rows: Sequence[Mapping[str, Any]], signal: str
) -> dict[str, Any]:
    if not rows:
        return {
            **{field: 0 for field in PAIR_COUNT_FIELDS},
            "pairwise_concordance_rate": None,
            "spearman_rho": _nullable_distribution([]),
            "kendall_tau_b": _nullable_distribution([]),
            "top1_oracle_agreement_count": 0,
            "top1_oracle_agreement_rate": None,
            "top2_oracle_coverage_count": 0,
            "top2_oracle_coverage_rate": None,
            "top1_regret_deg": _nullable_distribution([]),
            "regret_gt_1_deg_count": 0,
            "regret_gt_1_deg_rate": None,
            "regret_gt_3_deg_count": 0,
            "regret_gt_3_deg_rate": None,
            "regret_gt_10_deg_count": 0,
            "regret_gt_10_deg_rate": None,
        }
    counts = {
        field: sum(int(row[f"{signal}_{field}"]) for row in rows)
        for field in PAIR_COUNT_FIELDS
    }
    comparable = counts["pairwise_comparable_count"]
    agreement_count = sum(
        bool(row[f"{signal}_top1_oracle_agreement"]) for row in rows
    )
    coverage_count = sum(
        bool(row[f"{signal}_top2_oracle_coverage"]) for row in rows
    )
    regrets = [row[f"{signal}_top1_regret_deg"] for row in rows]
    result = {
        **counts,
        "pairwise_concordance_rate": (
            counts["concordant_pair_count"] / comparable if comparable else None
        ),
        "spearman_rho": _nullable_distribution(
            [row[f"{signal}_spearman_rho"] for row in rows]
        ),
        "kendall_tau_b": _nullable_distribution(
            [row[f"{signal}_kendall_tau_b"] for row in rows]
        ),
        "top1_oracle_agreement_count": agreement_count,
        "top1_oracle_agreement_rate": agreement_count / len(rows),
        "top2_oracle_coverage_count": coverage_count,
        "top2_oracle_coverage_rate": coverage_count / len(rows),
        "top1_regret_deg": _nullable_distribution(regrets),
    }
    for threshold in (1, 3, 10):
        count = sum(_finite(value) > threshold for value in regrets)
        result[f"regret_gt_{threshold}_deg_count"] = count
        result[f"regret_gt_{threshold}_deg_rate"] = count / len(rows)
    return result


def _dimension_support(
    rows: Sequence[Mapping[str, Any]], signal: str, dimension: str
) -> bool:
    return sum(
        int(
            row["sample_count"] > 0
            and row["pairwise_concordance_rate"] is not None
            and _finite(row["pairwise_concordance_rate"]) >= 0.55
            and _finite(row["top2_oracle_coverage_rate"]) >= 0.65
        )
        for row in rows
        if row["signal"] == signal and row["dimension"] == dimension
    ) >= 2


def _nullable_distribution(values: Sequence[Any]) -> dict[str, Any]:
    defined = sorted(_finite(value) for value in values if value is not None)
    result: dict[str, Any] = {
        "defined_count": len(defined),
        "null_count": len(values) - len(defined),
        "mean": None,
        "median": None,
        "p05": None,
        "p95": None,
        "min": None,
        "max": None,
    }
    if defined:
        result.update(
            {
                "mean": math.fsum(defined) / len(defined),
                "median": _quantile(defined, 0.50),
                "p05": _quantile(defined, 0.05),
                "p95": _quantile(defined, 0.95),
                "min": defined[0],
                "max": defined[-1],
            }
        )
    return result


def _quantile(values: Sequence[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _rho_bin(row: Mapping[str, Any]) -> str:
    value = _finite(row["rho"])
    if value not in RHO_VALUES:
        raise ValueError(f"rho is outside frozen protocol: {value}")
    return str(value)


def _snr_bin(row: Mapping[str, Any]) -> str:
    value = _finite(row["snr_db"])
    for label, lower, upper, inclusive_upper in SNR_BINS:
        if lower <= value < upper or (inclusive_upper and value == upper):
            return label
    raise ValueError(f"snr_db is outside frozen protocol: {value}")


def _snapshot_bin(row: Mapping[str, Any]) -> str:
    value = row["snapshot_count"]
    if isinstance(value, bool) or value not in SNAPSHOT_VALUES:
        raise ValueError(f"snapshot_count is outside frozen protocol: {value}")
    return str(value)


def _cohort_bin(row: Mapping[str, Any]) -> str:
    value = row["threshold_cohort"]
    if value not in THRESHOLD_COHORTS:
        raise ValueError(f"unknown threshold cohort: {value}")
    return str(value)


def _finite(value: Any) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError("report metric must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("report metric must be finite") from error
    if not math.isfinite(number):
        raise ValueError("report metric must be finite")
    return number


def _optional_finite(value: Any) -> float | None:
    return None if value is None else _finite(value)


def _at_least(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def _at_most(value: float | None, threshold: float) -> bool:
    return value is not None and value <= threshold


def _require_json_finite(value: Any) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        _finite(value)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _require_json_finite(item)
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            _require_json_finite(item)
        return
    raise ValueError(f"report value is not JSON-compatible: {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write CSV without rows")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False, sort_keys=True)
                        if isinstance(value, (Mapping, tuple, list))
                        else value
                    )
                    for key, value in row.items()
                }
            )
