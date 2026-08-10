"""Schema-v1 reports and frozen decisions for teacher-confidence diagnosis."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

from multisource_doa.diagnostics.teacher_confidence import (
    SCALE_SIZES,
    THRESHOLD_COHORTS,
    TeacherDiagnosticResult,
)


TEACHER_DIAGNOSTIC_SCHEMA_VERSION = 1
RHO_VALUES = (0.8, 0.9, 0.99, 1.0)
SNR_BINS = (
    ("[-5,0)", -5.0, 0.0, False),
    ("[0,5)", 0.0, 5.0, False),
    ("[5,10]", 5.0, 10.0, True),
)
SNAPSHOT_VALUES = (8, 20, 50)
SUMMARY_METRICS = (
    "teacher_entropy_current",
    "teacher_entropy_counterfactual",
    "teacher_max_probability_current",
    "teacher_max_probability_counterfactual",
    "teacher_score_margin",
    "teacher_score_margin_over_tau",
    "student_entropy_normalized",
    "student_max_probability",
    "teacher_student_kl",
    "teacher_student_js",
    "teacher_regret_deg",
)
_PROBABILITY_PREFIXES = (
    "teacher_p_current_L",
    "teacher_p_counterfactual_L",
    "student_p_L",
)


def build_teacher_summary(
    sample_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the frozen overall teacher/student/oracle summary."""

    rows = list(sample_rows)
    if not rows:
        raise ValueError("teacher summary requires at least one sample")
    for row in rows:
        _validate_sample_row(row)
    agreement_count = sum(bool(row["teacher_oracle_agreement"]) for row in rows)
    return {
        "sample_count": len(rows),
        "engineering_integrity": True,
        "metrics": {
            metric: _distribution([row[metric] for row in rows])
            for metric in SUMMARY_METRICS
        },
        "teacher_oracle_agreement_count": agreement_count,
        "teacher_oracle_agreement_rate": agreement_count / len(rows),
        "dominant_scale_counts": {
            "teacher_current": _dominant_counts(rows, "teacher_dominant_scale_current"),
            "teacher_counterfactual": _dominant_counts(
                rows, "teacher_dominant_scale_counterfactual"
            ),
            "student": _dominant_counts(rows, "student_dominant_scale"),
        },
    }


def build_teacher_stratified_summary(
    sample_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Enumerate all 17 frozen bins, including explicit empty bins."""

    rows = list(sample_rows)
    if not rows:
        raise ValueError("teacher stratification requires at least one sample")
    for row in rows:
        _validate_sample_row(row)
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
    summaries: list[dict[str, Any]] = []
    for dimension, bin_names, bin_for_row in dimensions:
        for bin_name in bin_names:
            selected = [row for row in rows if bin_for_row(row) == bin_name]
            summaries.append(_stratum_row(dimension, bin_name, selected))
        if sum(
            row["sample_count"]
            for row in summaries
            if row["dimension"] == dimension
        ) != len(rows):
            raise ValueError(f"{dimension} strata do not account for every sample")
    return summaries


def build_teacher_decision(
    summary: Mapping[str, Any],
    stratified_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the conjunctive scientific gate without authorizing training."""

    metrics = summary["metrics"]
    entropy_current = _finite(metrics["teacher_entropy_current"]["median"], "entropy")
    entropy_counterfactual = _finite(
        metrics["teacher_entropy_counterfactual"]["median"], "entropy"
    )
    pmax_current = _finite(
        metrics["teacher_max_probability_current"]["median"], "max probability"
    )
    pmax_counterfactual = _finite(
        metrics["teacher_max_probability_counterfactual"]["median"],
        "max probability",
    )
    entropy_drop = entropy_current - entropy_counterfactual
    pmax_rise = pmax_counterfactual - pmax_current
    dimension_support = {
        dimension: sum(
            int(
                row["sample_count"] > 0
                and _finite(row["teacher_entropy_current_median"], "stratum entropy")
                >= 0.90
                and _finite(
                    row["teacher_entropy_drop_median"], "stratum entropy drop"
                )
                >= 0.05
            )
            for row in stratified_rows
            if row["dimension"] == dimension
        )
        >= 2
        for dimension in ("rho", "snr_db", "snapshot_count")
    }
    gates = {
        "teacher_entropy_high": entropy_current >= 0.90,
        "counterfactual_entropy_drop": entropy_drop >= 0.05,
        "counterfactual_pmax_rise": pmax_rise >= 0.05,
        "oracle_agreement": _finite(
            summary["teacher_oracle_agreement_rate"], "oracle agreement"
        )
        >= 0.40,
        "median_regret": _finite(
            metrics["teacher_regret_deg"]["median"], "teacher regret"
        )
        <= 1.0,
        "stratified_support": sum(dimension_support.values()) >= 2,
        "engineering_integrity": summary.get("engineering_integrity") is True,
    }
    allowed = all(gates.values())
    return {
        "allow_tau_preregistration": allowed,
        "candidate_change": "tau_scale: 0.10 -> 0.05" if allowed else None,
        "training_authorized": False,
        "gates": gates,
        "dimension_support": dimension_support,
        "observed_entropy_drop_median": entropy_drop,
        "observed_pmax_rise_median": pmax_rise,
        "reason": (
            "all frozen scientific and engineering gates passed"
            if allowed
            else "one or more frozen scientific or engineering gates failed"
        ),
    }


def write_teacher_diagnostic_report(
    result: TeacherDiagnosticResult,
    output_directory: str | Path,
    *,
    diagnostic_config: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> Path:
    """Write six immutable artifacts after complete in-memory validation."""

    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    sample_rows = list(result.sample_rows)
    summary = build_teacher_summary(sample_rows)
    strata = build_teacher_stratified_summary(sample_rows)
    decision = build_teacher_decision(summary, strata)
    manifest = {
        **source_manifest,
        "teacher_diagnostic_schema_version": TEACHER_DIAGNOSTIC_SCHEMA_VERSION,
    }
    if manifest.get("no_model_forward") is not True:
        raise ValueError("source manifest must declare no_model_forward=true")
    if manifest.get("training_performed") is not False:
        raise ValueError("source manifest must declare training_performed=false")
    if manifest.get("sample_count") != len(sample_rows):
        raise ValueError("source manifest sample_count mismatch")
    for payload in (
        diagnostic_config,
        manifest,
        sample_rows,
        summary,
        strata,
        decision,
    ):
        _require_json_finite(payload)

    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "diagnostic_config.json", diagnostic_config)
    _write_json(output / "source_manifest.json", manifest)
    _write_csv(output / "teacher_sample_diagnostics.csv", sample_rows)
    _write_json(output / "teacher_summary.json", summary)
    _write_csv(output / "teacher_stratified_summary.csv", strata)
    _write_json(output / "decision.json", decision)
    return output


def _stratum_row(
    dimension: str,
    bin_name: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dimension": dimension,
        "bin": bin_name,
        "sample_count": len(rows),
    }
    for metric in SUMMARY_METRICS:
        distribution = _distribution([row[metric] for row in rows]) if rows else _empty_distribution()
        for statistic, value in distribution.items():
            result[f"{metric}_{statistic}"] = value
    current = result["teacher_entropy_current_median"]
    colder = result["teacher_entropy_counterfactual_median"]
    current_pmax = result["teacher_max_probability_current_median"]
    colder_pmax = result["teacher_max_probability_counterfactual_median"]
    result["teacher_entropy_drop_median"] = (
        current - colder if current is not None and colder is not None else None
    )
    result["teacher_pmax_rise_median"] = (
        colder_pmax - current_pmax
        if current_pmax is not None and colder_pmax is not None
        else None
    )
    agreement_count = sum(bool(row["teacher_oracle_agreement"]) for row in rows)
    result["teacher_oracle_agreement_count"] = agreement_count
    result["teacher_oracle_agreement_rate"] = (
        agreement_count / len(rows) if rows else None
    )
    for prefix, field in (
        ("teacher_current", "teacher_dominant_scale_current"),
        ("teacher_counterfactual", "teacher_dominant_scale_counterfactual"),
        ("student", "student_dominant_scale"),
    ):
        counts = _dominant_counts(rows, field)
        for size in SCALE_SIZES:
            result[f"{prefix}_dominant_L{size}_count"] = counts[str(size)]
    return result


def _validate_sample_row(row: Mapping[str, Any]) -> None:
    for field in (
        "sample_seed",
        "true_angle_1_deg",
        "true_angle_2_deg",
        "rho",
        "snr_db",
        "snapshot_count",
        "separation_deg",
        *SUMMARY_METRICS,
        *(f"teacher_score_L{size}" for size in SCALE_SIZES),
        *(f"fbss_L{size}_sample_rmspe_deg" for size in SCALE_SIZES),
    ):
        _finite(_required(row, field), field)
    _rho_bin(row)
    _snr_bin(row)
    _snapshot_bin(row)
    _cohort_bin(row)
    for prefix in _PROBABILITY_PREFIXES:
        probabilities = [
            _finite(_required(row, f"{prefix}{size}"), f"{prefix}{size}")
            for size in SCALE_SIZES
        ]
        if any(value < 0.0 for value in probabilities) or not math.isclose(
            math.fsum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(f"{prefix} probabilities must be nonnegative and sum to one")
    for field in (
        "teacher_dominant_scale_current",
        "teacher_dominant_scale_counterfactual",
        "student_dominant_scale",
    ):
        if _required(row, field) not in SCALE_SIZES:
            raise ValueError(f"{field} must be a frozen scale")
    if not isinstance(_required(row, "teacher_oracle_agreement"), bool):
        raise ValueError("teacher_oracle_agreement must be bool")
    oracle = _required(row, "oracle_best_scales")
    if not isinstance(oracle, (tuple, list)) or not oracle:
        raise ValueError("oracle_best_scales must be a nonempty sequence")
    if any(size not in SCALE_SIZES for size in oracle):
        raise ValueError("oracle_best_scales contains an unknown scale")


def _distribution(values: Sequence[Any]) -> dict[str, float | int]:
    ordered = sorted(_finite(value, "distribution value") for value in values)
    if not ordered:
        raise ValueError("distribution requires at least one value")
    return {
        "count": len(ordered),
        "mean": math.fsum(ordered) / len(ordered),
        "median": _quantile(ordered, 0.50),
        "p05": _quantile(ordered, 0.05),
        "p95": _quantile(ordered, 0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _empty_distribution() -> dict[str, None | int]:
    return {
        "count": 0,
        "mean": None,
        "median": None,
        "p05": None,
        "p95": None,
        "min": None,
        "max": None,
    }


def _dominant_counts(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, int]:
    return {
        str(size): sum(row[field] == size for row in rows) for size in SCALE_SIZES
    }


def _rho_bin(row: Mapping[str, Any]) -> str:
    value = _finite(_required(row, "rho"), "rho")
    if value not in RHO_VALUES:
        raise ValueError(f"rho is outside the frozen protocol: {value}")
    return str(value)


def _snr_bin(row: Mapping[str, Any]) -> str:
    value = _finite(_required(row, "snr_db"), "snr_db")
    for label, lower, upper, inclusive_upper in SNR_BINS:
        if lower <= value < upper or (inclusive_upper and value == upper):
            return label
    raise ValueError(f"snr_db is outside the frozen protocol: {value}")


def _snapshot_bin(row: Mapping[str, Any]) -> str:
    value = _required(row, "snapshot_count")
    if isinstance(value, bool) or value not in SNAPSHOT_VALUES:
        raise ValueError(f"snapshot_count is outside the frozen protocol: {value}")
    return str(value)


def _cohort_bin(row: Mapping[str, Any]) -> str:
    value = _required(row, "threshold_cohort")
    if value not in THRESHOLD_COHORTS:
        raise ValueError(f"unknown threshold_cohort: {value}")
    return str(value)


def _quantile(values: Sequence[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _required(row: Mapping[str, Any], field: str) -> Any:
    try:
        return row[field]
    except KeyError as error:
        raise ValueError(f"sample row is missing {field}") from error


def _finite(value: Any, name: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _require_json_finite(value: Any) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        _finite(value, "report value")
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
        raise ValueError("cannot write a CSV without rows")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True, ensure_ascii=False)
                        if isinstance(value, (Mapping, tuple, list))
                        else value
                    )
                    for key, value in row.items()
                }
            )
