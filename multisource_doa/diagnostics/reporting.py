"""Schema-v1 reports for frozen near-resolution diagnostics."""

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from multisource_doa.diagnostics.near_resolution import (
    ERROR_THRESHOLDS_DEG,
    NearDiagnosticResult,
)
from multisource_doa.training.artifacts import prepare_run_directory


SNR_BINS = ((-5.0, 0.0, "[-5,0)"), (0.0, 5.0, "[0,5)"), (5.0, 10.0, "[5,10]"))
RHO_VALUES = (0.8, 0.9, 0.99, 1.0)
SNAPSHOT_VALUES = (8, 20, 50)
THRESHOLD_COHORTS = (
    "estimation_failure",
    "separation_failure",
    "resolved",
    "near_miss_1_1p25",
    "near_miss_1p25_1p5",
    "near_miss_1p5_2",
    "far_miss_gt_2",
)
DIAGNOSTIC_SCHEMA_VERSION = 1

MECHANISM_METRICS = (
    "scale_entropy_normalized",
    "residual_magnitude_p50",
    "residual_magnitude_p95",
    "residual_magnitude_max",
    "saturated_lag_rate",
    "train_projection_change",
    "eval_projection_change",
    "total_projection_change",
)

_SCALE_METRICS = ("p_L4", "p_L5", "p_L6", "p_L7")
_PROJECTION_METRICS = (
    "train_projection_change",
    "eval_projection_change",
    "total_projection_change",
)


def build_stratified_summary(
    sample_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate every frozen sample into fixed, protocol-approved strata."""

    rows = list(sample_rows)
    for row in rows:
        _validate_sample_row(row)

    summaries: list[dict[str, Any]] = []
    dimensions = (
        ("rho", lambda row: _rho_bin(row["rho"])),
        ("snr_db", lambda row: _snr_bin(row["snr_db"])),
        ("snapshot_count", lambda row: _snapshot_bin(row["snapshot_count"])),
        ("threshold_cohort", lambda row: _cohort_bin(row["threshold_cohort"])),
    )
    for dimension, bin_for_row in dimensions:
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(bin_for_row(row), []).append(row)
        summaries.extend(
            _stratum_summary(dimension, bin_name, group_rows)
            for bin_name, group_rows in grouped.items()
        )
    return summaries


def build_mechanism_summary(sample_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize learned scale and projection mechanisms by outcome cohort."""

    rows = list(sample_rows)
    for row in rows:
        _validate_sample_row(row)

    cohorts: dict[str, dict[str, Any]] = {}
    for cohort in THRESHOLD_COHORTS:
        cohort_rows = [row for row in rows if row["threshold_cohort"] == cohort]
        if not cohort_rows:
            continue
        cohorts[cohort] = {
            "sample_count": len(cohort_rows),
            "metrics": {
                metric: _finite_distribution_summary([row[metric] for row in cohort_rows])
                for metric in MECHANISM_METRICS
            },
            "dominant_scale_counts": {
                str(size): sum(row["dominant_scale"] == size for row in cohort_rows)
                for size in (4, 5, 6, 7)
            },
        }
    return {"sample_count": len(rows), "cohorts": cohorts}


def write_near_diagnostic_report(
    result: NearDiagnosticResult,
    output_directory: str | Path,
    *,
    diagnostic_config: dict[str, Any],
    source_manifest: dict[str, Any],
    refuse_overwrite: bool = True,
) -> Path:
    """Write the six immutable schema-v1 artifacts for a frozen diagnosis."""

    output = prepare_run_directory(
        output_directory,
        refuse_overwrite=refuse_overwrite,
    )
    sample_rows = list(result.sample_rows)
    threshold_summary = _build_threshold_summary(sample_rows)
    stratified_rows = build_stratified_summary(sample_rows)
    mechanism_summary = build_mechanism_summary(sample_rows)
    _require_finite_or_null(diagnostic_config)
    _require_finite_or_null(source_manifest)
    _require_finite_or_null(sample_rows)
    _require_finite_or_null(threshold_summary)
    _require_finite_or_null(stratified_rows)
    _require_finite_or_null(mechanism_summary)
    _write_json(output / "diagnostic_config.json", diagnostic_config)
    _write_json(
        output / "source_manifest.json",
        {**source_manifest, "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION},
    )
    _write_csv(output / "near_sample_diagnostics.csv", sample_rows)
    _write_json(output / "threshold_summary.json", threshold_summary)
    _write_csv(output / "stratified_summary.csv", stratified_rows)
    _write_json(output / "mechanism_summary.json", mechanism_summary)
    return output


def _build_threshold_summary(sample_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    for row in sample_rows:
        _validate_sample_row(row)
    maximum_errors = [
        max(float(row["absolute_error_1_deg"]), float(row["absolute_error_2_deg"]))
        for row in sample_rows
    ]
    sample_count = len(sample_rows)
    summary: dict[str, Any] = {"sample_count": sample_count}
    for threshold in ERROR_THRESHOLDS_DEG:
        count = sum(error <= threshold for error in maximum_errors)
        summary[_threshold_key(threshold)] = {
            "count": count,
            "rate": _rate(count, sample_count),
        }
    resolved_count = sum(row["threshold_cohort"] == "resolved" for row in sample_rows)
    summary["resolved"] = {
        "count": resolved_count,
        "rate": _rate(resolved_count, sample_count),
    }
    return summary


def _stratum_summary(
    dimension: str,
    bin_name: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    sample_count = len(rows)
    maximum_errors = [
        max(float(row["absolute_error_1_deg"]), float(row["absolute_error_2_deg"]))
        for row in rows
    ]
    summary: dict[str, Any] = {
        "dimension": dimension,
        "bin": bin_name,
        "sample_count": sample_count,
    }
    for threshold in ERROR_THRESHOLDS_DEG:
        summary[f"{_threshold_key(threshold)}_rate"] = _rate(
            sum(error <= threshold for error in maximum_errors), sample_count
        )
    summary["resolved_rate"] = _rate(
        sum(row["threshold_cohort"] == "resolved" for row in rows), sample_count
    )
    for metric in _SCALE_METRICS:
        summary[f"{metric}_mean"] = _mean([row[metric] for row in rows])
    entropy = _finite_distribution_summary([row["scale_entropy_normalized"] for row in rows])
    summary["scale_entropy_normalized_mean"] = entropy["mean"]
    summary["scale_entropy_normalized_median"] = entropy["median"]
    summary["saturated_lag_rate"] = _mean([row["saturated_lag_rate"] for row in rows])
    for metric in _PROJECTION_METRICS:
        distribution = _finite_distribution_summary([row[metric] for row in rows])
        summary[f"{metric}_mean"] = distribution["mean"]
        summary[f"{metric}_median"] = distribution["median"]
    return summary


def _finite_distribution_summary(values: Sequence[Any]) -> dict[str, float | int]:
    """Return fixed distribution fields while rejecting every non-finite value."""

    finite_values = [_finite_float(value, "distribution value") for value in values]
    if not finite_values:
        raise ValueError("distribution summary requires at least one value")
    ordered = sorted(finite_values)
    return {
        "count": len(ordered),
        "mean": math.fsum(ordered) / len(ordered),
        "median": _quantile(ordered, 0.50),
        "p05": _quantile(ordered, 0.05),
        "p95": _quantile(ordered, 0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _validate_sample_row(row: Mapping[str, Any]) -> None:
    for field in (
        "rho",
        "snr_db",
        "absolute_error_1_deg",
        "absolute_error_2_deg",
        *_SCALE_METRICS,
        *MECHANISM_METRICS,
    ):
        _finite_float(_required_field(row, field), field)
    _rho_bin(_required_field(row, "rho"))
    _snr_bin(_required_field(row, "snr_db"))
    _snapshot_bin(_required_field(row, "snapshot_count"))
    _cohort_bin(_required_field(row, "threshold_cohort"))
    if _required_field(row, "dominant_scale") not in (4, 5, 6, 7):
        raise ValueError("dominant_scale must be one of 4, 5, 6, 7")


def _rho_bin(value: Any) -> str:
    rho = _finite_float(value, "rho")
    if rho not in RHO_VALUES:
        raise ValueError(f"rho is outside the frozen protocol: {rho}")
    return str(rho)


def _snr_bin(value: Any) -> str:
    snr_db = _finite_float(value, "snr_db")
    for lower, upper, label in SNR_BINS:
        if lower <= snr_db < upper or (upper == 10.0 and snr_db == upper):
            return label
    raise ValueError(f"snr_db is outside the frozen protocol: {snr_db}")


def _snapshot_bin(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, int) or value not in SNAPSHOT_VALUES:
        raise ValueError(f"snapshot_count is outside the frozen protocol: {value}")
    return str(value)


def _cohort_bin(value: Any) -> str:
    if value not in THRESHOLD_COHORTS:
        raise ValueError(f"unknown threshold_cohort: {value}")
    return str(value)


def _required_field(row: Mapping[str, Any], field: str) -> Any:
    try:
        return row[field]
    except KeyError as error:
        raise ValueError(f"sample row is missing {field}") from error


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _mean(values: Sequence[Any]) -> float:
    return _finite_distribution_summary(values)["mean"]  # type: ignore[return-value]


def _quantile(values: Sequence[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _rate(count: int, total: int) -> float | None:
    return float(count / total) if total else None


def _threshold_key(threshold: float) -> str:
    encoded = f"{threshold:.2f}".replace(".", "p")
    return f"max_error_le_{encoded}_deg"


def _require_finite_or_null(value: Any) -> None:
    """Recursively reject NaN and infinity before JSON/CSV serialization."""

    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        _finite_float(value, "report value")
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _require_finite_or_null(item)
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            _require_finite_or_null(item)
        return
    raise ValueError(f"report value is not JSON-compatible: {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    _require_finite_or_null(payload)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write a CSV without rows")
    _require_finite_or_null(rows)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return value
