"""Read-only labels for the frozen near-resolution audit input."""

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ERROR_THRESHOLDS_DEG = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
EXPECTED_EVALUATOR_CODE_SHA = "129c3ba3b9fc1919451eef5c67376f04b4b24680"

_PCNSS_ALGORITHM = "pcnss_root_music"
_FBSS_L7_ALGORITHM = "fbss_root_music_L7"
_METADATA_FIELDS = (
    "true_angle_1_deg",
    "true_angle_2_deg",
    "rho",
    "snr_db",
    "snapshot_count",
    "separation_deg",
)
_FLOAT_FIELDS = (
    "true_angle_1_deg",
    "true_angle_2_deg",
    "absolute_error_1_deg",
    "absolute_error_2_deg",
    "rho",
    "snr_db",
    "separation_deg",
)
_BOOLEAN_FIELDS = ("success", "estimated_separation_at_least_half_true")


@dataclass(frozen=True)
class NearAuditLabel:
    sample_seed: int
    rho: float
    snr_db: float
    snapshot_count: int
    separation_deg: float
    pcnss_row: dict[str, Any]
    fbss_l7_row: dict[str, Any]
    threshold_cohort: str


@dataclass(frozen=True)
class NearAuditSelection:
    labels: tuple[NearAuditLabel, ...]
    source_manifest: dict[str, Any]
    input_sha256: dict[str, str]


def classify_threshold_cohort(
    estimate_success: bool,
    separation_pass: bool,
    max_angle_error_deg: float,
) -> str:
    if not estimate_success:
        return "estimation_failure"
    if not separation_pass:
        return "separation_failure"
    if max_angle_error_deg <= 1.0:
        return "resolved"
    if max_angle_error_deg <= 1.25:
        return "near_miss_1_1p25"
    if max_angle_error_deg <= 1.5:
        return "near_miss_1p25_1p5"
    if max_angle_error_deg <= 2.0:
        return "near_miss_1p5_2"
    return "far_miss_gt_2"


def build_threshold_summary(rows: list[dict[str, Any]], algorithm: str) -> dict[str, Any]:
    """Summarize maximum matched-angle error for one near-separation algorithm."""

    algorithm_rows = [row for row in rows if row.get("algorithm") == algorithm]
    sample_count = len(algorithm_rows)
    summary: dict[str, Any] = {"sample_count": sample_count}
    maximum_errors = [
        max(
            _as_finite_float(row, "absolute_error_1_deg"),
            _as_finite_float(row, "absolute_error_2_deg"),
        )
        for row in algorithm_rows
    ]
    for threshold in ERROR_THRESHOLDS_DEG:
        count = sum(error <= threshold for error in maximum_errors)
        summary[_threshold_key(threshold)] = {
            "count": count,
            "rate": float(count / sample_count) if sample_count else None,
        }
    return summary


def load_near_audit(
    report_directory: str | Path,
    checkpoint_path: str | Path,
    *,
    expected_code_sha: str = EXPECTED_EVALUATOR_CODE_SHA,
    expected_near_count: int = 1270,
) -> NearAuditSelection:
    """Load and authenticate paired PC-NSS and L7 near-separation predictions."""

    report = Path(report_directory)
    checkpoint = Path(checkpoint_path)
    run_config = json.loads((report / "run_config.json").read_text(encoding="utf-8"))
    summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (report / "source_manifest.json").read_text(encoding="utf-8")
    )
    if run_config.get("stage") != "evaluate_validation":
        raise ValueError("diagnostic source must be evaluate_validation")
    if run_config.get("split") != "validation" or summary.get("split") != "validation":
        raise ValueError("diagnostic source must be validation")
    if summary.get("report_schema_version") != 2:
        raise ValueError("diagnostic source must use report schema v2")
    if manifest.get("code_sha") != expected_code_sha:
        raise ValueError("unexpected evaluator code SHA")
    if _sha256(checkpoint) != manifest.get("checkpoint_sha"):
        raise ValueError("checkpoint SHA does not match source manifest")
    prediction_path = report / "predictions.csv"
    pcnss_rows, fbss_rows = _read_algorithm_rows(prediction_path)
    labels = _pair_and_validate_near_rows(
        pcnss_rows,
        fbss_rows,
        expected_near_count=expected_near_count,
    )
    return NearAuditSelection(
        labels=tuple(labels),
        source_manifest=manifest,
        input_sha256={
            name: _sha256(report / name)
            for name in (
                "run_config.json",
                "summary.json",
                "source_manifest.json",
                "predictions.csv",
            )
        },
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_algorithm_rows(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pcnss_rows: list[dict[str, Any]] = []
    fbss_rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw_row in csv.DictReader(handle):
            algorithm = raw_row.get("algorithm")
            if algorithm == _PCNSS_ALGORITHM:
                pcnss_rows.append(_convert_prediction_row(raw_row))
            elif algorithm == _FBSS_L7_ALGORITHM:
                fbss_rows.append(_convert_prediction_row(raw_row))
    return pcnss_rows, fbss_rows


def _pair_and_validate_near_rows(
    pcnss_rows: list[dict[str, Any]],
    fbss_rows: list[dict[str, Any]],
    *,
    expected_near_count: int,
) -> list[NearAuditLabel]:
    pcnss_by_seed = _index_rows_by_seed(pcnss_rows, _PCNSS_ALGORITHM)
    fbss_by_seed = _index_rows_by_seed(fbss_rows, _FBSS_L7_ALGORITHM)
    if pcnss_by_seed.keys() != fbss_by_seed.keys():
        raise ValueError("PC-NSS/L7 sample_seed sets do not match")

    labels = []
    for sample_seed in sorted(pcnss_by_seed):
        pcnss_row = pcnss_by_seed[sample_seed]
        fbss_row = fbss_by_seed[sample_seed]
        for field in _METADATA_FIELDS:
            if pcnss_row[field] != fbss_row[field]:
                raise ValueError(f"metadata mismatch for {field}")
        separation = pcnss_row["separation_deg"]
        if not 2.0 <= separation < 4.0:
            continue
        max_error = max(
            pcnss_row["absolute_error_1_deg"],
            pcnss_row["absolute_error_2_deg"],
        )
        labels.append(
            NearAuditLabel(
                sample_seed=sample_seed,
                rho=pcnss_row["rho"],
                snr_db=pcnss_row["snr_db"],
                snapshot_count=pcnss_row["snapshot_count"],
                separation_deg=separation,
                pcnss_row=pcnss_row,
                fbss_l7_row=fbss_row,
                threshold_cohort=classify_threshold_cohort(
                    pcnss_row["success"],
                    pcnss_row["estimated_separation_at_least_half_true"],
                    max_error,
                ),
            )
        )
    if len(labels) != expected_near_count:
        raise ValueError(
            "near row count does not match expected_near_count: "
            f"{len(labels)} != {expected_near_count}"
        )
    return labels


def _convert_prediction_row(raw_row: dict[str, str | None]) -> dict[str, Any]:
    row = dict(raw_row)
    if _required_value(row, "split") != "validation":
        raise ValueError("prediction rows must be validation")
    try:
        row["sample_seed"] = int(_required_value(row, "sample_seed"))
        row["snapshot_count"] = int(_required_value(row, "snapshot_count"))
        for field in _FLOAT_FIELDS:
            row[field] = _as_finite_float(row, field)
        for field in _BOOLEAN_FIELDS:
            row[field] = _as_bool(row, field)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid selected prediction row") from error
    return row


def _index_rows_by_seed(
    rows: list[dict[str, Any]], algorithm: str
) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        sample_seed = row["sample_seed"]
        if sample_seed in indexed:
            raise ValueError(f"duplicate sample_seed in {algorithm}: {sample_seed}")
        indexed[sample_seed] = row
    return indexed


def _as_finite_float(row: dict[str, Any], field: str) -> float:
    value = float(_required_value(row, field))
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def _as_bool(row: dict[str, Any], field: str) -> bool:
    value = _required_value(row, field)
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{field} must be true or false")


def _required_value(row: dict[str, Any], field: str) -> Any:
    value = row.get(field)
    if value is None or value == "":
        raise ValueError(f"missing {field}")
    return value


def _threshold_key(threshold: float) -> str:
    encoded = f"{threshold:.2f}".replace(".", "p")
    return f"max_error_le_{encoded}_deg"
