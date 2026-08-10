"""Authenticated inputs for the frozen teacher-confidence diagnosis."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from multisource_doa.data.simulator import DOASample
from multisource_doa.training.engine import collate_samples
from multisource_doa.training.teacher import (
    build_scale_teacher,
    scale_probabilities_from_scores,
)


SCALE_SIZES = (4, 5, 6, 7)
ALGORITHMS = ("pcnss_root_music",) + tuple(
    f"fbss_root_music_L{size}" for size in SCALE_SIZES
)
METADATA_FIELDS = (
    "true_angle_1_deg",
    "true_angle_2_deg",
    "rho",
    "snr_db",
    "snapshot_count",
    "separation_deg",
)
THRESHOLD_COHORTS = (
    "estimation_failure",
    "separation_failure",
    "resolved",
    "near_miss_1_1p25",
    "near_miss_1p25_1p5",
    "near_miss_1p5_2",
    "far_miss_gt_2",
)
CURRENT_TAU = 0.10
COUNTERFACTUAL_TAU = 0.05
ORACLE_TIE_TOLERANCE_DEG = 1e-6
DIVERGENCE_EPSILON = 1e-8


@dataclass(frozen=True)
class TeacherAuthorityLabel:
    sample_seed: int
    true_angles_deg: tuple[float, float]
    rho: float
    snr_db: float
    snapshot_count: int
    separation_deg: float
    threshold_cohort: str
    student_probabilities: tuple[float, float, float, float]
    fixed_rmspe_deg: dict[int, float]


@dataclass(frozen=True)
class TeacherDiagnosticInputs:
    labels_by_seed: dict[int, TeacherAuthorityLabel]
    source_manifest: dict[str, Any]
    input_sha256: dict[str, str]


@dataclass(frozen=True)
class TeacherDiagnosticResult:
    sample_rows: tuple[dict[str, Any], ...]


def load_teacher_diagnostic_inputs(
    report_directory: str | Path,
    task14_directory: str | Path,
    *,
    expected_source_count: int = 5000,
    expected_near_count: int = 1270,
) -> TeacherDiagnosticInputs:
    """Authenticate and join audit-v4 and Task 14 without reading a checkpoint."""

    if expected_source_count <= 0 or expected_near_count <= 0:
        raise ValueError("expected counts must be positive")
    audit = Path(report_directory)
    task14 = Path(task14_directory)
    run_config = _read_json(audit / "run_config.json")
    summary = _read_json(audit / "summary.json")
    audit_manifest = _read_json(audit / "source_manifest.json")
    task14_manifest = _read_json(task14 / "source_manifest.json")
    _require_validation_schema_v2(run_config, summary)
    _require_task14_manifest(task14_manifest, expected_near_count)

    audit_hashes = {
        name: _sha256(audit / name)
        for name in (
            "run_config.json",
            "summary.json",
            "source_manifest.json",
            "predictions.csv",
        )
    }
    if task14_manifest.get("audit_input_sha256") != audit_hashes:
        raise ValueError("audit input SHA mismatch")
    if task14_manifest.get("checkpoint_sha") != audit_manifest.get("checkpoint_sha"):
        raise ValueError("checkpoint SHA mismatch")

    algorithm_rows = _read_and_validate_algorithms(
        audit / "predictions.csv",
        expected_source_count=expected_source_count,
    )
    student_rows = _read_and_validate_task14_rows(
        task14 / "near_sample_diagnostics.csv",
        expected_near_count=expected_near_count,
    )
    labels = _join_near_authority(algorithm_rows, student_rows)
    return TeacherDiagnosticInputs(
        labels_by_seed={label.sample_seed: label for label in labels},
        source_manifest={"audit": audit_manifest, "task14": task14_manifest},
        input_sha256={
            **{f"audit/{name}": digest for name, digest in audit_hashes.items()},
            "task14/source_manifest.json": _sha256(task14 / "source_manifest.json"),
            "task14/near_sample_diagnostics.csv": _sha256(
                task14 / "near_sample_diagnostics.csv"
            ),
        },
    )


def distribution_metrics(probabilities: Sequence[float]) -> dict[str, Any]:
    """Return normalized entropy, maximum mass, and deterministic top scale."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(SCALE_SIZES),):
        raise ValueError("probabilities must contain exactly four scales")
    _validate_probability_vector(tuple(float(value) for value in values), "probabilities")
    positive = values > 0.0
    entropy = -float(np.sum(values[positive] * np.log(values[positive])))
    dominant_index = int(np.argmax(values))
    return {
        "entropy_normalized": entropy / math.log(len(SCALE_SIZES)),
        "max_probability": float(values[dominant_index]),
        "dominant_scale": SCALE_SIZES[dominant_index],
    }


def build_teacher_sample_row(
    label: TeacherAuthorityLabel,
    scores: torch.Tensor,
    probabilities_current: torch.Tensor,
    probabilities_counterfactual: torch.Tensor,
    *,
    tau_current: float,
) -> dict[str, Any]:
    """Join frozen authority with teacher/student/oracle metrics for one sample."""

    raw = _finite_vector(scores, "teacher scores")
    current = _finite_vector(probabilities_current, "current teacher probabilities")
    counterfactual = _finite_vector(
        probabilities_counterfactual,
        "counterfactual teacher probabilities",
    )
    if raw.shape != (len(SCALE_SIZES),):
        raise ValueError("teacher scores must contain exactly four scales")
    current_metrics = distribution_metrics(current)
    counterfactual_metrics = distribution_metrics(counterfactual)
    student = np.asarray(label.student_probabilities, dtype=np.float64)
    student_metrics = distribution_metrics(student)
    fixed_rmspe = {
        size: _finite_number(label.fixed_rmspe_deg.get(size), f"L{size} sample_rmspe_deg")
        for size in SCALE_SIZES
    }

    sorted_scores = np.sort(raw)
    score_margin = float(sorted_scores[-1] - sorted_scores[-2])
    oracle_min = min(fixed_rmspe.values())
    oracle_scales = tuple(
        size
        for size in SCALE_SIZES
        if fixed_rmspe[size] - oracle_min <= ORACLE_TIE_TOLERANCE_DEG
    )
    teacher_scale = int(current_metrics["dominant_scale"])
    if counterfactual_metrics["dominant_scale"] != teacher_scale:
        raise ValueError("temperature change altered teacher score ranking")
    kl = _kl_divergence(current, student)
    midpoint = 0.5 * (current + student)
    js = 0.5 * _kl_divergence(current, midpoint) + 0.5 * _kl_divergence(
        student, midpoint
    )
    row: dict[str, Any] = {
        "sample_seed": label.sample_seed,
        "true_angle_1_deg": label.true_angles_deg[0],
        "true_angle_2_deg": label.true_angles_deg[1],
        "rho": label.rho,
        "snr_db": label.snr_db,
        "snapshot_count": label.snapshot_count,
        "separation_deg": label.separation_deg,
        "threshold_cohort": label.threshold_cohort,
        **{
            f"teacher_score_L{size}": float(raw[index])
            for index, size in enumerate(SCALE_SIZES)
        },
        **{
            f"teacher_p_current_L{size}": float(current[index])
            for index, size in enumerate(SCALE_SIZES)
        },
        **{
            f"teacher_p_counterfactual_L{size}": float(counterfactual[index])
            for index, size in enumerate(SCALE_SIZES)
        },
        **{
            f"student_p_L{size}": float(student[index])
            for index, size in enumerate(SCALE_SIZES)
        },
        "teacher_entropy_current": current_metrics["entropy_normalized"],
        "teacher_entropy_counterfactual": counterfactual_metrics[
            "entropy_normalized"
        ],
        "teacher_max_probability_current": current_metrics["max_probability"],
        "teacher_max_probability_counterfactual": counterfactual_metrics[
            "max_probability"
        ],
        "teacher_dominant_scale": teacher_scale,
        "teacher_dominant_scale_current": teacher_scale,
        "teacher_dominant_scale_counterfactual": counterfactual_metrics[
            "dominant_scale"
        ],
        "student_entropy_normalized": student_metrics["entropy_normalized"],
        "student_max_probability": student_metrics["max_probability"],
        "student_dominant_scale": student_metrics["dominant_scale"],
        "teacher_score_margin": score_margin,
        "teacher_score_margin_over_tau": score_margin / tau_current,
        "teacher_student_kl": kl,
        "teacher_student_js": max(0.0, float(js)),
        "oracle_best_scales": oracle_scales,
        "teacher_oracle_agreement": teacher_scale in oracle_scales,
        "teacher_regret_deg": fixed_rmspe[teacher_scale] - oracle_min,
        **{
            f"fbss_L{size}_sample_rmspe_deg": fixed_rmspe[size]
            for size in SCALE_SIZES
        },
    }
    _require_finite_row(row)
    return row


def diagnose_teacher_samples(
    samples: Sequence[DOASample],
    labels_by_seed: Mapping[int, TeacherAuthorityLabel],
    *,
    batch_size: int = 128,
    tau_current: float = CURRENT_TAU,
    tau_counterfactual: float = COUNTERFACTUAL_TAU,
) -> TeacherDiagnosticResult:
    """Compute the frozen physical teacher on CPU without a neural model."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not math.isclose(tau_current, CURRENT_TAU, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"tau_current must be fixed at {CURRENT_TAU}")
    if not math.isclose(
        tau_counterfactual,
        COUNTERFACTUAL_TAU,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"tau_counterfactual must be fixed at {COUNTERFACTUAL_TAU}"
        )
    sample_seeds = [sample.sample_seed for sample in samples]
    if len(sample_seeds) != len(set(sample_seeds)):
        raise ValueError("samples contain duplicate sample_seed values")
    if set(sample_seeds) != set(labels_by_seed):
        missing = sorted(set(sample_seeds) - set(labels_by_seed))
        extra = sorted(set(labels_by_seed) - set(sample_seeds))
        raise ValueError(f"missing authority label or extra label: missing={missing}, extra={extra}")

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch_samples = list(samples[start : start + batch_size])
            batch = collate_samples(batch_samples)
            if any(tensor.device.type != "cpu" for tensor in batch.fbss_covariances.values()):
                raise ValueError("teacher diagnosis must run on CPU")
            teacher = build_scale_teacher(
                batch.fbss_covariances,
                batch.true_angles_deg,
                tau_scale=tau_current,
            )
            counterfactual = scale_probabilities_from_scores(
                teacher.scale_scores,
                tau_scale=tau_counterfactual,
            )
            for sample, scores, current, colder in zip(
                batch_samples,
                teacher.scale_scores,
                teacher.scale_probabilities,
                counterfactual,
                strict=True,
            ):
                label = labels_by_seed[sample.sample_seed]
                validate_regenerated_metadata(sample, label)
                rows.append(
                    build_teacher_sample_row(
                        label,
                        scores,
                        current,
                        colder,
                        tau_current=tau_current,
                    )
                )
    return TeacherDiagnosticResult(sample_rows=tuple(rows))


def validate_regenerated_metadata(
    sample: DOASample,
    label: TeacherAuthorityLabel,
) -> None:
    """Require deterministic validation regeneration to match authority exactly."""

    separation = float(abs(np.diff(sample.angles_deg)[0]))
    if (
        sample.sample_seed != label.sample_seed
        or not np.allclose(
            sample.angles_deg,
            label.true_angles_deg,
            rtol=0.0,
            atol=1e-9,
        )
        or sample.rho != label.rho
        or sample.snr_db != label.snr_db
        or sample.snapshot_count != label.snapshot_count
        or not math.isclose(
            separation,
            label.separation_deg,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError(f"metadata mismatch for sample_seed {sample.sample_seed}")


def _read_and_validate_algorithms(
    path: Path,
    *,
    expected_source_count: int,
) -> dict[str, dict[int, dict[str, Any]]]:
    rows_by_algorithm: dict[str, dict[int, dict[str, Any]]] = {
        algorithm: {} for algorithm in ALGORITHMS
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw_row in csv.DictReader(handle):
            algorithm = str(raw_row.get("algorithm", ""))
            if algorithm not in rows_by_algorithm:
                continue
            row = _convert_prediction_row(raw_row)
            if row["split"] != "validation":
                raise ValueError("selected prediction rows must be validation")
            sample_seed = row["sample_seed"]
            if sample_seed in rows_by_algorithm[algorithm]:
                raise ValueError(
                    f"duplicate sample_seed {sample_seed} for {algorithm}"
                )
            rows_by_algorithm[algorithm][sample_seed] = row

    for algorithm, indexed_rows in rows_by_algorithm.items():
        if len(indexed_rows) != expected_source_count:
            raise ValueError(
                f"{algorithm} expected_source_count={expected_source_count}, "
                f"got {len(indexed_rows)}"
            )
    reference_seeds = set(rows_by_algorithm[ALGORITHMS[0]])
    for algorithm in ALGORITHMS[1:]:
        if set(rows_by_algorithm[algorithm]) != reference_seeds:
            raise ValueError(f"sample_seed sets differ for {algorithm}")
    for sample_seed in sorted(reference_seeds):
        reference = rows_by_algorithm[ALGORITHMS[0]][sample_seed]
        for algorithm in ALGORITHMS[1:]:
            candidate = rows_by_algorithm[algorithm][sample_seed]
            for field in METADATA_FIELDS:
                if candidate[field] != reference[field]:
                    raise ValueError(
                        f"metadata mismatch for sample_seed {sample_seed}: {field}"
                    )
    return rows_by_algorithm


def _read_and_validate_task14_rows(
    path: Path,
    *,
    expected_near_count: int,
) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw_row in csv.DictReader(handle):
            sample_seed = _as_int(raw_row, "sample_seed")
            if sample_seed in rows:
                raise ValueError(f"duplicate sample_seed {sample_seed} in Task 14")
            probabilities = tuple(
                _as_finite_float(raw_row, f"p_L{size}") for size in SCALE_SIZES
            )
            _validate_probability_vector(probabilities, "student probabilities")
            separation = _as_finite_float(raw_row, "separation_deg")
            if not 2.0 <= separation < 4.0:
                raise ValueError("Task 14 row is outside [2,4)")
            cohort = _required(raw_row, "threshold_cohort")
            if cohort not in THRESHOLD_COHORTS:
                raise ValueError(f"unknown threshold_cohort: {cohort}")
            rows[sample_seed] = {
                "sample_seed": sample_seed,
                "split": _required(raw_row, "split"),
                "true_angle_1_deg": _optional_finite_float(
                    raw_row, "true_angle_1_deg"
                ),
                "true_angle_2_deg": _optional_finite_float(
                    raw_row, "true_angle_2_deg"
                ),
                "rho": _as_finite_float(raw_row, "rho"),
                "snr_db": _as_finite_float(raw_row, "snr_db"),
                "snapshot_count": _as_int(raw_row, "snapshot_count"),
                "separation_deg": separation,
                "threshold_cohort": cohort,
                "student_probabilities": probabilities,
            }
    if len(rows) != expected_near_count:
        raise ValueError(
            f"Task 14 expected_near_count={expected_near_count}, got {len(rows)}"
        )
    if any(row["split"] != "validation" for row in rows.values()):
        raise ValueError("Task 14 rows must be validation")
    return rows


def _join_near_authority(
    algorithm_rows: Mapping[str, Mapping[int, Mapping[str, Any]]],
    student_rows: Mapping[int, Mapping[str, Any]],
) -> tuple[TeacherAuthorityLabel, ...]:
    pcnss_rows = algorithm_rows[ALGORITHMS[0]]
    near_seeds = {
        sample_seed
        for sample_seed, row in pcnss_rows.items()
        if 2.0 <= row["separation_deg"] < 4.0
    }
    if set(student_rows) != near_seeds:
        raise ValueError("near sample_seed set does not match Task 14")

    labels = []
    for sample_seed in sorted(near_seeds):
        reference = pcnss_rows[sample_seed]
        student = student_rows[sample_seed]
        for field in METADATA_FIELDS:
            student_value = student.get(field)
            if student_value is not None and student_value != reference[field]:
                raise ValueError(
                    f"Task 14 metadata mismatch for sample_seed {sample_seed}: {field}"
                )
        labels.append(
            TeacherAuthorityLabel(
                sample_seed=sample_seed,
                true_angles_deg=(
                    reference["true_angle_1_deg"],
                    reference["true_angle_2_deg"],
                ),
                rho=reference["rho"],
                snr_db=reference["snr_db"],
                snapshot_count=reference["snapshot_count"],
                separation_deg=reference["separation_deg"],
                threshold_cohort=str(student["threshold_cohort"]),
                student_probabilities=student["student_probabilities"],
                fixed_rmspe_deg={
                    size: algorithm_rows[f"fbss_root_music_L{size}"][sample_seed][
                        "sample_rmspe_deg"
                    ]
                    for size in SCALE_SIZES
                },
            )
        )
    return tuple(labels)


def _convert_prediction_row(raw_row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "split": _required(raw_row, "split"),
        "sample_seed": _as_int(raw_row, "sample_seed"),
        "algorithm": _required(raw_row, "algorithm"),
        "true_angle_1_deg": _as_finite_float(raw_row, "true_angle_1_deg"),
        "true_angle_2_deg": _as_finite_float(raw_row, "true_angle_2_deg"),
        "rho": _as_finite_float(raw_row, "rho"),
        "snr_db": _as_finite_float(raw_row, "snr_db"),
        "snapshot_count": _as_int(raw_row, "snapshot_count"),
        "separation_deg": _as_finite_float(raw_row, "separation_deg"),
        "sample_rmspe_deg": _as_finite_float(raw_row, "sample_rmspe_deg"),
    }


def _require_validation_schema_v2(
    run_config: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    if run_config.get("stage") != "evaluate_validation":
        raise ValueError("diagnostic source must be evaluate_validation")
    if run_config.get("split") != "validation" or summary.get("split") != "validation":
        raise ValueError("diagnostic source must be validation")
    if summary.get("report_schema_version") != 2:
        raise ValueError("diagnostic source must use report schema v2")


def _require_task14_manifest(manifest: Mapping[str, Any], expected_count: int) -> None:
    if manifest.get("diagnostic_schema_version") != 1:
        raise ValueError("Task 14 source must use diagnostic schema v1")
    if manifest.get("sample_count") != expected_count:
        raise ValueError("Task 14 manifest sample_count mismatch")
    if manifest.get("no_model_forward") is not True:
        raise ValueError("Task 14 schema-complete source must declare no_model_forward")


def _validate_probability_vector(values: tuple[float, ...], name: str) -> None:
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(f"{name} must be finite and nonnegative")
    if not math.isclose(math.fsum(values), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{name} must sum to one")


def _finite_vector(values: torch.Tensor, name: str) -> np.ndarray:
    array = np.asarray(torch.as_tensor(values).detach().cpu(), dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _kl_divergence(left: np.ndarray, right: np.ndarray) -> float:
    positive = left > 0.0
    safe_right = np.clip(right, DIVERGENCE_EPSILON, None)
    value = float(np.sum(left[positive] * np.log(left[positive] / safe_right[positive])))
    if value < 0.0 and value >= -1e-12:
        return 0.0
    if value < 0.0 or not math.isfinite(value):
        raise ValueError("divergence must be finite and nonnegative")
    return value


def _require_finite_row(row: Mapping[str, Any]) -> None:
    for key, value in row.items():
        if isinstance(value, bool) or value is None or isinstance(value, str):
            continue
        if isinstance(value, tuple):
            if not all(isinstance(item, int) for item in value):
                raise ValueError(f"{key} tuple must contain integer scales")
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            continue
        raise ValueError(f"{key} must be finite or schema-compatible")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _required(row: Mapping[str, Any], field: str) -> Any:
    value = row.get(field)
    if value is None or value == "":
        raise ValueError(f"row is missing {field}")
    return value


def _as_finite_float(row: Mapping[str, Any], field: str) -> float:
    value = _required(row, field)
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _optional_finite_float(row: Mapping[str, Any], field: str) -> float | None:
    value = row.get(field)
    if value is None or value == "":
        return None
    return _as_finite_float(row, field)


def _as_int(row: Mapping[str, Any], field: str) -> int:
    value = _required(row, field)
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        numeric = float(value)
        result = int(numeric)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field} must be an integer") from error
    if not math.isfinite(numeric) or numeric != result:
        raise ValueError(f"{field} must be an integer")
    return result
