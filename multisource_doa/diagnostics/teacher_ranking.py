"""Pure ranking metrics for the frozen Task 16 teacher diagnosis."""

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
from multisource_doa.training.teacher import normalized_music_denominator


SCALE_SIZES = (4, 5, 6, 7)
ORACLE_TIE_TOLERANCE_DEG = 1e-6
TASK15_SCORE_RTOL = 1e-6
TASK15_SCORE_ATOL = 1e-7
THRESHOLD_COHORTS = (
    "estimation_failure",
    "separation_failure",
    "resolved",
    "near_miss_1_1p25",
    "near_miss_1p25_1p5",
    "near_miss_1p5_2",
    "far_miss_gt_2",
)
TASK15_REPORT_FILES = (
    "diagnostic_config.json",
    "source_manifest.json",
    "teacher_sample_diagnostics.csv",
    "teacher_summary.json",
    "teacher_stratified_summary.csv",
    "decision.json",
)


@dataclass(frozen=True)
class TeacherRankingLabel:
    sample_seed: int
    true_angles_deg: tuple[float, float]
    rho: float
    snr_db: float
    snapshot_count: int
    separation_deg: float
    threshold_cohort: str
    task15_scores: tuple[float, float, float, float]
    fixed_rmspe_deg: dict[int, float]


@dataclass(frozen=True)
class TeacherRankingInputs:
    labels_by_seed: dict[int, TeacherRankingLabel]
    task15_manifest: dict[str, Any]
    task15_summary: dict[str, Any]
    task15_decision: dict[str, Any]
    task15_sha256: dict[str, str]
    upstream_sha256: dict[str, str]
    validation_split_seed: int


@dataclass(frozen=True)
class TeacherRankingResult:
    sample_rows: tuple[dict[str, Any], ...]


def load_teacher_ranking_inputs(
    task15_directory: str | Path,
    *,
    expected_count: int = 1270,
) -> TeacherRankingInputs:
    """Authenticate the complete Task 15 report and its upstream source files."""

    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count <= 0:
        raise ValueError("expected_count must be a positive integer")
    directory = Path(task15_directory)
    for name in TASK15_REPORT_FILES:
        if not (directory / name).is_file():
            raise ValueError(f"Task 15 report is missing {name}")
    task15_hashes = {
        name: _sha256(directory / name) for name in TASK15_REPORT_FILES
    }
    config = _read_json(directory / "diagnostic_config.json")
    manifest = _read_json(directory / "source_manifest.json")
    task15_summary = _read_json(directory / "teacher_summary.json")
    task15_decision = _read_json(directory / "decision.json")
    if manifest.get("teacher_diagnostic_schema_version") != 1:
        raise ValueError("Task 15 schema version must be 1")
    expected_manifest_fields = {
        "sample_count": expected_count,
        "device": "cpu",
        "batch_size": 128,
        "no_model_forward": True,
        "training_performed": False,
    }
    for field, expected in expected_manifest_fields.items():
        if manifest.get(field) != expected:
            raise ValueError(f"Task 15 manifest {field} mismatch")
    if config.get("stage") != "diagnose_validation_teacher" or config.get("split") != "validation":
        raise ValueError("Task 15 config must identify the validation diagnosis")
    if config.get("device") != "cpu" or config.get("batch_size") != 128:
        raise ValueError("Task 15 config must use CPU batch size 128")

    recorded_upstream = manifest.get("input_sha256")
    if not isinstance(recorded_upstream, dict):
        raise ValueError("Task 15 manifest is missing upstream input hashes")
    audit_directory = Path(str(config.get("report_directory", "")))
    task14_directory = Path(str(config.get("task14_directory", "")))
    upstream_paths = {
        "audit/run_config.json": audit_directory / "run_config.json",
        "audit/summary.json": audit_directory / "summary.json",
        "audit/source_manifest.json": audit_directory / "source_manifest.json",
        "audit/predictions.csv": audit_directory / "predictions.csv",
        "task14/source_manifest.json": task14_directory / "source_manifest.json",
        "task14/near_sample_diagnostics.csv": task14_directory
        / "near_sample_diagnostics.csv",
    }
    if set(recorded_upstream) != set(upstream_paths):
        raise ValueError("Task 15 upstream input hash set mismatch")
    current_upstream: dict[str, str] = {}
    for name, path in upstream_paths.items():
        if not path.is_file():
            raise ValueError(f"Task 15 upstream source is missing: {name}")
        current_upstream[name] = _sha256(path)
        if current_upstream[name] != recorded_upstream[name]:
            raise ValueError(f"Task 15 upstream SHA mismatch: {name}")

    labels = _read_task15_sample_rows(
        directory / "teacher_sample_diagnostics.csv",
        expected_count=expected_count,
    )
    split_seed = manifest.get("validation_split_seed")
    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise ValueError("Task 15 validation_split_seed must be an integer")
    return TeacherRankingInputs(
        labels_by_seed={label.sample_seed: label for label in labels},
        task15_manifest=manifest,
        task15_summary=task15_summary,
        task15_decision=task15_decision,
        task15_sha256=task15_hashes,
        upstream_sha256=current_upstream,
        validation_split_seed=split_seed,
    )


def diagnose_teacher_ranking_samples(
    samples: Sequence[DOASample],
    labels_by_seed: Mapping[int, TeacherRankingLabel],
    *,
    batch_size: int = 128,
) -> TeacherRankingResult:
    """Rebuild physical score components on CPU without any neural model."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    seeds = [sample.sample_seed for sample in samples]
    if len(seeds) != len(set(seeds)):
        raise ValueError("samples contain duplicate sample_seed values")
    if set(seeds) != set(labels_by_seed):
        raise ValueError("samples and Task 15 labels must have identical seed sets")

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch_samples = list(samples[start : start + batch_size])
            batch = collate_samples(batch_samples)
            if any(
                covariance.device.type != "cpu"
                for covariance in batch.fbss_covariances.values()
            ):
                raise ValueError("teacher ranking diagnosis must run on CPU")
            midpoint = batch.true_angles_deg.mean(dim=-1, keepdim=True)
            queries = torch.cat((batch.true_angles_deg, midpoint), dim=-1)
            q_by_scale: dict[int, torch.Tensor] = {}
            for size in SCALE_SIZES:
                q_by_scale[size] = normalized_music_denominator(
                    batch.fbss_covariances[size].detach(),
                    queries,
                    source_count=2,
                )
            for index, sample in enumerate(batch_samples):
                label = labels_by_seed[sample.sample_seed]
                _validate_regenerated_metadata(sample, label)
                q1 = tuple(float(q_by_scale[size][index, 0]) for size in SCALE_SIZES)
                q2 = tuple(float(q_by_scale[size][index, 1]) for size in SCALE_SIZES)
                qmid = tuple(float(q_by_scale[size][index, 2]) for size in SCALE_SIZES)
                components = component_ranking_diagnostics(
                    q_true_1=q1,
                    q_true_2=q2,
                    q_midpoint=qmid,
                    rmspe_deg=tuple(label.fixed_rmspe_deg[size] for size in SCALE_SIZES),
                )
                if not np.allclose(
                    components["current_score"],
                    label.task15_scores,
                    rtol=TASK15_SCORE_RTOL,
                    atol=TASK15_SCORE_ATOL,
                ):
                    raise ValueError(
                        f"Task 15 score mismatch for sample_seed {sample.sample_seed}"
                    )
                rows.append(_build_sample_row(label, q1, q2, qmid, components))
    return TeacherRankingResult(sample_rows=tuple(rows))


def rank_signal_against_rmspe(
    signal: Sequence[float],
    rmspe_deg: Sequence[float],
) -> dict[str, object]:
    """Compare a higher-is-better signal with lower-is-better scale RMSPE."""

    signal_values = _validated_vector(signal, "signal")
    rmspe_values = _validated_vector(rmspe_deg, "rmspe")
    ordered_indices = sorted(
        range(4), key=lambda index: (-signal_values[index], SCALE_SIZES[index])
    )
    minimum_rmspe = min(rmspe_values)
    oracle_indices = tuple(
        index
        for index, value in enumerate(rmspe_values)
        if abs(value - minimum_rmspe) <= ORACLE_TIE_TOLERANCE_DEG
    )

    concordant = 0
    discordant = 0
    teacher_ties = 0
    oracle_ties = 0
    ties_both = 0
    exact_signal_ties = 0
    for left in range(4):
        for right in range(left + 1, 4):
            signal_difference = signal_values[left] - signal_values[right]
            rmspe_difference = rmspe_values[left] - rmspe_values[right]
            signal_tie = signal_difference == 0.0
            oracle_tie = abs(rmspe_difference) <= ORACLE_TIE_TOLERANCE_DEG
            exact_signal_ties += int(signal_tie)
            if signal_tie and oracle_tie:
                ties_both += 1
            elif oracle_tie:
                oracle_ties += 1
            elif signal_tie:
                teacher_ties += 1
            elif signal_difference * rmspe_difference < 0.0:
                concordant += 1
            else:
                discordant += 1

    comparable = concordant + discordant + teacher_ties
    pairwise_rate = concordant / comparable if comparable else None
    kendall_denominator = math.sqrt(
        (concordant + discordant + teacher_ties)
        * (concordant + discordant + oracle_ties)
    )
    kendall = (
        (concordant - discordant) / kendall_denominator
        if kendall_denominator
        else None
    )
    spearman = _pearson_correlation(
        _average_ranks(signal_values, higher_is_better=True, tolerance=0.0),
        _average_ranks(
            rmspe_values,
            higher_is_better=False,
            tolerance=ORACLE_TIE_TOLERANCE_DEG,
        ),
    )

    top1_index = ordered_indices[0]
    top2_indices = ordered_indices[:2]
    return {
        "spearman_rho": spearman,
        "kendall_tau_b": kendall,
        "concordant_pair_count": concordant,
        "discordant_pair_count": discordant,
        "teacher_tie_pair_count": teacher_ties,
        "oracle_tie_pair_count": oracle_ties + ties_both,
        "both_tie_pair_count": ties_both,
        "pairwise_comparable_count": comparable,
        "pairwise_concordance_rate": pairwise_rate,
        "exact_signal_tie_pair_count": exact_signal_ties,
        "top1_scale": SCALE_SIZES[top1_index],
        "top2_scales": tuple(SCALE_SIZES[index] for index in top2_indices),
        "oracle_best_scales": tuple(SCALE_SIZES[index] for index in oracle_indices),
        "top1_oracle_agreement": top1_index in oracle_indices,
        "top2_oracle_coverage": bool(set(top2_indices).intersection(oracle_indices)),
        "top1_regret_deg": rmspe_values[top1_index] - minimum_rmspe,
    }


def component_ranking_diagnostics(
    *,
    q_true_1: Sequence[float],
    q_true_2: Sequence[float],
    q_midpoint: Sequence[float],
    rmspe_deg: Sequence[float],
) -> dict[str, object]:
    """Decompose the teacher score and evaluate all three ranking signals."""

    q1 = _validated_vector(q_true_1, "q_true_1")
    q2 = _validated_vector(q_true_2, "q_true_2")
    midpoint = _validated_vector(q_midpoint, "q_midpoint")
    rmspe = _validated_vector(rmspe_deg, "rmspe")
    truth_mean = tuple((left + right) / 2.0 for left, right in zip(q1, q2))
    negative_truth = tuple(-value for value in truth_mean)
    current_score = tuple(
        middle - truth for middle, truth in zip(midpoint, truth_mean)
    )
    midpoint_range = _range(midpoint)
    truth_range = _range(truth_mean)
    score_range = _range(current_score)
    cancellation_denominator = midpoint_range + truth_range
    return {
        "q_truth_mean": truth_mean,
        "negative_truth_mean": negative_truth,
        "current_score": current_score,
        "q_midpoint_range": midpoint_range,
        "q_midpoint_std": _population_std(midpoint),
        "negative_truth_mean_range": truth_range,
        "negative_truth_mean_std": _population_std(negative_truth),
        "current_score_range": score_range,
        "current_score_std": _population_std(current_score),
        "cancellation_ratio": (
            score_range / cancellation_denominator
            if cancellation_denominator > 0.0
            else None
        ),
        "cancellation_denominator_zero": cancellation_denominator == 0.0,
        "signals": {
            "current_score": rank_signal_against_rmspe(current_score, rmspe),
            "q_midpoint": rank_signal_against_rmspe(midpoint, rmspe),
            "negative_truth_mean": rank_signal_against_rmspe(
                negative_truth, rmspe
            ),
        },
    }


def _validated_vector(values: Sequence[float], name: str) -> tuple[float, ...]:
    if len(values) != 4:
        raise ValueError(f"{name} must contain four scale values")
    converted = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in converted):
        raise ValueError(f"{name} values must be finite")
    return converted


def _range(values: Sequence[float]) -> float:
    return max(values) - min(values)


def _population_std(values: Sequence[float]) -> float:
    mean = math.fsum(values) / len(values)
    return math.sqrt(math.fsum((value - mean) ** 2 for value in values) / len(values))


def _average_ranks(
    values: Sequence[float],
    *,
    higher_is_better: bool,
    tolerance: float,
) -> tuple[float, ...]:
    indices = sorted(
        range(len(values)),
        key=lambda index: (-values[index] if higher_is_better else values[index], index),
    )
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indices):
        end = start + 1
        anchor = values[indices[start]]
        while end < len(indices) and abs(values[indices[end]] - anchor) <= tolerance:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[indices[position]] = average_rank
        start = end
    return tuple(ranks)


def _pearson_correlation(
    left: Sequence[float], right: Sequence[float]
) -> float | None:
    left_mean = math.fsum(left) / len(left)
    right_mean = math.fsum(right) / len(right)
    numerator = math.fsum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_square = math.fsum((value - left_mean) ** 2 for value in left)
    right_square = math.fsum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_square * right_square)
    return numerator / denominator if denominator else None


def _read_task15_sample_rows(
    path: Path,
    *,
    expected_count: int,
) -> tuple[TeacherRankingLabel, ...]:
    labels: list[TeacherRankingLabel] = []
    seen: set[int] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_seed = _csv_int(row, "sample_seed")
            if sample_seed in seen:
                raise ValueError(f"duplicate Task 15 sample_seed {sample_seed}")
            seen.add(sample_seed)
            scores = tuple(
                _csv_float(row, f"teacher_score_L{size}") for size in SCALE_SIZES
            )
            for prefix in (
                "teacher_p_current_L",
                "teacher_p_counterfactual_L",
                "student_p_L",
            ):
                probabilities = tuple(
                    _csv_float(row, f"{prefix}{size}") for size in SCALE_SIZES
                )
                if any(value < 0.0 for value in probabilities) or not math.isclose(
                    math.fsum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6
                ):
                    raise ValueError(f"Task 15 {prefix} probabilities are invalid")
            separation = _csv_float(row, "separation_deg")
            if not 2.0 <= separation < 4.0:
                raise ValueError("Task 15 sample is outside frozen [2,4) separation")
            cohort = _csv_required(row, "threshold_cohort")
            if cohort not in THRESHOLD_COHORTS:
                raise ValueError(f"unknown Task 15 threshold cohort: {cohort}")
            labels.append(
                TeacherRankingLabel(
                    sample_seed=sample_seed,
                    true_angles_deg=(
                        _csv_float(row, "true_angle_1_deg"),
                        _csv_float(row, "true_angle_2_deg"),
                    ),
                    rho=_csv_float(row, "rho"),
                    snr_db=_csv_float(row, "snr_db"),
                    snapshot_count=_csv_int(row, "snapshot_count"),
                    separation_deg=separation,
                    threshold_cohort=cohort,
                    task15_scores=scores,
                    fixed_rmspe_deg={
                        size: _csv_float(
                            row, f"fbss_L{size}_sample_rmspe_deg"
                        )
                        for size in SCALE_SIZES
                    },
                )
            )
    if len(labels) != expected_count:
        raise ValueError(
            f"Task 15 expected_count={expected_count}, got {len(labels)}"
        )
    seeds = [label.sample_seed for label in labels]
    if seeds != sorted(seeds):
        raise ValueError("Task 15 sample seeds must be strictly ascending")
    return tuple(labels)


def _validate_regenerated_metadata(
    sample: DOASample,
    label: TeacherRankingLabel,
) -> None:
    separation = float(abs(np.diff(sample.angles_deg)[0]))
    if (
        sample.sample_seed != label.sample_seed
        or not np.allclose(
            sample.angles_deg, label.true_angles_deg, rtol=0.0, atol=1e-9
        )
        or sample.rho != label.rho
        or sample.snr_db != label.snr_db
        or sample.snapshot_count != label.snapshot_count
        or not math.isclose(
            separation, label.separation_deg, rel_tol=0.0, abs_tol=1e-9
        )
    ):
        raise ValueError(f"metadata mismatch for sample_seed {sample.sample_seed}")


def _build_sample_row(
    label: TeacherRankingLabel,
    q1: tuple[float, ...],
    q2: tuple[float, ...],
    qmid: tuple[float, ...],
    components: Mapping[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "split": "validation",
        "sample_seed": label.sample_seed,
        "true_angle_1_deg": label.true_angles_deg[0],
        "true_angle_2_deg": label.true_angles_deg[1],
        "rho": label.rho,
        "snr_db": label.snr_db,
        "snapshot_count": label.snapshot_count,
        "separation_deg": label.separation_deg,
        "threshold_cohort": label.threshold_cohort,
        "q_midpoint_range": components["q_midpoint_range"],
        "q_midpoint_std": components["q_midpoint_std"],
        "negative_truth_mean_range": components["negative_truth_mean_range"],
        "negative_truth_mean_std": components["negative_truth_mean_std"],
        "current_score_range": components["current_score_range"],
        "current_score_std": components["current_score_std"],
        "cancellation_ratio": components["cancellation_ratio"],
        "cancellation_denominator_zero": components[
            "cancellation_denominator_zero"
        ],
    }
    truth_mean = components["q_truth_mean"]
    negative_truth = components["negative_truth_mean"]
    current_score = components["current_score"]
    for index, size in enumerate(SCALE_SIZES):
        row.update(
            {
                f"q_true_1_L{size}": q1[index],
                f"q_true_2_L{size}": q2[index],
                f"q_truth_mean_L{size}": truth_mean[index],
                f"q_midpoint_L{size}": qmid[index],
                f"negative_truth_mean_L{size}": negative_truth[index],
                f"current_score_L{size}": current_score[index],
                f"task15_score_L{size}": label.task15_scores[index],
                f"fbss_L{size}_sample_rmspe_deg": label.fixed_rmspe_deg[size],
            }
        )
    for signal_name, metrics in components["signals"].items():
        for metric_name, value in metrics.items():
            row[f"{signal_name}_{metric_name}"] = value
    return row


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_required(row: Mapping[str, str], field: str) -> str:
    value = row.get(field)
    if value is None or value == "":
        raise ValueError(f"Task 15 row is missing {field}")
    return value


def _csv_float(row: Mapping[str, str], field: str) -> float:
    try:
        value = float(_csv_required(row, field))
    except ValueError as error:
        raise ValueError(f"Task 15 {field} must be finite") from error
    if not math.isfinite(value):
        raise ValueError(f"Task 15 {field} must be finite")
    return value


def _csv_int(row: Mapping[str, str], field: str) -> int:
    text = _csv_required(row, field)
    try:
        value = int(text)
    except ValueError as error:
        raise ValueError(f"Task 15 {field} must be an integer") from error
    return value
