"""Read-only frozen validation audit for the Task 17 teacher experiment."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

from multisource_doa.training.teacher_cache import sha256_file


REPORT_FILES = (
    "run_config.json",
    "source_manifest.json",
    "predictions.csv",
    "summary.json",
    "paired_comparisons.csv",
    "failure_reasons.csv",
    "runtime_summary.json",
)
PCNSS = "pcnss_root_music"
L7 = "fbss_root_music_L7"
TIE_TOLERANCE_DEG = 1e-6


@dataclass(frozen=True)
class TeacherExperimentResult:
    decision: Mapping[str, Any]
    paired_rows: tuple[Mapping[str, Any], ...]
    transition_rows: tuple[Mapping[str, Any], ...]
    stratified_rows: tuple[Mapping[str, Any], ...]
    source_sha256: Mapping[str, Mapping[str, str]]


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid boolean: {value}")


def _finite(value: Any, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _load_report(directory: str | Path, expected_count: int) -> dict[str, Any]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"validation report not found: {root}")
    if {path.name for path in root.iterdir()} != set(REPORT_FILES):
        raise ValueError("validation report must contain exactly seven files")
    summary = _json(root / "summary.json")
    manifest = _json(root / "source_manifest.json")
    run_config = _json(root / "run_config.json")
    if summary.get("report_schema_version") != 2 or summary.get("split") != "validation":
        raise ValueError("schema-v2 validation report required")
    if summary.get("best_fixed_fbss_scale") != 7:
        raise ValueError("frozen comparison requires best fixed FBSS L7")
    with (root / "predictions.csv").open("r", encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for raw in raw_rows:
        if raw.get("split") != "validation":
            raise ValueError("prediction rows must be validation only")
        algorithm = raw["algorithm"]
        seed = int(raw["sample_seed"])
        key = (algorithm, seed)
        if key in indexed:
            raise ValueError(f"duplicate prediction row: {key}")
        row = dict(raw)
        row.update({
            "sample_seed": seed,
            "absolute_error_1_deg": _finite(raw["absolute_error_1_deg"], "error"),
            "absolute_error_2_deg": _finite(raw["absolute_error_2_deg"], "error"),
            "sample_rmspe_deg": _finite(raw["sample_rmspe_deg"], "sample_rmspe_deg"),
            "rho": _finite(raw["rho"], "rho"),
            "snr_db": _finite(raw["snr_db"], "snr_db"),
            "snapshot_count": int(raw["snapshot_count"]),
            "separation_deg": _finite(raw["separation_deg"], "separation_deg"),
            "success": _bool(raw["success"]),
            "resolved": _bool(raw["resolved"]),
        })
        indexed[key] = row
    algorithms = {algorithm for algorithm, _ in indexed}
    if PCNSS not in algorithms or L7 not in algorithms:
        raise ValueError("PC-NSS and fixed FBSS L7 predictions are required")
    for algorithm in algorithms:
        if sum(key[0] == algorithm for key in indexed) != expected_count:
            raise ValueError(f"prediction count mismatch for {algorithm}")
    seeds_by_algorithm = {
        algorithm: {seed for name, seed in indexed if name == algorithm}
        for algorithm in algorithms
    }
    if len({frozenset(seeds) for seeds in seeds_by_algorithm.values()}) != 1:
        raise ValueError("algorithms must contain identical sample seeds")
    seeds = next(iter(seeds_by_algorithm.values()))
    if (
        manifest.get("split") != "validation"
        or manifest.get("size") != expected_count
        or manifest.get("sample_seed_start") != min(seeds)
        or manifest.get("sample_seed_end") != max(seeds)
    ):
        raise ValueError("validation manifest seed identity mismatch")
    metadata_fields = (
        "true_angle_1_deg", "true_angle_2_deg", "rho", "snr_db",
        "snapshot_count", "separation_deg",
    )
    for seed in seeds:
        reference = indexed[(PCNSS, seed)]
        for algorithm in algorithms:
            if any(indexed[(algorithm, seed)][field] != reference[field] for field in metadata_fields):
                raise ValueError("sample metadata mismatch across algorithms")
    return {
        "root": root,
        "summary": summary,
        "manifest": manifest,
        "run_config": run_config,
        "rows": indexed,
        "algorithms": algorithms,
        "sha": {name: sha256_file(root / name) for name in REPORT_FILES},
    }


def _algorithm_rows(report: Mapping[str, Any], algorithm: str) -> list[dict[str, Any]]:
    return [report["rows"][(algorithm, seed)] for seed in sorted(
        seed for name, seed in report["rows"] if name == algorithm
    )]


def _metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    errors = [
        float(row[field])
        for row in rows
        for field in ("absolute_error_1_deg", "absolute_error_2_deg")
    ]
    return {
        "failure_aware_rmspe_deg": float(np.sqrt(np.mean(np.square(errors)))),
        "resolution_rate": float(np.mean([bool(row["resolved"]) for row in rows])),
        "failure_count": int(sum(not bool(row["success"]) for row in rows)),
    }


def _near_resolution(rows: list[Mapping[str, Any]]) -> float:
    near = [row for row in rows if 2.0 <= float(row["separation_deg"]) < 4.0]
    if not near:
        raise ValueError("validation report has no [2,4) samples")
    return float(np.mean([bool(row["resolved"]) for row in near]))


def _protocol_identity(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    ignored_manifest = {"checkpoint_sha", "code_sha", "training_metadata"}
    a_manifest = {k: v for k, v in a["manifest"].items() if k not in ignored_manifest}
    b_manifest = {k: v for k, v in b["manifest"].items() if k not in ignored_manifest}
    a_seeds = {seed for name, seed in a["rows"] if name == PCNSS}
    b_seeds = {seed for name, seed in b["rows"] if name == PCNSS}
    l7_fields = (
        "true_angle_1_deg", "true_angle_2_deg", "absolute_error_1_deg",
        "absolute_error_2_deg", "sample_rmspe_deg", "success", "resolved",
        "rho", "snr_db", "snapshot_count", "separation_deg", "failure_reason",
    )
    l7_same = a_seeds == b_seeds and all(
        all(a["rows"][(L7, seed)][field] == b["rows"][(L7, seed)][field] for field in l7_fields)
        for seed in a_seeds
    )
    sample_metadata = ("true_angle_1_deg", "true_angle_2_deg", "rho", "snr_db", "snapshot_count", "separation_deg")
    pcnss_metadata_same = a_seeds == b_seeds and all(
        all(a["rows"][(PCNSS, seed)][field] == b["rows"][(PCNSS, seed)][field] for field in sample_metadata)
        for seed in a_seeds
    )
    frozen_run_keys = (
        "model_seed", "sample_count", "device", "evaluation_batch_size",
        "selected_best_fbss_scale",
    )
    run_identity = all(
        a["run_config"].get(key) == b["run_config"].get(key)
        for key in frozen_run_keys
    )
    return bool(
        a_manifest == b_manifest
        and a["algorithms"] == b["algorithms"]
        and a["run_config"].get("model_seed") == b["run_config"].get("model_seed") == 2026
        and run_identity
        and l7_same
        and pcnss_metadata_same
        and (
            a["manifest"].get("training_metadata") is None
            or a["manifest"].get("training_metadata", {}).get("teacher_mode")
            == "physical"
        )
        and b["manifest"].get("training_metadata", {}).get("teacher_mode") == "failure_aware_error"
        and b["manifest"].get("training_metadata", {}).get("scale_distillation_target_source") == "train_only_failure_aware_rmspe"
        and b["manifest"].get("training_metadata", {}).get("dominance_target_source") == "physical_music_score"
    )


def _stratum_label(name: str, row: Mapping[str, Any]) -> str:
    value = float(row[name]) if name != "snapshot_count" else int(row[name])
    if name == "separation_deg":
        return next(label for lo, hi, label in ((2,4,"[2,4)"),(4,6,"[4,6)"),(6,8,"[6,8)"),(8,10.000001,"[8,10]")) if lo <= value < hi)
    if name == "snr_db":
        return next(label for lo, hi, label in ((-5,0,"[-5,0)"),(0,5,"[0,5)"),(5,10.000001,"[5,10]")) if lo <= value < hi)
    return str(value)


def audit_teacher_experiment(
    baseline_validation_directory: str | Path,
    candidate_validation_directory: str | Path,
    *,
    expected_sample_count: int = 5_000,
) -> TeacherExperimentResult:
    a = _load_report(baseline_validation_directory, expected_sample_count)
    b = _load_report(candidate_validation_directory, expected_sample_count)
    a_rows = _algorithm_rows(a, PCNSS)
    b_rows = _algorithm_rows(b, PCNSS)
    l7_rows = _algorithm_rows(a, L7)
    if [row["sample_seed"] for row in a_rows] != [row["sample_seed"] for row in b_rows]:
        raise ValueError("A/B prediction seeds differ")
    a_metrics, b_metrics, l7_metrics = _metrics(a_rows), _metrics(b_rows), _metrics(l7_rows)
    a_near, b_near, l7_near = map(_near_resolution, (a_rows, b_rows, l7_rows))
    protocol_identity = _protocol_identity(a, b)
    gates = {
        "near_resolution_improves_over_original": b_near > a_near,
        "near_resolution_not_below_fbss_L7": b_near >= l7_near,
        "overall_rmspe_not_worse": b_metrics["failure_aware_rmspe_deg"] <= a_metrics["failure_aware_rmspe_deg"],
        "overall_resolution_not_worse": b_metrics["resolution_rate"] >= a_metrics["resolution_rate"],
        "failure_count_not_worse": b_metrics["failure_count"] <= a_metrics["failure_count"],
        "protocol_identity": protocol_identity,
    }
    paired = []
    transitions = Counter()
    strata: dict[tuple[str, str], list[tuple[dict, dict, dict]]] = defaultdict(list)
    for left, right, fixed in zip(a_rows, b_rows, l7_rows, strict=True):
        delta = float(right["sample_rmspe_deg"]) - float(left["sample_rmspe_deg"])
        outcome = "tie" if abs(delta) <= TIE_TOLERANCE_DEG else ("win" if delta < 0 else "loss")
        transition = f"A{int(left['resolved'])}_B{int(right['resolved'])}"
        transitions[transition] += 1
        paired.append({
            "row_type": "paired", "sample_seed": left["sample_seed"],
            "separation_deg": left["separation_deg"], "a_rmspe_deg": left["sample_rmspe_deg"],
            "b_rmspe_deg": right["sample_rmspe_deg"], "l7_rmspe_deg": fixed["sample_rmspe_deg"],
            "outcome": outcome, "transition": transition,
        })
        for name in ("separation_deg", "snr_db", "rho", "snapshot_count"):
            strata[(name, _stratum_label(name, left))].append((left, right, fixed))
    improved, regressed = transitions["A0_B1"], transitions["A1_B0"]
    discordant = improved + regressed
    if discordant:
        test = binomtest(improved, discordant, 0.5)
        interval = test.proportion_ci(confidence_level=0.95, method="exact")
        paired_statistics = {"mcnemar_exact_p_value": float(test.pvalue), "discordant_improvement_fraction": improved / discordant, "discordant_improvement_ci95": [float(interval.low), float(interval.high)]}
    else:
        paired_statistics = {"mcnemar_exact_p_value": 1.0, "discordant_improvement_fraction": None, "discordant_improvement_ci95": [None, None]}
    transition_rows = tuple({"row_type": "transition", "transition": name, "count": int(transitions[name])} for name in ("A0_B0", "A0_B1", "A1_B0", "A1_B1"))
    stratified = []
    for (name, label), triples in sorted(strata.items()):
        lefts, rights, fixeds = map(list, zip(*triples))
        outcomes = Counter(
            "tie" if abs(float(r["sample_rmspe_deg"]) - float(l["sample_rmspe_deg"])) <= TIE_TOLERANCE_DEG else ("win" if float(r["sample_rmspe_deg"]) < float(l["sample_rmspe_deg"]) else "loss")
            for l, r in zip(lefts, rights, strict=True)
        )
        stratified.append({"stratum": name, "bin": label, "count": len(triples), "a_rmspe_deg": _metrics(lefts)["failure_aware_rmspe_deg"], "b_rmspe_deg": _metrics(rights)["failure_aware_rmspe_deg"], "l7_rmspe_deg": _metrics(fixeds)["failure_aware_rmspe_deg"], "a_resolution": _metrics(lefts)["resolution_rate"], "b_resolution": _metrics(rights)["resolution_rate"], "l7_resolution": _metrics(fixeds)["resolution_rate"], "a_failures": _metrics(lefts)["failure_count"], "b_failures": _metrics(rights)["failure_count"], "l7_failures": _metrics(fixeds)["failure_count"], "win": outcomes["win"], "tie": outcomes["tie"], "loss": outcomes["loss"]})
    passed = all(gates.values())
    decision = {
        "gates": gates,
        "source_values": {"a_overall": a_metrics, "b_overall": b_metrics, "l7_overall": l7_metrics, "a_near_resolution": a_near, "b_near_resolution": b_near, "l7_near_resolution": l7_near},
        "paired_statistics": paired_statistics,
        "outlier_counts": {
            side: {
                cohort: {
                    f"sample_rmspe_gt_{threshold}_deg": sum(
                        float(row["sample_rmspe_deg"]) > threshold
                        for row in cohort_rows
                    )
                    for threshold in (10, 30, 60)
                }
                for cohort, cohort_rows in (
                    ("overall", rows),
                    ("near", [row for row in rows if 2.0 <= float(row["separation_deg"]) < 4.0]),
                )
            }
            for side, rows in (("a", a_rows), ("b", b_rows), ("l7", l7_rows))
        },
        "conclusion": "seed2026_gate_passed" if passed else "experiment_failed",
        "development_authorized": False,
        "multi_seed_authorized": False,
        "locked_test_authorized": False,
        "required_action": "request_development_approval" if passed else "stop_without_tuning",
    }
    return TeacherExperimentResult(decision=decision, paired_rows=tuple(paired), transition_rows=transition_rows, stratified_rows=tuple(stratified), source_sha256={"baseline_validation": a["sha"], "candidate_validation": b["sha"]})
