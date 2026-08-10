"""Five-file immutable report writer for Task 17 experiment audits."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from multisource_doa.evaluation.teacher_experiment import TeacherExperimentResult


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader(); writer.writerows(rows)


def write_teacher_experiment_report(
    result: TeacherExperimentResult,
    output_directory: str | Path,
    *,
    run_config: Mapping[str, Any],
) -> Path:
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    _json(output / "experiment_audit_config.json", dict(run_config))
    _json(output / "source_manifest.json", {"teacher_experiment_audit_schema_version": 1, "source_sha256": result.source_sha256, "no_model_forward": True, "training_performed": False, "locked_test_accessed": False})
    _csv(output / "paired_and_transitions.csv", [dict(row) for row in (*result.paired_rows, *result.transition_rows)])
    _csv(output / "stratified_summary.csv", [dict(row) for row in result.stratified_rows])
    _json(output / "decision.json", dict(result.decision))
    return output
