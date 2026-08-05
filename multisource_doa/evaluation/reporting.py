"""Standard-library CSV/JSON output for unified evaluation."""

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from multisource_doa.evaluation.runner import EvaluationRunResult
from multisource_doa.training.artifacts import prepare_run_directory


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_evaluation_report(
    result: EvaluationRunResult,
    output_directory: str | Path,
    *,
    run_config: dict,
    source_manifest: dict,
    code_sha: str,
    checkpoint_sha: str,
    refuse_overwrite: bool = True,
) -> Path:
    output = prepare_run_directory(
        output_directory,
        refuse_overwrite=refuse_overwrite,
    )
    _write_json(
        output / "run_config.json",
        {**run_config, "split": result.split.value},
    )
    _write_json(
        output / "source_manifest.json",
        {
            **source_manifest,
            "code_sha": code_sha,
            "checkpoint_sha": checkpoint_sha,
        },
    )
    prediction_rows = list(result.predictions)
    _write_csv(
        output / "predictions.csv",
        prediction_rows,
        list(prediction_rows[0]),
    )
    _write_json(
        output / "summary.json",
        {
            "framework_validation": True,
            "research_acceptance": "not_run",
            "split": result.split.value,
            "best_fixed_fbss_scale": result.best_fixed_fbss_scale,
            "algorithms": result.summaries,
        },
    )

    paired_rows = []
    for comparison_name, groups in result.paired_comparisons.items():
        for group_name, bins in groups.items():
            if group_name == "overall":
                paired_rows.append(
                    {
                        "comparison": comparison_name,
                        "group": "overall",
                        "bin": "all",
                        **bins,
                    }
                )
            else:
                for bin_name, counts in bins.items():
                    paired_rows.append(
                        {
                            "comparison": comparison_name,
                            "group": group_name,
                            "bin": bin_name,
                            **counts,
                        }
                    )
    _write_csv(
        output / "paired_comparisons.csv",
        paired_rows,
        ["comparison", "group", "bin", "win", "tie", "loss"],
    )

    failure_rows = []
    for algorithm, summary in result.summaries.items():
        reasons = summary["failure_reasons"]
        if not reasons:
            failure_rows.append(
                {"algorithm": algorithm, "failure_reason": "", "count": 0}
            )
        for reason, count in reasons.items():
            failure_rows.append(
                {"algorithm": algorithm, "failure_reason": reason, "count": count}
            )
    _write_csv(
        output / "failure_reasons.csv",
        failure_rows,
        ["algorithm", "failure_reason", "count"],
    )
    _write_json(output / "runtime_summary.json", result.runtime_seconds)
    return output
