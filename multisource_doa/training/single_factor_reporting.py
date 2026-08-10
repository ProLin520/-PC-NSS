"""Three-file reporting for the Task 17 single-factor identity audit."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from multisource_doa.training.single_factor_audit import SingleFactorAuditResult


SINGLE_FACTOR_AUDIT_SCHEMA_VERSION = 1


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_single_factor_audit_report(
    result: SingleFactorAuditResult,
    output_directory: str | Path,
    *,
    run_config: Mapping[str, Any],
) -> Path:
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "audit_config.json", dict(run_config))
    _write_json(
        output / "source_manifest.json",
        {
            "single_factor_audit_schema_version": SINGLE_FACTOR_AUDIT_SCHEMA_VERSION,
            "source_sha256": result.source_sha256,
            "no_model_forward": True,
            "training_performed": False,
            "locked_test_accessed": False,
        },
    )
    _write_json(
        output / "single_factor_audit.json",
        {
            "baseline_reuse_allowed": result.baseline_reuse_allowed,
            "gates": dict(result.gates),
            "evidence": dict(result.evidence),
            "training_authorized": False,
        },
    )
    return output
