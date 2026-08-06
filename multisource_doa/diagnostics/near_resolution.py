"""Read-only labels for the frozen near-resolution audit input."""

import csv
import hashlib
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from multisource_doa.data.simulator import DOASample
from multisource_doa.models.pc_nss import MultiScalePCNSS
from multisource_doa.physics.projection import (
    ProjectionResult,
    dykstra_structured_projection,
)
from multisource_doa.training.engine import PCNSSBatch, collate_samples
from multisource_doa.training.losses import aggregate_scale_weights


ERROR_THRESHOLDS_DEG = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
EXPECTED_EVALUATOR_CODE_SHA = "129c3ba3b9fc1919451eef5c67376f04b4b24680"

_SCALE_SIZES = (4, 5, 6, 7)

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


@dataclass(frozen=True)
class NearDiagnosticResult:
    sample_rows: tuple[dict[str, Any], ...]


def scale_weight_diagnostics(
    scale_weights: torch.Tensor,
    valid_mask: torch.Tensor,
    effective_counts: torch.Tensor,
) -> tuple[dict[str, Any], ...]:
    """Compute reliable scale mass and entropy for each frozen sample."""

    distribution = aggregate_scale_weights(
        scale_weights,
        valid_mask,
        effective_counts,
    )
    rows: list[dict[str, Any]] = []
    for weights, mask, probabilities in zip(
        scale_weights.detach().cpu(),
        valid_mask.detach().cpu().bool(),
        distribution.detach().cpu(),
        strict=True,
    ):
        entropy = -torch.sum(
            probabilities.clamp_min(1e-12) * probabilities.clamp_min(1e-12).log()
        ) / math.log(len(_SCALE_SIZES))
        dominant_index = int(torch.argmax(probabilities).item())
        row: dict[str, Any] = {
            f"p_L{size}": float(probabilities[index].item())
            for index, size in enumerate(_SCALE_SIZES)
        }
        row["scale_entropy_normalized"] = float(entropy.item())
        row["dominant_scale"] = _SCALE_SIZES[dominant_index]
        lag_entropies: list[float | None] = []
        for lag in range(weights.shape[-1]):
            valid_indices = mask[:, lag]
            valid_count = int(valid_indices.sum().item())
            for index, size in enumerate(_SCALE_SIZES):
                if bool(mask[index, lag]):
                    row[f"scale_weight_L{size}_lag{lag}"] = float(
                        weights[index, lag].item()
                    )
            if valid_count < 2:
                lag_entropies.append(None)
                continue
            lag_probabilities = weights[:, lag][valid_indices]
            lag_probabilities = lag_probabilities / lag_probabilities.sum().clamp_min(1e-12)
            lag_entropy = -torch.sum(
                lag_probabilities.clamp_min(1e-12)
                * lag_probabilities.clamp_min(1e-12).log()
            ) / math.log(valid_count)
            lag_entropies.append(float(lag_entropy.item()))
        row["lag_entropy_normalized"] = tuple(lag_entropies)
        rows.append(row)
    return tuple(rows)


def residual_diagnostics(
    lag_residual_ri: torch.Tensor,
    *,
    residual_limit: float,
) -> tuple[dict[str, Any], ...]:
    """Summarize bounded lag residual magnitude and saturation per sample."""

    if lag_residual_ri.ndim != 3 or lag_residual_ri.shape[-1] != 2:
        raise ValueError("lag_residual_ri must have shape [batch, lag, real_imag]")
    if residual_limit <= 0.0:
        raise ValueError("residual_limit must be positive")
    magnitudes = torch.linalg.vector_norm(lag_residual_ri.detach().cpu(), dim=-1)
    saturated = magnitudes / residual_limit >= 0.95
    rows: list[dict[str, Any]] = []
    for magnitude, is_saturated in zip(magnitudes, saturated, strict=True):
        row = {
            "residual_magnitude_p50": float(torch.quantile(magnitude, 0.50).item()),
            "residual_magnitude_p95": float(torch.quantile(magnitude, 0.95).item()),
            "residual_magnitude_max": float(magnitude.max().item()),
            "saturated_lag_count": int(is_saturated.sum().item()),
            "saturated_lag_rate": float(is_saturated.float().mean().item()),
            "lag_residual_mean": tuple(float(value) for value in magnitude.tolist()),
            "lag_residual_p95": tuple(float(value) for value in magnitude.tolist()),
            "lag_saturated_rate": tuple(
                float(value) for value in is_saturated.to(torch.float32).tolist()
            ),
        }
        rows.append(row)
    return tuple(rows)


def projection_diagnostics(
    candidate_covariances: np.ndarray,
    train_projected_covariances: np.ndarray,
    *,
    projection_fn: Callable[[np.ndarray], ProjectionResult] = dykstra_structured_projection,
) -> tuple[dict[str, Any], ...]:
    """Measure the train and evaluation structural projections separately."""

    rows = []
    for candidate, train_projected in zip(
        candidate_covariances,
        train_projected_covariances,
        strict=True,
    ):
        final = projection_fn(train_projected)
        candidate_norm = max(float(np.linalg.norm(candidate, ord="fro")), 1e-12)
        train_norm = max(float(np.linalg.norm(train_projected, ord="fro")), 1e-12)
        rows.append(
            {
                "train_projection_change": float(
                    np.linalg.norm(train_projected - candidate, ord="fro")
                    / candidate_norm
                ),
                "eval_projection_change": float(
                    np.linalg.norm(final.matrix - train_projected, ord="fro")
                    / train_norm
                ),
                "total_projection_change": float(
                    np.linalg.norm(final.matrix - candidate, ord="fro")
                    / candidate_norm
                ),
                "dykstra_converged": bool(final.converged),
                "dykstra_iterations": int(final.iterations),
                "final_hermitian_error": float(final.hermitian_error),
                "final_toeplitz_error": float(final.toeplitz_error),
                "final_trace_error": float(final.trace_error),
                "final_min_eigenvalue": float(final.min_eigenvalue),
            }
        )
    return tuple(rows)


def diagnose_near_samples(
    samples: Sequence[DOASample],
    labels_by_seed: Mapping[int, NearAuditLabel],
    model: MultiScalePCNSS,
    *,
    device: torch.device,
    batch_size: int,
) -> NearDiagnosticResult:
    """Run frozen forward diagnostics and join authority labels by sample seed."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    _require_unique_sample_seeds(samples)
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            sample_batch = list(samples[start : start + batch_size])
            batch = _batch_to_device(collate_samples(sample_batch), device)
            output = model(
                batch.raw_lags_ri,
                batch.fbss_lags_ri,
                batch.valid_mask,
                batch.effective_counts,
                batch.quality_features,
            )
            scale_rows = scale_weight_diagnostics(
                output.scale_weights,
                batch.valid_mask,
                batch.effective_counts,
            )
            residual_rows = residual_diagnostics(
                output.lag_residual_ri,
                residual_limit=model.residual_fraction,
            )
            projection_rows = projection_diagnostics(
                output.candidate_covariance.detach().cpu().numpy(),
                output.covariance.detach().cpu().numpy(),
            )
            for sample, scale_row, residual_row, projection_row in zip(
                sample_batch,
                scale_rows,
                residual_rows,
                projection_rows,
                strict=True,
            ):
                try:
                    label = labels_by_seed[sample.sample_seed]
                except KeyError as error:
                    raise ValueError(
                        f"missing authority label for sample_seed {sample.sample_seed}"
                    ) from error
                _validate_label_metadata(sample, label)
                rows.append(
                    {
                        **label.pcnss_row,
                        "sample_seed": sample.sample_seed,
                        "true_angle_1_deg": float(sample.angles_deg[0]),
                        "true_angle_2_deg": float(sample.angles_deg[1]),
                        "rho": float(sample.rho),
                        "snr_db": float(sample.snr_db),
                        "snapshot_count": int(sample.snapshot_count),
                        "separation_deg": float(abs(np.diff(sample.angles_deg)[0])),
                        "threshold_cohort": label.threshold_cohort,
                        **scale_row,
                        **residual_row,
                        **projection_row,
                    }
                )
    return NearDiagnosticResult(sample_rows=tuple(rows))


def _batch_to_device(batch: PCNSSBatch, device: torch.device) -> PCNSSBatch:
    values: dict[str, Any] = {}
    for item in fields(batch):
        value = getattr(batch, item.name)
        if isinstance(value, torch.Tensor):
            values[item.name] = value.to(device)
        elif isinstance(value, dict):
            values[item.name] = {key: tensor.to(device) for key, tensor in value.items()}
        else:
            values[item.name] = value
    return PCNSSBatch(**values)


def _require_unique_sample_seeds(samples: Sequence[DOASample]) -> None:
    seeds = [sample.sample_seed for sample in samples]
    if len(seeds) != len(set(seeds)):
        raise ValueError("samples contain duplicate sample_seed values")


def _validate_label_metadata(sample: DOASample, label: NearAuditLabel) -> None:
    if (
        sample.rho != label.rho
        or sample.snr_db != label.snr_db
        or sample.snapshot_count != label.snapshot_count
        or not math.isclose(
            float(abs(np.diff(sample.angles_deg)[0])),
            label.separation_deg,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(f"metadata mismatch for sample_seed {sample.sample_seed}")


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
