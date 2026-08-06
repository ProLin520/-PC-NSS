"""Unified evaluation of classical estimators and PC-NSS."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch

from multisource_doa.baselines.classical import (
    DOAEstimate,
    esprit,
    evaluate_fixed_scale_family,
    music_scan,
    root_music_raw,
)
from multisource_doa.config import SplitName
from multisource_doa.data.simulator import DOASample
from multisource_doa.evaluation.metrics import (
    NEAR_SEPARATION_BIN,
    SampleScore,
    aggregate_near_separation_audit,
    aggregate_metrics,
    paired_comparison,
    score_doa_sample,
)
from multisource_doa.models.pc_nss import MultiScalePCNSS
from multisource_doa.physics.covariance import sample_covariance
from multisource_doa.physics.projection import dykstra_structured_projection
from multisource_doa.physics.root_music import estimate_root_music
from multisource_doa.training.engine import collate_samples


@dataclass(frozen=True)
class EvaluationRunResult:
    split: SplitName
    summaries: dict[str, dict]
    predictions: tuple[dict, ...]
    scores_by_algorithm: dict[str, tuple[SampleScore, ...]]
    best_fixed_fbss_scale: int | None
    paired_comparisons: dict[str, dict]
    near_separation_audit: dict
    runtime_seconds: dict[str, float | int]


DEFAULT_INFERENCE_BATCH_SIZE = 128


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _infer_pcnss_covariances(
    samples: list[DOASample],
    model: MultiScalePCNSS,
    device: torch.device,
    *,
    batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
) -> tuple[np.ndarray, float]:
    if (
        isinstance(batch_size, bool)
        or int(batch_size) != batch_size
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    model.eval()
    covariance_batches = []
    neural_runtime = 0.0
    with torch.no_grad():
        for start in range(0, len(samples), int(batch_size)):
            batch = collate_samples(samples[start : start + int(batch_size)])
            _synchronize_if_cuda(device)
            started = perf_counter()
            output = model(
                batch.raw_lags_ri.to(device),
                batch.fbss_lags_ri.to(device),
                batch.valid_mask.to(device),
                batch.effective_counts.to(device),
                batch.quality_features.to(device),
            )
            _synchronize_if_cuda(device)
            neural_runtime += perf_counter() - started
            covariance_batches.append(output.covariance.detach().cpu().numpy())
    return np.concatenate(covariance_batches, axis=0), neural_runtime


def _pcnss_estimates(
    samples: list[DOASample],
    model: MultiScalePCNSS,
    device: torch.device,
    inference_batch_size: int,
) -> tuple[list[DOAEstimate], float]:
    covariances, neural_runtime = _infer_pcnss_covariances(
        samples,
        model,
        device,
        batch_size=inference_batch_size,
    )
    estimates = []
    for covariance in covariances:
        item_started = perf_counter()
        projection = dykstra_structured_projection(covariance)
        if projection.converged:
            root = estimate_root_music(projection.matrix, source_count=2)
            angles = root.angles_deg
            success = root.success
            reason = root.failure_reason
            metadata = {
                "projection_iterations": projection.iterations,
                "projection_converged": True,
                "candidate_count": root.candidate_count,
            }
        else:
            angles = np.empty(0, dtype=np.float64)
            success = False
            reason = "projection_not_converged"
            metadata = {
                "projection_iterations": projection.iterations,
                "projection_converged": False,
            }
        estimates.append(
            DOAEstimate(
                algorithm="pcnss_root_music",
                angles_deg=angles,
                success=success,
                failure_reason=reason,
                runtime_seconds=(
                    neural_runtime / len(samples) + perf_counter() - item_started
                ),
                metadata=metadata,
            )
        )
    return estimates, neural_runtime


def _prediction_row(
    split: SplitName,
    sample: DOASample,
    estimate: DOAEstimate,
    score: SampleScore,
) -> dict:
    matched = score.match.estimated_angles_deg
    errors = score.match.absolute_errors_deg
    return {
        "split": split.value,
        "sample_seed": sample.sample_seed,
        "algorithm": estimate.algorithm,
        "true_angle_1_deg": float(sample.angles_deg[0]),
        "true_angle_2_deg": float(sample.angles_deg[1]),
        "estimated_angle_1_deg": float(matched[0]),
        "estimated_angle_2_deg": float(matched[1]),
        "absolute_error_1_deg": float(errors[0]),
        "absolute_error_2_deg": float(errors[1]),
        "sample_rmspe_deg": score.sample_rmspe_deg,
        "success": score.success,
        "both_angle_errors_within_1_deg": score.both_angle_errors_within_1_deg,
        "estimated_separation_at_least_half_true": (
            score.estimated_separation_at_least_half_true
        ),
        "resolved": score.resolved,
        "failure_reason": score.failure_reason or "",
        "rho": sample.rho,
        "snr_db": sample.snr_db,
        "snapshot_count": sample.snapshot_count,
        "separation_deg": float(sample.angles_deg[1] - sample.angles_deg[0]),
        "runtime_seconds": estimate.runtime_seconds,
    }


def evaluate_samples(
    samples: list[DOASample],
    model: MultiScalePCNSS,
    *,
    split: SplitName,
    device: torch.device,
    selected_best_fbss_scale: int | None = None,
    inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
) -> EvaluationRunResult:
    split_name = SplitName(split)
    if split_name is SplitName.LOCKED_TEST:
        raise PermissionError("locked_test evaluation requires separate approval")
    if split_name not in (SplitName.VALIDATION, SplitName.DEVELOPMENT):
        raise PermissionError("evaluation runner accepts validation/development only")
    if not samples:
        raise ValueError("at least one sample is required")
    pcnss, neural_runtime = _pcnss_estimates(
        samples,
        model.to(device),
        device,
        inference_batch_size,
    )
    estimates_by_algorithm: dict[str, list[DOAEstimate]] = {}
    classical_started = perf_counter()
    for sample in samples:
        covariance = sample_covariance(sample.snapshots)
        sample_estimates = {
            "music": music_scan(covariance, source_count=2),
            "root_music": root_music_raw(covariance, source_count=2),
            "esprit": esprit(covariance, source_count=2),
        }
        sample_estimates.update(
            evaluate_fixed_scale_family(
                sample.snapshots,
                subarray_sizes=(4, 5, 6, 7),
                source_count=2,
            )
        )
        for name, estimate in sample_estimates.items():
            estimates_by_algorithm.setdefault(name, []).append(estimate)
    classical_runtime = perf_counter() - classical_started
    estimates_by_algorithm["pcnss_root_music"] = pcnss

    scores_by_algorithm: dict[str, tuple[SampleScore, ...]] = {}
    predictions = []
    summaries = {}
    for algorithm, estimates in estimates_by_algorithm.items():
        algorithm_scores = []
        for sample, estimate in zip(samples, estimates):
            score = score_doa_sample(
                sample.sample_seed,
                sample.angles_deg,
                estimate.angles_deg,
                estimate_success=estimate.success,
                failure_reason=estimate.failure_reason,
                strata={
                    "separation_deg": float(np.diff(sample.angles_deg)[0]),
                    "snr_db": sample.snr_db,
                    "snapshot_count": sample.snapshot_count,
                    "rho": sample.rho,
                },
            )
            algorithm_scores.append(score)
            predictions.append(_prediction_row(split_name, sample, estimate, score))
        scores_tuple = tuple(algorithm_scores)
        scores_by_algorithm[algorithm] = scores_tuple
        summaries[algorithm] = aggregate_metrics(scores_tuple)

    best_scale = selected_best_fbss_scale
    if split_name is SplitName.VALIDATION:
        best_scale = min(
            (4, 5, 6, 7),
            key=lambda size: (
                summaries[f"fbss_root_music_L{size}"]["failure_aware_rmspe_deg"],
                size,
            ),
        )
    elif best_scale is not None and best_scale not in (4, 5, 6, 7):
        raise ValueError("selected_best_fbss_scale must be one of 4,5,6,7")

    paired = {
        "pcnss_vs_root_music": paired_comparison(
            scores_by_algorithm["root_music"],
            scores_by_algorithm["pcnss_root_music"],
        )
    }
    if best_scale is not None:
        paired[f"pcnss_vs_fbss_L{best_scale}"] = paired_comparison(
            scores_by_algorithm[f"fbss_root_music_L{best_scale}"],
            scores_by_algorithm["pcnss_root_music"],
        )
    near_audit = {
        "separation_bin": NEAR_SEPARATION_BIN,
        "algorithms": {
            algorithm: aggregate_near_separation_audit(scores)
            for algorithm, scores in scores_by_algorithm.items()
        },
    }
    return EvaluationRunResult(
        split=split_name,
        summaries=summaries,
        predictions=tuple(predictions),
        scores_by_algorithm=scores_by_algorithm,
        best_fixed_fbss_scale=best_scale,
        paired_comparisons=paired,
        near_separation_audit=near_audit,
        runtime_seconds={
            "classical_total": classical_runtime,
            "pcnss_neural_total": neural_runtime,
            "pcnss_inference_batch_size": int(inference_batch_size),
            "overall_total": classical_runtime
            + sum(item.runtime_seconds for item in pcnss),
        },
    )
