"""Hermitian Toeplitz, PSD and trace-constrained covariance projection."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class StructureErrors:
    hermitian_error: float
    toeplitz_error: float
    trace_error: float
    min_eigenvalue: float


@dataclass(frozen=True)
class ProjectionResult:
    matrix: np.ndarray
    converged: bool
    iterations: int
    hermitian_error: float
    toeplitz_error: float
    trace_error: float
    min_eigenvalue: float


def _require_square(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.complex128)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be square")
    if not np.isfinite(array).all():
        raise ValueError("matrix must be finite")
    return array


def project_hermitian(matrix: np.ndarray) -> np.ndarray:
    array = _require_square(matrix)
    return 0.5 * (array + array.conj().T)


def project_toeplitz(matrix: np.ndarray) -> np.ndarray:
    """Orthogonally project a matrix onto Hermitian Toeplitz structure."""

    hermitian = project_hermitian(matrix)
    size = hermitian.shape[0]
    projected = np.zeros_like(hermitian)
    for lag in range(size):
        value = np.diag(hermitian, k=-lag).mean()
        if lag == 0:
            value = complex(value.real, 0.0)
        rows = np.arange(lag, size)
        columns = rows - lag
        projected[rows, columns] = value
        projected[columns, rows] = np.conj(value)
    return projected


def _project_vector_to_simplex(values: np.ndarray, total: float) -> np.ndarray:
    if total < 0.0:
        raise ValueError("simplex total must be non-negative")
    if total == 0.0:
        return np.zeros_like(values)
    sorted_values = np.sort(values)[::-1]
    cumulative = np.cumsum(sorted_values) - total
    indices = np.arange(1, values.size + 1)
    active = sorted_values - cumulative / indices > 0.0
    rho = np.nonzero(active)[0][-1]
    threshold = cumulative[rho] / float(rho + 1)
    return np.maximum(values - threshold, 0.0)


def project_psd_trace(
    matrix: np.ndarray,
    target_trace: float,
    eigenvalue_floor: float,
) -> np.ndarray:
    """Project onto Hermitian matrices with fixed trace and eigenvalue floor."""

    hermitian = project_hermitian(matrix)
    size = hermitian.shape[0]
    minimum_trace = size * eigenvalue_floor
    if target_trace < minimum_trace:
        raise ValueError("target_trace is incompatible with eigenvalue_floor")
    values, vectors = np.linalg.eigh(hermitian)
    budget = target_trace - minimum_trace
    shifted = _project_vector_to_simplex(values - eigenvalue_floor, budget)
    projected_values = shifted + eigenvalue_floor
    projected = (vectors * projected_values[None, :]) @ vectors.conj().T
    return project_hermitian(projected)


def structure_errors(
    matrix: np.ndarray,
    target_trace: float,
) -> StructureErrors:
    array = _require_square(matrix)
    scale = max(float(np.linalg.norm(array, ord="fro")), 1e-12)
    hermitian_error = float(
        np.linalg.norm(array - array.conj().T, ord="fro") / scale
    )
    toeplitz_error = float(
        np.linalg.norm(array - project_toeplitz(array), ord="fro") / scale
    )
    trace_error = float(
        abs(np.trace(array).real - target_trace) / max(abs(target_trace), 1e-12)
    )
    min_eigenvalue = float(np.linalg.eigvalsh(project_hermitian(array)).min())
    return StructureErrors(
        hermitian_error=hermitian_error,
        toeplitz_error=toeplitz_error,
        trace_error=trace_error,
        min_eigenvalue=min_eigenvalue,
    )


def dykstra_structured_projection(
    matrix: np.ndarray,
    target_trace: float = 8.0,
    tolerance: float = 1e-7,
    max_iterations: int = 100,
    eigenvalue_floor: float = 1e-6,
) -> ProjectionResult:
    """Project onto the intersection of Hermitian Toeplitz and PSD-trace sets."""

    if tolerance <= 0.0 or max_iterations <= 0:
        raise ValueError("tolerance and max_iterations must be positive")
    current = _require_square(matrix).copy()
    toeplitz_correction = np.zeros_like(current)
    psd_correction = np.zeros_like(current)
    converged = False
    completed_iterations = 0
    for iteration in range(1, max_iterations + 1):
        toeplitz_input = current + toeplitz_correction
        toeplitz_value = project_toeplitz(toeplitz_input)
        toeplitz_correction = toeplitz_input - toeplitz_value

        psd_input = toeplitz_value + psd_correction
        updated = project_psd_trace(
            psd_input,
            target_trace=target_trace,
            eigenvalue_floor=eigenvalue_floor,
        )
        psd_correction = psd_input - updated
        relative_change = np.linalg.norm(updated - current, ord="fro") / max(
            np.linalg.norm(current, ord="fro"),
            1e-12,
        )
        current = updated
        completed_iterations = iteration
        errors = structure_errors(current, target_trace)
        if (
            relative_change <= tolerance
            and errors.hermitian_error <= tolerance
            and errors.toeplitz_error <= tolerance
            and errors.trace_error <= tolerance
            and errors.min_eigenvalue >= eigenvalue_floor - tolerance
        ):
            converged = True
            break
    errors = structure_errors(current, target_trace)
    return ProjectionResult(
        matrix=current,
        converged=converged,
        iterations=completed_iterations,
        hermitian_error=errors.hermitian_error,
        toeplitz_error=errors.toeplitz_error,
        trace_error=errors.trace_error,
        min_eigenvalue=errors.min_eigenvalue,
    )


def hermitian_toeplitz_projection_torch(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("matrix must have square trailing dimensions")
    hermitian = 0.5 * (matrix + matrix.mH)
    size = matrix.shape[-1]
    projected = torch.zeros_like(hermitian)
    for lag in range(size):
        value = hermitian.diagonal(
            offset=-lag,
            dim1=-2,
            dim2=-1,
        ).mean(dim=-1)
        if lag == 0:
            value = torch.complex(value.real, torch.zeros_like(value.real))
        rows = torch.arange(lag, size, device=matrix.device)
        columns = rows - lag
        projected[..., rows, columns] = value.unsqueeze(-1)
        projected[..., columns, rows] = value.conj().unsqueeze(-1)
    return projected


def _psd_trace_projection_torch(
    matrix: torch.Tensor,
    target_trace: float,
    eigenvalue_floor: float,
) -> torch.Tensor:
    hermitian = 0.5 * (matrix + matrix.mH)
    values, vectors = torch.linalg.eigh(hermitian)
    size = matrix.shape[-1]
    budget = target_trace - size * eigenvalue_floor
    if budget < 0.0:
        raise ValueError("target_trace is incompatible with eigenvalue_floor")
    excess = torch.relu(values - eigenvalue_floor)
    excess_sum = excess.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(excess, budget / size)
    scaled = excess / excess_sum.clamp_min(1e-12) * budget
    allocated = torch.where(excess_sum > 1e-12, scaled, uniform)
    projected_values = allocated + eigenvalue_floor
    return (
        vectors
        @ torch.diag_embed(projected_values).to(matrix.dtype)
        @ vectors.mH
    )


def _repair_toeplitz_psd_trace_torch(
    matrix: torch.Tensor,
    target_trace: float,
    eigenvalue_floor: float,
) -> torch.Tensor:
    """Add the minimum scalar diagonal shift that survives trace normalization."""

    toeplitz = hermitian_toeplitz_projection_torch(matrix)
    size = toeplitz.shape[-1]
    denominator = target_trace - size * eigenvalue_floor
    if denominator <= 0.0:
        raise ValueError("target_trace is incompatible with eigenvalue_floor")
    trace = toeplitz.diagonal(dim1=-2, dim2=-1).real.sum(dim=-1)
    minimum = torch.linalg.eigvalsh(toeplitz).amin(dim=-1)
    shift_for_floor = eigenvalue_floor - minimum
    shift_after_scaling = (
        eigenvalue_floor * trace - target_trace * minimum
    ) / denominator
    shift = torch.maximum(shift_for_floor, shift_after_scaling).clamp_min(0.0)
    identity = torch.eye(
        size,
        dtype=toeplitz.dtype,
        device=toeplitz.device,
    )
    shifted = toeplitz + shift[..., None, None] * identity
    shifted_trace = shifted.diagonal(dim1=-2, dim2=-1).real.sum(dim=-1)
    scale = target_trace / shifted_trace.clamp_min(1e-12)
    return shifted * scale[..., None, None]


def structured_projection_torch(
    covariance: torch.Tensor,
    target_trace: float = 8.0,
    iterations: int = 4,
    eigenvalue_floor: float = 1e-6,
) -> torch.Tensor:
    """Fixed-iteration differentiable approximation used during training."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    projected = covariance
    for _ in range(iterations):
        projected = hermitian_toeplitz_projection_torch(projected)
        projected = _psd_trace_projection_torch(
            projected,
            target_trace=target_trace,
            eigenvalue_floor=eigenvalue_floor,
        )
    return _repair_toeplitz_psd_trace_torch(
        projected,
        target_trace=target_trace,
        eigenvalue_floor=eigenvalue_floor,
    )
