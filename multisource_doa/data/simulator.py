"""Deterministic coherent two-source snapshot simulation."""

from dataclasses import dataclass

import numpy as np

from multisource_doa.config import ExperimentConfig


GENERATOR_VERSION = "pcnss-two-source-v1"


@dataclass(frozen=True)
class DOASample:
    snapshots: np.ndarray
    angles_deg: np.ndarray
    rho: float
    snr_db: float
    snapshot_count: int
    sample_seed: int
    source_correlation: complex
    source_powers: np.ndarray
    noise_power: float
    steering_matrix: np.ndarray
    target_covariance: np.ndarray


def complex_normal(
    rng: np.random.Generator,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Return circular complex Gaussian samples with unit total variance."""

    scale = 1.0 / np.sqrt(2.0)
    return scale * (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    )


def steering_vector(
    angles_deg: np.ndarray,
    sensor_count: int = 8,
    spacing_wavelengths: float = 0.5,
) -> np.ndarray:
    """Construct the positive-phase ULA steering matrix."""

    angles = np.atleast_1d(np.asarray(angles_deg, dtype=np.float64))
    sensors = np.arange(sensor_count, dtype=np.float64)[:, None]
    phase = (
        2.0
        * np.pi
        * spacing_wavelengths
        * sensors
        * np.sin(np.deg2rad(angles))[None, :]
    )
    return np.exp(1j * phase)


def _empirical_correlation(left: np.ndarray, right: np.ndarray) -> complex:
    denominator = np.sqrt(np.vdot(left, left).real * np.vdot(right, right).real)
    if denominator <= 0.0:
        return 0.0 + 0.0j
    return complex(np.vdot(left, right) / denominator)


def _select_stratified_value(values: tuple, index: int, stride: int = 1):
    return values[(index // stride) % len(values)]


def generate_two_source_sample(
    config: ExperimentConfig,
    split_seed: int,
    index: int,
    *,
    rho: float | None = None,
    snr_db: float | None = None,
    snapshot_count: int | None = None,
    center_deg: float | None = None,
    separation_deg: float | None = None,
) -> DOASample:
    """Generate one reproducible two-source sample without touching global RNG state."""

    if index < 0:
        raise IndexError("sample index must be non-negative")
    sample_seed = int(split_seed) + int(index)
    rng = np.random.default_rng(sample_seed)

    rho_value = float(
        _select_stratified_value(config.data.train_rhos, index)
        if rho is None
        else rho
    )
    if not 0.0 <= rho_value <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    snapshot_value = int(
        _select_stratified_value(
            config.data.train_snapshot_counts,
            index,
            stride=len(config.data.train_rhos),
        )
        if snapshot_count is None
        else snapshot_count
    )
    if snapshot_value <= 0:
        raise ValueError("snapshot_count must be positive")
    snr_value = float(
        rng.uniform(*config.data.train_snr_db) if snr_db is None else snr_db
    )

    center = float(
        rng.uniform(*config.data.center_limits_deg)
        if center_deg is None
        else center_deg
    )
    separation = float(
        rng.uniform(*config.data.separation_limits_deg)
        if separation_deg is None
        else separation_deg
    )
    if separation <= 0.0:
        raise ValueError("separation_deg must be positive")
    angles = np.sort(
        np.asarray(
            [center - 0.5 * separation, center + 0.5 * separation],
            dtype=np.float64,
        )
    )
    lower, upper = config.array.angle_limits_deg
    if angles[0] < lower or angles[-1] > upper:
        raise ValueError("generated angles exceed the configured array limits")

    steering = steering_vector(
        angles,
        sensor_count=config.array.sensor_count,
        spacing_wavelengths=config.array.spacing_wavelengths,
    )
    first_source = complex_normal(rng, (snapshot_value,))
    independent_source = complex_normal(rng, (snapshot_value,))
    relative_phase = rng.uniform(-np.pi, np.pi)
    second_source = (
        rho_value * np.exp(1j * relative_phase) * first_source
        + np.sqrt(max(0.0, 1.0 - rho_value**2)) * independent_source
    )
    sources = np.stack([first_source, second_source], axis=0)
    clean_snapshots = steering @ sources
    signal_power = float(np.mean(np.abs(clean_snapshots) ** 2))
    noise_power = signal_power / (10.0 ** (snr_value / 10.0))
    noise = np.sqrt(noise_power) * complex_normal(rng, clean_snapshots.shape)
    snapshots = clean_snapshots + noise

    source_powers = np.mean(np.abs(sources) ** 2, axis=1).astype(np.float64)
    target_covariance = (
        steering @ np.diag(source_powers) @ steering.conj().T
        + noise_power * np.eye(config.array.sensor_count, dtype=np.complex128)
    )
    return DOASample(
        snapshots=np.asarray(snapshots, dtype=np.complex128),
        angles_deg=angles,
        rho=rho_value,
        snr_db=snr_value,
        snapshot_count=snapshot_value,
        sample_seed=sample_seed,
        source_correlation=_empirical_correlation(first_source, second_source),
        source_powers=source_powers,
        noise_power=float(noise_power),
        steering_matrix=np.asarray(steering, dtype=np.complex128),
        target_covariance=np.asarray(target_covariance, dtype=np.complex128),
    )
