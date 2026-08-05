"""Frozen first-round experiment protocol for Multi-Scale PC-NSS."""

from dataclasses import dataclass, field
from enum import Enum


class SplitName(str, Enum):
    """Audited dataset split names."""

    TRAIN = "train"
    VALIDATION = "validation"
    DEVELOPMENT = "development"
    LOCKED_TEST = "locked_test"


@dataclass(frozen=True)
class ArrayConfig:
    sensor_count: int = 8
    spacing_wavelengths: float = 0.5
    angle_limits_deg: tuple[float, float] = (-60.0, 60.0)


@dataclass(frozen=True)
class DataConfig:
    source_count: int = 2
    train_rhos: tuple[float, ...] = (0.8, 0.9, 0.99, 1.0)
    train_snr_db: tuple[float, float] = (-5.0, 10.0)
    train_snapshot_counts: tuple[int, ...] = (8, 20, 50)
    center_limits_deg: tuple[float, float] = (-50.0, 50.0)
    separation_limits_deg: tuple[float, float] = (2.0, 10.0)


@dataclass(frozen=True)
class PhysicsConfig:
    fbss_subarray_sizes: tuple[int, ...] = (4, 5, 6, 7)
    projection_iterations_train: int = 4
    projection_max_iterations_eval: int = 100
    projection_tolerance: float = 1e-7
    eigenvalue_floor: float = 1e-6


@dataclass(frozen=True)
class TrainingConfig:
    stage_one_epochs: int = 10
    total_epochs: int = 50
    learning_rate: float = 1e-3
    batch_size: int = 128
    tau_scale: float = 0.1
    peak_margin: float = 0.05
    residual_fraction: float = 0.10
    loading_fraction: float = 0.05


@dataclass(frozen=True)
class SplitConfig:
    sizes: dict[SplitName, int] = field(
        default_factory=lambda: {
            SplitName.TRAIN: 40_000,
            SplitName.VALIDATION: 5_000,
            SplitName.DEVELOPMENT: 5_000,
            SplitName.LOCKED_TEST: 10_000,
        }
    )
    seeds: dict[SplitName, int] = field(
        default_factory=lambda: {
            SplitName.TRAIN: 202_608_040,
            SplitName.VALIDATION: 202_708_040,
            SplitName.DEVELOPMENT: 202_808_040,
            SplitName.LOCKED_TEST: 202_908_040,
        }
    )
    allow_locked_test: bool = False

    def require_access(self, split: SplitName) -> None:
        if split is SplitName.LOCKED_TEST and not self.allow_locked_test:
            raise PermissionError("locked_test is frozen until explicit approval")


@dataclass(frozen=True)
class RuntimeConfig:
    dry_run: bool = True
    output_root: str = "outputs/multiscale_pcnss"
    refuse_overwrite: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    array: ArrayConfig = field(default_factory=ArrayConfig)
    data: DataConfig = field(default_factory=DataConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
