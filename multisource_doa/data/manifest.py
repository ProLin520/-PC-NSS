"""Auditable split manifests and seed-range checks."""

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from collections.abc import Mapping

from multisource_doa.config import ExperimentConfig, SplitName
from multisource_doa.data.simulator import GENERATOR_VERSION


def split_seed_interval(
    config: ExperimentConfig,
    split: SplitName,
) -> tuple[int, int]:
    split_name = SplitName(split)
    start = int(config.split.seeds[split_name])
    return start, start + int(config.split.sizes[split_name]) - 1


def assert_split_seed_ranges_disjoint(config: ExperimentConfig) -> None:
    intervals = sorted(
        (split_seed_interval(config, split), split)
        for split in SplitName
    )
    for ((_, left_end), left_split), ((right_start, _), right_split) in zip(
        intervals,
        intervals[1:],
    ):
        if left_end >= right_start:
            raise ValueError(
                f"sample seed ranges overlap: {left_split.value} and "
                f"{right_split.value}"
            )


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {
            str(_to_jsonable(key)): _to_jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_to_jsonable(item) for item in value]
    return value


def build_split_manifest(
    config: ExperimentConfig,
    split: SplitName,
) -> dict[str, Any]:
    split_name = SplitName(split)
    config.split.require_access(split_name)
    assert_split_seed_ranges_disjoint(config)
    start, end = split_seed_interval(config, split_name)
    return {
        "split": split_name.value,
        "size": int(config.split.sizes[split_name]),
        "sample_seed_start": start,
        "sample_seed_end": end,
        "generator_version": GENERATOR_VERSION,
        "experiment_config": _to_jsonable(config),
    }


def write_split_manifest(
    path: str | Path,
    config: ExperimentConfig,
    split: SplitName,
    *,
    extra_metadata: Mapping[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_split_manifest(config, split)
    if extra_metadata is not None:
        payload["training_metadata"] = _to_jsonable(dict(extra_metadata))
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination
