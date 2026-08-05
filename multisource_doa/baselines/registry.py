"""Names and integration status for first-stage and publication baselines."""

from dataclasses import dataclass
from enum import Enum


class ExternalBaselineStatus(str, Enum):
    NOT_INTEGRATED = "not_integrated"
    AVAILABLE = "available"
    FAILED_REPRODUCTION = "failed_reproduction"


@dataclass(frozen=True)
class BaselineRegistration:
    name: str
    category: str
    status: ExternalBaselineStatus
    note: str


def build_baseline_registry() -> dict[str, BaselineRegistration]:
    registry: dict[str, BaselineRegistration] = {}
    classical_names = ["music", "root_music", "esprit", "pcnss_root_music"]
    classical_names.extend(
        f"{prefix}_root_music_L{size}"
        for prefix in ("sps", "fbss")
        for size in (4, 5, 6, 7)
    )
    for name in classical_names:
        category = "pcnss" if name == "pcnss_root_music" else "first_stage"
        registry[name] = BaselineRegistration(
            name=name,
            category=category,
            status=ExternalBaselineStatus.AVAILABLE,
            note="implemented under the locked local protocol",
        )
    for name in ("subspacenet", "da_music", "deepmusic"):
        registry[name] = BaselineRegistration(
            name=name,
            category="publication_comparison",
            status=ExternalBaselineStatus.NOT_INTEGRATED,
            note="reserved for author-code reproduction after the foundation stage",
        )
    return registry
