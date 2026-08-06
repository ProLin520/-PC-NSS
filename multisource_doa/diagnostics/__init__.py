"""Read-only diagnostics for frozen evaluation reports."""

from multisource_doa.diagnostics.reporting import (
    DIAGNOSTIC_SCHEMA_VERSION,
    MECHANISM_METRICS,
    RHO_VALUES,
    SNR_BINS,
    SNAPSHOT_VALUES,
    THRESHOLD_COHORTS,
    build_mechanism_summary,
    build_stratified_summary,
    write_near_diagnostic_report,
)

__all__ = [
    "DIAGNOSTIC_SCHEMA_VERSION",
    "MECHANISM_METRICS",
    "RHO_VALUES",
    "SNR_BINS",
    "SNAPSHOT_VALUES",
    "THRESHOLD_COHORTS",
    "build_mechanism_summary",
    "build_stratified_summary",
    "write_near_diagnostic_report",
]
