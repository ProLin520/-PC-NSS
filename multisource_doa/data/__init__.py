"""Deterministic data generation and split auditing."""

from .dataset import PCNSSDataset
from .simulator import DOASample, generate_two_source_sample, steering_vector

__all__ = [
    "DOASample",
    "PCNSSDataset",
    "generate_two_source_sample",
    "steering_vector",
]
