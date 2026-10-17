"""Data generation and schema validation module for EconoCausal."""

from .schema import (
    REQUIRED_COLUMNS,
    TREATMENT_COL,
    OUTCOME_COL,
    OBSERVED_CONFOUNDERS,
    POST_TREATMENT_VARS,
    validate_dataset,
)
from .generator import generate_synthetic_campaign_data

__all__ = [
    "REQUIRED_COLUMNS",
    "TREATMENT_COL",
    "OUTCOME_COL",
    "OBSERVED_CONFOUNDERS",
    "POST_TREATMENT_VARS",
    "validate_dataset",
    "generate_synthetic_campaign_data",
]
