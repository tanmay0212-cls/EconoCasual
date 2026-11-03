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

# Week 3 additive extension — multi-arm optimization dataset generator.
# Import is isolated so that any missing optional dependency never breaks
# the existing Week 1/2 imports above.
from .optimization_generator import (
    generate_optimization_dataset,
    OPT_REQUIRED_COLUMNS,
    DISCOUNT_LEVELS,
    CUSTOMER_SEGMENTS,
    DGP_PARAMETERS as OPTIMIZATION_DGP_PARAMETERS,
)

__all__ = [
    # ── Week 1/2 (unchanged) ──────────────────────────────────────────────
    "REQUIRED_COLUMNS",
    "TREATMENT_COL",
    "OUTCOME_COL",
    "OBSERVED_CONFOUNDERS",
    "POST_TREATMENT_VARS",
    "validate_dataset",
    "generate_synthetic_campaign_data",
    # ── Week 3 extension ──────────────────────────────────────────────────
    "generate_optimization_dataset",
    "OPT_REQUIRED_COLUMNS",
    "DISCOUNT_LEVELS",
    "CUSTOMER_SEGMENTS",
    "OPTIMIZATION_DGP_PARAMETERS",
]
