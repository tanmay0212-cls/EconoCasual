"""Schema specifications, variable taxonomy, and dataset validation for EconoCausal."""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Canonical variable definitions
TREATMENT_COL: str = "discount_offered"
OUTCOME_COL: str = "purchased"

# Strictly pre-treatment observed confounders (Common Causes)
# These variables temporally precede discount assignment and confound Treatment -> Outcome
OBSERVED_CONFOUNDERS: List[str] = [
    "prior_purchases",
    "prior_spend",
    "engagement_score",
    "customer_tenure",
    "campaign_channel",
]

# Post-treatment downstream variables
# CRITICAL: These must NEVER be used as confounders (avoids post-treatment conditioning / collider/mediator bias)
POST_TREATMENT_VARS: List[str] = [
    "discount_amount",
    "revenue",
]

# Full required schema
REQUIRED_COLUMNS: List[str] = [
    "customer_id",
    "prior_purchases",
    "prior_spend",
    "engagement_score",
    "customer_tenure",
    "discount_offered",
    "discount_amount",
    "purchased",
    "revenue",
    "campaign_channel",
]


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate customer campaign dataset against schema, domain, and causal integrity rules.

    Args:
        df: Input pandas DataFrame.

    Returns:
        Dictionary with validation status, metrics, and any error messages.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Non-empty check
    if df is None or len(df) == 0:
        return {
            "is_valid": False,
            "row_count": 0,
            "errors": ["Dataset is empty or None."],
            "warnings": [],
            "metrics": {},
        }

    row_count = len(df)

    # 2. Minimum row requirement
    if row_count < 1000:
        warnings.append(
            f"Dataset has {row_count} records. Week 1 specifications recommend at least 1,000 records."
        )

    # 3. Missing column check
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # If missing core columns, cannot proceed with domain validation
    if errors:
        return {
            "is_valid": False,
            "row_count": row_count,
            "errors": errors,
            "warnings": warnings,
            "metrics": {},
        }

    # 4. Null value check
    null_counts = df[REQUIRED_COLUMNS].isnull().sum().to_dict()
    total_nulls = sum(null_counts.values())
    if total_nulls > 0:
        cols_with_nulls = {k: v for k, v in null_counts.items() if v > 0}
        errors.append(f"Null values detected in required columns: {cols_with_nulls}")

    # 5. Binary domain checks for treatment and outcome
    for binary_col in [TREATMENT_COL, OUTCOME_COL]:
        unique_vals = set(df[binary_col].dropna().unique())
        if not unique_vals.issubset({0, 1}):
            errors.append(
                f"Column '{binary_col}' must be strictly binary (0 or 1), found values: {unique_vals}"
            )

    # 6. Value domain checks
    if (df["prior_purchases"] < 0).any():
        errors.append("Column 'prior_purchases' contains negative values.")
    if (df["prior_spend"] < 0).any():
        errors.append("Column 'prior_spend' contains negative values.")
    if (df["customer_tenure"] < 0).any():
        errors.append("Column 'customer_tenure' contains negative values.")
    if (df["discount_amount"] < 0).any():
        errors.append("Column 'discount_amount' contains negative values.")
    if (df["revenue"] < 0).any():
        errors.append("Column 'revenue' contains negative values.")

    # 7. Post-treatment logic consistency
    # When discount_offered == 0, discount_amount should be 0
    non_zero_discount_without_offer = (
        (df[TREATMENT_COL] == 0) & (df["discount_amount"] > 0)
    ).sum()
    if non_zero_discount_without_offer > 0:
        errors.append(
            f"Found {non_zero_discount_without_offer} records where discount_offered=0 but discount_amount > 0."
        )

    # When purchased == 0, revenue should be 0
    non_zero_revenue_without_purchase = (
        (df[OUTCOME_COL] == 0) & (df["revenue"] > 0)
    ).sum()
    if non_zero_revenue_without_purchase > 0:
        errors.append(
            f"Found {non_zero_revenue_without_purchase} records where purchased=0 but revenue > 0."
        )

    # 8. Customer ID uniqueness
    duplicate_ids = df["customer_id"].duplicated().sum()
    if duplicate_ids > 0:
        warnings.append(f"Found {duplicate_ids} duplicate customer_id values.")

    treatment_rate = float(df[TREATMENT_COL].mean()) if TREATMENT_COL in df else 0.0
    conversion_rate = float(df[OUTCOME_COL].mean()) if OUTCOME_COL in df else 0.0

    return {
        "is_valid": len(errors) == 0,
        "row_count": row_count,
        "errors": errors,
        "warnings": warnings,
        "metrics": {
            "treatment_rate": round(treatment_rate, 4),
            "conversion_rate": round(conversion_rate, 4),
            "total_revenue": round(float(df["revenue"].sum()), 2),
            "channels": list(df["campaign_channel"].unique()),
        },
    }
