"""Unit tests for synthetic data generation and schema validation."""

import pytest
import pandas as pd
import numpy as np

from src.data.generator import generate_synthetic_campaign_data
from src.data.schema import (
    REQUIRED_COLUMNS,
    TREATMENT_COL,
    OUTCOME_COL,
    OBSERVED_CONFOUNDERS,
    POST_TREATMENT_VARS,
    validate_dataset,
)


def test_reproducible_generation():
    """Verify that using the same random seed yields identical data."""
    df1 = generate_synthetic_campaign_data(n_samples=1000, random_seed=42)
    df2 = generate_synthetic_campaign_data(n_samples=1000, random_seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_minimum_sample_size_requirement():
    """Verify dataset meets the Week 1 requirement of at least 1,000 records."""
    df = generate_synthetic_campaign_data(n_samples=1200, random_seed=123)
    assert len(df) == 1200
    assert len(df) >= 1000

    # Ensure generator raises error if requested samples < 1000
    with pytest.raises(ValueError, match="at least 1,000"):
        generate_synthetic_campaign_data(n_samples=500)


def test_required_columns_present():
    """Verify all 10 canonical columns are present in output."""
    df = generate_synthetic_campaign_data(n_samples=1000, random_seed=42)
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Required column '{col}' missing from generated DataFrame."


def test_no_null_values():
    """Verify generated dataset contains zero null or NaN values."""
    df = generate_synthetic_campaign_data(n_samples=1000, random_seed=42)
    null_count = df.isnull().sum().sum()
    assert null_count == 0, f"Found {null_count} null values in generated dataset."


def test_binary_variable_domains():
    """Verify treatment and outcome variables are strictly binary (0 or 1)."""
    df = generate_synthetic_campaign_data(n_samples=1000, random_seed=42)

    assert set(df[TREATMENT_COL].unique()).issubset({0, 1})
    assert set(df[OUTCOME_COL].unique()).issubset({0, 1})

    # Both classes should be present
    assert len(df[TREATMENT_COL].unique()) == 2
    assert len(df[OUTCOME_COL].unique()) == 2


def test_post_treatment_logic_consistency():
    """Verify downstream variables adhere to causal ordering and business logic."""
    df = generate_synthetic_campaign_data(n_samples=1000, random_seed=42)

    # 1. When no discount offered, discount_amount must be 0
    no_discount_mask = df[TREATMENT_COL] == 0
    assert (df.loc[no_discount_mask, "discount_amount"] == 0.0).all()

    # 2. When discount offered, discount_amount must be positive
    discount_mask = df[TREATMENT_COL] == 1
    assert (df.loc[discount_mask, "discount_amount"] > 0.0).all()

    # 3. When no purchase made, revenue must be 0
    no_purchase_mask = df[OUTCOME_COL] == 0
    assert (df.loc[no_purchase_mask, "revenue"] == 0.0).all()

    # 4. When purchase made, revenue must be positive
    purchase_mask = df[OUTCOME_COL] == 1
    assert (df.loc[purchase_mask, "revenue"] > 0.0).all()


def test_schema_validator_catches_errors():
    """Verify schema validator detects missing columns, bad domains, and nulls."""
    df_valid = generate_synthetic_campaign_data(n_samples=1000, random_seed=42)
    report_valid = validate_dataset(df_valid)
    assert report_valid["is_valid"] is True
    assert len(report_valid["errors"]) == 0

    # Test missing column detection
    df_missing = df_valid.drop(columns=["engagement_score"])
    report_missing = validate_dataset(df_missing)
    assert report_missing["is_valid"] is False
    assert any("engagement_score" in err for err in report_missing["errors"])

    # Test non-binary treatment detection
    df_bad_treatment = df_valid.copy()
    df_bad_treatment.loc[0, TREATMENT_COL] = 5
    report_bad_treatment = validate_dataset(df_bad_treatment)
    assert report_bad_treatment["is_valid"] is False
    assert any("strictly binary" in err for err in report_bad_treatment["errors"])

    # Test negative spend detection
    df_neg = df_valid.copy()
    df_neg.loc[0, "prior_spend"] = -50.0
    report_neg = validate_dataset(df_neg)
    assert report_neg["is_valid"] is False
    assert any("negative values" in err for err in report_neg["errors"])
