"""Unit tests for customer-level treatment effect interpreter and estimation layer."""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.generator import generate_synthetic_campaign_data
from src.causal.dml import EconoCausalDML, DEFAULT_CONFOUNDERS
from src.causal.interpreter import (
    CustomerEffectInterpreter,
    DEFAULT_NEAR_ZERO_THRESHOLD,
)


@pytest.fixture(scope="module")
def fitted_dml_model() -> tuple[EconoCausalDML, pd.DataFrame]:
    """Train a small reproducible DML model for fast fixture sharing."""
    df = generate_synthetic_campaign_data(n_samples=1000, random_seed=42)
    train_df, eval_df = train_test_split(df, test_size=0.20, random_state=42)

    model = EconoCausalDML(
        model_type="linear",
        base_estimator="random_forest",
        n_estimators=30,
        max_depth=3,
        cv=3,
        random_state=42,
    )
    model.fit(train_df)
    return model, eval_df


def test_customer_effects_structure(fitted_dml_model):
    """Verify estimate_customer_effects produces required columns, correct lengths, and finite values."""
    model, eval_df = fitted_dml_model
    interpreter = CustomerEffectInterpreter(model)

    results_df = interpreter.estimate_customer_effects(eval_df)

    expected_cols = [
        "customer_id",
        "estimated_effect",
        "estimated_effect_percentage_points",
        "ci_lower",
        "ci_upper",
        "ci_lower_percentage_points",
        "ci_upper_percentage_points",
        "effect_category",
    ]
    for col in expected_cols:
        assert col in results_df.columns, f"Missing required column: {col}"

    assert len(results_df) == len(eval_df)
    assert np.all(np.isfinite(results_df["estimated_effect"]))
    assert np.all(np.isfinite(results_df["ci_lower"]))
    assert np.all(np.isfinite(results_df["ci_upper"]))

    # Verify percentage points match raw * 100
    np.testing.assert_allclose(
        results_df["estimated_effect_percentage_points"],
        results_df["estimated_effect"] * 100.0,
        rtol=1e-3,
    )


def test_row_to_customer_alignment(fitted_dml_model):
    """Verify 1-to-1 customer ID alignment and row-order preservation."""
    model, eval_df = fitted_dml_model
    interpreter = CustomerEffectInterpreter(model)

    results_df = interpreter.estimate_customer_effects(eval_df)
    assert list(results_df["customer_id"]) == list(eval_df["customer_id"])

    # Test reverse order preserves alignment
    reversed_df = eval_df.iloc[::-1].copy()
    reversed_results = interpreter.estimate_customer_effects(reversed_df)
    assert list(reversed_results["customer_id"]) == list(reversed_df["customer_id"])
    np.testing.assert_allclose(
        reversed_results["estimated_effect"].values,
        results_df["estimated_effect"].values[::-1],
    )


def test_customer_id_quarantined_from_model(fitted_dml_model):
    """Verify customer_id is never included in model features and predictions don't depend on it."""
    model, eval_df = fitted_dml_model
    interpreter = CustomerEffectInterpreter(model)

    assert "customer_id" not in interpreter.feature_cols

    # Modify customer_ids completely
    mutated_df = eval_df.copy()
    mutated_df["customer_id"] = [f"NEW_ID_{i}" for i in range(len(eval_df))]

    res_orig = interpreter.estimate_customer_effects(eval_df)
    res_mut = interpreter.estimate_customer_effects(mutated_df)

    # Effects must be bitwise identical because features didn't change
    np.testing.assert_allclose(res_orig["estimated_effect"], res_mut["estimated_effect"])
    assert list(res_mut["customer_id"]) == list(mutated_df["customer_id"])


def test_missing_features_validation(fitted_dml_model):
    """Verify validator flags missing features with informative exception."""
    model, eval_df = fitted_dml_model
    interpreter = CustomerEffectInterpreter(model)

    missing_df = eval_df.drop(columns=["prior_spend"])
    with pytest.raises(ValueError, match="missing required pre-treatment features.*prior_spend"):
        interpreter.estimate_customer_effects(missing_df)


def test_null_features_validation(fitted_dml_model):
    """Verify validator flags null or NaN feature inputs."""
    model, eval_df = fitted_dml_model
    interpreter = CustomerEffectInterpreter(model)

    null_df = eval_df.copy()
    null_df.loc[null_df.index[0], "engagement_score"] = np.nan
    with pytest.raises(ValueError, match="Null values detected.*engagement_score"):
        interpreter.estimate_customer_effects(null_df)


def test_non_finite_features_and_empty_input(fitted_dml_model):
    """Verify interpreter rejects non-finite (inf/-inf) features and empty inputs."""
    model, eval_df = fitted_dml_model
    interpreter = CustomerEffectInterpreter(model)

    # Infinite value
    inf_df = eval_df.copy()
    inf_df.loc[inf_df.index[0], "prior_spend"] = np.inf
    with pytest.raises(ValueError, match="Non-finite.*detected in incoming customer features"):
        interpreter.estimate_customer_effects(inf_df)

    # Empty DataFrame
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Input DataFrame is empty or None"):
        interpreter.estimate_customer_effects(empty_df)



def test_configurable_threshold(fitted_dml_model):
    """Verify configurable threshold correctly shifts effect classifications."""
    model, eval_df = fitted_dml_model

    # Default threshold (0.02)
    interpreter_default = CustomerEffectInterpreter(model, near_zero_threshold=0.02)
    res_default = interpreter_default.estimate_customer_effects(eval_df)
    summary_default = interpreter_default.summarize_effects(res_default)

    # High threshold (e.g. 0.20)
    interpreter_high = CustomerEffectInterpreter(model, near_zero_threshold=0.20)
    res_high = interpreter_high.estimate_customer_effects(eval_df)
    summary_high = interpreter_high.summarize_effects(res_high)

    # Customers with effects <= 0.20 should now be categorized as near_zero
    assert summary_high["category_distribution"]["near_zero"]["count"] > summary_default["category_distribution"]["near_zero"]["count"]

    # Negative threshold should raise error
    with pytest.raises(ValueError, match="near_zero_threshold must be non-negative"):
        CustomerEffectInterpreter(model, near_zero_threshold=-0.05)


def test_export_results_to_csv_and_markdown(fitted_dml_model):
    """Verify export_results writes both CSV table and markdown report cleanly."""
    model, eval_df = fitted_dml_model
    interpreter = CustomerEffectInterpreter(model)
    results_df = interpreter.estimate_customer_effects(eval_df)

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "test_effects.csv")
        summary_path = os.path.join(tmp_dir, "test_summary.md")

        summary = interpreter.export_results(results_df, csv_path, summary_path)

        assert os.path.exists(csv_path)
        assert os.path.exists(summary_path)

        # Verify CSV is loadable and has correct record count
        loaded_csv = pd.read_csv(csv_path)
        assert len(loaded_csv) == len(eval_df)
        assert "customer_id" in loaded_csv.columns
        assert "estimated_effect" in loaded_csv.columns
        assert "effect_category" in loaded_csv.columns

        # Verify category counts sum to total customers
        cat_dist = summary["category_distribution"]
        total_counted = sum(v["count"] for v in cat_dist.values())
        assert total_counted == len(eval_df)
