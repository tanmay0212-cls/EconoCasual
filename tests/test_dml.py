"""Unit tests for Double Machine Learning (DML) estimator in EconoCausal."""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.generator import generate_synthetic_campaign_data
from src.causal.dml import (
    EconoCausalDML,
    DEFAULT_CONFOUNDERS,
    PROHIBITED_LEAKAGE_VARS,
    SYNTHETIC_SAMPLE_ATE,
)


@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    """Generate small reproducible dataset for fast unit testing."""
    return generate_synthetic_campaign_data(n_samples=1000, random_seed=42)


def test_dml_fit_and_effect_shapes(sample_dataset):
    """Verify DML fits cleanly and outputs correctly shaped, finite effect estimates."""
    train_df, eval_df = train_test_split(sample_dataset, test_size=0.20, random_state=42)

    model = EconoCausalDML(
        model_type="linear",
        base_estimator="random_forest",
        n_estimators=30,  # Fast tree count for testing
        max_depth=3,
        cv=3,
        random_state=42,
    )
    model.fit(train_df)

    assert model.is_fitted_ is True

    # 1. Individual treatment effects (CATE)
    cate = model.effect(eval_df)
    assert isinstance(cate, np.ndarray)
    assert cate.shape == (len(eval_df),)
    assert np.all(np.isfinite(cate))

    # 2. Average treatment effect (ATE)
    ate = model.ate(eval_df)
    assert isinstance(ate, float)
    assert np.isfinite(ate)

    # 3. Confidence intervals
    ate_lb, ate_ub = model.ate_interval(eval_df, alpha=0.05)
    assert np.isfinite(ate_lb)
    assert np.isfinite(ate_ub)
    assert ate_lb <= ate <= ate_ub

    # 4. Evaluation metrics summary
    metrics = model.evaluate(eval_df)
    assert "ate_estimate" in metrics
    assert "cate_summary" in metrics
    assert metrics["evaluation_sample_size"] == len(eval_df)


def test_anti_leakage_guards(sample_dataset):
    """Verify estimator strictly forbids post-treatment variables and identifiers."""
    train_df = sample_dataset.copy()
    model = EconoCausalDML()

    # Attempting to include discount_amount
    with pytest.raises(ValueError, match="Data leakage detected.*discount_amount"):
        model.fit(train_df, feature_cols=DEFAULT_CONFOUNDERS + ["discount_amount"])

    # Attempting to include revenue
    with pytest.raises(ValueError, match="Data leakage detected.*revenue"):
        model.fit(train_df, feature_cols=DEFAULT_CONFOUNDERS + ["revenue"])

    # Attempting to include customer_id
    with pytest.raises(ValueError, match="Data leakage detected.*customer_id"):
        model.fit(train_df, feature_cols=DEFAULT_CONFOUNDERS + ["customer_id"])

    # Attempting to include outcome in features
    with pytest.raises(ValueError, match="outcome column.*cannot be included"):
        model.fit(train_df, feature_cols=DEFAULT_CONFOUNDERS + ["purchased"])


def test_binary_domain_validation(sample_dataset):
    """Verify validator rejects non-binary treatment or outcome columns."""
    train_df = sample_dataset.copy()
    model = EconoCausalDML()

    # Non-binary treatment
    train_df_bad_t = train_df.copy()
    train_df_bad_t.loc[0, "discount_offered"] = 9
    with pytest.raises(ValueError, match="Treatment column.*must be strictly binary"):
        model.fit(train_df_bad_t)

    # Single-class treatment
    train_df_single_t = train_df.copy()
    train_df_single_t["discount_offered"] = 0
    with pytest.raises(ValueError, match="must contain both 0 and 1"):
        model.fit(train_df_single_t)

    # Non-binary outcome
    train_df_bad_y = train_df.copy()
    train_df_bad_y.loc[0, "purchased"] = 5
    with pytest.raises(ValueError, match="Outcome column.*must be strictly binary"):
        model.fit(train_df_bad_y)


def test_missing_features_validation(sample_dataset):
    """Verify validator flags missing required pre-treatment confounders."""
    train_df = sample_dataset.drop(columns=["engagement_score"])
    model = EconoCausalDML()

    with pytest.raises(ValueError, match="Default confounder features missing"):
        model.fit(train_df)


def test_insufficient_sample_size_validation():
    """Verify validator raises error if sample size is too small for DML cross-fitting."""
    tiny_df = pd.DataFrame(
        {
            "prior_purchases": [1, 2, 3],
            "prior_spend": [10.0, 20.0, 30.0],
            "engagement_score": [50.0, 60.0, 70.0],
            "customer_tenure": [12, 24, 36],
            "campaign_channel": ["email", "push", "sms"],
            "discount_offered": [0, 1, 0],
            "purchased": [0, 1, 0],
        }
    )
    model = EconoCausalDML()
    with pytest.raises(ValueError, match="minimum 100 rows required"):
        model.fit(tiny_df)


def test_model_serialization_and_reload(sample_dataset):
    """Verify trained model and metadata serialize and reload with exact identical predictions."""
    train_df, eval_df = train_test_split(sample_dataset, test_size=0.20, random_state=42)

    model = EconoCausalDML(
        model_type="linear",
        base_estimator="random_forest",
        n_estimators=30,
        max_depth=3,
        cv=3,
        random_state=42,
    )
    model.fit(train_df)

    original_ate = model.ate(eval_df)
    original_cate = model.effect(eval_df)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_file = os.path.join(tmp_dir, "test_dml.joblib")
        meta_file = os.path.join(tmp_dir, "test_dml_meta.json")

        model.save(model_file, meta_file, extra_metadata={"test_tag": "ci_run"})

        assert os.path.exists(model_file)
        assert os.path.exists(meta_file)

        # Reload model
        reloaded_model = EconoCausalDML.load(model_file)
        assert reloaded_model.is_fitted_ is True

        reloaded_ate = reloaded_model.ate(eval_df)
        reloaded_cate = reloaded_model.effect(eval_df)

        assert np.isclose(original_ate, reloaded_ate)
        np.testing.assert_allclose(original_cate, reloaded_cate)


def test_out_of_sample_inference_without_outcomes(sample_dataset):
    """Verify evaluation inference does not require outcome or treatment columns (zero leakage)."""
    train_df, eval_df = train_test_split(sample_dataset, test_size=0.20, random_state=42)

    model = EconoCausalDML(
        model_type="linear",
        base_estimator="random_forest",
        n_estimators=30,
        max_depth=3,
        cv=3,
        random_state=42,
    )
    model.fit(train_df)

    # Strip outcome, treatment, and post-treatment columns completely from eval_df
    features_only_df = eval_df[DEFAULT_CONFOUNDERS].copy()

    # effect() and ate() should compute cleanly without outcome or treatment present
    cate_clean = model.effect(features_only_df)
    ate_clean = model.ate(features_only_df)

    assert len(cate_clean) == len(eval_df)
    assert np.all(np.isfinite(cate_clean))
    assert np.isfinite(ate_clean)


def test_dml_random_seed_reproducibility(sample_dataset):
    """Verify that identical random_state generates bitwise identical ATE and CATE estimates."""
    train_df, eval_df = train_test_split(sample_dataset, test_size=0.20, random_state=42)

    model1 = EconoCausalDML(
        model_type="linear",
        base_estimator="random_forest",
        n_estimators=30,
        max_depth=3,
        cv=3,
        random_state=42,
    )
    model1.fit(train_df)

    model2 = EconoCausalDML(
        model_type="linear",
        base_estimator="random_forest",
        n_estimators=30,
        max_depth=3,
        cv=3,
        random_state=42,
    )
    model2.fit(train_df)

    cate1 = model1.effect(eval_df)
    cate2 = model2.effect(eval_df)
    ate1 = model1.ate(eval_df)
    ate2 = model2.ate(eval_df)

    np.testing.assert_allclose(cate1, cate2, err_msg="CATE estimates must be reproducible with same seed")
    assert np.isclose(ate1, ate2), "ATE estimates must be reproducible with same seed"


def test_non_finite_features_validation(sample_dataset):
    """Verify estimator rejects non-finite (inf / -inf / NaN) feature values."""
    train_df, eval_df = train_test_split(sample_dataset, test_size=0.20, random_state=42)
    model = EconoCausalDML(
        model_type="linear",
        base_estimator="random_forest",
        n_estimators=30,
        max_depth=3,
        cv=3,
        random_state=42,
    )

    # 1. NaN in training features
    train_nan = train_df.copy()
    train_nan.loc[train_nan.index[0], "engagement_score"] = np.nan
    with pytest.raises(ValueError, match="Null values detected in training columns"):
        model.fit(train_nan)

    # 2. Inf in training features
    train_inf = train_df.copy()
    train_inf.loc[train_inf.index[0], "prior_spend"] = np.inf
    with pytest.raises(ValueError, match="Non-finite.*detected in training columns"):
        model.fit(train_inf)

    # Fit clean model for inference check
    model.fit(train_df)

    # 3. Inf in inference features
    eval_inf = eval_df.copy()
    eval_inf.loc[eval_inf.index[0], "engagement_score"] = np.inf
    with pytest.raises(ValueError, match="Non-finite.*detected in inference features"):
        model.effect(eval_inf)

    # 4. NaN in inference features
    eval_nan = eval_df.copy()
    eval_nan.loc[eval_nan.index[0], "prior_purchases"] = np.nan
    with pytest.raises(ValueError, match="Null values detected in inference features"):
        model.effect(eval_nan)


def test_empty_dataframe_validation(sample_dataset):
    """Verify estimator rejects empty DataFrames at fit and inference time."""
    model = EconoCausalDML()
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="Input DataFrame is empty or None"):
        model.fit(empty_df)

    # Fit on valid data
    train_df = sample_dataset.copy()
    model.fit(train_df)

    with pytest.raises(ValueError, match="Input DataFrame is empty or None"):
        model.effect(empty_df)

