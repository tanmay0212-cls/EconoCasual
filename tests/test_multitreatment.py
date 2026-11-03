"""Unit tests for Multi-Treatment Causal Estimator in EconoCausal Week 3."""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd

from src.causal.multi_tier_dml import (
    MultiTierCausalDML,
    DEFAULT_PRE_TREATMENT_CONFOUNDERS,
    PROHIBITED_LEAKAGE_COLUMNS,
    MULTI_TREATMENT_LEVELS,
)


@pytest.fixture
def synthetic_sample_data() -> pd.DataFrame:
    """Generate small synthetic dataset for fast deterministic unit testing."""
    rng = np.random.default_rng(seed=42)
    n = 600
    age = rng.integers(18, 70, size=n)
    loyalty = np.round(rng.uniform(5.0, 95.0, size=n), 1)
    prev_purch = rng.poisson(lam=4.0, size=n)
    segments = rng.choice(["budget_seeker", "casual_shopper", "regular_buyer", "vip_loyalist"], size=n)
    channels = rng.choice(["email", "push", "sms", "social"], size=n)
    treatments = rng.choice([0, 10, 20], size=n, p=[0.33, 0.34, 0.33])

    # True outcome logic
    logit = -0.5 + 0.02 * (age - 40) + 0.01 * loyalty + (0.5 * (treatments == 10)) + (0.8 * (treatments == 20))
    prob = 1.0 / (1.0 + np.exp(-logit))
    purchase = rng.binomial(n=1, p=prob)

    cust_ids = [f"TEST_{i+1:05d}" for i in range(n)]

    return pd.DataFrame({
        "customer_id": cust_ids,
        "age": age,
        "loyalty_score": loyalty,
        "previous_purchases": prev_purch,
        "customer_segment": segments,
        "campaign_channel": channels,
        "discount_level": treatments,
        "purchase": purchase,
    })


def test_multitreatment_fit_and_shapes(synthetic_sample_data):
    """Verify MultiTierCausalDML fits and returns correctly shaped finite estimates."""
    train_df = synthetic_sample_data.iloc[:450].copy()
    eval_df = synthetic_sample_data.iloc[450:].copy()

    model = MultiTierCausalDML(
        n_estimators_nuisance=20,
        max_depth_nuisance=3,
        min_samples_leaf=5,
        cv=3,
        random_state=42,
    )
    model.fit(train_df)
    assert model.is_fitted_ is True

    # 1. CATE for Tier 10
    cate_10 = model.effect(eval_df, T0=0, T1=10)
    assert isinstance(cate_10, np.ndarray)
    assert cate_10.shape == (len(eval_df),)
    assert np.all(np.isfinite(cate_10))

    # 2. CATE for Tier 20
    cate_20 = model.effect(eval_df, T0=0, T1=20)
    assert isinstance(cate_20, np.ndarray)
    assert cate_20.shape == (len(eval_df),)
    assert np.all(np.isfinite(cate_20))

    # 3. ATE and ATE interval
    ate_10 = model.ate(eval_df, T0=0, T1=10)
    assert isinstance(ate_10, float)
    assert np.isfinite(ate_10)

    ate_10_lb, ate_10_ub = model.ate_interval(eval_df, T0=0, T1=10, alpha=0.05)
    assert np.isfinite(ate_10_lb) and np.isfinite(ate_10_ub)
    assert ate_10_lb <= ate_10 <= ate_10_ub

    # 4. Effect intervals
    eff_lb, eff_ub = model.effect_interval(eval_df, T0=0, T1=10, alpha=0.05)
    assert eff_lb.shape == (len(eval_df),)
    assert eff_ub.shape == (len(eval_df),)
    assert np.all(eff_lb <= eff_ub)


def test_predicted_probabilities_bounds(synthetic_sample_data):
    """Verify counterfactual probabilities are strictly bounded in [0, 1]."""
    train_df = synthetic_sample_data.iloc[:450].copy()
    eval_df = synthetic_sample_data.iloc[450:].copy()

    model = MultiTierCausalDML(
        n_estimators_nuisance=20,
        max_depth_nuisance=3,
        cv=3,
        random_state=42,
    )
    model.fit(train_df)

    probs_df = model.predict_probabilities(eval_df)
    assert "p_purchase_t0" in probs_df.columns
    assert "p_purchase_t10" in probs_df.columns
    assert "p_purchase_t20" in probs_df.columns
    assert "customer_id" in probs_df.columns
    assert len(probs_df) == len(eval_df)

    for col in ["p_purchase_t0", "p_purchase_t10", "p_purchase_t20"]:
        assert (probs_df[col] >= 0.0).all()
        assert (probs_df[col] <= 1.0).all()
        assert np.all(np.isfinite(probs_df[col]))


def test_model_serialization_and_reload(synthetic_sample_data):
    """Verify model can be serialized and reloaded with bit-exact inference predictions."""
    train_df = synthetic_sample_data.iloc[:450].copy()
    eval_df = synthetic_sample_data.iloc[450:].copy()

    model = MultiTierCausalDML(
        n_estimators_nuisance=20,
        max_depth_nuisance=3,
        cv=3,
        random_state=42,
    )
    model.fit(train_df)

    orig_cate10 = model.effect(eval_df, T0=0, T1=10)
    orig_probs = model.predict_probabilities(eval_df)

    with tempfile.TemporaryDirectory() as tmpdir:
        mod_path = os.path.join(tmpdir, "test_multi_model.joblib")
        meta_path = os.path.join(tmpdir, "test_multi_meta.json")
        model.save(mod_path, meta_path)

        assert os.path.exists(mod_path)
        assert os.path.exists(meta_path)

        reloaded = MultiTierCausalDML.load(mod_path)
        assert reloaded.is_fitted_ is True

        reloaded_cate10 = reloaded.effect(eval_df, T0=0, T1=10)
        reloaded_probs = reloaded.predict_probabilities(eval_df)

        np.testing.assert_allclose(orig_cate10, reloaded_cate10, rtol=1e-5)
        np.testing.assert_allclose(orig_probs["p_purchase_t10"], reloaded_probs["p_purchase_t10"], rtol=1e-5)


def test_anti_leakage_rejection(synthetic_sample_data):
    """Verify anti-leakage guards reject post-treatment or ground truth features."""
    train_df = synthetic_sample_data.copy()
    train_df["gross_revenue"] = 50.0
    train_df["tau_10_true"] = 0.15

    model = MultiTierCausalDML(n_estimators_nuisance=10, cv=2, random_state=42)

    with pytest.raises(ValueError, match="Data leakage detected"):
        model.fit(train_df, feature_cols=DEFAULT_PRE_TREATMENT_CONFOUNDERS + ["gross_revenue"])

    with pytest.raises(ValueError, match="Data leakage detected"):
        model.fit(train_df, feature_cols=DEFAULT_PRE_TREATMENT_CONFOUNDERS + ["tau_10_true"])

    with pytest.raises(ValueError, match="Data leakage"):
        model.fit(train_df, feature_cols=DEFAULT_PRE_TREATMENT_CONFOUNDERS + ["purchase"])


def test_missing_and_non_finite_features(synthetic_sample_data):
    """Verify validator catches missing columns and non-finite values."""
    train_df = synthetic_sample_data.iloc[:450].copy()
    model = MultiTierCausalDML(n_estimators_nuisance=10, cv=2, random_state=42)
    model.fit(train_df)

    eval_df = synthetic_sample_data.iloc[450:].copy()

    # Missing column
    corrupted_df = eval_df.drop(columns=["age"])
    with pytest.raises(ValueError, match="missing required features"):
        model.effect(corrupted_df, T0=0, T1=10)

    # Inf value
    inf_df = eval_df.copy()
    inf_df.iloc[0, inf_df.columns.get_loc("loyalty_score")] = np.inf
    with pytest.raises(ValueError, match="Non-finite"):
        model.effect(inf_df, T0=0, T1=10)


def test_unsupported_treatment_levels(synthetic_sample_data):
    """Verify validator rejects invalid or corrupted treatment values."""
    corrupted_df = synthetic_sample_data.copy()
    corrupted_df.loc[0, "discount_level"] = 99

    model = MultiTierCausalDML(n_estimators_nuisance=10, cv=2, random_state=42)
    with pytest.raises(ValueError, match="invalid levels"):
        model.fit(corrupted_df)

    # Valid fit, invalid query
    clean_df = synthetic_sample_data.copy()
    model.fit(clean_df)
    with pytest.raises(ValueError, match="T0 and T1 must be in"):
        model.effect(clean_df, T0=0, T1=99)


def test_seed_reproducibility(synthetic_sample_data):
    """Verify identical random seeds produce identical CATE estimates."""
    df = synthetic_sample_data.copy()
    m1 = MultiTierCausalDML(n_estimators_nuisance=20, max_depth_nuisance=3, cv=3, random_state=42)
    m1.fit(df)
    eff1 = m1.effect(df, T0=0, T1=10)

    m2 = MultiTierCausalDML(n_estimators_nuisance=20, max_depth_nuisance=3, cv=3, random_state=42)
    m2.fit(df)
    eff2 = m2.effect(df, T0=0, T1=10)

    np.testing.assert_allclose(eff1, eff2, rtol=1e-5)


def test_overlap_diagnostics(synthetic_sample_data):
    """Verify propensity check returns expected metrics and positivity verdict."""
    df = synthetic_sample_data.copy()
    model = MultiTierCausalDML(n_estimators_nuisance=20, max_depth_nuisance=3, cv=3, random_state=42)
    model.fit(df)

    overlap = model.check_overlap(df, propensity_threshold=0.05)
    assert "positivity_verdict" in overlap
    assert "min_propensity_observed" in overlap
    assert overlap["min_propensity_observed"] > 0.0
    assert "T=0" in overlap["propensity_distribution"]
    assert "T=10" in overlap["propensity_distribution"]
    assert "T=20" in overlap["propensity_distribution"]

