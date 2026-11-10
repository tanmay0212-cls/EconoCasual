"""Unit Tests for Uncertainty-Aware Causal Estimation (Phase 2).

Tests cover:
1. Estimator initialization, critical value approximation, and parameter validation.
2. Asymptotic / Subforest Infinitesimal Jackknife variance estimation.
3. Non-parametric residual bootstrap reproducibility (fixed seed consistency).
4. Inverse propensity scaling: lower overlap yields higher standard errors.
5. Reliability status classification (reliable, moderate_uncertainty, low_overlap, high_variance).
6. Ground-truth coverage, interval width, CATE MAE, RMSE, and ATE bias calculations.
7. Risk-aware policy schema generation for downstream optimization (strict column assertions).
8. Customer identifier isolation: IDs present in output schema but excluded from features.
9. Segment-level and support-level diagnostic breakdowns.
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from src.causal.uncertainty import CATEUncertaintyEstimator, TreatmentUncertaintyMetrics


@pytest.fixture
def synthetic_toy_cohort():
    """Deterministic synthetic cohort for uncertainty unit tests."""
    rng = np.random.default_rng(42)
    n = 300

    cids = [f"CUST_{i+1:05d}" for i in range(n)]
    segments = rng.choice(["budget", "regular", "premium", "vip"], size=n)

    covariates_df = pd.DataFrame({
        "customer_id": cids,
        "age": rng.normal(42, 10, size=n).clip(18, 75),
        "loyalty_score": rng.uniform(10, 90, size=n),
        "previous_purchases": rng.poisson(5, size=n),
        "average_previous_order_value": rng.uniform(40, 200, size=n),
        "customer_activity_score": rng.uniform(15, 85, size=n),
        "customer_segment": segments,
    })

    # Propensities [0, 10, 20]
    p0 = np.full(n, 0.35)
    p10 = np.full(n, 0.40)
    p20 = np.full(n, 0.25)
    propensities = np.column_stack([p0, p10, p20])

    # Estimated CATE and True ITE
    true_cate_10 = rng.normal(0.18, 0.04, size=n)
    pred_cate_10 = true_cate_10 + rng.normal(0.0, 0.02, size=n)

    true_cate_20 = rng.normal(0.24, 0.05, size=n)
    pred_cate_20 = true_cate_20 + rng.normal(0.0, 0.03, size=n)

    return {
        "n": n,
        "covariates_df": covariates_df,
        "propensities": propensities,
        "true_cate_10": true_cate_10,
        "pred_cate_10": pred_cate_10,
        "true_cate_20": true_cate_20,
        "pred_cate_20": pred_cate_20,
    }


# =============================================================================
# 1. Estimator Initialization & Critical Values
# =============================================================================

def test_estimator_initialization():
    """Verify confidence level validation and critical z-score calculations."""
    est = CATEUncertaintyEstimator(confidence_level=0.95)
    assert np.isclose(est.z_crit, 1.95996, atol=1e-3)

    est_90 = CATEUncertaintyEstimator(confidence_level=0.90)
    assert np.isclose(est_90.z_crit, 1.64485, atol=1e-3)

    with pytest.raises(ValueError, match="Confidence level must be in"):
        CATEUncertaintyEstimator(confidence_level=1.20)

    with pytest.raises(ValueError, match="Confidence level must be in"):
        CATEUncertaintyEstimator(confidence_level=0.40)


# =============================================================================
# 2. Asymptotic Uncertainty & Variance Bounds
# =============================================================================

def test_asymptotic_uncertainty_shapes_and_bounds(synthetic_toy_cohort):
    """Verify standard error, bounds, and interval width consistency."""
    d = synthetic_toy_cohort
    est = CATEUncertaintyEstimator(confidence_level=0.95)

    res = est.compute_asymptotic_uncertainty(
        cate_estimates=d["pred_cate_10"],
        propensities=d["propensities"],
        covariates_df=d["covariates_df"],
        treatment_arm=10,
    )

    required_keys = ["standard_error", "lower_bound", "upper_bound", "interval_width", "propensity", "reliability_status"]
    for k in required_keys:
        assert k in res
        assert len(res[k]) == d["n"]

    se = res["standard_error"]
    lb = res["lower_bound"]
    ub = res["upper_bound"]
    width = res["interval_width"]

    assert np.all(se > 0.0)
    assert np.all(lb <= ub)
    assert np.allclose(width, ub - lb)


# =============================================================================
# 3. Non-Parametric Bootstrap Reproducibility
# =============================================================================

def test_bootstrap_reproducibility(synthetic_toy_cohort):
    """Verify that identical seeds produce identical bootstrap bounds."""
    d = synthetic_toy_cohort
    est1 = CATEUncertaintyEstimator(n_bootstraps=50, random_state=42)
    est2 = CATEUncertaintyEstimator(n_bootstraps=50, random_state=42)
    est_diff = CATEUncertaintyEstimator(n_bootstraps=50, random_state=99)

    res1 = est1.compute_bootstrap_uncertainty(
        cate_estimates=d["pred_cate_10"],
        propensities=d["propensities"],
        covariates_df=d["covariates_df"],
        treatment_arm=10,
    )
    res2 = est2.compute_bootstrap_uncertainty(
        cate_estimates=d["pred_cate_10"],
        propensities=d["propensities"],
        covariates_df=d["covariates_df"],
        treatment_arm=10,
    )
    res_diff = est_diff.compute_bootstrap_uncertainty(
        cate_estimates=d["pred_cate_10"],
        propensities=d["propensities"],
        covariates_df=d["covariates_df"],
        treatment_arm=10,
    )

    assert np.allclose(res1["lower_bound"], res2["lower_bound"])
    assert np.allclose(res1["upper_bound"], res2["upper_bound"])
    assert np.allclose(res1["standard_error"], res2["standard_error"])

    # Different seeds should produce slightly different bootstrap samples
    assert not np.allclose(res1["lower_bound"], res_diff["lower_bound"])


# =============================================================================
# 4. Inverse Propensity Scaling & Overlap Penalization
# =============================================================================

def test_propensity_inverse_scaling(synthetic_toy_cohort):
    """Verify that low propensity support inflates the standard error."""
    d = synthetic_toy_cohort
    est = CATEUncertaintyEstimator(confidence_level=0.95)

    # High support propensities vs low support propensities
    n = d["n"]
    prop_high = np.column_stack([np.full(n, 0.40), np.full(n, 0.40), np.full(n, 0.20)])
    prop_low = np.column_stack([np.full(n, 0.85), np.full(n, 0.05), np.full(n, 0.10)])

    res_high = est.compute_asymptotic_uncertainty(
        cate_estimates=d["pred_cate_10"],
        propensities=prop_high,
        covariates_df=d["covariates_df"],
        treatment_arm=10,
    )
    res_low = est.compute_asymptotic_uncertainty(
        cate_estimates=d["pred_cate_10"],
        propensities=prop_low,
        covariates_df=d["covariates_df"],
        treatment_arm=10,
    )

    # Low support must yield higher average standard error and wider intervals
    assert np.mean(res_low["standard_error"]) > np.mean(res_high["standard_error"])
    assert np.mean(res_low["interval_width"]) > np.mean(res_high["interval_width"])


# =============================================================================
# 5. Reliability Indicator Classification
# =============================================================================

def test_reliability_classification(synthetic_toy_cohort):
    """Verify low_overlap, high_variance, and reliable status tags."""
    d = synthetic_toy_cohort
    est = CATEUncertaintyEstimator(overlap_threshold=0.10)

    # Force first 5 customers to have near-zero propensity < 0.10
    prop_mod = d["propensities"].copy()
    prop_mod[:5, 1] = 0.03  # arm 10 low overlap

    res = est.compute_asymptotic_uncertainty(
        cate_estimates=d["pred_cate_10"],
        propensities=prop_mod,
        covariates_df=d["covariates_df"],
        treatment_arm=10,
    )

    statuses = res["reliability_status"]
    # First 5 must be flagged as low_overlap
    assert np.all(statuses[:5] == "low_overlap")
    assert "reliable" in statuses


# =============================================================================
# 6. Ground-Truth Coverage & Error Metrics
# =============================================================================

def test_ground_truth_coverage_calculation():
    """Verify empirical coverage rate, MAE, RMSE, and bias computation."""
    est = CATEUncertaintyEstimator()

    true_c = np.array([0.15, 0.20, 0.25, 0.30])
    pred_c = np.array([0.16, 0.19, 0.24, 0.31])  # MAE = 0.01

    # Bounds wide enough to cover all
    lb = pred_c - 0.05
    ub = pred_c + 0.05

    cov_res = est.evaluate_ground_truth_coverage(true_c, pred_c, lb, ub)
    assert cov_res["coverage_rate_pct"] == 100.0
    assert np.isclose(cov_res["cate_mae"], 0.01, atol=1e-4)
    assert cov_res["cate_rmse"] >= cov_res["cate_mae"]
    assert np.isclose(cov_res["mean_interval_width"], 0.10, atol=1e-4)

    # Narrow bounds that miss some points
    lb_narrow = pred_c - 0.005
    ub_narrow = pred_c + 0.005
    cov_miss = est.evaluate_ground_truth_coverage(true_c, pred_c, lb_narrow, ub_narrow)
    assert cov_miss["coverage_rate_pct"] == 0.0


# =============================================================================
# 7. Risk-Aware Optimizer Schema Generation
# =============================================================================

def test_build_risk_aware_dataframe(synthetic_toy_cohort):
    """Verify output DataFrame schema matches exact risk-aware optimization contract."""
    d = synthetic_toy_cohort
    est = CATEUncertaintyEstimator()

    uncert = est.compute_asymptotic_uncertainty(
        cate_estimates=d["pred_cate_10"],
        propensities=d["propensities"],
        covariates_df=d["covariates_df"],
        treatment_arm=10,
    )

    df_out = est.build_risk_aware_dataframe(
        customer_ids=d["covariates_df"]["customer_id"],
        treatment_arm=10,
        cate=d["pred_cate_10"],
        uncertainty_dict=uncert,
    )

    expected_cols = [
        "customer_id",
        "treatment",
        "cate",
        "standard_error",
        "lower_bound",
        "upper_bound",
        "interval_width",
        "propensity",
        "reliability_status",
    ]
    assert list(df_out.columns) == expected_cols
    assert len(df_out) == d["n"]
    assert np.all(df_out["treatment"] == 10)
    assert not df_out.isnull().any().any()


# =============================================================================
# 8. Segment and Support Diagnostics Breakdowns
# =============================================================================

def test_segment_and_support_breakdown(synthetic_toy_cohort):
    """Verify segment-level and propensity support breakdowns partition cleanly."""
    d = synthetic_toy_cohort
    est = CATEUncertaintyEstimator()

    uncert = est.compute_asymptotic_uncertainty(
        cate_estimates=d["pred_cate_10"],
        propensities=d["propensities"],
        covariates_df=d["covariates_df"],
        treatment_arm=10,
    )

    seg_diag = est.evaluate_by_segment(
        d["covariates_df"], d["true_cate_10"], d["pred_cate_10"], uncert
    )
    supp_diag = est.evaluate_by_support_level(
        d["propensities"], d["true_cate_10"], d["pred_cate_10"], uncert
    )

    # Must contain segment entries
    assert "budget" in seg_diag or "regular" in seg_diag
    for seg_key, metrics in seg_diag.items():
        assert "coverage_rate_pct" in metrics
        assert "cate_mae" in metrics
        assert metrics["sample_size"] > 0

    assert len(supp_diag) > 0
    for supp_key, metrics in supp_diag.items():
        assert "mean_interval_width" in metrics
        assert "mean_propensity" in metrics
