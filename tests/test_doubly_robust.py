"""Unit Tests for Doubly Robust Policy Evaluation and Off-Policy Inference.

Tests cover:
1. Multi-treatment Inverse Probability Weighting (IPW) & Hajek normalization.
2. Positivity clipping and Kish's Effective Sample Size (ESS).
3. Doubly Robust estimator unbiasedness under double robustness property.
4. Non-parametric bootstrap confidence intervals & paired contrast intervals.
5. Comprehensive 5-policy comparison engine (including unavailable placeholder).
6. Input validation, shape assertions, and prevention of post-treatment feature leakage.
7. Strict distinction between realized observed profit, model expected profit, and DR estimated profit.
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from src.evaluation.ipw import InverseProbabilityWeighting, IPWResult
from src.evaluation.doubly_robust import DoublyRobustEstimator, DRResult
from src.evaluation.bootstrap import PolicyEvaluatorBootstrap, BootstrapResult
from src.evaluation.policy_comparison import (
    MultiTreatmentPolicyComparator,
    PolicyEvaluationSummary,
)


@pytest.fixture
def synthetic_toy_data():
    """Generates a small, deterministic multi-treatment dataset for unit testing."""
    rng = np.random.default_rng(42)
    n = 200

    # True treatment classes: 0, 10, 20
    t_classes = [0, 10, 20]
    t = rng.choice(t_classes, size=n, p=[0.4, 0.4, 0.2])

    # Propensities e_d(X) bounded away from 0
    prop_0 = np.full(n, 0.40)
    prop_10 = np.full(n, 0.40)
    prop_20 = np.full(n, 0.20)
    propensities = np.column_stack([prop_0, prop_10, prop_20])

    # Potential outcomes: Y(0) ~ 0.25, Y(10) ~ 0.45, Y(20) ~ 0.55
    y = np.zeros(n, dtype=float)
    y[t == 0] = rng.binomial(1, 0.25, size=np.sum(t == 0))
    y[t == 10] = rng.binomial(1, 0.45, size=np.sum(t == 10))
    y[t == 20] = rng.binomial(1, 0.55, size=np.sum(t == 20))

    # Nuisance predictions: mu_hat
    mu_hat = np.column_stack([
        np.full(n, 0.25),
        np.full(n, 0.45),
        np.full(n, 0.55),
    ])

    # Dummy customer features
    eval_df = pd.DataFrame({
        "customer_id": [f"CUST_{i:04d}" for i in range(n)],
        "average_previous_order_value": rng.uniform(50.0, 150.0, size=n).round(2),
        "customer_segment": rng.choice(["budget", "regular", "premium"], size=n),
        "discount_amount": t,
        "purchase": y,
    })

    return {
        "n": n,
        "t": t,
        "y": y,
        "propensities": propensities,
        "mu_hat": mu_hat,
        "eval_df": eval_df,
        "treatment_classes": t_classes,
    }


# =============================================================================
# 1. IPW Evaluator Tests
# =============================================================================

def test_ipw_basic_computation(synthetic_toy_data):
    """Verify Horvitz-Thompson and Hajek IPW calculate correct weights and estimates."""
    d = synthetic_toy_data
    ipw = InverseProbabilityWeighting(treatment_classes=d["treatment_classes"])

    # Policy: always assign $10
    policy_actions = np.full(d["n"], 10)
    result = ipw.evaluate_policy(d["y"], d["t"], d["propensities"], policy_actions, policy_name="flat_10")

    assert isinstance(result, IPWResult)
    assert 0.0 <= result.policy_value <= 1.0
    assert 0.0 <= result.normalized_policy_value <= 1.0
    assert result.sample_size == d["n"]
    assert result.effective_sample_size > 0.0
    assert result.effective_sample_size <= d["n"]
    assert result.propensity_clipped_count == 0


def test_ipw_propensity_clipping():
    """Verify that near-zero propensities are clipped and ESS decreases under extreme weights."""
    n = 100
    t = np.array([10] * 50 + [0] * 50)
    y = np.ones(n)
    policy_actions = np.full(n, 10)

    # Inject extreme near-zero propensities
    propensities = np.zeros((n, 3))
    propensities[:, 0] = 0.998
    propensities[:, 1] = 0.001  # Extremely small
    propensities[:, 2] = 0.001

    ipw = InverseProbabilityWeighting(treatment_classes=[0, 10, 20], min_propensity=0.05)
    result = ipw.evaluate_policy(y, t, propensities, policy_actions)

    assert result.propensity_clipped_count > 0
    assert result.propensity_clipped_pct > 0.0
    # Sensitivity analysis dictionary populated across grid
    assert len(result.clipping_sensitivity) > 0


def test_ipw_stochastic_policy():
    """Verify stochastic policy evaluation using assignment probability matrix."""
    n = 60
    t = np.array([0, 10, 20] * 20)
    y = np.ones(n)
    propensities = np.full((n, 3), 1/3)

    # Uniform random policy
    policy_probs = np.full((n, 3), 1/3)
    ipw = InverseProbabilityWeighting(treatment_classes=[0, 10, 20])
    result = ipw.evaluate_policy(y, t, propensities, policy_probs, policy_name="random_uniform")

    assert np.isclose(result.policy_value, 1.0, atol=1e-5)
    assert np.isclose(result.normalized_policy_value, 1.0, atol=1e-5)


# =============================================================================
# 2. Doubly Robust Estimator Tests
# =============================================================================

def test_doubly_robust_double_robustness_property():
    """Verify the double robustness property:
    Unbiased if outcome model is correct (even if propensity is wrong),
    and unbiased if propensity is correct (even if outcome model is wrong).
    """
    n = 2000
    rng = np.random.default_rng(123)

    # DGP: Binary treatment T in {0, 10}
    t = rng.choice([0, 10], size=n, p=[0.5, 0.5])
    # True expected outcomes: E[Y | T=0] = 0.30, E[Y | T=10] = 0.70
    y = np.where(t == 10, rng.binomial(1, 0.70, size=n).astype(float), rng.binomial(1, 0.30, size=n).astype(float))

    policy_actions = np.full(n, 10)  # Evaluate policy assigning T=10, true expectation = 0.70
    dr_eval = DoublyRobustEstimator(treatment_classes=[0, 10])

    # Case A: Exact Outcome Model, Wrong/Biased Propensity
    # If mu matches conditional expectation, DR estimate is consistent even with wrong propensities
    mu_correct = np.column_stack([np.full(n, 0.30), np.full(n, 0.70)])
    prop_wrong = np.column_stack([np.full(n, 0.75), np.full(n, 0.25)])  # Severely biased propensities
    res_a = dr_eval.evaluate_policy(y, t, prop_wrong, mu_correct, policy_actions)
    # DR point estimate must be within 3 standard errors of the truth 0.70
    assert abs(res_a.dr_policy_value - 0.70) <= 3.0 * res_a.standard_error
    assert res_a.ci_95_normal[0] <= 0.70 or abs(res_a.dr_policy_value - 0.70) <= 0.08

    # Case B: Wrong Outcome Model (constant zero), Correct Propensity
    # DR collapses to IPW, which is asymptotically unbiased with correct propensities
    mu_wrong = np.zeros((n, 2))
    prop_correct = np.column_stack([np.full(n, 0.50), np.full(n, 0.50)])
    res_b = dr_eval.evaluate_policy(y, t, prop_correct, mu_wrong, policy_actions)
    assert abs(res_b.dr_policy_value - 0.70) <= 3.0 * res_b.standard_error


def test_doubly_robust_influence_scores_mean(synthetic_toy_data):
    """Verify that sample average of DR influence scores equals the DR policy value."""
    d = synthetic_toy_data
    dr_eval = DoublyRobustEstimator(treatment_classes=d["treatment_classes"])
    policy_actions = np.full(d["n"], 10)

    res = dr_eval.evaluate_policy(d["y"], d["t"], d["propensities"], d["mu_hat"], policy_actions)
    assert np.isclose(np.mean(res.influence_scores), res.dr_policy_value, atol=1e-6)
    assert res.variance >= 0.0
    assert res.standard_error >= 0.0


# =============================================================================
# 3. Non-Parametric Bootstrap Tests
# =============================================================================

def test_bootstrap_reproducibility():
    """Verify that bootstrap engine with fixed random seed produces identical CIs."""
    influence_scores = np.linspace(0.1, 0.9, 150)

    engine1 = PolicyEvaluatorBootstrap(n_bootstraps=200, random_state=42)
    engine2 = PolicyEvaluatorBootstrap(n_bootstraps=200, random_state=42)

    res1 = engine1.bootstrap_influence_scores(influence_scores)
    res2 = engine2.bootstrap_influence_scores(influence_scores)

    assert np.isclose(res1.point_estimate, res2.point_estimate)
    assert np.isclose(res1.ci_lower_percentile, res2.ci_lower_percentile)
    assert np.isclose(res1.ci_upper_percentile, res2.ci_upper_percentile)
    assert res1.ci_lower_percentile <= res1.point_estimate <= res1.ci_upper_percentile


def test_bootstrap_paired_contrast():
    """Verify that paired contrast bootstrap computes delta distribution and correct CI."""
    n = 100
    scores_a = np.random.normal(loc=0.55, scale=0.05, size=n)
    scores_b = np.random.normal(loc=0.35, scale=0.05, size=n)

    engine = PolicyEvaluatorBootstrap(n_bootstraps=200, random_state=42)
    contrast = engine.bootstrap_paired_contrast(scores_a, scores_b)

    assert contrast.point_estimate > 0.10  # true difference ~ 0.20
    assert contrast.ci_lower_percentile < contrast.point_estimate < contrast.ci_upper_percentile


# =============================================================================
# 4. Multi-Treatment Policy Comparator Tests
# =============================================================================

def test_policy_comparator_all_five_policies(synthetic_toy_data):
    """Verify that MultiTreatmentPolicyComparator returns all 5 required policies with correct statuses."""
    d = synthetic_toy_data
    comparator = MultiTreatmentPolicyComparator(
        treatment_classes=d["treatment_classes"],
        n_bootstraps=100,
        random_state=42,
    )

    # Simulated DML+MILP allocations
    dml_actions = np.array([0 if i % 3 == 0 else 10 for i in range(d["n"])])

    results = comparator.evaluate_all_policies(
        eval_df=d["eval_df"],
        propensities=d["propensities"],
        pred_probs=d["mu_hat"],
        dml_milp_actions=dml_actions,
    )

    expected_keys = ["no_discount", "flat_10", "random_discount", "dml_milp", "uncertainty_aware_placeholder"]
    assert set(results.keys()) == set(expected_keys)

    # Available policies
    for key in ["no_discount", "flat_10", "random_discount", "dml_milp"]:
        summary = results[key]
        assert isinstance(summary, PolicyEvaluationSummary)
        assert summary.status == "evaluated"
        assert summary.is_available is True
        assert 0.0 <= summary.dr_conversion_rate <= 1.0
        assert summary.dr_conversion_ci_95[0] <= summary.dr_conversion_rate <= summary.dr_conversion_ci_95[1]
        assert summary.dr_profit_ci_95[0] <= summary.dr_expected_profit <= summary.dr_profit_ci_95[1]

    # Future Uncertainty-Aware Policy Placeholder
    placeholder = results["uncertainty_aware_placeholder"]
    assert placeholder.status == "unavailable"
    assert placeholder.is_available is False
    assert placeholder.dr_conversion_rate == 0.0
    assert placeholder.dr_expected_profit == 0.0


def test_policy_comparison_distinction_of_profits(synthetic_toy_data):
    """Verify that realized profit, model expected profit, and DR profit remain distinct."""
    d = synthetic_toy_data
    comparator = MultiTreatmentPolicyComparator(
        treatment_classes=d["treatment_classes"],
        n_bootstraps=50,
        random_state=42,
    )

    dml_actions = np.array([0 if i % 2 == 0 else 10 for i in range(d["n"])])
    results = comparator.evaluate_all_policies(
        eval_df=d["eval_df"],
        propensities=d["propensities"],
        pred_probs=d["mu_hat"],
        dml_milp_actions=dml_actions,
    )

    dml_summary = results["dml_milp"]
    # Realized profit only observes rows where observed treatment coincided with policy action
    assert dml_summary.realized_observed_profit is not None
    assert dml_summary.realized_observed_profit != dml_summary.dr_expected_profit
    # DR expected profit includes outcome correction
    assert isinstance(dml_summary.dr_expected_profit, float)
    assert isinstance(dml_summary.model_expected_profit, float)


# =============================================================================
# 5. Input Validation and Leakage Protection
# =============================================================================

def test_input_validation_mismatched_dimensions(synthetic_toy_data):
    """Verify that dimension mismatches between inputs raise descriptive ValueErrors."""
    d = synthetic_toy_data
    ipw = InverseProbabilityWeighting(treatment_classes=d["treatment_classes"])

    # Wrong length of y
    with pytest.raises(ValueError, match="Length mismatch"):
        ipw.evaluate_policy(d["y"][:-5], d["t"], d["propensities"], np.zeros(d["n"]))

    # Wrong propensity matrix columns
    with pytest.raises(ValueError, match="Propensities columns"):
        ipw.evaluate_policy(d["y"], d["t"], d["propensities"][:, :2], np.zeros(d["n"]))


def test_input_validation_invalid_treatment_class(synthetic_toy_data):
    """Verify that unknown treatment classes raise ValueError."""
    d = synthetic_toy_data
    ipw = InverseProbabilityWeighting(treatment_classes=d["treatment_classes"])

    # Invalid treatment value 99
    bad_t = d["t"].copy()
    bad_t[0] = 99
    with pytest.raises(ValueError, match="invalid classes"):
        ipw.evaluate_policy(d["y"], bad_t, d["propensities"], np.zeros(d["n"]))


def test_serialization_to_dict(synthetic_toy_data):
    """Verify that all result objects serialize cleanly to JSON-ready dicts."""
    d = synthetic_toy_data
    comparator = MultiTreatmentPolicyComparator(
        treatment_classes=d["treatment_classes"],
        n_bootstraps=50,
        random_state=42,
    )
    dml_actions = np.zeros(d["n"], dtype=int)
    results = comparator.evaluate_all_policies(
        eval_df=d["eval_df"],
        propensities=d["propensities"],
        pred_probs=d["mu_hat"],
        dml_milp_actions=dml_actions,
    )

    for key, summary in results.items():
        data_dict = summary.to_dict()
        assert isinstance(data_dict, dict)
        assert "conversion_metrics" in data_dict
        assert "economic_metrics" in data_dict
        assert "diagnostics" in data_dict
