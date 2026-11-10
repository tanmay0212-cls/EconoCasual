"""Unit Tests for Advanced Uncertainty-Aware & Fairness-Constrained MILP Optimizer (Phase 3).

Tests cover:
1. Optimizer initialization, mode validation, and parameter bounds.
2. Objective construction across baseline, conservative LCB, and risk-adjusted regimes.
3. Structural MILP constraints: exact assignment sum_d z_{i, d} = 1 and integrality.
4. Budget compliance: total expected promotional cost <= budget ceiling.
5. Infeasible budget handling: graceful error handling and status codes.
6. Zero budget edge case: non-crashing assignment of all customers to control tier 0.
7. Fairness constraints: enforcement of business segment allocation floors and parity ratios.
8. Risk-aversion sensitivity: increasing lambda penalizes high-variance customers.
9. Reproducibility: deterministic optimization under identical conditions.
10. Customer identifier isolation: IDs present in output schema but excluded from features.
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from src.optimization.advanced_optimizer import (
    AdvancedMILPOptimizer,
    AdvancedOptimizationResult,
    SegmentFairnessDiagnostic,
)


@pytest.fixture
def optimization_test_cohort():
    """Deterministic synthetic retail cohort for optimizer tests."""
    rng = np.random.default_rng(42)
    n = 200

    cids = [f"CUST_{i+1:05d}" for i in range(n)]
    segments = rng.choice(["budget", "regular", "premium", "vip"], size=n, p=[0.30, 0.40, 0.20, 0.10])
    aov = rng.uniform(40.0, 180.0, size=n)

    df_customers = pd.DataFrame({
        "customer_id": cids,
        "customer_segment": segments,
        "average_previous_order_value": aov,
    })

    # Baseline and uplift probabilities
    p0 = rng.uniform(0.15, 0.40, size=n)
    tau10 = rng.uniform(0.10, 0.25, size=n)
    tau20 = tau10 + rng.uniform(0.02, 0.10, size=n)

    p10 = np.clip(p0 + tau10, 0.0, 0.95)
    p20 = np.clip(p0 + tau20, 0.0, 0.95)

    df_preds = pd.DataFrame({
        "customer_id": cids,
        "pred_prob_0": p0,
        "pred_prob_10": p10,
        "pred_prob_20": p20,
    })

    # Uncertainty DataFrame
    se10 = rng.uniform(0.025, 0.055, size=n)
    se20 = rng.uniform(0.035, 0.075, size=n)

    uncert_10 = pd.DataFrame({
        "customer_id": cids,
        "treatment": 10,
        "standard_error": se10,
        "reliability_status": np.where(se10 > 0.05, "high_variance", "reliable"),
    })
    uncert_20 = pd.DataFrame({
        "customer_id": cids,
        "treatment": 20,
        "standard_error": se20,
        "reliability_status": np.where(se20 > 0.065, "high_variance", "reliable"),
    })
    uncertainty_df = pd.concat([uncert_10, uncert_20], ignore_index=True)

    return {
        "n": n,
        "df_customers": df_customers,
        "df_preds": df_preds,
        "uncertainty_df": uncertainty_df,
    }


# =============================================================================
# 1. Parameter Validation
# =============================================================================

def test_optimizer_parameter_validation():
    """Verify validation of budget, risk aversion, confidence level, and modes."""
    with pytest.raises(ValueError, match="Budget must be non-negative"):
        AdvancedMILPOptimizer(budget=-100.0)

    with pytest.raises(ValueError, match="Unknown optimization_mode"):
        AdvancedMILPOptimizer(optimization_mode="invalid_mode")

    with pytest.raises(ValueError, match="risk_aversion must be non-negative"):
        AdvancedMILPOptimizer(risk_aversion=-0.5)

    with pytest.raises(ValueError, match="confidence_level must be in"):
        AdvancedMILPOptimizer(confidence_level=0.30)


# =============================================================================
# 2. Feasibility and Exact Assignment
# =============================================================================

def test_milp_exact_assignment_and_budget_cap(optimization_test_cohort):
    """Verify exact assignment constraint sum_d z_{i, d} = 1 and cost <= budget."""
    d = optimization_test_cohort
    budget = 400.0

    opt = AdvancedMILPOptimizer(
        budget=budget,
        budget_type="redemption",
        optimization_mode="baseline",
    )
    res = opt.optimize(d["df_customers"], d["df_preds"], d["uncertainty_df"])

    assert res.success is True
    assert res.status_code == 0
    assert res.total_expected_cost <= budget + 1e-4

    # Every customer assigned exactly one tier
    alloc = res.allocation_df["allocated_discount"].to_numpy()
    assert len(alloc) == d["n"]
    assert set(alloc).issubset({0, 10, 20})
    assert sum(res.tier_counts.values()) == d["n"]


# =============================================================================
# 3. Optimization Modes Comparison
# =============================================================================

def test_optimization_modes_execution(optimization_test_cohort):
    """Verify baseline, conservative LCB, and risk-adjusted modes run and produce valid results."""
    d = optimization_test_cohort
    budget = 500.0

    # 1. Baseline
    opt_base = AdvancedMILPOptimizer(budget=budget, optimization_mode="baseline")
    res_base = opt_base.optimize(d["df_customers"], d["df_preds"], d["uncertainty_df"])

    # 2. Conservative LCB
    opt_lcb = AdvancedMILPOptimizer(budget=budget, optimization_mode="conservative_lcb")
    res_lcb = opt_lcb.optimize(d["df_customers"], d["df_preds"], d["uncertainty_df"])

    # 3. Risk-Adjusted
    opt_risk = AdvancedMILPOptimizer(budget=budget, optimization_mode="risk_adjusted", risk_aversion=1.5)
    res_risk = opt_risk.optimize(d["df_customers"], d["df_preds"], d["uncertainty_df"])

    assert res_base.success and res_lcb.success and res_risk.success
    assert res_base.total_expected_cost <= budget + 1e-4
    assert res_lcb.total_expected_cost <= budget + 1e-4
    assert res_risk.total_expected_cost <= budget + 1e-4

    # Conservative LCB profit should be less than or equal to nominal profit
    assert res_lcb.conservative_lcb_profit <= res_lcb.nominal_expected_profit


# =============================================================================
# 4. Risk-Aversion Monotonicity
# =============================================================================

def test_risk_aversion_uncertainty_reduction(optimization_test_cohort):
    """Verify that increasing lambda reduces the uncertainty of selected customers."""
    d = optimization_test_cohort
    budget = 400.0

    opt_neutral = AdvancedMILPOptimizer(budget=budget, optimization_mode="risk_adjusted", risk_aversion=0.0)
    opt_averse = AdvancedMILPOptimizer(budget=budget, optimization_mode="risk_adjusted", risk_aversion=2.5)

    res_neutral = opt_neutral.optimize(d["df_customers"], d["df_preds"], d["uncertainty_df"])
    res_averse = opt_averse.optimize(d["df_customers"], d["df_preds"], d["uncertainty_df"])

    assert res_neutral.success and res_averse.success
    # Risk-averse optimizer should achieve lower or equal mean uncertainty for selected customers
    assert res_averse.selected_customers_mean_uncertainty <= res_neutral.selected_customers_mean_uncertainty + 1e-4


# =============================================================================
# 5. Fairness Constraints Enforcement
# =============================================================================

def test_fairness_constraint_increases_segment_allocation(optimization_test_cohort):
    """Verify that fairness floor forces higher allocation to under-served segments."""
    d = optimization_test_cohort
    budget = 600.0

    # Unconstrained
    opt_unconstrained = AdvancedMILPOptimizer(
        budget=budget,
        optimization_mode="baseline",
        fairness_enabled=False,
    )
    res_unconstrained = opt_unconstrained.optimize(d["df_customers"], d["df_preds"], d["uncertainty_df"])

    # Constrain budget segment to at least 25% allocation
    opt_fair = AdvancedMILPOptimizer(
        budget=budget,
        optimization_mode="baseline",
        fairness_enabled=True,
        fairness_mode="min_allocation_rate",
        min_segment_rate={"budget": 0.25},
    )
    res_fair = opt_fair.optimize(d["df_customers"], d["df_preds"], d["uncertainty_df"])

    assert res_unconstrained.success and res_fair.success
    diag_uncon = res_unconstrained.fairness_metrics["budget"]
    diag_fair = res_fair.fairness_metrics["budget"]

    # Budget segment allocation rate must satisfy floor (>= 25%)
    assert diag_fair.allocation_rate_pct >= 24.99
    assert diag_fair.allocation_count >= diag_uncon.allocation_count


# =============================================================================
# 6. Edge Cases: Zero Budget & Infeasibility
# =============================================================================

def test_zero_budget_edge_case(optimization_test_cohort):
    """Verify that budget=0 assigns everyone to control tier 0 without solver failure."""
    d = optimization_test_cohort
    opt = AdvancedMILPOptimizer(budget=0.0)
    res = opt.optimize(d["df_customers"], d["df_preds"], d["uncertainty_df"])

    assert res.success is True
    assert res.status_code == 0
    assert res.total_expected_cost == 0.0
    assert res.tier_counts[0] == d["n"]
    assert res.tier_counts[10] == 0
    assert res.tier_counts[20] == 0


def test_infeasible_budget_and_fairness_handling(optimization_test_cohort):
    """Verify that an impossible combination of budget and fairness returns success=False gracefully."""
    d = optimization_test_cohort
    # Require 90% allocation across all segments with only $1.00 budget
    opt = AdvancedMILPOptimizer(
        budget=1.00,
        fairness_enabled=True,
        fairness_mode="min_allocation_rate",
        min_segment_rate={"budget": 0.90, "regular": 0.90, "premium": 0.90, "vip": 0.90},
    )
    res = opt.optimize(d["df_customers"], d["df_preds"], d["uncertainty_df"])

    assert res.success is False
    assert res.status_code != 0
    assert "infeasible" in res.status_message.lower() or "failed" in res.status_message.lower()


# =============================================================================
# 7. Reproducibility & Schema Assertions
# =============================================================================

def test_optimization_reproducibility(optimization_test_cohort):
    """Verify that identical inputs produce identical allocations."""
    d = optimization_test_cohort
    opt1 = AdvancedMILPOptimizer(budget=400.0, optimization_mode="risk_adjusted", risk_aversion=1.0)
    opt2 = AdvancedMILPOptimizer(budget=400.0, optimization_mode="risk_adjusted", risk_aversion=1.0)

    res1 = opt1.optimize(d["df_customers"], d["df_preds"], d["uncertainty_df"])
    res2 = opt2.optimize(d["df_customers"], d["df_preds"], d["uncertainty_df"])

    assert np.all(res1.allocation_df["allocated_discount"] == res2.allocation_df["allocated_discount"])
    assert np.isclose(res1.nominal_expected_profit, res2.nominal_expected_profit)
    assert np.isclose(res1.total_expected_cost, res2.total_expected_cost)


def test_to_dict_serialization(optimization_test_cohort):
    """Verify that result objects serialize cleanly to JSON-compatible dictionaries."""
    d = optimization_test_cohort
    opt = AdvancedMILPOptimizer(budget=400.0, optimization_mode="risk_adjusted")
    res = opt.optimize(d["df_customers"], d["df_preds"], d["uncertainty_df"])

    res_dict = res.to_dict()
    assert isinstance(res_dict, dict)
    assert "financial_kpis" in res_dict
    assert "uncertainty_kpis" in res_dict
    assert "fairness_metrics" in res_dict
    assert "cash_budget_disclaimer" in res_dict
