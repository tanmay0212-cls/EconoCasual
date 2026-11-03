"""Unit tests for Prescriptive Budget Optimization Engine in EconoCausal Week 3."""

import pytest
from typing import Tuple
import numpy as np
import pandas as pd

from src.optimization.budget_optimizer import (
    SciPyBudgetOptimizer,
    DISCOUNT_TIERS,
    DEFAULT_BUDGET,
    OptimizationResult,
)


@pytest.fixture
def mock_customer_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate small customer population with known economics for deterministic testing."""
    rng = np.random.default_rng(seed=42)
    N = 100

    prices = np.round(rng.uniform(30.0, 120.0, size=N), 2)
    margins = np.round(rng.uniform(0.30, 0.60, size=N), 4)

    # Probabilities: p0 in [0.2, 0.6], p10 > p0, p20 >= p10
    p0 = np.round(rng.uniform(0.20, 0.60, size=N), 4)
    tau10 = np.round(rng.uniform(0.08, 0.22, size=N), 4)
    tau20 = np.round(tau10 + rng.uniform(0.02, 0.08, size=N), 4)

    p10 = np.clip(p0 + tau10, 0.0, 1.0)
    p20 = np.clip(p0 + tau20, 0.0, 1.0)

    customers_df = pd.DataFrame({
        "customer_id": [f"CUST_{i+1:04d}" for i in range(N)],
        "product_price": prices,
        "gross_margin_rate": margins,
        "age": rng.integers(20, 65, size=N),
        "loyalty_score": np.round(rng.uniform(10.0, 90.0, size=N), 1),
        "previous_purchases": rng.poisson(lam=4, size=N),
        "customer_segment": rng.choice(["budget_seeker", "casual_shopper", "regular_buyer", "vip_loyalist"], size=N),
        "campaign_channel": rng.choice(["email", "push", "sms", "social"], size=N),
    })

    probs_df = pd.DataFrame({
        "customer_id": customers_df["customer_id"].values,
        "p_purchase_t0": p0,
        "p_purchase_t10": p10,
        "p_purchase_t20": p20,
    })

    return customers_df, probs_df


def test_budget_constraint_strictly_satisfied(mock_customer_data):
    """Verify solver never violates the configured budget ceiling."""
    customers_df, probs_df = mock_customer_data

    for test_budget in [100.0, 250.0, 500.0]:
        optimizer = SciPyBudgetOptimizer(budget=test_budget)
        res = optimizer.optimize(customers_df, precomputed_probabilities=probs_df)

        assert res.success is True
        assert res.total_cost <= test_budget + 1e-4, f"Cost {res.total_cost} exceeded budget {test_budget}"
        assert res.budget_utilization_pct <= 100.001


def test_mutually_exclusive_single_action(mock_customer_data):
    """Verify each customer receives exactly one valid discrete action."""
    customers_df, probs_df = mock_customer_data

    optimizer = SciPyBudgetOptimizer(budget=300.0)
    res = optimizer.optimize(customers_df, precomputed_probabilities=probs_df)
    mat = res.allocation_matrix

    assert len(mat) == len(customers_df)
    assert set(mat["recommended_discount"].unique()).issubset({0, 10, 20})
    assert mat["customer_id"].tolist() == customers_df["customer_id"].tolist()

    # Required columns check
    expected_cols = [
        "customer_id",
        "predicted_purchase_probability_0",
        "predicted_purchase_probability_10",
        "predicted_purchase_probability_20",
        "predicted_effect_10_vs_0",
        "predicted_effect_20_vs_0",
        "recommended_discount",
        "expected_discount_cost",
        "expected_revenue",
        "expected_net_profit",
    ]
    for col in expected_cols:
        assert col in mat.columns


def test_zero_budget_edge_case(mock_customer_data):
    """Verify zero budget ($0) assigns $0 to 100% of customers deterministically."""
    customers_df, probs_df = mock_customer_data

    optimizer = SciPyBudgetOptimizer(budget=0.0)
    res = optimizer.optimize(customers_df, precomputed_probabilities=probs_df)

    assert res.success is True
    assert res.total_cost == 0.0
    assert res.tier_counts[0] == len(customers_df)
    assert res.tier_counts.get(10, 0) == 0
    assert res.tier_counts.get(20, 0) == 0


def test_tiny_budget_insufficient_for_all(mock_customer_data):
    """Verify tiny budget (e.g. $8) assigns discounts only up to feasible limit."""
    customers_df, probs_df = mock_customer_data

    optimizer = SciPyBudgetOptimizer(budget=8.0)
    res = optimizer.optimize(customers_df, precomputed_probabilities=probs_df)

    assert res.success is True
    assert res.total_cost <= 8.0
    assert res.tier_counts[0] >= len(customers_df) - 2


def test_unconstrained_large_budget(mock_customer_data):
    """Verify massive budget allows unconstrained profit maximization."""
    customers_df, probs_df = mock_customer_data

    optimizer = SciPyBudgetOptimizer(budget=1_000_000.0)
    res = optimizer.optimize(customers_df, precomputed_probabilities=probs_df)

    assert res.success is True
    assert res.total_cost <= 1_000_000.0
    # Should allocate to tiers where marginal profit > 0
    assert res.expected_net_profit >= 0.0


def test_strategy_comparison_table(mock_customer_data):
    """Verify comparison table generates all 4 strategies with correct relative properties."""
    customers_df, probs_df = mock_customer_data

    optimizer = SciPyBudgetOptimizer(budget=250.0)
    comp_df = optimizer.compare_strategies(customers_df, precomputed_probabilities=probs_df)

    assert len(comp_df) == 4
    strategies = comp_df["strategy"].tolist()
    assert any("No Discount" in s for s in strategies)
    assert any("Blanket $10" in s for s in strategies)
    assert any("Blanket $20" in s for s in strategies)
    assert any("Optimized Personalized" in s for s in strategies)

    opt_row = comp_df[comp_df["strategy"].str.contains("Optimized")].iloc[0]
    no_row = comp_df[comp_df["strategy"].str.contains("No Discount")].iloc[0]

    assert opt_row["budget_compliant"] == "YES"
    assert opt_row["expected_net_profit"] >= no_row["expected_net_profit"]


def test_reproducibility(mock_customer_data):
    """Verify repeated runs yield identical bit-exact allocations."""
    customers_df, probs_df = mock_customer_data

    opt1 = SciPyBudgetOptimizer(budget=300.0)
    res1 = opt1.optimize(customers_df, precomputed_probabilities=probs_df)

    opt2 = SciPyBudgetOptimizer(budget=300.0)
    res2 = opt2.optimize(customers_df, precomputed_probabilities=probs_df)

    assert res1.total_cost == res2.total_cost
    assert res1.expected_net_profit == res2.expected_net_profit
    assert res1.allocation_matrix["recommended_discount"].tolist() == res2.allocation_matrix["recommended_discount"].tolist()


def test_invalid_parameters_raise_errors(mock_customer_data):
    """Verify validator catches negative budget, unknown objectives, or missing columns."""
    customers_df, probs_df = mock_customer_data

    with pytest.raises(ValueError, match="Budget must be non-negative"):
        SciPyBudgetOptimizer(budget=-500.0)

    with pytest.raises(ValueError, match="Unknown objective_type"):
        SciPyBudgetOptimizer(objective_type="invalid_objective")

    corrupted_df = customers_df.drop(columns=["product_price"])
    optimizer = SciPyBudgetOptimizer(budget=500.0)
    with pytest.raises(ValueError, match="Missing 'product_price'"):
        optimizer.optimize(corrupted_df, precomputed_probabilities=probs_df)
