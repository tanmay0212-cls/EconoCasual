"""Unit tests for allocation_validator in EconoCausal."""

import pytest
import pandas as pd
from src.optimization.allocation_validator import validate_allocation


def test_valid_allocation():
    df = pd.DataFrame({
        "customer_id": ["C1", "C2", "C3"],
        "allocated_discount": [0, 10, 20],
        "expected_purchase_probability_selected": [0.3, 0.5, 0.7],
        "expected_discount_cost_selected": [0.0, 5.0, 14.0],
    })
    res = validate_allocation(df, budget=25.0)
    assert res["valid"] is True
    assert res["budget_satisfied"] is True
    assert res["tier_counts"] == {0: 1, 10: 1, 20: 1}


def test_budget_violation_caught():
    df = pd.DataFrame({
        "customer_id": ["C1", "C2"],
        "allocated_discount": [10, 20],
        "expected_discount_cost_selected": [5.0, 12.0],
    })
    res = validate_allocation(df, budget=10.0)
    assert res["valid"] is False
    assert res["budget_satisfied"] is False
    assert any("Budget constraint violated" in iss for iss in res["issues"])


def test_invalid_tier_caught():
    df = pd.DataFrame({
        "customer_id": ["C1"],
        "allocated_discount": [15],  # Invalid tier
        "expected_discount_cost_selected": [5.0],
    })
    res = validate_allocation(df, budget=100.0)
    assert res["valid"] is False
    assert any("Invalid discount tiers" in iss for iss in res["issues"])


def test_duplicate_customers_caught():
    df = pd.DataFrame({
        "customer_id": ["C1", "C1"],  # Duplicate
        "allocated_discount": [0, 10],
        "expected_discount_cost_selected": [0.0, 5.0],
    })
    res = validate_allocation(df, budget=100.0)
    assert res["valid"] is False
    assert any("duplicate customer IDs" in iss for iss in res["issues"])


def test_ground_truth_leakage_caught():
    df = pd.DataFrame({
        "customer_id": ["C1"],
        "allocated_discount": [0],
        "expected_discount_cost_selected": [0.0],
        "true_ite_10_vs_0": [0.15],  # Leaked column
    })
    res = validate_allocation(df, budget=100.0)
    assert res["valid"] is False
    assert res["no_leakage_detected"] is False

