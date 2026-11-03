"""Unit tests for ProfitCalculator in EconoCausal."""

import pytest
import numpy as np
import pandas as pd

from src.optimization.profit_calculator import ProfitCalculator


def test_margin_rate_estimation():
    calc = ProfitCalculator()
    aov = np.array([50.0, 100.0, 150.0, 40.0])
    seg = np.array(["budget", "regular", "premium", "vip"])
    margins = calc.estimate_gross_margin_rate(aov, seg)

    assert len(margins) == 4
    # budget segment penalty (-0.03)
    assert margins[0] == pytest.approx(0.52 - 0.03, abs=1e-3)
    # regular baseline (0.52)
    assert margins[1] == pytest.approx(0.52, abs=1e-3)
    # high aov bonus (+0.04)
    assert margins[2] == pytest.approx(0.52 + 0.04, abs=1e-3)
    # bounds [0.35, 0.70]
    assert np.all(margins >= 0.35) and np.all(margins <= 0.70)


def test_zero_discount_cost():
    calc = ProfitCalculator()
    df = pd.DataFrame({
        "customer_id": ["C1", "C2"],
        "average_previous_order_value": [80.0, 100.0],
        "customer_segment": ["regular", "vip"],
        "pred_prob_0": [0.3, 0.5],
        "pred_prob_10": [0.5, 0.7],
        "pred_prob_20": [0.6, 0.8],
    })
    res = calc.calculate_customer_economics(df)

    assert np.all(res["expected_discount_cost_0"] == 0.0)
    assert res.loc[0, "expected_discount_cost_10"] == pytest.approx(10.0 * 0.5, abs=1e-4)
    assert res.loc[1, "expected_discount_cost_20"] == pytest.approx(20.0 * 0.8, abs=1e-4)


def test_profit_accounting_identity():
    """Verify ExpectedProfit = P(Y=1) * GrossMarginBeforeDiscount - ExpectedDiscountCost."""
    calc = ProfitCalculator()
    df = pd.DataFrame({
        "customer_id": ["C1"],
        "average_previous_order_value": [100.0],
        "customer_segment": ["regular"],
        "pred_prob_0": [0.4],
        "pred_prob_10": [0.6],
        "pred_prob_20": [0.7],
    })
    res = calc.calculate_customer_economics(df)

    # For $10:
    # order_val = 100 * 1.03 = 103.0
    # gross_margin_before = 103.0 * 0.52 = 53.56
    # exp_cost = 10.0 * 0.6 = 6.0
    # exp_profit = 0.6 * (53.56 - 10.0) = 26.136
    p10 = 0.6
    val10 = 103.0 * 0.52
    expected_profit_10 = p10 * (val10 - 10.0)
    assert res.loc[0, "expected_profit_10"] == pytest.approx(expected_profit_10, abs=1e-3)

