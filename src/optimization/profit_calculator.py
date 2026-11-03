"""Economic Profit & Financial Liability Accounting Calculator.

This module formalizes the economic accounting definitions for promotional discounting:
1. Expected Gross Margin Before Discount
2. Expected Discount Redemption Cost: DiscountAmount * PurchaseProbability
3. Expected Net Profit / Contribution: ExpectedGrossMarginBeforeDiscount - ExpectedDiscountCost
4. Issued Face-Value Coupon Exposure (Non-probabilistic worst-case budget exposure)

All estimations rely strictly on legitimate pre-treatment customer features:
- baseline basket estimation from customer's historical order value (average_previous_order_value)
- category gross margin rate based on customer segment & spending tier
- coupon basket expansion elasticity (3% for $10, 6% for $20)
- model-predicted counterfactual purchase probabilities (P0, P10, P20)
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class ProfitCalculator:
    """Calculates customer-level expected financial outcomes under promotional discount options."""

    def __init__(
        self,
        base_margin_rate: float = 0.52,
        high_aov_threshold: float = 120.0,
        high_aov_margin_bonus: float = 0.04,
        budget_segment_margin_penalty: float = 0.03,
        basket_expansion_10: float = 1.03,
        basket_expansion_20: float = 1.06,
    ):
        self.base_margin_rate = base_margin_rate
        self.high_aov_threshold = high_aov_threshold
        self.high_aov_margin_bonus = high_aov_margin_bonus
        self.budget_segment_margin_penalty = budget_segment_margin_penalty
        self.basket_expansion_10 = basket_expansion_10
        self.basket_expansion_20 = basket_expansion_20

    def estimate_gross_margin_rate(
        self,
        average_previous_order_value: np.ndarray,
        customer_segment: np.ndarray,
    ) -> np.ndarray:
        """Estimate customer-level gross margin rate from pre-treatment attributes."""
        aov = np.asarray(average_previous_order_value, dtype=float)
        seg = np.asarray(customer_segment, dtype=str)

        margin_rate = (
            self.base_margin_rate
            + self.high_aov_margin_bonus * (aov > self.high_aov_threshold).astype(float)
            - self.budget_segment_margin_penalty * (seg == "budget").astype(float)
        )
        return np.clip(margin_rate, 0.35, 0.70)

    def calculate_customer_economics(
        self,
        customer_data: pd.DataFrame,
        prob_0_col: str = "pred_prob_0",
        prob_10_col: str = "pred_prob_10",
        prob_20_col: str = "pred_prob_20",
    ) -> pd.DataFrame:
        """Compute expected revenue, gross margin, discount cost, and net profit per tier."""
        n_customers = len(customer_data)
        
        # 1. Pre-treatment baseline basket size
        if "average_previous_order_value" in customer_data.columns:
            aov = customer_data["average_previous_order_value"].to_numpy(dtype=float)
        else:
            # Fallback if AOV not in DataFrame: use population median $85.00
            aov = np.full(n_customers, 85.00)

        # 2. Pre-treatment segment-informed margin rate (prefer observed margin rate if present)
        if "gross_margin_rate" in customer_data.columns:
            margin_rate = customer_data["gross_margin_rate"].to_numpy(dtype=float)
        elif "customer_segment" in customer_data.columns:
            seg = customer_data["customer_segment"].to_numpy(dtype=str)
            margin_rate = self.estimate_gross_margin_rate(aov, seg)
        else:
            margin_rate = np.full(n_customers, self.base_margin_rate)

        # 3. Treatment-specific expected order values (including coupon basket expansion)
        order_val_0 = aov * 1.00
        order_val_10 = aov * self.basket_expansion_10
        order_val_20 = aov * self.basket_expansion_20

        # 4. Expected Gross Margin Before Discount
        gross_margin_before_0 = order_val_0 * margin_rate
        gross_margin_before_10 = order_val_10 * margin_rate
        gross_margin_before_20 = order_val_20 * margin_rate

        # 5. Model-predicted counterfactual conversion probabilities
        p0 = customer_data[prob_0_col].to_numpy(dtype=float)
        p10 = customer_data[prob_10_col].to_numpy(dtype=float)
        p20 = customer_data[prob_20_col].to_numpy(dtype=float)

        # 6. Expected Discount Redemption Cost: DiscountAmount * PurchaseProbability
        exp_cost_0 = np.zeros(n_customers, dtype=float)
        exp_cost_10 = 10.0 * p10
        exp_cost_20 = 20.0 * p20

        # 7. Expected Gross Revenue Net of Discount
        exp_rev_0 = order_val_0 * p0
        exp_rev_10 = np.maximum(0.0, order_val_10 - 10.0) * p10
        exp_rev_20 = np.maximum(0.0, order_val_20 - 20.0) * p20

        # 8. Expected Net Profit (Contribution Margin):
        # E[Profit_d] = P(Y=1|d) * GrossMarginBeforeDiscount_d - ExpectedDiscountCost_d
        #             = P(Y=1|d) * (GrossMarginBeforeDiscount_d - d)
        exp_profit_0 = p0 * gross_margin_before_0
        exp_profit_10 = p10 * (gross_margin_before_10 - 10.0)
        exp_profit_20 = p20 * (gross_margin_before_20 - 20.0)

        results = pd.DataFrame(
            {
                "customer_id": customer_data["customer_id"].values if "customer_id" in customer_data.columns else np.arange(n_customers),
                "predicted_purchase_probability_0": np.round(p0, 4),
                "predicted_purchase_probability_10": np.round(p10, 4),
                "predicted_purchase_probability_20": np.round(p20, 4),
                "expected_gross_margin_before_0": np.round(gross_margin_before_0, 2),
                "expected_gross_margin_before_10": np.round(gross_margin_before_10, 2),
                "expected_gross_margin_before_20": np.round(gross_margin_before_20, 2),
                "expected_discount_cost_0": np.round(exp_cost_0, 4),
                "expected_discount_cost_10": np.round(exp_cost_10, 4),
                "expected_discount_cost_20": np.round(exp_cost_20, 4),
                "expected_revenue_0": np.round(exp_rev_0, 2),
                "expected_revenue_10": np.round(exp_rev_10, 2),
                "expected_revenue_20": np.round(exp_rev_20, 2),
                "expected_profit_0": np.round(exp_profit_0, 4),
                "expected_profit_10": np.round(exp_profit_10, 4),
                "expected_profit_20": np.round(exp_profit_20, 4),
            },
            index=customer_data.index,
        )

        return results

