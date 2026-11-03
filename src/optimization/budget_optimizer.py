"""Prescriptive Budget-Constrained Optimization Engine for EconoCausal Week 3.

This module provides the optimization engine that allocates personalized promotional
discounts ($0, $10, $20) to customers to maximize expected business value subject
to a strict marketing budget cap.

MATHEMATICAL FORMULATION:
Multiple-Choice Knapsack Problem (MCKP) / Binary Integer Linear Program (BILP):

    max_{z_{i, d}}  sum_{i=1}^N sum_{d in {0, 10, 20}} V_{i, d} * z_{i, d}

    subject to:
        1. Exact Assignment (Mutually Exclusive Tiers):
           sum_{d in {0, 10, 20}} z_{i, d} = 1      for all i in {1, ..., N}

        2. Strict Campaign Budget Cap:
           sum_{i=1}^N sum_{d in {10, 20}} Cost_{i, d} * z_{i, d} <= Budget

        3. Binary Domain:
           z_{i, d} in {0, 1}                       for all i, d

ECONOMIC ACCOUNTING ASSUMPTIONS (REDEMPTION MODEL):
- Cost is incurred ONLY when a customer receives an offer AND completes a purchase.
- Expected Discount Cost:
      Cost_{i, d} = d * P(purchase = 1 | X_i, d)
- Expected Gross Revenue:
      GrossRev_{i, d} = price_i * P(purchase = 1 | X_i, d)
- Expected Net Revenue:
      NetRev_{i, d} = (price_i - d) * P(purchase = 1 | X_i, d)
- Expected Gross Profit (Pre-Discount):
      GrossProfit_{i, d} = price_i * margin_i * P(purchase = 1 | X_i, d)
- Expected Net Profit (Post-Discount Contribution):
      NetProfit_{i, d} = (price_i * margin_i - d) * P(purchase = 1 | X_i, d)

OBJECTIVE TYPES:
1. 'net_profit' (Primary Business Objective):
   Maximizes expected bottom-line contribution after deducting discount expenses
   and COGS (captured via gross margin rate).
2. 'net_revenue' (Alternative PDF Objective):
   Maximizes top-line sales dollars after coupon deductions: (price - d) * P.
3. 'gross_revenue':
   Maximizes top-line sales dollars (price * P) subject to budget.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

from src.causal.multi_tier_dml import MultiTierCausalDML

DISCOUNT_TIERS: List[int] = [0, 10, 20]
DEFAULT_BUDGET: float = 5000.0


@dataclass
class OptimizationResult:
    """Encapsulates the output of a budget optimization run."""
    success: bool
    status_code: int
    status_message: str
    budget: float
    total_cost: float
    budget_utilization_pct: float
    expected_gross_revenue: float
    expected_net_revenue: float
    expected_net_profit: float
    expected_conversions: float
    tier_counts: Dict[int, int]
    allocation_matrix: pd.DataFrame
    objective_type: str


class SciPyBudgetOptimizer:
    """Mixed-Integer Linear Programming budget optimizer for personalized dynamic pricing.

    Attributes:
        budget: Configurable marketing budget ceiling in dollars (default: 5000.0).
        discount_tiers: Eligible discrete discount amounts (default: [0, 10, 20]).
        objective_type: Target metric ('net_profit', 'net_revenue', or 'gross_revenue').
    """

    def __init__(
        self,
        budget: float = DEFAULT_BUDGET,
        discount_tiers: Optional[List[int]] = None,
        objective_type: str = "net_profit",
    ) -> None:
        """Initialize SciPyBudgetOptimizer configuration.

        Args:
            budget: Total marketing budget cap (must be non-negative).
            discount_tiers: List of available discount face values (default: [0, 10, 20]).
            objective_type: Optimization objective ('net_profit', 'net_revenue', or 'gross_revenue').

        Raises:
            ValueError: If budget is negative or objective_type is unknown.
        """
        if budget < 0:
            raise ValueError(f"Budget must be non-negative, received: {budget}")
        if objective_type not in ["net_profit", "net_revenue", "gross_revenue"]:
            raise ValueError(
                f"Unknown objective_type '{objective_type}'. Expected 'net_profit', 'net_revenue', or 'gross_revenue'."
            )

        self.budget = float(budget)
        self.discount_tiers = list(discount_tiers) if discount_tiers is not None else list(DISCOUNT_TIERS)
        self.objective_type = objective_type

        # Verify discount tiers are sorted non-negative integers
        if any(t < 0 for t in self.discount_tiers):
            raise ValueError(f"Discount tiers must be non-negative, received: {self.discount_tiers}")
        if 0 not in self.discount_tiers:
            raise ValueError(f"Discount tiers must include control arm 0, received: {self.discount_tiers}")

    def compute_unit_economics(
        self,
        df_customers: pd.DataFrame,
        model: Optional[MultiTierCausalDML] = None,
        precomputed_probabilities: Optional[pd.DataFrame] = None,
    ) -> Dict[str, np.ndarray]:
        """Compute predicted purchase probabilities and expected economics for all customers.

        Args:
            df_customers: DataFrame containing customer features, 'product_price', and 'gross_margin_rate'.
            model: Optional fitted MultiTierCausalDML model (used if precomputed_probabilities is None).
            precomputed_probabilities: Optional DataFrame containing columns 'p_purchase_t0', 'p_purchase_t10', 'p_purchase_t20'.

        Returns:
            Dictionary of arrays containing per-customer probabilities, revenues, costs, and profits.

        Raises:
            ValueError: If required price, margin, or probability inputs are missing.
        """
        if df_customers is None or len(df_customers) == 0:
            raise ValueError("Input customer DataFrame is empty or None.")

        # Price and margin checks
        if "product_price" not in df_customers.columns:
            raise ValueError("Missing 'product_price' in customer DataFrame.")
        if "gross_margin_rate" not in df_customers.columns:
            raise ValueError("Missing 'gross_margin_rate' in customer DataFrame.")

        prices = df_customers["product_price"].to_numpy(dtype=float)
        margins = df_customers["gross_margin_rate"].to_numpy(dtype=float)

        if np.any(prices < 0) or np.any(margins < 0) or np.any(margins > 1.0):
            raise ValueError("Invalid values in 'product_price' or 'gross_margin_rate'.")

        # Retrieve or compute probabilities
        if precomputed_probabilities is not None:
            p0 = precomputed_probabilities["p_purchase_t0"].to_numpy(dtype=float)
            p10 = precomputed_probabilities["p_purchase_t10"].to_numpy(dtype=float)
            p20 = precomputed_probabilities["p_purchase_t20"].to_numpy(dtype=float)
        elif model is not None:
            if not model.is_fitted_:
                raise RuntimeError("Provided causal model is not fitted.")
            probs_df = model.predict_probabilities(df_customers)
            p0 = probs_df["p_purchase_t0"].to_numpy(dtype=float)
            p10 = probs_df["p_purchase_t10"].to_numpy(dtype=float)
            p20 = probs_df["p_purchase_t20"].to_numpy(dtype=float)
        else:
            raise ValueError("Either 'model' or 'precomputed_probabilities' must be provided.")

        # Clip probabilities strictly to [0, 1]
        p0 = np.clip(p0, 0.0, 1.0)
        p10 = np.clip(p10, 0.0, 1.0)
        p20 = np.clip(p20, 0.0, 1.0)

        # Expected Gross Revenue: price * p
        gross_rev_0 = prices * p0
        gross_rev_10 = prices * p10
        gross_rev_20 = prices * p20

        # Expected Discount Cost (Redemption model: d * p)
        cost_0 = np.zeros_like(p0)
        cost_10 = 10.0 * p10
        cost_20 = 20.0 * p20

        # Expected Net Revenue: (price - d) * p
        net_rev_0 = gross_rev_0
        net_rev_10 = gross_rev_10 - cost_10
        net_rev_20 = gross_rev_20 - cost_20

        # Expected Gross Profit (Pre-Discount): price * margin * p
        gross_prof_0 = prices * margins * p0
        gross_prof_10 = prices * margins * p10
        gross_prof_20 = prices * margins * p20

        # Expected Net Profit (Post-Discount Contribution): (price * margin - d) * p
        net_prof_0 = gross_prof_0 - cost_0
        net_prof_10 = gross_prof_10 - cost_10
        net_prof_20 = gross_prof_20 - cost_20

        # CATE lifts relative to T = 0
        tau_10 = p10 - p0
        tau_20 = p20 - p0

        return {
            "p0": p0,
            "p10": p10,
            "p20": p20,
            "tau_10": tau_10,
            "tau_20": tau_20,
            "gross_rev_0": gross_rev_0,
            "gross_rev_10": gross_rev_10,
            "gross_rev_20": gross_rev_20,
            "cost_0": cost_0,
            "cost_10": cost_10,
            "cost_20": cost_20,
            "net_rev_0": net_rev_0,
            "net_rev_10": net_rev_10,
            "net_rev_20": net_rev_20,
            "net_prof_0": net_prof_0,
            "net_prof_10": net_prof_10,
            "net_prof_20": net_prof_20,
            "prices": prices,
            "margins": margins,
        }

    def optimize(
        self,
        df_customers: pd.DataFrame,
        model: Optional[MultiTierCausalDML] = None,
        precomputed_probabilities: Optional[pd.DataFrame] = None,
        budget: Optional[float] = None,
        objective_type: Optional[str] = None,
    ) -> OptimizationResult:
        """Solve the budget-constrained personalized discount allocation problem via SciPy MILP.

        Args:
            df_customers: DataFrame containing customer records.
            model: Optional fitted MultiTierCausalDML instance.
            precomputed_probabilities: Optional DataFrame of predicted purchase probabilities.
            budget: Optional override for campaign budget.
            objective_type: Optional override for objective type.

        Returns:
            OptimizationResult containing allocation table, objective values, and solver diagnostics.
        """
        active_budget = float(budget) if budget is not None else self.budget
        active_obj = objective_type if objective_type is not None else self.objective_type

        N = len(df_customers)
        K = len(self.discount_tiers)  # 3 tiers: 0, 10, 20

        econ = self.compute_unit_economics(
            df_customers,
            model=model,
            precomputed_probabilities=precomputed_probabilities,
        )

        # Determine objective coefficients (SciPy milp MINIMIZES c^T x, so c = -values)
        if active_obj == "net_profit":
            v0 = econ["net_prof_0"]
            v10 = econ["net_prof_10"]
            v20 = econ["net_prof_20"]
        elif active_obj == "net_revenue":
            v0 = econ["net_rev_0"]
            v10 = econ["net_rev_10"]
            v20 = econ["net_rev_20"]
        elif active_obj == "gross_revenue":
            v0 = econ["gross_rev_0"]
            v10 = econ["gross_rev_10"]
            v20 = econ["gross_rev_20"]
        else:
            raise ValueError(f"Unknown objective_type: {active_obj}")

        c0 = econ["cost_0"]
        c10 = econ["cost_10"]
        c20 = econ["cost_20"]

        # Interleave decision variables: [z_{0, 0}, z_{0, 10}, z_{0, 20}, z_{1, 0}, ...]
        values = np.empty(K * N, dtype=float)
        values[0::3] = v0
        values[1::3] = v10
        values[2::3] = v20

        costs = np.empty(K * N, dtype=float)
        costs[0::3] = c0
        costs[1::3] = c10
        costs[2::3] = c20

        # Objective vector: minimize -values
        c = -values

        # Constraint 1: Mutually exclusive allocation (sum_d z_{i, d} = 1 for each customer i)
        # Sparse matrix of shape (N, K * N)
        rows_eq = np.repeat(np.arange(N), K)
        cols_eq = np.arange(K * N)
        data_eq = np.ones(K * N, dtype=float)
        A_eq = sp.csc_matrix((data_eq, (rows_eq, cols_eq)), shape=(N, K * N))

        # Constraint 2: Budget cap (sum_{i, d} cost_{i, d} * z_{i, d} <= budget)
        A_budget = sp.csc_matrix(costs.reshape(1, -1))

        # Stack into unified constraint matrix (N + 1, K * N)
        A = sp.vstack([A_eq, A_budget], format="csc")
        lb = np.concatenate([np.ones(N, dtype=float), [-np.inf]])
        ub = np.concatenate([np.ones(N, dtype=float), [active_budget]])

        constraints = LinearConstraint(A, lb=lb, ub=ub)
        integrality = np.ones(K * N)  # All variables binary {0, 1}
        bounds = Bounds(0, 1)

        # Handle zero budget edge case immediately or solve with HiGHS
        if active_budget == 0.0:
            # Deterministic: everyone gets $0
            selected_indices = np.zeros(N, dtype=int)
            res_success = True
            res_status = 0
            res_message = "Zero budget constraint solved deterministically (All customers assigned $0)."
        else:
            res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
            res_success = bool(res.success)
            res_status = int(res.status)
            res_message = str(res.message)

            if not res_success:
                raise RuntimeError(
                    f"SciPy MILP optimization failed with status {res_status}: {res_message}. "
                    f"Check feasibility of budget ${active_budget:,.2f}."
                )

            # Extract selected tiers per customer
            x_sol = res.x.reshape(N, K)
            selected_indices = np.argmax(x_sol, axis=1)

        # Map selected index (0, 1, 2) to tier dollar value (0, 10, 20)
        tier_map = np.array(self.discount_tiers)
        recommended_discounts = tier_map[selected_indices]

        # Assemble per-customer realized expected metrics under optimal recommendation
        opt_cost = np.where(recommended_discounts == 0, c0, np.where(recommended_discounts == 10, c10, c20))
        opt_gross_rev = np.where(recommended_discounts == 0, econ["gross_rev_0"], np.where(recommended_discounts == 10, econ["gross_rev_10"], econ["gross_rev_20"]))
        opt_net_rev = np.where(recommended_discounts == 0, econ["net_rev_0"], np.where(recommended_discounts == 10, econ["net_rev_10"], econ["net_rev_20"]))
        opt_net_prof = np.where(recommended_discounts == 0, econ["net_prof_0"], np.where(recommended_discounts == 10, econ["net_prof_10"], econ["net_prof_20"]))
        opt_prob = np.where(recommended_discounts == 0, econ["p0"], np.where(recommended_discounts == 10, econ["p10"], econ["p20"]))

        customer_ids = df_customers["customer_id"].astype(str).values if "customer_id" in df_customers.columns else [f"CUST_{i+1:06d}" for i in range(N)]

        # Allocation Matrix DataFrame
        allocation_df = pd.DataFrame({
            "customer_id": customer_ids,
            "predicted_purchase_probability_0": np.round(econ["p0"], 6),
            "predicted_purchase_probability_10": np.round(econ["p10"], 6),
            "predicted_purchase_probability_20": np.round(econ["p20"], 6),
            "predicted_effect_10_vs_0": np.round(econ["tau_10"], 6),
            "predicted_effect_20_vs_0": np.round(econ["tau_20"], 6),
            "recommended_discount": recommended_discounts.astype(int),
            "expected_discount_cost": np.round(opt_cost, 4),
            "expected_revenue": np.round(opt_gross_rev, 4),
            "expected_net_profit": np.round(opt_net_prof, 4),
        })

        total_cost = float(np.sum(opt_cost))
        utilization = (total_cost / active_budget * 100.0) if active_budget > 0 else 0.0
        counts = {t: int(np.sum(recommended_discounts == t)) for t in self.discount_tiers}

        return OptimizationResult(
            success=res_success,
            status_code=res_status,
            status_message=res_message,
            budget=active_budget,
            total_cost=round(total_cost, 2),
            budget_utilization_pct=round(utilization, 2),
            expected_gross_revenue=round(float(np.sum(opt_gross_rev)), 2),
            expected_net_revenue=round(float(np.sum(opt_net_rev)), 2),
            expected_net_profit=round(float(np.sum(opt_net_prof)), 2),
            expected_conversions=round(float(np.sum(opt_prob)), 2),
            tier_counts=counts,
            allocation_matrix=allocation_df,
            objective_type=active_obj,
        )

    def compare_strategies(
        self,
        df_customers: pd.DataFrame,
        model: Optional[MultiTierCausalDML] = None,
        precomputed_probabilities: Optional[pd.DataFrame] = None,
        budget: Optional[float] = None,
        allocation_result: Optional[OptimizationResult] = None,
    ) -> pd.DataFrame:
        """Compare the 4 marketing strategies on the exact same customer population.

        Strategies:
        1. No-Discount ($0 for all customers)
        2. Blanket $10 ($10 for all customers)
        3. Blanket $20 ($20 for all customers)
        4. Optimized Personalized Allocation (under budget cap)

        Args:
            df_customers: Customer DataFrame.
            model: Optional MultiTierCausalDML model.
            precomputed_probabilities: Optional DataFrame of predicted probabilities.
            budget: Optional budget override.
            allocation_result: Optional pre-run OptimizationResult.

        Returns:
            Summary comparison DataFrame.
        """
        active_budget = float(budget) if budget is not None else self.budget

        econ = self.compute_unit_economics(
            df_customers,
            model=model,
            precomputed_probabilities=precomputed_probabilities,
        )

        N = len(df_customers)

        # Run optimization if not provided
        if allocation_result is None:
            allocation_result = self.optimize(
                df_customers,
                model=model,
                precomputed_probabilities=precomputed_probabilities,
                budget=active_budget,
            )

        # Baseline: No discount
        no_cost = float(np.sum(econ["cost_0"]))
        no_gross_rev = float(np.sum(econ["gross_rev_0"]))
        no_net_rev = float(np.sum(econ["net_rev_0"]))
        no_net_prof = float(np.sum(econ["net_prof_0"]))
        no_conv = float(np.sum(econ["p0"]))

        # Blanket $10
        b10_cost = float(np.sum(econ["cost_10"]))
        b10_gross_rev = float(np.sum(econ["gross_rev_10"]))
        b10_net_rev = float(np.sum(econ["net_rev_10"]))
        b10_net_prof = float(np.sum(econ["net_prof_10"]))
        b10_conv = float(np.sum(econ["p10"]))

        # Blanket $20
        b20_cost = float(np.sum(econ["cost_20"]))
        b20_gross_rev = float(np.sum(econ["gross_rev_20"]))
        b20_net_rev = float(np.sum(econ["net_rev_20"]))
        b20_net_prof = float(np.sum(econ["net_prof_20"]))
        b20_conv = float(np.sum(econ["p20"]))

        # Optimized Personalized
        opt_cost = allocation_result.total_cost
        opt_gross_rev = allocation_result.expected_gross_revenue
        opt_net_rev = allocation_result.expected_net_revenue
        opt_net_prof = allocation_result.expected_net_profit
        opt_conv = allocation_result.expected_conversions

        strategies = [
            {
                "strategy": "No Discount (All $0)",
                "budget_cap": f"${active_budget:,.2f}",
                "expected_discount_cost": round(no_cost, 2),
                "budget_utilization": "0.0%",
                "budget_compliant": "YES",
                "expected_conversions": round(no_conv, 1),
                "expected_gross_revenue": round(no_gross_rev, 2),
                "expected_net_revenue": round(no_net_rev, 2),
                "expected_net_profit": round(no_net_prof, 2),
                "profit_lift_vs_baseline": "$0.00",
                "profit_lift_pct": "0.0%",
            },
            {
                "strategy": "Blanket $10 Discount",
                "budget_cap": f"${active_budget:,.2f}",
                "expected_discount_cost": round(b10_cost, 2),
                "budget_utilization": f"{(b10_cost / active_budget * 100.0):.1f}%",
                "budget_compliant": "YES" if b10_cost <= active_budget else "NO (Over Budget)",
                "expected_conversions": round(b10_conv, 1),
                "expected_gross_revenue": round(b10_gross_rev, 2),
                "expected_net_revenue": round(b10_net_rev, 2),
                "expected_net_profit": round(b10_net_prof, 2),
                "profit_lift_vs_baseline": f"${(b10_net_prof - no_net_prof):+,.2f}",
                "profit_lift_pct": f"{((b10_net_prof - no_net_prof) / no_net_prof * 100.0):+.1f}%",
            },
            {
                "strategy": "Blanket $20 Discount",
                "budget_cap": f"${active_budget:,.2f}",
                "expected_discount_cost": round(b20_cost, 2),
                "budget_utilization": f"{(b20_cost / active_budget * 100.0):.1f}%",
                "budget_compliant": "YES" if b20_cost <= active_budget else "NO (Over Budget)",
                "expected_conversions": round(b20_conv, 1),
                "expected_gross_revenue": round(b20_gross_rev, 2),
                "expected_net_revenue": round(b20_net_rev, 2),
                "expected_net_profit": round(b20_net_prof, 2),
                "profit_lift_vs_baseline": f"${(b20_net_prof - no_net_prof):+,.2f}",
                "profit_lift_pct": f"{((b20_net_prof - no_net_prof) / no_net_prof * 100.0):+.1f}%",
            },
            {
                "strategy": f"Optimized Personalized ({allocation_result.objective_type})",
                "budget_cap": f"${active_budget:,.2f}",
                "expected_discount_cost": round(opt_cost, 2),
                "budget_utilization": f"{allocation_result.budget_utilization_pct:.1f}%",
                "budget_compliant": "YES",
                "expected_conversions": round(opt_conv, 1),
                "expected_gross_revenue": round(opt_gross_rev, 2),
                "expected_net_revenue": round(opt_net_rev, 2),
                "expected_net_profit": round(opt_net_prof, 2),
                "profit_lift_vs_baseline": f"${(opt_net_prof - no_net_prof):+,.2f}",
                "profit_lift_pct": f"{((opt_net_prof - no_net_prof) / no_net_prof * 100.0):+.1f}%",
            },
        ]

        return pd.DataFrame(strategies)

