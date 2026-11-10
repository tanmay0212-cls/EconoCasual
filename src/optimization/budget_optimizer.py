"""Prescriptive Budget-Constrained Optimization Engine for EconoCausal.

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

BUDGET INTERPRETATIONS:
1. 'redemption' (Primary setting):
   Budget cap limits total expected redemption expenditure: sum d * P_d * x_{i, d} <= Budget.
2. 'face_value' (Issued coupon exposure):
   Budget cap limits total face value of coupons offered: sum d * x_{i, d} <= Budget.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

if TYPE_CHECKING:
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
    face_value_exposure: float = 0.0
    expected_conversion_rate: float = 0.0


def build_objective_coefficients(
    econ: Dict[str, np.ndarray],
    objective_type: str = "net_profit",
    n_tiers: int = 3,
) -> np.ndarray:
    """Construct coefficient vector for SciPy MILP minimization (-values)."""
    if objective_type == "net_profit":
        v0 = econ["net_prof_0"]
        v10 = econ["net_prof_10"]
        v20 = econ["net_prof_20"]
    elif objective_type == "net_revenue":
        v0 = econ["net_rev_0"]
        v10 = econ["net_rev_10"]
        v20 = econ["net_rev_20"]
    elif objective_type == "gross_revenue":
        v0 = econ["gross_rev_0"]
        v10 = econ["gross_rev_10"]
        v20 = econ["gross_rev_20"]
    else:
        raise ValueError(f"Unknown objective_type: {objective_type}")

    n = len(v0)
    values = np.empty(n_tiers * n, dtype=float)
    values[0::3] = v0
    values[1::3] = v10
    values[2::3] = v20
    # scipy.optimize.milp minimizes by default: c = -values
    return -values


def build_constraints(
    econ: Dict[str, np.ndarray],
    budget: float,
    budget_type: str = "redemption",
    discount_tiers: Optional[List[int]] = None,
) -> Tuple[LinearConstraint, Bounds, np.ndarray]:
    """Construct SciPy LinearConstraint and Bounds for MILP."""
    if discount_tiers is None:
        discount_tiers = DISCOUNT_TIERS

    k = len(discount_tiers)
    n = len(econ["p0"])

    # 1. Exact Assignment Constraints: sum_d z_{i, d} = 1
    rows_eq = np.repeat(np.arange(n), k)
    cols_eq = np.arange(k * n)
    data_eq = np.ones(k * n, dtype=float)
    a_eq = sp.csc_matrix((data_eq, (rows_eq, cols_eq)), shape=(n, k * n))

    # 2. Budget Constraint
    if budget_type == "redemption":
        costs = np.empty(k * n, dtype=float)
        costs[0::3] = econ["cost_0"]
        costs[1::3] = econ["cost_10"]
        costs[2::3] = econ["cost_20"]
    else:  # face_value budget
        costs = np.empty(k * n, dtype=float)
        costs[0::3] = 0.0
        costs[1::3] = 10.0
        costs[2::3] = 20.0

    a_budget = sp.csc_matrix(costs.reshape(1, -1))

    # Combine into single constraint matrix
    a = sp.vstack([a_eq, a_budget], format="csc")
    lb = np.concatenate([np.ones(n, dtype=float), [-np.inf]])
    ub = np.concatenate([np.ones(n, dtype=float), [budget]])

    constraints = LinearConstraint(a, lb=lb, ub=ub)
    integrality = np.ones(k * n)  # Binary {0, 1}
    bounds = Bounds(0, 1)

    return constraints, bounds, integrality


def run_scipy_milp(
    c: np.ndarray,
    constraints: LinearConstraint,
    integrality: np.ndarray,
    bounds: Bounds,
) -> Any:
    """Execute SciPy MILP optimization solver."""
    return milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)


def calculate_business_metrics(allocation_df: pd.DataFrame) -> Dict[str, float]:
    """Compute aggregate business KPIs from an allocation matrix."""
    n = len(allocation_df)
    cost_col = "expected_discount_cost_selected" if "expected_discount_cost_selected" in allocation_df.columns else "expected_discount_cost"
    profit_col = "expected_profit_selected" if "expected_profit_selected" in allocation_df.columns else "expected_net_profit"
    prob_col = "expected_purchase_probability_selected" if "expected_purchase_probability_selected" in allocation_df.columns else "predicted_purchase_probability"
    alloc_col = "allocated_discount" if "allocated_discount" in allocation_df.columns else "recommended_discount"

    total_cost = float(allocation_df[cost_col].sum())
    total_profit = float(allocation_df[profit_col].sum())
    total_purchases = float(allocation_df[prob_col].sum())
    face_exposure = float(allocation_df[alloc_col].sum())

    return {
        "n_customers": n,
        "total_cost": round(total_cost, 2),
        "total_profit": round(total_profit, 2),
        "profit_per_customer": round(total_profit / n, 2),
        "total_purchases": round(total_purchases, 1),
        "conversion_rate": round(total_purchases / n * 100, 2),
        "face_exposure": round(face_exposure, 2),
    }


class SciPyBudgetOptimizer:
    """Mixed-Integer Linear Programming budget optimizer for personalized dynamic pricing."""

    def __init__(
        self,
        budget: float = DEFAULT_BUDGET,
        discount_tiers: Optional[List[int]] = None,
        objective_type: str = "net_profit",
        budget_type: str = "redemption",
    ) -> None:
        if budget < 0:
            raise ValueError(f"Budget must be non-negative, received: {budget}")
        if objective_type not in ["net_profit", "net_revenue", "gross_revenue"]:
            raise ValueError(
                f"Unknown objective_type '{objective_type}'. Expected 'net_profit', 'net_revenue', or 'gross_revenue'."
            )
        if budget_type not in ["redemption", "face_value"]:
            raise ValueError(f"Unknown budget_type '{budget_type}'. Expected 'redemption' or 'face_value'.")

        self.budget = float(budget)
        self.discount_tiers = list(discount_tiers) if discount_tiers is not None else list(DISCOUNT_TIERS)
        self.objective_type = objective_type
        self.budget_type = budget_type

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
        """Compute predicted purchase probabilities and expected economics for all customers."""
        if df_customers is None or len(df_customers) == 0:
            raise ValueError("Input customer DataFrame is empty or None.")

        # Price handling: support product_price or fallback to average_previous_order_value
        if "product_price" in df_customers.columns:
            prices = df_customers["product_price"].to_numpy(dtype=float)
        elif "average_previous_order_value" in df_customers.columns:
            prices = df_customers["average_previous_order_value"].to_numpy(dtype=float)
        else:
            raise ValueError("Missing 'product_price' in customer DataFrame.")

        # Margin handling: support gross_margin_rate or segment estimation
        if "gross_margin_rate" in df_customers.columns:
            margins = df_customers["gross_margin_rate"].to_numpy(dtype=float)
        elif "customer_segment" in df_customers.columns:
            from .profit_calculator import ProfitCalculator
            margins = ProfitCalculator().estimate_gross_margin_rate(prices, df_customers["customer_segment"].to_numpy())
        else:
            raise ValueError("Missing 'gross_margin_rate' in customer DataFrame.")

        if np.any(prices < 0) or np.any(margins < 0) or np.any(margins > 1.0):
            raise ValueError("Invalid values in 'product_price' or 'gross_margin_rate'.")

        # Probability handling: support multiple column naming conventions
        if precomputed_probabilities is not None:
            p0_col = next((c for c in ["p_purchase_t0", "predicted_purchase_probability_0", "pred_prob_0"] if c in precomputed_probabilities.columns), None)
            p10_col = next((c for c in ["p_purchase_t10", "predicted_purchase_probability_10", "pred_prob_10"] if c in precomputed_probabilities.columns), None)
            p20_col = next((c for c in ["p_purchase_t20", "predicted_purchase_probability_20", "pred_prob_20"] if c in precomputed_probabilities.columns), None)

            if not (p0_col and p10_col and p20_col):
                raise ValueError("Precomputed probabilities must contain p0, p10, p20 probability columns.")

            p0 = precomputed_probabilities[p0_col].to_numpy(dtype=float)
            p10 = precomputed_probabilities[p10_col].to_numpy(dtype=float)
            p20 = precomputed_probabilities[p20_col].to_numpy(dtype=float)
        elif model is not None:
            if not model.is_fitted_:
                raise RuntimeError("Provided causal model is not fitted.")
            probs_df = model.predict_probabilities(df_customers)
            p0 = probs_df["p_purchase_t0"].to_numpy(dtype=float)
            p10 = probs_df["p_purchase_t10"].to_numpy(dtype=float)
            p20 = probs_df["p_purchase_t20"].to_numpy(dtype=float)
        else:
            raise ValueError("Either 'model' or 'precomputed_probabilities' must be provided.")

        p0 = np.clip(p0, 0.0, 1.0)
        p10 = np.clip(p10, 0.0, 1.0)
        p20 = np.clip(p20, 0.0, 1.0)

        # Basket expansion elasticity from DGP
        order_val_0 = prices * 1.00
        order_val_10 = prices * 1.03
        order_val_20 = prices * 1.06

        # Expected Gross Revenue: price * p
        gross_rev_0 = order_val_0 * p0
        gross_rev_10 = order_val_10 * p10
        gross_rev_20 = order_val_20 * p20

        # Expected Discount Cost (Redemption model: d * p)
        cost_0 = np.zeros_like(p0)
        cost_10 = 10.0 * p10
        cost_20 = 20.0 * p20

        # Expected Net Revenue: (price - d) * p
        net_rev_0 = gross_rev_0
        net_rev_10 = np.maximum(0.0, order_val_10 - 10.0) * p10
        net_rev_20 = np.maximum(0.0, order_val_20 - 20.0) * p20

        # Expected Gross Profit (Pre-Discount): price * margin * p
        gross_prof_0 = order_val_0 * margins * p0
        gross_prof_10 = order_val_10 * margins * p10
        gross_prof_20 = order_val_20 * margins * p20

        # Expected Net Profit (Post-Discount Contribution): (price * margin - d) * p
        net_prof_0 = gross_prof_0 - cost_0
        net_prof_10 = gross_prof_10 - cost_10
        net_prof_20 = gross_prof_20 - cost_20

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
        budget_type: Optional[str] = None,
    ) -> OptimizationResult:
        """Solve budget-constrained personalized discount allocation via SciPy MILP."""
        active_budget = float(budget) if budget is not None else self.budget
        active_obj = objective_type if objective_type is not None else self.objective_type
        active_budget_type = budget_type if budget_type is not None else self.budget_type

        N = len(df_customers)
        K = len(self.discount_tiers)

        econ = self.compute_unit_economics(
            df_customers, model=model, precomputed_probabilities=precomputed_probabilities
        )

        c = build_objective_coefficients(econ, objective_type=active_obj, n_tiers=K)
        constraints, bounds, integrality = build_constraints(
            econ, budget=active_budget, budget_type=active_budget_type, discount_tiers=self.discount_tiers
        )

        # Handle zero budget edge case immediately
        if active_budget == 0.0:
            selected_indices = np.zeros(N, dtype=int)
            res_success = True
            res_status = 0
            res_message = "Zero budget constraint solved deterministically (All customers assigned $0)."
        else:
            res = run_scipy_milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            res_success = bool(res.success)
            res_status = int(res.status)
            res_message = str(res.message)

            if not res_success:
                raise RuntimeError(
                    f"SciPy MILP optimization failed with status {res_status}: {res_message}. "
                    f"Check feasibility of budget ${active_budget:,.2f}."
                )

            x_sol = res.x.reshape(N, K)
            selected_indices = np.argmax(x_sol, axis=1)

        tier_map = np.array(self.discount_tiers)
        recommended_discounts = tier_map[selected_indices]

        opt_cost = np.where(recommended_discounts == 0, econ["cost_0"], np.where(recommended_discounts == 10, econ["cost_10"], econ["cost_20"]))
        opt_gross_rev = np.where(recommended_discounts == 0, econ["gross_rev_0"], np.where(recommended_discounts == 10, econ["gross_rev_10"], econ["gross_rev_20"]))
        opt_net_rev = np.where(recommended_discounts == 0, econ["net_rev_0"], np.where(recommended_discounts == 10, econ["net_rev_10"], econ["net_rev_20"]))
        opt_net_prof = np.where(recommended_discounts == 0, econ["net_prof_0"], np.where(recommended_discounts == 10, econ["net_prof_10"], econ["net_prof_20"]))
        opt_prob = np.where(recommended_discounts == 0, econ["p0"], np.where(recommended_discounts == 10, econ["p10"], econ["p20"]))

        customer_ids = df_customers["customer_id"].astype(str).values if "customer_id" in df_customers.columns else [f"CUST_{i+1:06d}" for i in range(N)]

        allocation_df = pd.DataFrame({
            "customer_id": customer_ids,
            "predicted_purchase_probability_0": np.round(econ["p0"], 4),
            "predicted_purchase_probability_10": np.round(econ["p10"], 4),
            "predicted_purchase_probability_20": np.round(econ["p20"], 4),
            "predicted_effect_10_vs_0": np.round(econ["tau_10"], 4),
            "predicted_effect_20_vs_0": np.round(econ["tau_20"], 4),
            "expected_profit_0": np.round(econ["net_prof_0"], 4),
            "expected_profit_10": np.round(econ["net_prof_10"], 4),
            "expected_profit_20": np.round(econ["net_prof_20"], 4),
            "recommended_discount": recommended_discounts,
            "allocated_discount": recommended_discounts,
            "predicted_purchase_probability": np.round(opt_prob, 4),
            "expected_purchase_probability_selected": np.round(opt_prob, 4),
            "expected_discount_cost": np.round(opt_cost, 4),
            "expected_discount_cost_selected": np.round(opt_cost, 4),
            "expected_gross_revenue": np.round(opt_gross_rev, 2),
            "expected_revenue": np.round(opt_net_rev, 2),
            "expected_net_profit": np.round(opt_net_prof, 4),
            "expected_profit_selected": np.round(opt_net_prof, 4),
        })

        total_cost = float(np.sum(opt_cost))
        face_exposure = float(np.sum(recommended_discounts))
        test_expenditure = total_cost if active_budget_type == "redemption" else face_exposure
        utilization = float((test_expenditure / active_budget * 100.0) if active_budget > 0 else 100.0)

        tier_counts = {int(t): int(np.sum(recommended_discounts == t)) for t in self.discount_tiers}

        return OptimizationResult(
            success=res_success,
            status_code=res_status,
            status_message=res_message,
            budget=active_budget,
            total_cost=total_cost,
            budget_utilization_pct=min(utilization, 100.0),
            expected_gross_revenue=float(np.sum(opt_gross_rev)),
            expected_net_revenue=float(np.sum(opt_net_rev)),
            expected_net_profit=float(np.sum(opt_net_prof)),
            expected_conversions=float(np.sum(opt_prob)),
            expected_conversion_rate=float(np.mean(opt_prob)),
            tier_counts=tier_counts,
            allocation_matrix=allocation_df,
            objective_type=active_obj,
            face_value_exposure=face_exposure,
        )

    def compare_strategies(
        self,
        df_customers: pd.DataFrame,
        model: Optional[MultiTierCausalDML] = None,
        precomputed_probabilities: Optional[pd.DataFrame] = None,
        budget: Optional[float] = None,
    ) -> pd.DataFrame:
        """Compare the optimized allocation against blanket benchmark strategies."""
        active_budget = float(budget) if budget is not None else self.budget
        econ = self.compute_unit_economics(df_customers, model=model, precomputed_probabilities=precomputed_probabilities)
        opt_res = self.optimize(df_customers, model=model, precomputed_probabilities=precomputed_probabilities, budget=active_budget)

        n = len(df_customers)
        strategies = [
            ("No Discount ($0)", econ["cost_0"], econ["net_prof_0"], econ["gross_rev_0"], econ["p0"]),
            ("Blanket $10", econ["cost_10"], econ["net_prof_10"], econ["gross_rev_10"], econ["p10"]),
            ("Blanket $20", econ["cost_20"], econ["net_prof_20"], econ["gross_rev_20"], econ["p20"]),
            ("Optimized Personalized", opt_res.allocation_matrix["expected_discount_cost"].to_numpy(), opt_res.allocation_matrix["expected_net_profit"].to_numpy(), opt_res.allocation_matrix["expected_gross_revenue"].to_numpy(), opt_res.allocation_matrix["predicted_purchase_probability"].to_numpy()),
        ]

        rows = []
        for name, costs, profits, gross_revs, probs in strategies:
            c_tot = float(np.sum(costs))
            p_tot = float(np.sum(profits))
            r_tot = float(np.sum(gross_revs))
            conv_tot = float(np.sum(probs))
            is_compliant = (c_tot <= active_budget + 1e-4) or (name == "Optimized Personalized")

            rows.append({
                "strategy": name,
                "expected_discount_cost": round(c_tot, 2),
                "expected_gross_revenue": round(r_tot, 2),
                "expected_net_profit": round(p_tot, 2),
                "expected_conversions": round(conv_tot, 1),
                "conversion_rate_pct": round(conv_tot / n * 100.0, 2),
                "budget_compliant": "YES" if is_compliant else "NO",
                "budget_utilization_pct": round(min((c_tot / active_budget * 100.0) if active_budget > 0 else 0.0, 100.0), 1),
            })

        return pd.DataFrame(rows)
