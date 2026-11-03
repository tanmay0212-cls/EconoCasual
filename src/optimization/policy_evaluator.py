"""Policy Evaluation and Multi-Strategy Benchmarking Engine.

This module evaluates and benchmarks promotional strategies:
1. Blanket $0 (Organic Control)
2. Blanket $10 (Uniform Moderate Discount)
3. Blanket $20 (Uniform Deep Discount)
4. Personalized DML Policy (SciPy MILP allocation)
5. Oracle Optimal Policy (Retrospective benchmark using isolated ground-truth)

All metrics clearly distinguish expected business metrics from actual historical outcomes.
Infeasible strategies that exceed the marketing budget are explicitly flagged.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


def evaluate_strategies(
    economics_df: pd.DataFrame,
    optimized_allocation_df: pd.DataFrame,
    budget: float,
    budget_type: str = "redemption",
    ground_truth_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compare all baseline and optimized strategies on a customer cohort."""
    n_customers = len(economics_df)
    results = []

    # Strategy 1: Blanket $0
    cost_0 = float(economics_df["expected_discount_cost_0"].sum())
    face_0 = 0.0
    profit_0 = float(economics_df["expected_profit_0"].sum())
    purchases_0 = float(economics_df["predicted_purchase_probability_0"].sum())
    conv_0 = purchases_0 / n_customers
    test_budget_0 = cost_0 if budget_type == "redemption" else face_0
    is_feas_0 = test_budget_0 <= budget + 1e-3

    results.append({
        "strategy": "Blanket $0 (Organic Control)",
        "status": "FEASIBLE" if is_feas_0 else "INFEASIBLE",
        "expected_total_profit": round(profit_0, 2),
        "expected_profit_per_customer": round(profit_0 / n_customers, 2),
        "expected_purchases": round(purchases_0, 1),
        "expected_conversion_rate": round(conv_0 * 100, 2),
        "expected_redemption_cost": round(cost_0, 2),
        "issued_face_value_exposure": round(face_0, 2),
        "count_0": n_customers,
        "count_10": 0,
        "count_20": 0,
        "budget_utilization_pct": round((test_budget_0 / budget * 100) if budget > 0 else 0.0, 2),
    })

    # Strategy 2: Blanket $10
    cost_10 = float(economics_df["expected_discount_cost_10"].sum())
    face_10 = float(10.0 * n_customers)
    profit_10 = float(economics_df["expected_profit_10"].sum())
    purchases_10 = float(economics_df["predicted_purchase_probability_10"].sum())
    conv_10 = purchases_10 / n_customers
    test_budget_10 = cost_10 if budget_type == "redemption" else face_10
    is_feas_10 = test_budget_10 <= budget + 1e-3

    results.append({
        "strategy": "Blanket $10 (Uniform Moderate)",
        "status": "FEASIBLE" if is_feas_10 else "INFEASIBLE",
        "expected_total_profit": round(profit_10, 2),
        "expected_profit_per_customer": round(profit_10 / n_customers, 2),
        "expected_purchases": round(purchases_10, 1),
        "expected_conversion_rate": round(conv_10 * 100, 2),
        "expected_redemption_cost": round(cost_10, 2),
        "issued_face_value_exposure": round(face_10, 2),
        "count_0": 0,
        "count_10": n_customers,
        "count_20": 0,
        "budget_utilization_pct": round((test_budget_10 / budget * 100) if budget > 0 else 0.0, 2),
    })

    # Strategy 3: Blanket $20
    cost_20 = float(economics_df["expected_discount_cost_20"].sum())
    face_20 = float(20.0 * n_customers)
    profit_20 = float(economics_df["expected_profit_20"].sum())
    purchases_20 = float(economics_df["predicted_purchase_probability_20"].sum())
    conv_20 = purchases_20 / n_customers
    test_budget_20 = cost_20 if budget_type == "redemption" else face_20
    is_feas_20 = test_budget_20 <= budget + 1e-3

    results.append({
        "strategy": "Blanket $20 (Uniform Deep)",
        "status": "FEASIBLE" if is_feas_20 else "INFEASIBLE",
        "expected_total_profit": round(profit_20, 2),
        "expected_profit_per_customer": round(profit_20 / n_customers, 2),
        "expected_purchases": round(purchases_20, 1),
        "expected_conversion_rate": round(conv_20 * 100, 2),
        "expected_redemption_cost": round(cost_20, 2),
        "issued_face_value_exposure": round(face_20, 2),
        "count_0": 0,
        "count_10": 0,
        "count_20": n_customers,
        "budget_utilization_pct": round((test_budget_20 / budget * 100) if budget > 0 else 0.0, 2),
    })

    # Strategy 4: Personalized DML Allocation (SciPy MILP)
    alloc_col = "allocated_discount" if "allocated_discount" in optimized_allocation_df.columns else "recommended_discount"
    cost_col = "expected_discount_cost_selected" if "expected_discount_cost_selected" in optimized_allocation_df.columns else "expected_discount_cost"
    profit_col = "expected_profit_selected" if "expected_profit_selected" in optimized_allocation_df.columns else "expected_net_profit"
    prob_col = "expected_purchase_probability_selected" if "expected_purchase_probability_selected" in optimized_allocation_df.columns else "predicted_purchase_probability"

    discounts = optimized_allocation_df[alloc_col].to_numpy()
    c0_count = int(np.sum(discounts == 0))
    c10_count = int(np.sum(discounts == 10))
    c20_count = int(np.sum(discounts == 20))

    opt_cost = float(optimized_allocation_df[cost_col].sum())
    opt_face = float((discounts).sum())
    opt_profit = float(optimized_allocation_df[profit_col].sum())
    opt_purchases = float(optimized_allocation_df[prob_col].sum())
    opt_conv = opt_purchases / n_customers
    test_budget_opt = opt_cost if budget_type == "redemption" else opt_face
    is_feas_opt = test_budget_opt <= budget + 1e-3

    results.append({
        "strategy": "Personalized DML Allocation (SciPy MILP)",
        "status": "FEASIBLE" if is_feas_opt else "INFEASIBLE",
        "expected_total_profit": round(opt_profit, 2),
        "expected_profit_per_customer": round(opt_profit / n_customers, 2),
        "expected_purchases": round(opt_purchases, 1),
        "expected_conversion_rate": round(opt_conv * 100, 2),
        "expected_redemption_cost": round(opt_cost, 2),
        "issued_face_value_exposure": round(opt_face, 2),
        "count_0": c0_count,
        "count_10": c10_count,
        "count_20": c20_count,
        "budget_utilization_pct": round((test_budget_opt / budget * 100) if budget > 0 else 100.0, 2),
    })

    # Strategy 5: Oracle Optimal Policy (Synthetic Ground Truth Retrospective Benchmark)
    if ground_truth_df is not None:
        merged_gt = pd.merge(optimized_allocation_df[["customer_id", alloc_col]], ground_truth_df, on="customer_id")
        opt_decision_col = "optimal_unconstrained_discount" if "optimal_unconstrained_discount" in merged_gt.columns else "optimal_discount"
        
        oracle_decisions = merged_gt[opt_decision_col].to_numpy()
        oracle_c0 = int(np.sum(oracle_decisions == 0))
        oracle_c10 = int(np.sum(oracle_decisions == 10))
        oracle_c20 = int(np.sum(oracle_decisions == 20))

        p0_col = "expected_net_profit_0" if "expected_net_profit_0" in merged_gt.columns else "exp_profit_0_pot"
        p10_col = "expected_net_profit_10" if "expected_net_profit_10" in merged_gt.columns else "exp_profit_10_pot"
        p20_col = "expected_net_profit_20" if "expected_net_profit_20" in merged_gt.columns else "exp_profit_20_pot"

        oracle_profits = np.where(
            oracle_decisions == 0,
            merged_gt[p0_col],
            np.where(oracle_decisions == 10, merged_gt[p10_col], merged_gt[p20_col])
        )
        oracle_total_profit = float(oracle_profits.sum())

        prob0_col = "potential_purchase_prob_0" if "potential_purchase_prob_0" in merged_gt.columns else "pred_prob_0"
        prob10_col = "potential_purchase_prob_10" if "potential_purchase_prob_10" in merged_gt.columns else "pred_prob_10"
        prob20_col = "potential_purchase_prob_20" if "potential_purchase_prob_20" in merged_gt.columns else "pred_prob_20"

        oracle_probs = np.where(
            oracle_decisions == 0,
            merged_gt[prob0_col],
            np.where(oracle_decisions == 10, merged_gt[prob10_col], merged_gt[prob20_col])
        )
        oracle_cost = float((oracle_decisions * oracle_probs).sum())
        oracle_face = float(oracle_decisions.sum())
        oracle_purchases = float(oracle_probs.sum())

        test_budget_oracle = oracle_cost if budget_type == "redemption" else oracle_face
        is_feas_oracle = test_budget_oracle <= budget + 1e-3

        results.append({
            "strategy": "Oracle Optimal Policy (Ground Truth Upper Bound)",
            "status": "FEASIBLE" if is_feas_oracle else "INFEASIBLE",
            "expected_total_profit": round(oracle_total_profit, 2),
            "expected_profit_per_customer": round(oracle_total_profit / n_customers, 2),
            "expected_purchases": round(oracle_purchases, 1),
            "expected_conversion_rate": round(oracle_purchases / n_customers * 100, 2),
            "expected_redemption_cost": round(oracle_cost, 2),
            "issued_face_value_exposure": round(oracle_face, 2),
            "count_0": oracle_c0,
            "count_10": oracle_c10,
            "count_20": oracle_c20,
            "budget_utilization_pct": round((test_budget_oracle / budget * 100) if budget > 0 else 0.0, 2),
        })

    return pd.DataFrame(results)

