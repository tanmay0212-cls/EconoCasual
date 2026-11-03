"""Script to run baseline strategy comparison and Oracle benchmark evaluation."""

import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.optimization.budget_optimizer import SciPyBudgetOptimizer
from src.optimization.profit_calculator import ProfitCalculator
from src.optimization.policy_evaluator import evaluate_strategies
from src.optimization.sensitivity_analysis import plot_policy_comparison_bars


def main():
    print("=" * 80)
    print("RUNNING MULTI-POLICY BENCHMARK COMPARISON & ORACLE EVALUATION")
    print("=" * 80)

    test_path = "data/synthetic_retail/synthetic_retail_test.csv"
    preds_path = "data/synthetic_retail/synthetic_counterfactual_predictions.csv"
    gt_path = "data/synthetic_retail/synthetic_retail_ground_truth.csv"
    output_dir = "data/optimization"
    os.makedirs(output_dir, exist_ok=True)

    df_customers = pd.read_csv(test_path)
    df_preds = pd.read_csv(preds_path)
    df_gt = pd.read_csv(gt_path)

    # Calculate customer economics using ProfitCalculator
    profit_calc = ProfitCalculator()
    merged_cust_preds = pd.merge(df_customers, df_preds[["customer_id", "pred_prob_0", "pred_prob_10", "pred_prob_20"]], on="customer_id")
    econ_df = profit_calc.calculate_customer_economics(merged_cust_preds)

    # 1. Primary Evaluation at Default $5,000 Budget
    budget_default = 5000.0
    optimizer = SciPyBudgetOptimizer(budget=budget_default, budget_type="redemption")
    opt_res = optimizer.optimize(df_customers, precomputed_probabilities=df_preds)

    comp_df_5k = evaluate_strategies(
        econ_df,
        opt_res.allocation_matrix,
        budget=budget_default,
        budget_type="redemption",
        ground_truth_df=df_gt,
    )

    print(f"\n[Strategy Comparison at Default Budget = ${budget_default:,.2f}]")
    print(comp_df_5k[["strategy", "status", "expected_total_profit", "expected_profit_per_customer", "expected_redemption_cost", "expected_conversion_rate", "budget_utilization_pct"]].to_string(index=False))

    plot_policy_comparison_bars(comp_df_5k, "reports/figures/policy_comparison_bars.png")

    # 2. Multi-Budget Comparison ($1,000, $2,500, $5,000, $10,000)
    test_budgets = [1000.0, 2500.0, 5000.0, 10000.0]
    multi_budget_rows = []

    print(f"\nEvaluating Across Specific Budget Tiers: {test_budgets}")
    for b in test_budgets:
        opt_b = SciPyBudgetOptimizer(budget=b, budget_type="redemption")
        res_b = opt_b.optimize(df_customers, precomputed_probabilities=df_preds)
        
        comp_b = evaluate_strategies(
            econ_df,
            res_b.allocation_matrix,
            budget=b,
            budget_type="redemption",
            ground_truth_df=df_gt,
        )
        comp_b.insert(0, "budget_cap", b)
        multi_budget_rows.append(comp_b)

    all_comparisons = pd.concat(multi_budget_rows, ignore_index=True)
    oracle_csv_path = os.path.join(output_dir, "oracle_evaluation.csv")
    all_comparisons.to_csv(oracle_csv_path, index=False)
    print(f"\n[OK] Oracle and multi-policy comparison exported to: {oracle_csv_path}")


if __name__ == "__main__":
    main()

