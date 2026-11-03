"""Script to run multi-budget sensitivity analysis and generate frontier visualizations."""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.optimization.sensitivity_analysis import run_budget_sensitivity, plot_sensitivity_curves


def main():
    print("=" * 80)
    print("RUNNING BUDGET SENSITIVITY ANALYSIS & FRONTIER MODELING")
    print("=" * 80)

    test_path = "data/synthetic_retail/synthetic_retail_test.csv"
    preds_path = "data/synthetic_retail/synthetic_counterfactual_predictions.csv"
    gt_path = "data/synthetic_retail/synthetic_retail_ground_truth.csv"
    output_dir = "data/optimization"
    figures_dir = "reports/figures"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    df_customers = pd.read_csv(test_path)
    df_preds = pd.read_csv(preds_path)
    df_gt = pd.read_csv(gt_path)

    # Compute Oracle unconstrained total profit on test set
    merged_gt = pd.merge(df_customers[["customer_id"]], df_gt, on="customer_id")
    opt_col = "optimal_unconstrained_discount" if "optimal_unconstrained_discount" in merged_gt.columns else "optimal_discount"
    p0_col = "expected_net_profit_0" if "expected_net_profit_0" in merged_gt.columns else "exp_profit_0_pot"
    p10_col = "expected_net_profit_10" if "expected_net_profit_10" in merged_gt.columns else "exp_profit_10_pot"
    p20_col = "expected_net_profit_20" if "expected_net_profit_20" in merged_gt.columns else "exp_profit_20_pot"
    
    oracle_decisions = merged_gt[opt_col].to_numpy()
    oracle_profits = np.where(
        oracle_decisions == 0,
        merged_gt[p0_col],
        np.where(oracle_decisions == 10, merged_gt[p10_col], merged_gt[p20_col])
    )
    oracle_total_profit = float(oracle_profits.sum())

    budget_sweep = [500.0, 1000.0, 2000.0, 2500.0, 3500.0, 5000.0, 7500.0, 10000.0, 15000.0, 25000.0]
    print(f"Sweeping {len(budget_sweep)} budget levels from ${budget_sweep[0]:,.0f} to ${budget_sweep[-1]:,.0f}...")

    sens_df = run_budget_sensitivity(
        customer_df=df_customers,
        precomputed_probs=df_preds,
        budgets=budget_sweep,
        budget_type="redemption",
        objective_type="net_profit",
    )

    print("\n[Budget Sensitivity Summary Table]")
    print(sens_df[["budget", "expected_total_profit", "expected_profit_per_customer", "expected_redemption_cost", "budget_utilization_pct", "count_0", "count_10", "count_20"]].to_string(index=False))

    sens_csv_path = os.path.join(output_dir, "budget_sensitivity.csv")
    sens_df.to_csv(sens_csv_path, index=False)
    print(f"\n[OK] Budget sensitivity table exported: {sens_csv_path}")

    # Generate Visualization Charts
    plot_sensitivity_curves(
        sensitivity_df=sens_df,
        oracle_profit=oracle_total_profit,
        output_dir=figures_dir,
    )
    print(f"[OK] Sensitivity figures generated in: {figures_dir}")


if __name__ == "__main__":
    main()
