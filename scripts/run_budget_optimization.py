"""Script to run SciPy MILP budget optimization and export customer allocation matrix."""

import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.optimization.budget_optimizer import SciPyBudgetOptimizer
from src.optimization.allocation_validator import validate_allocation


def main():
    print("=" * 80)
    print("RUNNING BUDGET-CONSTRAINED PERSONALIZED DISCOUNT OPTIMIZATION")
    print("=" * 80)

    test_path = "data/synthetic_retail/synthetic_retail_test.csv"
    preds_path = "data/synthetic_retail/synthetic_counterfactual_predictions.csv"
    output_dir = "data/optimization"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading test cohort from: {test_path}")
    df_customers = pd.read_csv(test_path)
    df_preds = pd.read_csv(preds_path)
    print(f"Loaded {len(df_customers):,} customer records.")

    budget_val = 5000.0
    print(f"\nConfiguring SciPy MILP Optimizer:")
    print(f"  - Marketing Budget Cap: ${budget_val:,.2f}")
    print(f"  - Budget Interpretation: Expected Redemption Cost (sum d * P(Y=1|d) * x_{{i, d}} <= Budget)")
    print(f"  - Objective: Maximize Expected Net Contribution (Post-Discount Net Profit)")
    print(f"  - Decision Variables: Binary x_{{i, d}} in {{0, 1}} for d in {{0, 10, 20}}")

    optimizer = SciPyBudgetOptimizer(
        budget=budget_val,
        discount_tiers=[0, 10, 20],
        objective_type="net_profit",
        budget_type="redemption",
    )

    opt_res = optimizer.optimize(df_customers, precomputed_probabilities=df_preds)
    print(f"\n[Optimization Solution Achieved: Status {opt_res.status_code}]")
    print(f"  - Total Expected Net Profit:       ${opt_res.expected_net_profit:,.2f}")
    print(f"  - Expected Profit per Customer:     ${opt_res.expected_net_profit / len(df_customers):.2f}")
    print(f"  - Total Expected Redemption Cost:   ${opt_res.total_cost:,.2f} / ${budget_val:,.2f}")
    print(f"  - Budget Utilization:               {opt_res.budget_utilization_pct:.2f}%")
    print(f"  - Issued Face-Value Exposure:       ${opt_res.face_value_exposure:,.2f}")
    print(f"  - Expected Customer Purchases:      {opt_res.expected_conversions:,.1f} ({opt_res.expected_conversion_rate*100:.2f}% conversion rate)")
    print(f"  - Customer Allocations:             $0={opt_res.tier_counts[0]:,} ({opt_res.tier_counts[0]/len(df_customers)*100:.1f}%), $10={opt_res.tier_counts[10]:,} ({opt_res.tier_counts[10]/len(df_customers)*100:.1f}%), $20={opt_res.tier_counts[20]:,} ({opt_res.tier_counts[20]/len(df_customers)*100:.1f}%)")

    # Validate Allocation Matrix
    alloc_mat = opt_res.allocation_matrix
    val_report = validate_allocation(alloc_mat, budget=budget_val, budget_type="redemption")
    print(f"\n[Allocation Validation Check]")
    print(f"  - Is Valid: {val_report['valid']}")
    print(f"  - Budget Satisfied: {val_report['budget_satisfied']}")
    print(f"  - Anti-Leakage Verified (0 ground-truth columns): {val_report['no_leakage_detected']}")
    assert val_report["valid"], f"Allocation failed validation: {val_report['issues']}"

    # Export Operational Customer Allocation Matrix (Part H specification)
    part_h_cols = [
        "customer_id",
        "predicted_purchase_probability_0",
        "predicted_purchase_probability_10",
        "predicted_purchase_probability_20",
        "expected_profit_0",
        "expected_profit_10",
        "expected_profit_20",
        "allocated_discount",
        "expected_purchase_probability_selected",
        "expected_profit_selected",
        "expected_discount_cost_selected",
    ]
    operational_df = alloc_mat[part_h_cols]
    alloc_csv_path = os.path.join(output_dir, "customer_allocation.csv")
    operational_df.to_csv(alloc_csv_path, index=False)
    print(f"\n[OK] Operational allocation matrix exported: {alloc_csv_path} ({len(operational_df):,} rows)")

    # Export Optimization Summary CSV
    summary_df = pd.DataFrame([{
        "budget": budget_val,
        "budget_type": "redemption",
        "status": "Optimal",
        "total_expected_profit": round(opt_res.expected_net_profit, 2),
        "profit_per_customer": round(opt_res.expected_net_profit / len(df_customers), 2),
        "expected_redemption_cost": round(opt_res.total_cost, 2),
        "issued_face_value_exposure": round(opt_res.face_value_exposure, 2),
        "budget_utilization_pct": round(opt_res.budget_utilization_pct, 2),
        "expected_conversions": round(opt_res.expected_conversions, 1),
        "expected_conversion_rate_pct": round(opt_res.expected_conversion_rate * 100, 2),
        "count_discount_0": opt_res.tier_counts[0],
        "count_discount_10": opt_res.tier_counts[10],
        "count_discount_20": opt_res.tier_counts[20],
    }])
    summary_csv_path = os.path.join(output_dir, "optimization_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"[OK] Optimization summary exported: {summary_csv_path}")


if __name__ == "__main__":
    main()
