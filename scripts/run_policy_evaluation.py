"""Standalone Runner for Multi-Treatment Off-Policy Evaluation (OPE).

Evaluates 5 candidate promotional discount policies on held-out test data (N = 6,005):
1. No-Discount Policy ($0 for all)
2. Flat $10 Discount Policy ($10 for all)
3. Random Discount Policy (uniform stochastic assignment over {0, 10, 20})
4. Existing DML + MILP Prescriptive Policy (SciPy HiGHS under $5,000 budget cap)
5. Future Uncertainty-Aware Policy Placeholder (clearly marked unavailable)

Saves full evaluation metrics, confidence intervals, diagnostics, and ground-truth
comparisons to `experiments/results/policy_evaluation.json`.
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Add repository root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.policy_comparison import MultiTreatmentPolicyComparator


def compute_dgp_propensities(df: pd.DataFrame) -> np.ndarray:
    """Compute true logging assignment probabilities P(T=d | X) using the known synthetic DGP."""
    n = len(df)
    activity = df["customer_activity_score"].to_numpy(dtype=float)
    loyalty = df["loyalty_score"].to_numpy(dtype=float)
    aov = df["average_previous_order_value"].to_numpy(dtype=float)
    segment = df["customer_segment"].to_numpy(dtype=str)

    logit_0 = np.zeros(n)

    logit_10 = (
        -0.10
        + 0.015 * (activity - 50.0)
        + 0.010 * (loyalty - 50.0)
        + 0.35 * (segment == "regular")
        + 0.20 * (segment == "budget")
        - 0.30 * (segment == "vip")
    )

    logit_20 = (
        -0.35
        + 0.020 * (activity - 50.0)
        + 0.015 * (aov - 90.0) / 30.0
        + 0.40 * (segment == "premium")
        + 0.55 * (segment == "vip")
        - 0.45 * (segment == "budget")
    )

    logits_matrix = np.column_stack([logit_0, logit_10, logit_20])
    exp_z = np.exp(logits_matrix - np.max(logits_matrix, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def main():
    print("=" * 80)
    print("ECONOCASUAL 2.0 — PHASE 1: DOUBLY ROBUST POLICY EVALUATION RUNNER")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # 1. Paths
    test_path = ROOT_DIR / "data" / "synthetic_retail" / "synthetic_retail_test.csv"
    preds_path = ROOT_DIR / "data" / "synthetic_retail" / "synthetic_counterfactual_predictions.csv"
    gt_path = ROOT_DIR / "data" / "synthetic_retail" / "synthetic_retail_ground_truth.csv"
    alloc_path = ROOT_DIR / "data" / "optimization" / "customer_allocation.csv"
    output_dir = ROOT_DIR / "experiments" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json_path = output_dir / "policy_evaluation.json"

    # 2. Load Datasets
    print("\n[1/5] Loading held-out evaluation datasets...")
    df_test = pd.read_csv(test_path)
    df_preds = pd.read_csv(preds_path)
    df_gt = pd.read_csv(gt_path)
    df_alloc = pd.read_csv(alloc_path)

    n_eval = len(df_test)
    print(f"  Evaluation customers (N): {n_eval:,}")

    # Validate customer_id alignment
    assert (df_test["customer_id"] == df_preds["customer_id"]).all(), "Customer ID mismatch in predictions!"
    assert (df_test["customer_id"] == df_alloc["customer_id"]).all(), "Customer ID mismatch in allocations!"

    # Align ground truth by customer_id
    gt_test = df_test[["customer_id"]].merge(df_gt, on="customer_id", how="left")

    # 3. Covariates and Logging Propensities
    print("\n[2/5] Computing logging treatment propensities...")
    propensities = compute_dgp_propensities(df_test)
    print(f"  Propensity matrix shape: {propensities.shape}")
    print(f"  P(T=$0 | X)  range: [{propensities[:, 0].min():.4f}, {propensities[:, 0].max():.4f}], mean: {propensities[:, 0].mean():.4f}")
    print(f"  P(T=$10 | X) range: [{propensities[:, 1].min():.4f}, {propensities[:, 1].max():.4f}], mean: {propensities[:, 1].mean():.4f}")
    print(f"  P(T=$20 | X) range: [{propensities[:, 2].min():.4f}, {propensities[:, 2].max():.4f}], mean: {propensities[:, 2].mean():.4f}")

    # 4. Model Predictions Matrix (N, 3)
    pred_probs = df_preds[["pred_prob_0", "pred_prob_10", "pred_prob_20"]].to_numpy(dtype=float)

    # 5. DML + MILP Prescriptive Actions
    dml_milp_actions = df_alloc["allocated_discount"].to_numpy(dtype=int)

    # 6. Initialize Policy Comparator
    print("\n[3/5] Initializing MultiTreatmentPolicyComparator...")
    comparator = MultiTreatmentPolicyComparator(
        treatment_classes=[0, 10, 20],
        min_propensity=0.01,
        n_bootstraps=500,
        random_state=42,
    )

    # 7. Evaluate All 5 Policies
    print("\n[4/5] Executing Doubly Robust Policy Evaluation & Bootstrap CI (500 replicates)...")
    results = comparator.evaluate_all_policies(
        eval_df=df_test,
        propensities=propensities,
        pred_probs=pred_probs,
        dml_milp_actions=dml_milp_actions,
        ground_truth_df=gt_test,
    )

    # 8. Print Summary Report Table
    print("\n" + "=" * 115)
    print("POLICY EVALUATION SCORECARD (HELD-OUT TEST SET, N = 6,005)")
    print("=" * 115)
    header = f"{'Policy':<34} | {'Status':<10} | {'DR Conv %':<10} | {'DR Profit ($)':<15} | {'Model Prof ($)':<14} | {'GT Prof ($)':<13} | {'Cost ($)':<10} | {'ROI %':<8}"
    print(header)
    print("-" * 115)

    json_payload = {
        "metadata": {
            "evaluation_timestamp": datetime.now().isoformat(),
            "evaluation_sample_size": n_eval,
            "treatment_classes": [0, 10, 20],
            "n_bootstraps": 500,
            "min_propensity_clip": 0.01,
            "random_seed": 42,
            "canonical_milestone3_benchmarks": {
                "organic_no_discount_profit": 90586.38,
                "dml_milp_expected_profit": 100786.25,
                "dml_milp_incremental_profit": 10199.87,
                "dml_milp_expected_cost": 4998.74,
                "dml_milp_incremental_profit_to_cost_pct": 204.05,
                "oracle_upper_bound_profit": 103994.65,
                "optimality_gap_pct": 0.986,
            },
        },
        "policies": {},
    }

    for key, summary in results.items():
        json_payload["policies"][key] = summary.to_dict()

        if summary.is_available:
            dr_conv_str = f"{summary.dr_conversion_rate * 100.0:.2f}%"
            dr_prof_str = f"${summary.dr_expected_profit:,.2f}"
            mod_prof_str = f"${summary.model_expected_profit:,.2f}"
            gt_prof_str = f"${summary.ground_truth_expected_profit:,.2f}" if summary.ground_truth_expected_profit else "N/A"
            cost_str = f"${summary.expected_promotional_cost:,.2f}"
            roi_str = f"{summary.incremental_profit_to_cost_ratio:.1f}%" if summary.incremental_profit_to_cost_ratio is not None else "N/A"
        else:
            dr_conv_str = "N/A"
            dr_prof_str = "N/A"
            mod_prof_str = "N/A"
            gt_prof_str = "N/A"
            cost_str = "N/A"
            roi_str = "N/A"

        print(f"{summary.policy_name:<34} | {summary.status:<10} | {dr_conv_str:<10} | {dr_prof_str:<15} | {mod_prof_str:<14} | {gt_prof_str:<13} | {cost_str:<10} | {roi_str:<8}")

    print("-" * 115)

    # 9. Export to experiments/results/policy_evaluation.json
    print(f"\n[5/5] Exporting results to: {output_json_path}")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    print(f"[SUCCESS] Off-Policy Evaluation Complete. Artifact created: {output_json_path}")
    return json_payload


if __name__ == "__main__":
    main()
