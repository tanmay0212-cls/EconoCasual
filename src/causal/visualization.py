"""Publication-Quality Visualizations for Multi-Treatment Causal Inference.

Generates all Milestone 2 figures:
1. Treatment group distribution charts
2. Estimated CATE distributions
3. True vs Estimated CATE scatter & calibration
4. Multi-arm uplift curves
5. Multi-arm Qini curves
6. Counterfactual purchase probability comparisons
7. Hillstrom experimental vs estimated treatment effect comparisons
8. Policy value comparison vs Oracle
"""

from typing import Dict, Any, List, Optional
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_treatment_distributions(
    df_synthetic: pd.DataFrame,
    df_hillstrom: pd.DataFrame,
    output_path: str = "reports/figures/treatment_distributions.png",
):
    """Plot treatment assignment distributions for both datasets side-by-side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Synthetic Retail
    syn_counts = df_synthetic["discount_amount"].value_counts().sort_index()
    labels_syn = ["$0 (Control)", "$10 Discount", "$20 Discount"]
    colors_syn = ["#3b82f6", "#10b981", "#8b5cf6"]
    bars1 = ax1.bar(labels_syn, syn_counts.values, color=colors_syn, alpha=0.85, edgecolor="white", width=0.55)
    ax1.set_title("Multi-Treatment Synthetic Retail (Confounded)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Customer Count", fontsize=11)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    for bar in bars1:
        yval = bar.get_height()
        pct = yval / len(df_synthetic) * 100
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 350, f"{yval:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)
    ax1.set_ylim(0, max(syn_counts.values) * 1.18)

    # Hillstrom RCT
    hill_map = {0: "No E-Mail\n(Control)", 1: "Men's\nE-Mail", 2: "Women's\nE-Mail"}
    hill_counts = df_hillstrom["treatment_group"].map(hill_map).value_counts()[list(hill_map.values())]
    colors_hill = ["#6b7280", "#0284c7", "#ec4899"]
    bars2 = ax2.bar(list(hill_map.values()), hill_counts.values, color=colors_hill, alpha=0.85, edgecolor="white", width=0.55)
    ax2.set_title("Hillstrom MineThatData (Randomized RCT)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Customer Count", fontsize=11)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    for bar in bars2:
        yval = bar.get_height()
        pct = yval / len(df_hillstrom) * 100
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 500, f"{yval:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)
    ax2.set_ylim(0, max(hill_counts.values) * 1.18)

    plt.suptitle("Experimental & Observational Treatment Allocations", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Figure] Treatment distributions saved: {output_path}")


def plot_synthetic_cate_distributions(
    preds_df: pd.DataFrame,
    output_path: str = "reports/figures/synthetic_cate_distributions.png",
):
    """Plot customer-level heterogeneous treatment effect (CATE) distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ite10 = preds_df["pred_ite_10_vs_0"] * 100.0
    ite20 = preds_df["pred_ite_20_vs_0"] * 100.0

    # $10 vs $0
    ax1.hist(ite10, bins=35, color="#10b981", alpha=0.75, edgecolor="white", density=True)
    ax1.axvline(ite10.mean(), color="#047857", linestyle="--", linewidth=2, label=f"Mean ATE: +{ite10.mean():.2f} pp")
    ax1.axvline(np.median(ite10), color="#d97706", linestyle=":", linewidth=2, label=f"Median: +{np.median(ite10):.2f} pp")
    ax1.set_title("Predicted CATE: $10 Discount vs $0", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Conversion Lift (Percentage Points)", fontsize=11)
    ax1.set_ylabel("Customer Density", fontsize=11)
    ax1.legend(frameon=True)
    ax1.grid(alpha=0.3, linestyle="--")

    # $20 vs $0
    ax2.hist(ite20, bins=35, color="#8b5cf6", alpha=0.75, edgecolor="white", density=True)
    ax2.axvline(ite20.mean(), color="#5b21b6", linestyle="--", linewidth=2, label=f"Mean ATE: +{ite20.mean():.2f} pp")
    ax2.axvline(np.median(ite20), color="#d97706", linestyle=":", linewidth=2, label=f"Median: +{np.median(ite20):.2f} pp")
    ax2.set_title("Predicted CATE: $20 Discount vs $0", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Conversion Lift (Percentage Points)", fontsize=11)
    ax2.set_ylabel("Customer Density", fontsize=11)
    ax2.legend(frameon=True)
    ax2.grid(alpha=0.3, linestyle="--")

    plt.suptitle("CausalForestDML: Heterogeneous Treatment Effect (CATE) Distributions", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Figure] CATE distributions saved: {output_path}")


def plot_true_vs_estimated_cate(
    merged_test_df: pd.DataFrame,
    output_path: str = "reports/figures/synthetic_true_vs_estimated_cate.png",
):
    """Plot scatter and calibration lines comparing estimated vs ground-truth CATE."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # $10 vs $0
    true10 = merged_test_df["true_ite_10_vs_0"] * 100.0
    pred10 = merged_test_df["pred_ite_10_vs_0"] * 100.0
    corr10 = float(true10.corr(pred10))

    ax1.scatter(true10, pred10, alpha=0.18, color="#059669", s=18, edgecolors="none")
    ax1.plot([0, 45], [0, 45], color="#dc2626", linestyle="--", linewidth=1.8, label="45° Perfect Oracle Line")
    ax1.set_title(f"ITE Calibration: $10 vs $0 (Pearson r = {corr10:.3f})", fontsize=13, fontweight="bold")
    ax1.set_xlabel("True Ground-Truth ITE (pp)", fontsize=11)
    ax1.set_ylabel("CausalForestDML Predicted ITE (pp)", fontsize=11)
    ax1.set_xlim(0, 40)
    ax1.set_ylim(0, 40)
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(alpha=0.3, linestyle="--")

    # $20 vs $0
    true20 = merged_test_df["true_ite_20_vs_0"] * 100.0
    pred20 = merged_test_df["pred_ite_20_vs_0"] * 100.0
    corr20 = float(true20.corr(pred20))

    ax2.scatter(true20, pred20, alpha=0.18, color="#7c3aed", s=18, edgecolors="none")
    ax2.plot([0, 50], [0, 50], color="#dc2626", linestyle="--", linewidth=1.8, label="45° Perfect Oracle Line")
    ax2.set_title(f"ITE Calibration: $20 vs $0 (Pearson r = {corr20:.3f})", fontsize=13, fontweight="bold")
    ax2.set_xlabel("True Ground-Truth ITE (pp)", fontsize=11)
    ax2.set_ylabel("CausalForestDML Predicted ITE (pp)", fontsize=11)
    ax2.set_xlim(0, 50)
    ax2.set_ylim(0, 50)
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(alpha=0.3, linestyle="--")

    plt.suptitle("Out-of-Sample CATE Estimation vs Ground Truth (Test Set N=6,005)", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Figure] True vs Estimated CATE saved: {output_path}")


def plot_counterfactual_probabilities(
    preds_df: pd.DataFrame,
    output_path: str = "reports/figures/synthetic_counterfactual_probabilities.png",
):
    """Plot customer purchase probability distributions under all three potential policies."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    p0 = preds_df["pred_prob_0"] * 100.0
    p10 = preds_df["pred_prob_10"] * 100.0
    p20 = preds_df["pred_prob_20"] * 100.0

    ax.hist(p0, bins=35, alpha=0.55, color="#3b82f6", label=f"P(Y|T=$0) [Mean: {p0.mean():.1f}%]", density=True, edgecolor="white")
    ax.hist(p10, bins=35, alpha=0.55, color="#10b981", label=f"P(Y|T=$10) [Mean: {p10.mean():.1f}%]", density=True, edgecolor="white")
    ax.hist(p20, bins=35, alpha=0.55, color="#8b5cf6", label=f"P(Y|T=$20) [Mean: {p20.mean():.1f}%]", density=True, edgecolor="white")

    ax.set_title("Customer Purchase Probability Distributions Under Potential Interventions", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Purchase Probability (%)", fontsize=11)
    ax.set_ylabel("Customer Density", fontsize=11)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_xlim(0, 100)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Figure] Counterfactual probabilities saved: {output_path}")


def plot_uplift_and_qini_curves(
    uplift_10: pd.DataFrame,
    uplift_20: pd.DataFrame,
    output_uplift: str = "reports/figures/multi_arm_uplift_curves.png",
    output_qini: str = "reports/figures/multi_arm_qini_curves.png",
):
    """Plot cumulative uplift and Qini curves for multi-treatment promotional tiers."""
    # 1. Cumulative Uplift
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = uplift_10["population_fraction"] * 100.0
    u10 = uplift_10["cumulative_uplift"] * 100.0
    u20 = uplift_20["cumulative_uplift"] * 100.0

    ax.plot(x, u10, marker="o", color="#10b981", linewidth=2.2, label="$10 vs $0 Cumulative Lift")
    ax.plot(x, u20, marker="s", color="#8b5cf6", linewidth=2.2, label="$20 vs $0 Cumulative Lift")
    ax.axhline(u10.iloc[-1], color="#047857", linestyle="--", alpha=0.7, label=f"$10 Overall ATE (+{u10.iloc[-1]:.1f} pp)")
    ax.axhline(u20.iloc[-1], color="#5b21b6", linestyle="--", alpha=0.7, label=f"$20 Overall ATE (+{u20.iloc[-1]:.1f} pp)")

    ax.set_title("Multi-Arm Cumulative Uplift Curves (Ranked by CATE)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Top Population Targeted (%)", fontsize=11)
    ax.set_ylabel("Cumulative Incremental Conversion (pp)", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_uplift, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Figure] Uplift curves saved: {output_uplift}")

    # 2. Qini Curves
    fig, ax = plt.subplots(figsize=(9, 5.5))
    q10 = uplift_10["qini_value"]
    q20 = uplift_20["qini_value"]

    ax.plot(x, q10, marker="o", color="#10b981", linewidth=2.2, label="$10 Promotion Qini Curve")
    ax.plot(x, q20, marker="s", color="#8b5cf6", linewidth=2.2, label="$20 Promotion Qini Curve")
    # Random targeting baselines
    ax.plot([0, 100], [0, q10.iloc[-1]], color="#6b7280", linestyle=":", label="Random Baseline ($10)")
    ax.plot([0, 100], [0, q20.iloc[-1]], color="#9ca3af", linestyle="--", label="Random Baseline ($20)")

    ax.set_title("Multi-Arm Qini Curves (Incremental Converted Customers)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Top Population Targeted (%)", fontsize=11)
    ax.set_ylabel("Incremental Buyers (Qini Metric)", fontsize=11)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_qini, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Figure] Qini curves saved: {output_qini}")


def plot_hillstrom_ate_comparison(
    hillstrom_results: Dict[str, Any],
    output_path: str = "reports/figures/hillstrom_ate_comparison.png",
):
    """Plot Hillstrom RCT Empirical ATE vs DML Estimated ATE across visit, conversion, and spend."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    outcomes = ["visit", "conversion", "spend"]
    units = ["pp", "pp", "$"]
    scales = [100.0, 100.0, 1.0]

    for idx, (out, unit, scale) in enumerate(zip(outcomes, units, scales)):
        ax = axes[idx]
        data = hillstrom_results[out]
        
        emp_1 = data["empirical_ate"]["mens_vs_control"] * scale
        emp_2 = data["empirical_ate"]["womens_vs_control"] * scale
        pred_1 = data["model_estimated_ate"]["mens_vs_control"] * scale
        pred_2 = data["model_estimated_ate"]["womens_vs_control"] * scale

        x = np.arange(2)
        w = 0.35

        ax.bar(x - w/2, [emp_1, emp_2], width=w, label="Gold Standard RCT Difference", color="#0284c7", alpha=0.85)
        ax.bar(x + w/2, [pred_1, pred_2], width=w, label="LinearDML Estimated ATE", color="#10b981", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(["Men's vs Control", "Women's vs Control"], fontsize=10)
        ax.set_title(f"Outcome: {out.upper()} ({unit})", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"Average Treatment Effect ({unit})", fontsize=10)
        ax.legend(loc="upper right", frameon=True, fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.suptitle("Hillstrom RCT Benchmark: Empirical Difference-in-Means vs DML Causal Estimates", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Figure] Hillstrom ATE comparison saved: {output_path}")


def plot_policy_value_comparison(
    policy_metrics: Dict[str, Any],
    output_path: str = "reports/figures/synthetic_model_policy_value.png",
):
    """Plot expected profit per customer across policies."""
    fig, ax = plt.subplots(figsize=(9, 5))

    policies = [
        "Blanket $0\n(Organic)",
        "Blanket $10\n(Moderate)",
        "Blanket $20\n(Deep)",
        "DML Personalized\nPolicy",
        "Oracle Optimal\nPolicy",
    ]
    profits = [
        policy_metrics["blanket_0_profit_per_customer"],
        policy_metrics["blanket_10_profit_per_customer"],
        policy_metrics["blanket_20_profit_per_customer"],
        policy_metrics["model_policy_profit_per_customer"],
        policy_metrics["oracle_optimal_profit_per_customer"],
    ]
    colors = ["#6b7280", "#3b82f6", "#ef4444", "#10b981", "#8b5cf6"]

    bars = ax.bar(policies, profits, color=colors, width=0.55, alpha=0.88, edgecolor="white")
    ax.set_title("Expected Net Contribution per Customer Across Strategic Policies", fontsize=13, fontweight="bold")
    ax.set_ylabel("Expected Net Profit ($ / Customer)", fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.35, f"${yval:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    ax.set_ylim(0, max(profits) * 1.18)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Figure] Policy value comparison saved: {output_path}")

