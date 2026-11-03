"""Budget Sensitivity Analysis and Frontier Visualizations.

This module evaluates optimal discount allocations across varying marketing budgets:
- Sweeps budget caps from tight budgets ($500) to unconstrained frontiers ($25,000+)
- Evaluates diminishing marginal returns and profit concavity
- Generates publication-ready figures for reports/figures/
"""

from typing import Dict, Any, List, Optional
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .budget_optimizer import SciPyBudgetOptimizer


def run_budget_sensitivity(
    customer_df: pd.DataFrame,
    precomputed_probs: Optional[pd.DataFrame] = None,
    budgets: Optional[List[float]] = None,
    budget_type: str = "redemption",
    objective_type: str = "net_profit",
) -> pd.DataFrame:
    """Run MILP optimization across a spectrum of budget caps."""
    if budgets is None:
        budgets = [500.0, 1000.0, 2000.0, 2500.0, 3500.0, 5000.0, 7500.0, 10000.0, 15000.0, 25000.0, 50000.0]

    records = []
    n_customers = len(customer_df)

    for b in budgets:
        optimizer = SciPyBudgetOptimizer(budget=b, budget_type=budget_type, objective_type=objective_type)
        res = optimizer.optimize(customer_df, precomputed_probabilities=precomputed_probs)

        records.append({
            "budget": b,
            "status": "Optimal" if res.success else "Failed",
            "expected_total_profit": round(res.expected_net_profit, 2),
            "expected_profit_per_customer": round(res.expected_net_profit / n_customers, 2),
            "expected_redemption_cost": round(res.total_cost, 2),
            "issued_face_value_exposure": round(res.face_value_exposure, 2),
            "budget_utilization_pct": round(res.budget_utilization_pct, 2),
            "expected_conversion_rate": round(res.expected_conversion_rate * 100, 2),
            "count_0": res.tier_counts.get(0, 0),
            "count_10": res.tier_counts.get(10, 0),
            "count_20": res.tier_counts.get(20, 0),
            "pct_0": round(res.tier_counts.get(0, 0) / n_customers * 100, 1),
            "pct_10": round(res.tier_counts.get(10, 0) / n_customers * 100, 1),
            "pct_20": round(res.tier_counts.get(20, 0) / n_customers * 100, 1),
        })

    return pd.DataFrame(records)


def plot_sensitivity_curves(
    sensitivity_df: pd.DataFrame,
    oracle_profit: Optional[float] = None,
    output_dir: str = "reports/figures",
):
    """Generate all Milestone 3 sensitivity and policy comparison charts."""
    os.makedirs(output_dir, exist_ok=True)
    budgets = sensitivity_df["budget"].to_numpy()
    profits = sensitivity_df["expected_total_profit"].to_numpy()
    costs = sensitivity_df["expected_redemption_cost"].to_numpy()

    # 1. Budget vs Total Expected Profit
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(budgets, profits, marker="o", color="#10b981", linewidth=2.5, label="Optimized DML Policy (MILP)")
    if oracle_profit is not None:
        ax.axhline(oracle_profit, color="#8b5cf6", linestyle="--", linewidth=2, label=f"Oracle Upper Bound (${oracle_profit:,.0f})")
    ax.set_title("Marketing Budget vs Total Expected Net Profit", fontsize=13, fontweight="bold")
    ax.set_xlabel("Marketing Budget Cap ($)", fontsize=11)
    ax.set_ylabel("Total Expected Net Profit ($)", fontsize=11)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "budget_vs_profit.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Budget vs Expected Redemption Cost
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(budgets, costs, marker="s", color="#3b82f6", linewidth=2.5, label="Actual Redeemed Expenditure")
    ax.plot(budgets, budgets, color="#ef4444", linestyle=":", label="100% Budget Utilization Line")
    ax.set_title("Marketing Budget vs Expected Discount Expenditure", fontsize=13, fontweight="bold")
    ax.set_xlabel("Configured Budget Cap ($)", fontsize=11)
    ax.set_ylabel("Expected Redemption Cost ($)", fontsize=11)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "budget_vs_discount_cost.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Allocation Distribution vs Budget
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = [f"${int(b):,}" if b < 10000 else f"${int(b/1000)}k" for b in budgets]
    p0 = sensitivity_df["pct_0"]
    p10 = sensitivity_df["pct_10"]
    p20 = sensitivity_df["pct_20"]

    x = np.arange(len(budgets))
    w = 0.6
    ax.bar(x, p0, width=w, label="$0 (No Discount)", color="#3b82f6", alpha=0.85)
    ax.bar(x, p10, width=w, bottom=p0, label="$10 Discount", color="#10b981", alpha=0.85)
    ax.bar(x, p20, width=w, bottom=p0 + p10, label="$20 Discount", color="#8b5cf6", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, fontsize=10)
    ax.set_title("Customer Allocation Mix Across Budget Constraints", fontsize=13, fontweight="bold")
    ax.set_xlabel("Marketing Budget ($)", fontsize=11)
    ax.set_ylabel("Customer Share (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "allocation_distribution_vs_budget.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Optimized vs Oracle Convergence Plot
    if oracle_profit is not None:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        efficiency = (profits / oracle_profit) * 100.0
        ax.plot(budgets, efficiency, marker="D", color="#6366f1", linewidth=2.5)
        ax.axhline(100.0, color="#8b5cf6", linestyle="--", label="Oracle Theoretical Frontier (100%)")
        ax.set_title("Policy Optimality: Frontier Efficiency vs Oracle Upper Bound", fontsize=13, fontweight="bold")
        ax.set_xlabel("Marketing Budget ($)", fontsize=11)
        ax.set_ylabel("Captured Share of Oracle Profit (%)", fontsize=11)
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "optimized_vs_oracle.png"), dpi=300, bbox_inches="tight")
        plt.close()


def plot_policy_comparison_bars(
    comp_df: pd.DataFrame,
    output_path: str = "reports/figures/policy_comparison_bars.png",
):
    """Plot horizontal comparison of expected profits across all baseline and optimized strategies."""
    fig, ax = plt.subplots(figsize=(10, 5))
    strategies = comp_df["strategy"].tolist()
    profits = comp_df["expected_profit_per_customer"].tolist()
    statuses = comp_df["status"].tolist()

    colors = []
    for s, stat in zip(strategies, statuses):
        if "Oracle" in s:
            colors.append("#8b5cf6")
        elif "Personalized" in s:
            colors.append("#10b981")
        elif stat == "INFEASIBLE":
            colors.append("#ef4444")
        else:
            colors.append("#3b82f6")

    y = np.arange(len(strategies))
    bars = ax.barh(y, profits, color=colors, height=0.55, alpha=0.88, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(strategies, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Expected Net Profit ($ / Customer)", fontsize=11)
    ax.set_title("Strategy Comparison: Expected Profit per Customer (Budget = $5,000)", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    for bar, stat in zip(bars, statuses):
        val = bar.get_width()
        tag = f"  ${val:.2f}" + (f" [{stat}]" if stat == "INFEASIBLE" else "")
        ax.text(val, bar.get_y() + bar.get_height()/2.0, tag, va="center", ha="left", fontsize=9, fontweight="bold")

    ax.set_xlim(0, max(profits) * 1.25)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

