"""Phase 3 Experiment Runner: Uncertainty-Aware and Fairness-Constrained MILP Optimization.

Compares 4 core optimization regimes on the held-out test cohort (N = 6,005):
1. Original MILP (Baseline nominal expected profit maximization)
2. Uncertainty-Aware MILP (Risk-adjusted expected profit maximization, lambda = 1.0)
3. Fairness-Constrained MILP (Demographic allocation parity across business segments)
4. Uncertainty-Aware & Fairness-Constrained MILP (Combined risk aversion and fairness)

Also runs:
- Risk-Aversion Sensitivity Sweep: lambda in [0.0, 0.5, 1.0, 1.5, 2.0]
- Fairness Trade-off Analysis: Budget segment floor sweep

Exports:
- `data/optimization/advanced_allocations.csv`
- `experiments/results/advanced_optimization_benchmarks.json`
"""

from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Add repository root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.optimization.advanced_optimizer import AdvancedMILPOptimizer


def main():
    print("=" * 90)
    print("ECONOCASUAL 2.0 — PHASE 3: UNCERTAINTY-AWARE & FAIRNESS-CONSTRAINED MILP EXPERIMENT")
    print("=" * 90)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # 1. Paths
    test_path = ROOT_DIR / "data" / "synthetic_retail" / "synthetic_retail_test.csv"
    preds_path = ROOT_DIR / "data" / "synthetic_retail" / "synthetic_counterfactual_predictions.csv"
    uncert_path = ROOT_DIR / "data" / "optimization" / "uncertainty_aware_counterfactuals.csv"
    opt_dir = ROOT_DIR / "data" / "optimization"
    opt_dir.mkdir(parents=True, exist_ok=True)
    out_csv_path = opt_dir / "advanced_allocations.csv"

    res_dir = ROOT_DIR / "experiments" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    out_json_path = res_dir / "advanced_optimization_benchmarks.json"

    # 2. Load Evaluation Data
    print("\n[1/6] Loading test dataset, predictions, and uncertainty features...")
    df_test = pd.read_csv(test_path)
    df_preds = pd.read_csv(preds_path)
    df_uncert = pd.read_csv(uncert_path) if os.path.exists(uncert_path) else None

    n_eval = len(df_test)
    budget = 5000.0
    print(f"  Evaluation customers: {n_eval:,}")
    print(f"  Promotional budget cap: ${budget:,.2f} (redemption model)")

    # -------------------------------------------------------------------------
    # 3. Core Experiment: Four Optimization Regimes
    # -------------------------------------------------------------------------
    print("\n[2/6] Executing 4 Core Optimization Regimes...")
    regimes = {}

    # Regime 1: Original MILP (Baseline)
    print("  --> Regime 1: Original Baseline MILP (Nominal Profit Maximization)")
    opt_regime1 = AdvancedMILPOptimizer(
        budget=budget,
        budget_type="redemption",
        optimization_mode="baseline",
        fairness_enabled=False,
    )
    res_regime1 = opt_regime1.optimize(df_test, df_preds, df_uncert)
    regimes["regime1_original_milp"] = res_regime1

    # Regime 2: Uncertainty-Aware MILP (Risk-Adjusted, lambda = 1.0)
    print("  --> Regime 2: Uncertainty-Aware MILP (Risk-Adjusted, lambda = 1.0)")
    opt_regime2 = AdvancedMILPOptimizer(
        budget=budget,
        budget_type="redemption",
        optimization_mode="risk_adjusted",
        risk_aversion=1.0,
        fairness_enabled=False,
    )
    res_regime2 = opt_regime2.optimize(df_test, df_preds, df_uncert)
    regimes["regime2_uncertainty_aware_milp"] = res_regime2

    # Regime 3: Fairness-Constrained MILP (Budget segment floor = 15%)
    print("  --> Regime 3: Fairness-Constrained MILP (Budget Segment Floor = 15%)")
    opt_regime3 = AdvancedMILPOptimizer(
        budget=budget,
        budget_type="redemption",
        optimization_mode="baseline",
        fairness_enabled=True,
        fairness_mode="min_allocation_rate",
        min_segment_rate={"budget": 0.15, "regular": 0.15, "premium": 0.15, "vip": 0.15},
    )
    res_regime3 = opt_regime3.optimize(df_test, df_preds, df_uncert)
    regimes["regime3_fairness_constrained_milp"] = res_regime3

    # Regime 4: Uncertainty-Aware and Fairness-Constrained MILP (Combined)
    print("  --> Regime 4: Uncertainty-Aware & Fairness-Constrained MILP (Combined)")
    opt_regime4 = AdvancedMILPOptimizer(
        budget=budget,
        budget_type="redemption",
        optimization_mode="risk_adjusted",
        risk_aversion=1.0,
        fairness_enabled=True,
        fairness_mode="min_allocation_rate",
        min_segment_rate={"budget": 0.15, "regular": 0.15, "premium": 0.15, "vip": 0.15},
    )
    res_regime4 = opt_regime4.optimize(df_test, df_preds, df_uncert)
    regimes["regime4_combined_milp"] = res_regime4

    # -------------------------------------------------------------------------
    # 4. Risk-Aversion Parameter Sweep (lambda in [0.0, 0.5, 1.0, 1.5, 2.0])
    # -------------------------------------------------------------------------
    print("\n[3/6] Running Risk-Aversion Sensitivity Sweep (lambda in [0.0, 0.5, 1.0, 1.5, 2.0])...")
    lambda_values = [0.0, 0.5, 1.0, 1.5, 2.0]
    risk_sweep_results = []

    for lam in lambda_values:
        opt_lam = AdvancedMILPOptimizer(
            budget=budget,
            budget_type="redemption",
            optimization_mode="risk_adjusted",
            risk_aversion=lam,
            fairness_enabled=False,
        )
        res_lam = opt_lam.optimize(df_test, df_preds, df_uncert)
        risk_sweep_results.append({
            "lambda": lam,
            "nominal_expected_profit": res_lam.nominal_expected_profit,
            "conservative_lcb_profit": res_lam.conservative_lcb_profit,
            "total_expected_cost": res_lam.total_expected_cost,
            "tier_counts": res_lam.tier_counts,
            "selected_customers_mean_uncertainty": res_lam.selected_customers_mean_uncertainty,
            "runtime_seconds": res_lam.runtime_seconds,
            "status": res_lam.status_message,
        })

    # -------------------------------------------------------------------------
    # 5. Fairness Sensitivity Sweep: Budget Segment Allocation Floor
    # -------------------------------------------------------------------------
    print("\n[4/6] Running Fairness Sensitivity Sweep (Budget Segment Allocation Floor)...")
    budget_floors = [0.089, 0.12, 0.15, 0.18, 0.20]
    fairness_sweep_results = []

    for fl in budget_floors:
        opt_fl = AdvancedMILPOptimizer(
            budget=budget,
            budget_type="redemption",
            optimization_mode="baseline",
            fairness_enabled=True,
            fairness_mode="min_allocation_rate",
            min_segment_rate={"budget": fl},
        )
        res_fl = opt_fl.optimize(df_test, df_preds, df_uncert)
        budget_diag = res_fl.fairness_metrics.get("budget")
        fairness_sweep_results.append({
            "budget_floor_pct": fl * 100.0,
            "actual_budget_allocation_rate_pct": budget_diag.allocation_rate_pct if budget_diag else 0.0,
            "actual_budget_parity_ratio": budget_diag.parity_ratio_vs_overall if budget_diag else 0.0,
            "nominal_expected_profit": res_fl.nominal_expected_profit,
            "price_of_fairness_profit_loss": round(res_regime1.nominal_expected_profit - res_fl.nominal_expected_profit, 2),
            "tier_counts": res_fl.tier_counts,
            "runtime_seconds": res_fl.runtime_seconds,
        })

    # -------------------------------------------------------------------------
    # 6. Build Multi-Regime Allocation CSV
    # -------------------------------------------------------------------------
    print("\n[5/6] Exporting Multi-Regime Allocations Table...")
    df_allocations = pd.DataFrame({
        "customer_id": df_test["customer_id"],
        "customer_segment": df_test["customer_segment"],
        "regime1_original_milp_tier": res_regime1.allocation_df["allocated_discount"],
        "regime2_uncertainty_aware_tier": res_regime2.allocation_df["allocated_discount"],
        "regime3_fairness_constrained_tier": res_regime3.allocation_df["allocated_discount"],
        "regime4_combined_tier": res_regime4.allocation_df["allocated_discount"],
    })
    df_allocations.to_csv(out_csv_path, index=False)
    print(f"  [OK] Exported {len(df_allocations):,} rows to: {out_csv_path}")

    # -------------------------------------------------------------------------
    # 7. Print CLI Scorecard Tables
    # -------------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("CORE REGIMES COMPARISON SCORECARD ($5,000 BUDGET CAP)")
    print("=" * 120)
    header = f"{'Optimization Regime':<38} | {'Exp. Profit ($)':<15} | {'LCB Profit ($)':<14} | {'Exp. Cost ($)':<13} | {'Tiers ($0/$10/$20)':<18} | {'Mean SE':<8} | {'Runtime':<7}"
    print(header)
    print("-" * 120)

    for r_name, r_res in regimes.items():
        tiers_str = f"{r_res.tier_counts.get(0, 0)}/{r_res.tier_counts.get(10, 0)}/{r_res.tier_counts.get(20, 0)}"
        print(
            f"{r_name:<38} | "
            f"${r_res.nominal_expected_profit:>13,.2f} | "
            f"${r_res.conservative_lcb_profit:>12,.2f} | "
            f"${r_res.total_expected_cost:>11,.2f} | "
            f"{tiers_str:<18} | "
            f"{r_res.selected_customers_mean_uncertainty:<8.4f} | "
            f"{r_res.runtime_seconds:<6.3f}s"
        )
    print("=" * 120)

    # Print Risk-Aversion Table
    print("\n" + "=" * 110)
    print("RISK-AVERSION PARAMETER SWEEP (MEAN-VARIANCE REGIME)")
    print("=" * 110)
    print(f"{'Risk Aversion (lambda)':<24} | {'Exp. Profit ($)':<15} | {'LCB Profit ($)':<14} | {'Exp. Cost ($)':<13} | {'Mean SE':<10} | {'Tiers ($0/$10/$20)':<18}")
    print("-" * 110)
    for r in risk_sweep_results:
        tiers_str = f"{r['tier_counts'].get(0, 0)}/{r['tier_counts'].get(10, 0)}/{r['tier_counts'].get(20, 0)}"
        print(f"lambda = {r['lambda']:<17.1f} | ${r['nominal_expected_profit']:>13,.2f} | ${r['conservative_lcb_profit']:>12,.2f} | ${r['total_expected_cost']:>11,.2f} | {r['selected_customers_mean_uncertainty']:<10.4f} | {tiers_str:<18}")
    print("=" * 110)

    # Print Fairness Diagnostics
    print("\n" + "=" * 110)
    print("BUSINESS SEGMENT FAIRNESS DIAGNOSTICS: ORIGINAL MILP VS FAIRNESS-CONSTRAINED")
    print("=" * 110)
    print(f"{'Segment':<10} | {'Size':<6} | {'Original Rate %':<17} | {'Original Parity':<16} | {'Fairness Rate %':<17} | {'Fairness Parity':<15}")
    print("-" * 110)
    for seg in sorted(df_test["customer_segment"].unique()):
        orig_d = res_regime1.fairness_metrics.get(seg)
        fair_d = res_regime3.fairness_metrics.get(seg)
        o_rate = f"{orig_d.allocation_rate_pct:.1f}%" if orig_d else "N/A"
        o_par = f"{orig_d.parity_ratio_vs_overall:.3f}" if orig_d else "N/A"
        f_rate = f"{fair_d.allocation_rate_pct:.1f}%" if fair_d else "N/A"
        f_par = f"{fair_d.parity_ratio_vs_overall:.3f}" if fair_d else "N/A"
        n_seg = orig_d.sample_size if orig_d else 0
        print(f"{seg:<10} | {n_seg:<6} | {o_rate:<17} | {o_par:<16} | {f_rate:<17} | {f_par:<15}")
    print("=" * 110)

    # -------------------------------------------------------------------------
    # 8. Export Benchmark JSON Artifact
    # -------------------------------------------------------------------------
    print(f"\n[6/6] Exporting benchmark results to: {out_json_path}")
    json_payload = {
        "metadata": {
            "experiment_timestamp": datetime.now().isoformat(),
            "n_customers": n_eval,
            "budget": budget,
            "budget_type": "redemption",
            "solver": "SciPy HiGHS MILP",
            "cash_budget_disclaimer": res_regime1.cash_budget_disclaimer,
        },
        "core_regimes": {
            k: v.to_dict() for k, v in regimes.items()
        },
        "risk_aversion_sweep": risk_sweep_results,
        "fairness_sensitivity_sweep": fairness_sweep_results,
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    print(f"[SUCCESS] Phase 3 Advanced Optimization Complete. Artifact created: {out_json_path}")
    return json_payload


if __name__ == "__main__":
    main()

