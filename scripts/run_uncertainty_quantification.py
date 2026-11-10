"""Phase 2 Experiment Runner: Uncertainty-Aware Causal Estimation and Diagnostics.

Evaluates CATE uncertainty quantification across:
1. Baseline Point-Estimate CATE Model
2. Uncertainty-Aware Asymptotic / Subforest Infinitesimal Jackknife Model
3. Uncertainty-Aware Non-Parametric Bootstrap Model (100 replicates)

Computes coverage, width, MAE, RMSE, bias, segment-level and support-level diagnostics.
Exports:
- `data/optimization/uncertainty_aware_counterfactuals.csv`
- `experiments/results/uncertainty_quantification.json`
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

from src.causal.uncertainty import CATEUncertaintyEstimator


def compute_dgp_propensities(df: pd.DataFrame) -> np.ndarray:
    """Compute true logging assignment probabilities P(T=d | X) using the synthetic DGP."""
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
    print("=" * 85)
    print("ECONOCASUAL 2.0 — PHASE 2: UNCERTAINTY-AWARE CAUSAL ESTIMATION EXPERIMENT")
    print("=" * 85)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # 1. Paths
    test_path = ROOT_DIR / "data" / "synthetic_retail" / "synthetic_retail_test.csv"
    preds_path = ROOT_DIR / "data" / "synthetic_retail" / "synthetic_counterfactual_predictions.csv"
    gt_path = ROOT_DIR / "data" / "synthetic_retail" / "synthetic_retail_ground_truth.csv"
    opt_dir = ROOT_DIR / "data" / "optimization"
    opt_dir.mkdir(parents=True, exist_ok=True)
    out_csv_path = opt_dir / "uncertainty_aware_counterfactuals.csv"

    res_dir = ROOT_DIR / "experiments" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    out_json_path = res_dir / "uncertainty_quantification.json"

    # 2. Load Datasets
    print("\n[1/5] Loading held-out test evaluation data and ground-truth counterfactuals...")
    df_test = pd.read_csv(test_path)
    df_preds = pd.read_csv(preds_path)
    df_gt = pd.read_csv(gt_path)

    n_eval = len(df_test)
    print(f"  Test evaluation customers: {n_eval:,}")

    # Align ground truth with test customer_ids
    gt_test = df_test[["customer_id"]].merge(df_gt, on="customer_id", how="left")

    # 3. Propensity Scores
    print("\n[2/5] Computing logging treatment propensities...")
    propensities = compute_dgp_propensities(df_test)

    # 4. Initialize Estimator
    estimator = CATEUncertaintyEstimator(
        confidence_level=0.95,
        overlap_threshold=0.10,
        n_bootstraps=100,
        random_state=42,
    )

    pred_cate_10 = df_preds["pred_ite_10_vs_0"].to_numpy(dtype=float)
    pred_cate_20 = df_preds["pred_ite_20_vs_0"].to_numpy(dtype=float)
    true_cate_10 = gt_test["true_ite_10_vs_0"].to_numpy(dtype=float)
    true_cate_20 = gt_test["true_ite_20_vs_0"].to_numpy(dtype=float)

    # -------------------------------------------------------------------------
    # Experiment 1: Point Estimate Baseline (No Uncertainty)
    # -------------------------------------------------------------------------
    print("\n[3/5] Evaluating Experiment Models...")
    print("  --> Model 1: Point-Estimate Baseline (EconML CausalForestDML Point Predictions)")
    t0_pt = time.time()
    errors_10 = pred_cate_10 - true_cate_10
    errors_20 = pred_cate_20 - true_cate_20
    mae_10_pt = float(np.mean(np.abs(errors_10)))
    rmse_10_pt = float(np.sqrt(np.mean(errors_10 ** 2)))
    bias_10_pt = float(np.abs(np.mean(pred_cate_10) - np.mean(true_cate_10)))
    mae_20_pt = float(np.mean(np.abs(errors_20)))
    rmse_20_pt = float(np.sqrt(np.mean(errors_20 ** 2)))
    bias_20_pt = float(np.abs(np.mean(pred_cate_20) - np.mean(true_cate_20)))
    time_pt = time.time() - t0_pt

    # -------------------------------------------------------------------------
    # Experiment 2: Uncertainty-Aware Asymptotic / Subforest Model
    # -------------------------------------------------------------------------
    print("  --> Model 2: Uncertainty-Aware Asymptotic / Subforest Infinitesimal Jackknife")
    t0_asymp = time.time()
    uncert_10_asymp = estimator.compute_asymptotic_uncertainty(
        cate_estimates=pred_cate_10,
        propensities=propensities,
        covariates_df=df_test,
        treatment_arm=10,
    )
    uncert_20_asymp = estimator.compute_asymptotic_uncertainty(
        cate_estimates=pred_cate_20,
        propensities=propensities,
        covariates_df=df_test,
        treatment_arm=20,
    )
    time_asymp = time.time() - t0_asymp

    cov_10_asymp = estimator.evaluate_ground_truth_coverage(
        true_cate=true_cate_10,
        pred_cate=pred_cate_10,
        lower_bound=uncert_10_asymp["lower_bound"],
        upper_bound=uncert_10_asymp["upper_bound"],
    )
    cov_20_asymp = estimator.evaluate_ground_truth_coverage(
        true_cate=true_cate_20,
        pred_cate=pred_cate_20,
        lower_bound=uncert_20_asymp["lower_bound"],
        upper_bound=uncert_20_asymp["upper_bound"],
    )

    seg_10_asymp = estimator.evaluate_by_segment(df_test, true_cate_10, pred_cate_10, uncert_10_asymp)
    seg_20_asymp = estimator.evaluate_by_segment(df_test, true_cate_20, pred_cate_20, uncert_20_asymp)
    supp_10_asymp = estimator.evaluate_by_support_level(propensities, true_cate_10, pred_cate_10, uncert_10_asymp)
    supp_20_asymp = estimator.evaluate_by_support_level(propensities, true_cate_20, pred_cate_20, uncert_20_asymp)

    # -------------------------------------------------------------------------
    # Experiment 3: Uncertainty-Aware Non-Parametric Bootstrap Model (100 Reps)
    # -------------------------------------------------------------------------
    print("  --> Model 3: Uncertainty-Aware Non-Parametric Bootstrap (B=100 Replicates)")
    t0_boot = time.time()
    uncert_10_boot = estimator.compute_bootstrap_uncertainty(
        cate_estimates=pred_cate_10,
        propensities=propensities,
        covariates_df=df_test,
        treatment_arm=10,
        n_bootstraps=100,
    )
    uncert_20_boot = estimator.compute_bootstrap_uncertainty(
        cate_estimates=pred_cate_20,
        propensities=propensities,
        covariates_df=df_test,
        treatment_arm=20,
        n_bootstraps=100,
    )
    time_boot = time.time() - t0_boot

    cov_10_boot = estimator.evaluate_ground_truth_coverage(
        true_cate=true_cate_10,
        pred_cate=pred_cate_10,
        lower_bound=uncert_10_boot["lower_bound"],
        upper_bound=uncert_10_boot["upper_bound"],
    )
    cov_20_boot = estimator.evaluate_ground_truth_coverage(
        true_cate=true_cate_20,
        pred_cate=pred_cate_20,
        lower_bound=uncert_20_boot["lower_bound"],
        upper_bound=uncert_20_boot["upper_bound"],
    )

    seg_10_boot = estimator.evaluate_by_segment(df_test, true_cate_10, pred_cate_10, uncert_10_boot)
    seg_20_boot = estimator.evaluate_by_segment(df_test, true_cate_20, pred_cate_20, uncert_20_boot)

    # -------------------------------------------------------------------------
    # 5. Build Risk-Aware Schema for Optimization
    # -------------------------------------------------------------------------
    print("\n[4/5] Generating Risk-Aware Policy Input Schema for Optimizer...")
    df_arm_10 = estimator.build_risk_aware_dataframe(
        customer_ids=df_test["customer_id"],
        treatment_arm=10,
        cate=pred_cate_10,
        uncertainty_dict=uncert_10_asymp,
    )
    df_arm_20 = estimator.build_risk_aware_dataframe(
        customer_ids=df_test["customer_id"],
        treatment_arm=20,
        cate=pred_cate_20,
        uncertainty_dict=uncert_20_asymp,
    )
    df_risk_aware = pd.concat([df_arm_10, df_arm_20], ignore_index=True)
    df_risk_aware.sort_values(by=["customer_id", "treatment"], inplace=True)
    df_risk_aware.to_csv(out_csv_path, index=False)
    print(f"  [OK] Exported {len(df_risk_aware):,} rows to: {out_csv_path}")

    # -------------------------------------------------------------------------
    # 6. Print CLI Scorecard
    # -------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("PHASE 2 EXPERIMENT BENCHMARK SCORECARD: POINT ESTIMATE VS UNCERTAINTY-AWARE CATE")
    print("=" * 115)
    header = f"{'Model':<36} | {'Contrast':<10} | {'MAE':<8} | {'RMSE':<8} | {'Bias':<8} | {'Cov %':<8} | {'Width':<8} | {'Runtime':<8}"
    print(header)
    print("-" * 115)

    print(f"{'1. Baseline Point Estimate':<36} | {'$10 vs $0':<10} | {mae_10_pt:.4f}   | {rmse_10_pt:.4f}   | {bias_10_pt:.4f}   | {'N/A':<8} | {'N/A':<8} | {time_pt:.3f}s")
    print(f"{'1. Baseline Point Estimate':<36} | {'$20 vs $0':<10} | {mae_20_pt:.4f}   | {rmse_20_pt:.4f}   | {bias_20_pt:.4f}   | {'N/A':<8} | {'N/A':<8} | {time_pt:.3f}s")
    print("-" * 115)
    print(f"{'2. Uncertainty-Aware (Asymptotic IJ)':<36} | {'$10 vs $0':<10} | {cov_10_asymp['cate_mae']:.4f}   | {cov_10_asymp['cate_rmse']:.4f}   | {cov_10_asymp['ate_bias']:.4f}   | {cov_10_asymp['coverage_rate_pct']:.1f}%   | {cov_10_asymp['mean_interval_width']:.4f}   | {time_asymp:.3f}s")
    print(f"{'2. Uncertainty-Aware (Asymptotic IJ)':<36} | {'$20 vs $0':<10} | {cov_20_asymp['cate_mae']:.4f}   | {cov_20_asymp['cate_rmse']:.4f}   | {cov_20_asymp['ate_bias']:.4f}   | {cov_20_asymp['coverage_rate_pct']:.1f}%   | {cov_20_asymp['mean_interval_width']:.4f}   | {time_asymp:.3f}s")
    print("-" * 115)
    print(f"{'3. Uncertainty-Aware (Bootstrap B=100)':<36} | {'$10 vs $0':<10} | {cov_10_boot['cate_mae']:.4f}   | {cov_10_boot['cate_rmse']:.4f}   | {cov_10_boot['ate_bias']:.4f}   | {cov_10_boot['coverage_rate_pct']:.1f}%   | {cov_10_boot['mean_interval_width']:.4f}   | {time_boot:.3f}s")
    print(f"{'3. Uncertainty-Aware (Bootstrap B=100)':<36} | {'$20 vs $0':<10} | {cov_20_boot['cate_mae']:.4f}   | {cov_20_boot['cate_rmse']:.4f}   | {cov_20_boot['ate_bias']:.4f}   | {cov_20_boot['coverage_rate_pct']:.1f}%   | {cov_20_boot['mean_interval_width']:.4f}   | {time_boot:.3f}s")
    print("=" * 115)

    # Reliability counts
    rel_counts_10 = pd.Series(uncert_10_asymp["reliability_status"]).value_counts().to_dict()
    rel_counts_20 = pd.Series(uncert_20_asymp["reliability_status"]).value_counts().to_dict()

    print("\nReliability Classification Counts ($10 vs $0):", rel_counts_10)
    print("Reliability Classification Counts ($20 vs $0):", rel_counts_20)

    # -------------------------------------------------------------------------
    # 7. Export Full JSON Experiment Report
    # -------------------------------------------------------------------------
    print(f"\n[5/5] Exporting experiment benchmark artifact to: {out_json_path}")
    json_payload = {
        "metadata": {
            "experiment_timestamp": datetime.now().isoformat(),
            "n_samples": n_eval,
            "confidence_level": 0.95,
            "overlap_threshold": 0.10,
            "n_bootstraps": 100,
            "random_state": 42,
            "methodology": "EconML CausalForestDML + Infinitesimal Jackknife & Residual Bootstrap",
        },
        "models": {
            "point_estimate_baseline": {
                "description": "Baseline EconML CausalForestDML point estimates without uncertainty quantification",
                "runtime_seconds": round(time_pt, 4),
                "contrasts": {
                    "10_vs_0": {
                        "cate_mae": mae_10_pt,
                        "cate_rmse": rmse_10_pt,
                        "ate_bias": bias_10_pt,
                        "coverage_rate_pct": None,
                        "mean_interval_width": None,
                    },
                    "20_vs_0": {
                        "cate_mae": mae_20_pt,
                        "cate_rmse": rmse_20_pt,
                        "ate_bias": bias_20_pt,
                        "coverage_rate_pct": None,
                        "mean_interval_width": None,
                    },
                },
            },
            "asymptotic_subforest_model": {
                "description": "Asymptotic / Subforest Infinitesimal Jackknife heteroskedastic variance estimator",
                "runtime_seconds": round(time_asymp, 4),
                "contrasts": {
                    "10_vs_0": {
                        "cate_mae": cov_10_asymp["cate_mae"],
                        "cate_rmse": cov_10_asymp["cate_rmse"],
                        "ate_bias": cov_10_asymp["ate_bias"],
                        "coverage_rate_pct": cov_10_asymp["coverage_rate_pct"],
                        "mean_interval_width": cov_10_asymp["mean_interval_width"],
                        "reliability_counts": {str(k): int(v) for k, v in rel_counts_10.items()},
                        "segment_diagnostics": seg_10_asymp,
                        "support_diagnostics": supp_10_asymp,
                    },
                    "20_vs_0": {
                        "cate_mae": cov_20_asymp["cate_mae"],
                        "cate_rmse": cov_20_asymp["cate_rmse"],
                        "ate_bias": cov_20_asymp["ate_bias"],
                        "coverage_rate_pct": cov_20_asymp["coverage_rate_pct"],
                        "mean_interval_width": cov_20_asymp["mean_interval_width"],
                        "reliability_counts": {str(k): int(v) for k, v in rel_counts_20.items()},
                        "segment_diagnostics": seg_20_asymp,
                        "support_diagnostics": supp_20_asymp,
                    },
                },
            },
            "bootstrap_residual_model": {
                "description": "Non-parametric residual bootstrap estimator (B=100 replicates, reproducible seed 42)",
                "runtime_seconds": round(time_boot, 4),
                "contrasts": {
                    "10_vs_0": {
                        "cate_mae": cov_10_boot["cate_mae"],
                        "cate_rmse": cov_10_boot["cate_rmse"],
                        "ate_bias": cov_10_boot["ate_bias"],
                        "coverage_rate_pct": cov_10_boot["coverage_rate_pct"],
                        "mean_interval_width": cov_10_boot["mean_interval_width"],
                        "segment_diagnostics": seg_10_boot,
                    },
                    "20_vs_0": {
                        "cate_mae": cov_20_boot["cate_mae"],
                        "cate_rmse": cov_20_boot["cate_rmse"],
                        "ate_bias": cov_20_boot["ate_bias"],
                        "coverage_rate_pct": cov_20_boot["coverage_rate_pct"],
                        "mean_interval_width": cov_20_boot["mean_interval_width"],
                        "segment_diagnostics": seg_20_boot,
                    },
                },
            },
        },
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    print(f"[SUCCESS] Phase 2 Experiment Completed. Artifact created: {out_json_path}")
    return json_payload


if __name__ == "__main__":
    main()

