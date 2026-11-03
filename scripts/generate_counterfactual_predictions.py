"""Script to generate calibrated counterfactual outcome predictions for all treatment options."""

import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.causal.synthetic_multitreatment import SyntheticMultiTreatmentDML
from src.causal.hillstrom_causal import HillstromCausalEngine
from src.causal.counterfactual_prediction import (
    generate_synthetic_counterfactuals,
    evaluate_probability_calibration,
    generate_hillstrom_counterfactuals,
)


def main():
    print("=" * 80)
    print("GENERATING COUNTERFACTUAL OUTCOME PREDICTIONS")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Synthetic Retail Counterfactual Predictions
    # -------------------------------------------------------------------------
    syn_model_dir = "models/synthetic_retail"
    syn_test_path = "data/synthetic_retail/synthetic_retail_test.csv"
    syn_output_path = "data/synthetic_retail/synthetic_counterfactual_predictions.csv"

    print(f"Loading trained synthetic causal engine from: {syn_model_dir}")
    syn_engine = SyntheticMultiTreatmentDML.load(syn_model_dir)

    df_syn_test = pd.read_csv(syn_test_path)
    print(f"Generating counterfactual predictions for test set (N={len(df_syn_test):,})...")
    syn_preds_df = generate_synthetic_counterfactuals(syn_engine, df_syn_test)

    # Calibration audit against actual observed purchases
    cal_metrics = evaluate_probability_calibration(df_syn_test, syn_preds_df)
    print("\n[Synthetic Probability Calibration Results]")
    print(f"  - Overall Brier Score: {cal_metrics['overall_brier_score']}")
    print(f"  - Overall Log Loss:    {cal_metrics['overall_log_loss']}")
    for d, m in cal_metrics["tier_calibrations"].items():
        print(f"  - Tier ${d}: Actual Rate={m['actual_conversion_rate']*100:.2f}% | Mean Predicted={m['predicted_mean_probability']*100:.2f}% | Calibration Error={m['absolute_calibration_error']*100:.2f} pp")

    syn_preds_df.to_csv(syn_output_path, index=False)
    print(f"\n[OK] Synthetic counterfactual predictions exported to: {syn_output_path}")

    # -------------------------------------------------------------------------
    # 2. Hillstrom Counterfactual Predictions
    # -------------------------------------------------------------------------
    hill_model_dir = "models/hillstrom"
    hill_test_path = "data/hillstrom/processed/hillstrom_test.csv"
    hill_output_path = "data/hillstrom/hillstrom_counterfactual_predictions.csv"

    print(f"\nLoading trained Hillstrom causal engine from: {hill_model_dir}")
    hill_engine = HillstromCausalEngine.load(hill_model_dir)

    df_hill_test = pd.read_csv(hill_test_path)
    print(f"Generating counterfactual predictions for Hillstrom test set (N={len(df_hill_test):,})...")
    hill_preds_df = generate_hillstrom_counterfactuals(hill_engine, df_hill_test)

    hill_preds_df.to_csv(hill_output_path, index=False)
    print(f"[OK] Hillstrom counterfactual predictions exported to: {hill_output_path}")


if __name__ == "__main__":
    main()

