"""Script to train the Multi-Treatment Synthetic Retail Causal Forest DML engine."""

import os
import sys
import time
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.causal.synthetic_multitreatment import SyntheticMultiTreatmentDML


def main():
    print("=" * 80)
    print("TRAINING MULTI-TREATMENT CAUSAL ENGINE (SYNTHETIC RETAIL)")
    print("=" * 80)

    train_path = "data/synthetic_retail/synthetic_retail_train.csv"
    val_path = "data/synthetic_retail/synthetic_retail_validation.csv"
    model_dir = "models/synthetic_retail"

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training dataset not found: {train_path}")

    print(f"Loading training data from: {train_path}")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    print(f"Loaded {len(df_train):,} training records and {len(df_val):,} validation records.")

    print("\nInitializing EconML CausalForestDML:")
    print("  - Treatment: discount_amount in {0, 10, 20} (discrete_treatment=True)")
    print("  - Outcome: purchase (binary conversion)")
    print("  - Nuisance Model Y: LightGBM Regressor (cross-fitted)")
    print("  - Nuisance Model T: Calibrated Multinomial Logistic Regression")
    print("  - Forest Parameters: n_estimators=120, subforest_size=4, min_samples_leaf=15, max_depth=8, cv=3")

    engine = SyntheticMultiTreatmentDML(
        n_estimators=120,
        subforest_size=4,
        min_samples_leaf=15,
        max_depth=8,
        cv=3,
        random_state=42,
        n_jobs=-1,
    )

    t0 = time.time()
    engine.fit(df_train, treatment_col="discount_amount", outcome_col="purchase")
    elapsed = time.time() - t0
    print(f"\n[OK] Model training completed in {elapsed:.2f} seconds.")

    # Validation out-of-sample predictions
    val_effects = engine.predict_effects(df_val)
    mean_tau_10 = float(val_effects["pred_ite_10_vs_0"].mean())
    mean_tau_20 = float(val_effects["pred_ite_20_vs_0"].mean())
    mean_tau_20_10 = float(val_effects["pred_ite_20_vs_10"].mean())

    print("\nValidation Out-of-Sample Estimated Treatment Effects (N=5,997):")
    print(f"  - Mean CATE ($10 vs $0):  +{mean_tau_10*100:.2f} pp")
    print(f"  - Mean CATE ($20 vs $0):  +{mean_tau_20*100:.2f} pp")
    print(f"  - Mean CATE ($20 vs $10): +{mean_tau_20_10*100:.2f} pp")

    # Save artifacts
    engine.save(model_dir=model_dir)

    # Save training log
    log_data = {
        "training_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_time_seconds": round(elapsed, 2),
        "train_rows": len(df_train),
        "val_rows": len(df_val),
        "validation_estimated_ate": {
            "tau_10_vs_0_pp": round(mean_tau_10 * 100, 2),
            "tau_20_vs_0_pp": round(mean_tau_20 * 100, 2),
            "tau_20_vs_10_pp": round(mean_tau_20_10 * 100, 2),
        },
    }
    with open(os.path.join(model_dir, "training_log.json"), "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"  [Log] Training log saved: {os.path.join(model_dir, 'training_log.json')}")


if __name__ == "__main__":
    main()

