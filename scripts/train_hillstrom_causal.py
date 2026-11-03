"""Script to train the Hillstrom 3-Arm Randomized Causal Benchmark Engine."""

import os
import sys
import time
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.causal.hillstrom_causal import HillstromCausalEngine


def main():
    print("=" * 80)
    print("TRAINING MULTI-TREATMENT CAUSAL ENGINE (HILLSTROM RCT BENCHMARK)")
    print("=" * 80)

    train_path = "data/hillstrom/processed/hillstrom_train.csv"
    val_path = "data/hillstrom/processed/hillstrom_validation.csv"
    model_dir = "models/hillstrom"

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Hillstrom training dataset not found: {train_path}")

    print(f"Loading Hillstrom training data from: {train_path}")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    print(f"Loaded {len(df_train):,} training records and {len(df_val):,} validation records.")

    print("\nInitializing Multi-Outcome LinearDML Causal Engine:")
    print("  - Treatment: treatment_group in {0: no_email, 1: mens_email, 2: womens_email}")
    print("  - Outcomes: visit (binary), conversion (binary), spend (continuous)")
    print("  - Nuisance Models: Cross-fitted gradient boosted regressors and Ridge estimators")

    engine = HillstromCausalEngine(random_state=42)
    t0 = time.time()
    engine.fit(df_train, treatment_col="treatment_group")
    elapsed = time.time() - t0
    print(f"\n[OK] Model training completed in {elapsed:.2f} seconds.")

    # Validation out-of-sample effects
    val_visit = engine.predict_effects(df_val, outcome="visit")
    val_conv = engine.predict_effects(df_val, outcome="conversion")
    val_spend = engine.predict_effects(df_val, outcome="spend")

    print("\nValidation Out-of-Sample Estimated Treatment Effects (N=9,598):")
    print(f"  - Visit Lift (Men's vs Control):     +{val_visit['ite_visit_mens_vs_control'].mean()*100:.2f} pp")
    print(f"  - Visit Lift (Women's vs Control):   +{val_visit['ite_visit_womens_vs_control'].mean()*100:.2f} pp")
    print(f"  - Conv Lift (Men's vs Control):      +{val_conv['ite_conversion_mens_vs_control'].mean()*100:.2f} pp")
    print(f"  - Conv Lift (Women's vs Control):    +{val_conv['ite_conversion_womens_vs_control'].mean()*100:.2f} pp")
    print(f"  - Spend Lift (Men's vs Control):     +${val_spend['ite_spend_mens_vs_control'].mean():.2f} / customer")
    print(f"  - Spend Lift (Women's vs Control):   +${val_spend['ite_spend_womens_vs_control'].mean():.2f} / customer")

    # Save artifacts
    engine.save(model_dir=model_dir)

    # Save training log
    log_data = {
        "training_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_time_seconds": round(elapsed, 2),
        "train_rows": len(df_train),
        "val_rows": len(df_val),
        "validation_estimated_ate": {
            "visit_mens_vs_control_pp": round(float(val_visit["ite_visit_mens_vs_control"].mean() * 100), 2),
            "visit_womens_vs_control_pp": round(float(val_visit["ite_visit_womens_vs_control"].mean() * 100), 2),
            "conv_mens_vs_control_pp": round(float(val_conv["ite_conversion_mens_vs_control"].mean() * 100), 2),
            "conv_womens_vs_control_pp": round(float(val_conv["ite_conversion_womens_vs_control"].mean() * 100), 2),
            "spend_mens_vs_control_usd": round(float(val_spend["ite_spend_mens_vs_control"].mean()), 2),
            "spend_womens_vs_control_usd": round(float(val_spend["ite_spend_womens_vs_control"].mean()), 2),
        },
    }
    with open(os.path.join(model_dir, "training_log.json"), "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"  [Log] Training log saved: {os.path.join(model_dir, 'training_log.json')}")


if __name__ == "__main__":
    main()

