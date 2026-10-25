"""Customer-level Treatment Effect Estimation Script for EconoCausal Week 2.

This script executes the customer-level causal estimation and interpretation workflow:
1. Loads the serialized DML model (models/dml_model.joblib) and its metadata (models/dml_metadata.json).
2. Partitions the dataset into the exact held-out evaluation split (500 records).
3. Validates feature schemas, order, and anti-leakage constraints.
4. Generates customer-level CATE estimates, 95% confidence intervals, and effect categories.
5. Saves results to reports/customer_treatment_effects.csv.
6. Generates an executive interpretation summary at reports/customer_treatment_effects_summary.md.
"""

from typing import Dict, Any
import os
import sys
import json
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.model_selection import train_test_split
from src.causal.dml import EconoCausalDML
from src.causal.interpreter import (
    CustomerEffectInterpreter,
    DEFAULT_NEAR_ZERO_THRESHOLD,
)


def load_model_and_metadata() -> tuple[EconoCausalDML, Dict[str, Any]]:
    """Load serialized model and metadata, verifying file integrity."""
    model_path = os.path.join(PROJECT_ROOT, "models", "dml_model.joblib")
    meta_path = os.path.join(PROJECT_ROOT, "models", "dml_metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Run scripts/train_dml.py first.")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}. Run scripts/train_dml.py first.")

    model = EconoCausalDML.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return model, meta


def load_evaluation_data() -> pd.DataFrame:
    """Load dataset and reproduce the exact held-out evaluation partition (N=500)."""
    paths = [
        os.path.join(PROJECT_ROOT, "data", "mock_retail_data.csv"),
        os.path.join(PROJECT_ROOT, "sample_data", "synthetic_retail_campaign.csv"),
    ]
    data_path = None
    for p in paths:
        if os.path.exists(p):
            data_path = p
            break

    if not data_path:
        raise FileNotFoundError(f"Dataset not found in expected locations: {paths}")

    df = pd.read_csv(data_path)

    # Replicate exact stratified split
    t_candidate = "discount_received" if "discount_received" in df.columns else "discount_offered"
    y_candidate = "purchase" if "purchase" in df.columns else "purchased"
    strat_col = df[t_candidate].astype(str) + "_" + df[y_candidate].astype(str)

    _, eval_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=strat_col,
    )
    return eval_df


def run_customer_estimation(near_zero_threshold: float = DEFAULT_NEAR_ZERO_THRESHOLD) -> None:
    print("=" * 75)
    print("  ECONOCAUSAL - CUSTOMER-LEVEL TREATMENT EFFECT ESTIMATION")
    print("  Week 2: CATE Inference, Confidence Intervals & Transparent Taxonomy")
    print("=" * 75)

    # 1. Load Model & Metadata
    print("\n[Step 1/5] Loading serialized DML model and configuration metadata...")
    model, meta = load_model_and_metadata()
    print(f"  -> Model class: {meta['model_class']} ({meta['model_type']})")
    print(f"  -> Base estimators: {meta['base_estimator']} (n_estimators={meta['n_estimators']}, max_depth={meta['max_depth']})")
    print(f"  -> Features required ({len(meta['features'])}): {meta['features']}")

    # 2. Load Evaluation Data
    print("\n[Step 2/5] Loading held-out evaluation dataset (N=500)...")
    eval_df = load_evaluation_data()
    print(f"  -> Loaded evaluation records: {len(eval_df):,}")
    print(f"  -> Customer IDs present: {'customer_id' in eval_df.columns}")

    # 3. Initialize Interpreter Layer
    print("\n[Step 3/5] Initializing CustomerEffectInterpreter layer...")
    interpreter = CustomerEffectInterpreter(model, near_zero_threshold=near_zero_threshold)
    print(f"  -> Near-zero threshold: +-{near_zero_threshold * 100:.2f} percentage points (+-{near_zero_threshold})")
    print("  -> Customer ID status: STRICTLY QUARANTINED (retained only for index alignment)")

    # 4. Estimate Customer Effects
    print("\n[Step 4/5] Computing CATE estimates and 95% Wald confidence intervals...")
    results_df = interpreter.estimate_customer_effects(eval_df, alpha=0.05)
    print(f"  -> Computed effects for {len(results_df):,} customers.")
    print(f"  -> Results schema: {list(results_df.columns)}")

    # 5. Export Results & Summary
    print("\n[Step 5/5] Exporting results table and interpretation summary report...")
    reports_dir = os.path.join(PROJECT_ROOT, "reports")
    csv_path = os.path.join(reports_dir, "customer_treatment_effects.csv")
    summary_path = os.path.join(reports_dir, "customer_treatment_effects_summary.md")

    summary = interpreter.export_results(results_df, csv_path, summary_path)
    print(f"  -> CSV table saved to: {csv_path}")
    print(f"  -> Summary report saved to: {summary_path}")

    # Display Metrics & Category Breakdown
    cats = summary["category_distribution"]
    q_pp = summary["cate_quantiles_percentage_points"]

    print("\n" + "-" * 75)
    print("  CUSTOMER EFFECT CATEGORY DISTRIBUTION:")
    for cat_name, info in cats.items():
        print(f"  * {cat_name.upper():12s}: {info['count']:4d} customers ({info['percentage']:5.1f}%) — {info['description']}")

    print("\n  CATE DISTRIBUTION (PERCENTAGE POINTS LIFT):")
    print(f"  * Mean (ATE):   +{q_pp['mean']:.2f} pp")
    print(f"  * Std Dev:       {q_pp['std']:.2f} pp")
    print(f"  * Min / Max:    +{q_pp['min']:.2f} pp / +{q_pp['max']:.2f} pp")
    print(f"  * 25% / 75%:    +{q_pp['q25']:.2f} pp / +{q_pp['q75']:.2f} pp")
    print(f"  * Median (50%): +{q_pp['median']:.2f} pp")

    print("\n  SAMPLE RESULTS PREVIEW (First 5 Customers):")
    sample_cols = [
        "customer_id",
        "estimated_effect_percentage_points",
        "ci_lower_percentage_points",
        "ci_upper_percentage_points",
        "effect_category",
    ]
    print(results_df[sample_cols].head(5).to_string(index=False))

    print("\n" + "-" * 75)
    print("  CRITICAL CAUSAL INTERPRETATION NOTE:")
    print("  These outputs represent CONDITIONAL AVERAGE TREATMENT EFFECTS (CATE) for customer")
    print("  segments with identical observed attributes X. They are NOT proven individualized")
    print("  counterfactuals (ITE). True causality relies on Week 1 DAG assumptions (unconfoundedness,")
    print("  positivity, no interference) and model specification validity.")
    print("-" * 75)
    print("\n[SUCCESS] Customer-level causal effect estimation completed successfully!\n")


if __name__ == "__main__":
    run_customer_estimation()
