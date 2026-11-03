"""
Export precomputed drift evaluation reports and distribution histograms to frontend/public/drift_data.json.
Enables instant interactive inspection in the React dashboard with or without live API connection.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd

from src.monitoring.drift_detector import EconoCausalDriftDetector

def compute_histogram(series_ref: pd.Series, series_curr: pd.Series, n_bins: int = 10):
    """Compute synchronized histogram bins across reference and current series."""
    combined = pd.concat([series_ref.dropna(), series_curr.dropna()])
    min_val, max_val = float(combined.min()), float(combined.max())
    if min_val == max_val:
        min_val -= 0.5
        max_val += 0.5
    
    bins = np.linspace(min_val, max_val, n_bins + 1)
    ref_counts, _ = np.histogram(series_ref.dropna(), bins=bins)
    curr_counts, _ = np.histogram(series_curr.dropna(), bins=bins)
    
    ref_pct = (ref_counts / len(series_ref.dropna()) * 100).round(2).tolist()
    curr_pct = (curr_counts / len(series_curr.dropna()) * 100).round(2).tolist()
    
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(n_bins)]
    return {
        "labels": labels,
        "ref_pct": ref_pct,
        "curr_pct": curr_pct,
    }

def compute_category_bars(series_ref: pd.Series, series_curr: pd.Series):
    """Compute relative percentages for categorical variables across both batches."""
    categories = sorted(list(set(series_ref.dropna().unique()) | set(series_curr.dropna().unique())))
    ref_counts = series_ref.value_counts(normalize=True) * 100
    curr_counts = series_curr.value_counts(normalize=True) * 100
    
    ref_pct = [round(float(ref_counts.get(cat, 0.0)), 2) for cat in categories]
    curr_pct = [round(float(curr_counts.get(cat, 0.0)), 2) for cat in categories]
    
    return {
        "labels": categories,
        "ref_pct": ref_pct,
        "curr_pct": curr_pct,
    }

def main():
    root = Path(__file__).resolve().parent.parent
    ref_path = root / "data" / "drift_reference.csv"
    curr_path = root / "data" / "drift_current.csv"
    output_path = root / "frontend" / "public" / "drift_data.json"

    df_ref = pd.read_csv(ref_path)
    df_curr = pd.read_csv(curr_path)
    
    detector = EconoCausalDriftDetector(reference_data=df_ref)
    
    # 1. Evaluate drifted cohort (Reference vs Current)
    shifted_report = detector.detect_drift(df_curr)
    
    # 2. Evaluate in-distribution cohort (Reference vs 50% split sample of Reference)
    df_clean = df_ref.sample(n=len(df_ref) // 2, random_state=42).copy()
    clean_report = detector.detect_drift(df_clean)
    
    # 3. Compute distribution shapes for charts
    numerical_cols = ["age", "loyalty_score", "previous_purchases", "product_price", "gross_margin_rate"]
    categorical_cols = ["customer_segment", "campaign_channel"]
    
    shifted_distributions = {}
    for col in numerical_cols:
        shifted_distributions[col] = compute_histogram(df_ref[col], df_curr[col], n_bins=8)
    for col in categorical_cols:
        shifted_distributions[col] = compute_category_bars(df_ref[col], df_curr[col])
        
    clean_distributions = {}
    for col in numerical_cols:
        clean_distributions[col] = compute_histogram(df_ref[col], df_clean[col], n_bins=8)
    for col in categorical_cols:
        clean_distributions[col] = compute_category_bars(df_ref[col], df_clean[col])
        
    payload = {
        "metadata": {
            "reference_file": "data/drift_reference.csv",
            "reference_size": len(df_ref),
            "current_shifted_file": "data/drift_current.csv",
            "current_shifted_size": len(df_curr),
        },
        "batches": {
            "shifted": {
                "report": shifted_report.to_dict(),
                "distributions": shifted_distributions,
            },
            "clean": {
                "report": clean_report.to_dict(),
                "distributions": clean_distributions,
            }
        }
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        
    print(f"[OK] Exported drift dashboard payload to {output_path}")

if __name__ == "__main__":
    main()
