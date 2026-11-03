"""Script to generate drift reference and simulated future customer drift batches.

This script creates:
1. data/drift_reference.csv: Clean baseline population (N=5,000) drawn from the reference distribution.
2. data/drift_current.csv: Simulated future campaign batch (N=3,500) exhibiting realistic covariate shifts:
   - Age shifted younger (e.g. social acquisition campaign)
   - Loyalty score shifted downward (influx of new first-time users)
   - Customer segment shifted toward 'budget_seeker'
   - Campaign channels shifted toward 'social' and 'push'
   - Baseline product price and gross margins remain unshifted.

NEVER overwrites data/mock_retail_data.csv or data/optimization_retail_data.csv.
"""

import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OPTIMIZATION_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "optimization_retail_data.csv")
DRIFT_REF_PATH = os.path.join(PROJECT_ROOT, "data", "drift_reference.csv")
DRIFT_CUR_PATH = os.path.join(PROJECT_ROOT, "data", "drift_current.csv")


def generate_drift_datasets():
    """Generate drift reference and current evaluation datasets."""
    print("Generating EconoCausal drift datasets...")

    if not os.path.exists(OPTIMIZATION_DATA_PATH):
        raise FileNotFoundError(f"Baseline dataset not found at: {OPTIMIZATION_DATA_PATH}")

    # Load baseline optimization dataset
    base_df = pd.read_csv(OPTIMIZATION_DATA_PATH)
    print(f"Loaded baseline dataset: {len(base_df):,} rows.")

    # 1. Create drift_reference.csv (5,000 rows sampled from the training partition)
    # Rows 0-12,000 correspond to training data
    np.random.seed(42)
    ref_indices = np.random.choice(12000, size=5000, replace=False)
    ref_df = base_df.iloc[ref_indices].copy().reset_index(drop=True)
    ref_df.to_csv(DRIFT_REF_PATH, index=False)
    print(f"Created drift reference dataset: {DRIFT_REF_PATH} ({len(ref_df):,} rows).")

    # 2. Create drift_current.csv (3,500 rows with controlled, realistic covariate shifts)
    np.random.seed(2026)
    n_current = 3500

    # Demographics: younger age distribution (e.g., social/influencer campaign)
    # Baseline mean ~42.1 -> Shifted mean ~33.5
    age = np.clip(np.random.normal(loc=33.5, scale=9.0, size=n_current), 18.0, 75.0).round(1)

    # Loyalty score: influx of new users with lower historical tenure and loyalty
    # Baseline mean ~51.4 -> Shifted mean ~36.8
    loyalty_score = np.clip(np.random.beta(a=2.0, b=3.5, size=n_current) * 100.0, 0.0, 100.0).round(2)

    # Previous purchases: lower purchase frequency
    # Baseline mean ~4.5 -> Shifted mean ~2.1
    previous_purchases = np.random.poisson(lam=2.1, size=n_current)

    # Customer segment: heavy shift toward 'budget_seeker' (45% vs baseline 25%)
    segments = ["budget_seeker", "casual_shopper", "regular_buyer", "vip_loyalist"]
    seg_probs = [0.48, 0.30, 0.16, 0.06]  # VIPs reduced from 20% to 6%
    customer_segment = np.random.choice(segments, size=n_current, p=seg_probs)

    # Campaign channel: shift toward mobile push and social media ads
    channels = ["email", "push", "sms", "social"]
    chan_probs = [0.15, 0.40, 0.10, 0.35]  # Email reduced from 40% to 15%
    campaign_channel = np.random.choice(channels, size=n_current, p=chan_probs)

    # Stable baseline product economics (Price and margin remain constant in catalog)
    # Mean price ~85.0, Mean margin ~0.40 (Identical to reference!)
    product_price = np.clip(np.random.normal(loc=85.0, scale=12.0, size=n_current), 40.0, 180.0).round(2)
    gross_margin_rate = np.clip(np.random.normal(loc=0.40, scale=0.04, size=n_current), 0.20, 0.60).round(3)

    # Treatment assignment and simulated purchase
    # 33% per tier
    discount_level = np.random.choice([0, 10, 20], size=n_current, p=[0.34, 0.33, 0.33])
    discount_amount = discount_level.astype(float)

    # Purchase outcome influenced by shifted features
    base_prob = 1.0 / (1.0 + np.exp(-(
        -0.8
        + 0.02 * loyalty_score
        - 0.015 * (age - 35)
        + 0.15 * (discount_level / 10.0)
        - 0.3 * (customer_segment == "budget_seeker")
    )))
    purchase = (np.random.uniform(0, 1, size=n_current) < base_prob).astype(int)

    customer_ids = [f"FUT_{i+1:06d}" for i in range(n_current)]

    cur_df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": age,
        "loyalty_score": loyalty_score,
        "previous_purchases": previous_purchases,
        "customer_segment": customer_segment,
        "campaign_channel": campaign_channel,
        "discount_level": discount_level,
        "discount_amount": discount_amount,
        "purchase": purchase,
        "product_price": product_price,
        "gross_margin_rate": gross_margin_rate,
    })

    cur_df.to_csv(DRIFT_CUR_PATH, index=False)
    print(f"Created simulated future drift dataset: {DRIFT_CUR_PATH} ({len(cur_df):,} rows).")
    print("Generation complete.")


if __name__ == "__main__":
    generate_drift_datasets()

