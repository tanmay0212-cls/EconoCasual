"""
Preprocess the Hillstrom MineThatData Email Marketing Dataset.
Creates clean analytical datasets in data/hillstrom/processed/
without modifying data/hillstrom/raw/hillstrom_raw.csv.
"""
from pathlib import Path
import pandas as pd
import numpy as np

def preprocess_hillstrom():
    root = Path(__file__).resolve().parent.parent
    raw_file = root / "data" / "hillstrom" / "raw" / "hillstrom_raw.csv"
    processed_dir = root / "data" / "hillstrom" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PREPROCESSING HILLSTROM EMAIL MARKETING DATASET")
    print("=" * 70)

    if not raw_file.exists():
        raise FileNotFoundError(f"Raw file missing at: {raw_file}. Run scripts/download_hillstrom.py first.")

    df_raw = pd.read_csv(raw_file)
    print(f"Loaded raw dataset: {len(df_raw):,} records.")

    # 1. Clean column names & assign customer identifiers
    df = df_raw.copy()
    df.insert(0, "customer_id", [f"HILL_{i+1:06d}" for i in range(len(df))])

    # 2. Documented Treatment Group Mapping
    # Control = 0 ("No E-Mail")
    # Treatment 1 = 1 ("Mens E-Mail")
    # Treatment 2 = 2 ("Womens E-Mail")
    treatment_map = {
        "No E-Mail": 0,
        "Mens E-Mail": 1,
        "Womens E-Mail": 2,
    }
    label_map = {
        "No E-Mail": "no_email",
        "Mens E-Mail": "mens_email",
        "Womens E-Mail": "womens_email",
    }
    
    df["treatment_group"] = df["segment"].map(treatment_map)
    df["treatment_name"] = df["segment"].map(label_map)

    # 3. Covariates and Outcomes Classification
    # Pre-treatment covariates (available prior to email campaign):
    covariate_cols = [
        "customer_id",
        "recency",
        "history_segment",
        "history",
        "mens",
        "womens",
        "zip_code",
        "newbie",
        "channel",
    ]
    # Treatment indicators:
    treatment_cols = ["treatment_group", "treatment_name", "segment"]
    # Post-treatment outcomes (must NOT be used as baseline features):
    outcome_cols = ["visit", "conversion", "spend"]

    # Reorder columns logically: ID, Pre-treatment features, Treatment, Outcomes
    ordered_cols = (
        covariate_cols
        + treatment_cols
        + outcome_cols
    )
    df_processed = df[ordered_cols].copy()

    # 4. Stratified Train / Validation / Test Splitting (70% / 15% / 15%)
    # Stratify by treatment_group and conversion to preserve rare positive conversions
    np.random.seed(42)
    strata = df_processed["treatment_group"].astype(str) + "_" + df_processed["conversion"].astype(str)
    
    # Shuffle indices per stratum
    train_indices = []
    val_indices = []
    test_indices = []

    for _, group in df_processed.groupby(strata):
        shuffled = np.random.permutation(group.index.to_numpy())
        n = len(shuffled)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        
        train_indices.extend(shuffled[:n_train])
        val_indices.extend(shuffled[n_train:n_train + n_val])
        test_indices.extend(shuffled[n_train + n_val:])

    train_df = df_processed.loc[train_indices].sort_values("customer_id").reset_index(drop=True)
    val_df = df_processed.loc[val_indices].sort_values("customer_id").reset_index(drop=True)
    test_df = df_processed.loc[test_indices].sort_values("customer_id").reset_index(drop=True)

    # 5. Export processed analytical datasets
    out_master = processed_dir / "hillstrom_processed.csv"
    out_train = processed_dir / "hillstrom_train.csv"
    out_val = processed_dir / "hillstrom_validation.csv"
    out_test = processed_dir / "hillstrom_test.csv"

    df_processed.to_csv(out_master, index=False)
    train_df.to_csv(out_train, index=False)
    val_df.to_csv(out_val, index=False)
    test_df.to_csv(out_test, index=False)

    print("\n[OK] Processed Hillstrom datasets successfully created:")
    print(f"  - Master: {out_master} ({len(df_processed):,} rows)")
    print(f"  - Train:  {out_train} ({len(train_df):,} rows, 70%)")
    print(f"  - Val:    {out_val} ({len(val_df):,} rows, 15%)")
    print(f"  - Test:   {out_test} ({len(test_df):,} rows, 15%)")

    # 6. Summary Statistics & Sanity Checks
    print("\nTreatment Group Breakdown:")
    for grp_id, name in [(0, "No E-Mail"), (1, "Mens E-Mail"), (2, "Womens E-Mail")]:
        sub = df_processed[df_processed["treatment_group"] == grp_id]
        conv_rate = sub["conversion"].mean() * 100
        visit_rate = sub["visit"].mean() * 100
        avg_spend = sub["spend"].mean()
        print(f"  - Group {grp_id} ({name}): N={len(sub):,} | Visit={visit_rate:.2f}% | Conversion={conv_rate:.2f}% | Avg Spend=${avg_spend:.2f}")

if __name__ == "__main__":
    preprocess_hillstrom()
