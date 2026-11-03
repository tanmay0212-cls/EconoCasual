"""Generate the multi-tier optimization retail dataset for EconoCausal Week 3.

Usage
-----
    python scripts/generate_optimization_dataset.py

Outputs
-------
    data/optimization_retail_data.csv        – 15,000-row synthetic dataset
    data/optimization_dgp_parameters.json   – Documented DGP parameter manifest

IMPORTANT: This script does NOT touch, read, or overwrite any Week 1/2 files.
"""

import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np

from src.data.optimization_generator import (
    generate_optimization_dataset,
    OPT_REQUIRED_COLUMNS,
    DISCOUNT_LEVELS,
    CUSTOMER_SEGMENTS,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR          = os.path.join(PROJECT_ROOT, "data")
OUTPUT_CSV        = os.path.join(DATA_DIR, "optimization_retail_data.csv")
OUTPUT_DGP_JSON   = os.path.join(DATA_DIR, "optimization_dgp_parameters.json")

# Safety guard: never overwrite the original Week 1 dataset
PROTECTED_CSV     = os.path.join(DATA_DIR, "mock_retail_data.csv")

N_SAMPLES   = 15_000
RANDOM_SEED = 42


def _section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main() -> None:
    # ── Guard ─────────────────────────────────────────────────────────────────
    if os.path.exists(PROTECTED_CSV):
        print(f"[GUARD] Protected file detected: {PROTECTED_CSV}")
        print("[GUARD] This script will NOT modify it. Proceeding with new file only.")

    # ── Generate ──────────────────────────────────────────────────────────────
    _section("Generating Multi-Tier Optimization Dataset")
    print(f"  n_samples   : {N_SAMPLES:,}")
    print(f"  random_seed : {RANDOM_SEED}")
    print(f"  Output CSV  : {OUTPUT_CSV}")
    print(f"  DGP JSON    : {OUTPUT_DGP_JSON}")

    df = generate_optimization_dataset(
        n_samples=N_SAMPLES,
        random_seed=RANDOM_SEED,
        output_filepath=OUTPUT_CSV,
        dgp_params_filepath=OUTPUT_DGP_JSON,
    )

    # ── Quick summary ──────────────────────────────────────────────────────────
    _section("Dataset Overview")
    print(f"  Shape        : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns      : {list(df.columns)}")
    print(f"\n  dtypes:\n{df.dtypes.to_string()}")

    # ── Treatment distribution ─────────────────────────────────────────────────
    _section("Treatment Distribution  (discount_level)")
    t_counts = df["discount_level"].value_counts().sort_index()
    for lvl, cnt in t_counts.items():
        print(f"  ${int(lvl):>2} discount : {cnt:,} customers ({cnt/N_SAMPLES*100:.1f}%)")

    # ── Treatment × Segment cross-tab ─────────────────────────────────────────
    _section("Treatment x Segment Cross-Tab (counts)")
    xtab = pd.crosstab(df["customer_segment"], df["discount_level"])
    xtab.columns = [f"T={c}" for c in xtab.columns]
    print(xtab.to_string())

    # ── Conversion rates by tier ───────────────────────────────────────────────
    _section("Conversion Rates by Discount Level")
    for lvl in DISCOUNT_LEVELS:
        sub = df[df["discount_level"] == lvl]
        rate = sub["purchase"].mean()
        print(f"  discount_level={lvl:>2} : {rate:.4f}  ({rate*100:.2f}%)")

    # ── Conversion rates by segment ────────────────────────────────────────────
    _section("Conversion Rates by Segment (across all tiers)")
    for seg in CUSTOMER_SEGMENTS:
        sub = df[df["customer_segment"] == seg]
        rate = sub["purchase"].mean()
        print(f"  {seg:<20}: {rate:.4f}  ({rate*100:.2f}%)")

    # ── Economics summary ──────────────────────────────────────────────────────
    _section("Economics Summary (Population Averages)")
    buyers = df[df["purchase"] == 1]
    print(f"  Overall purchase rate          : {df['purchase'].mean():.4f}")
    print(f"  Total gross revenue            : ${df['gross_revenue'].sum():>12,.2f}")
    print(f"  Total discount cost            : ${df['discount_cost'].sum():>12,.2f}")
    print(f"  Total net revenue              : ${df['net_revenue'].sum():>12,.2f}")
    print(f"  Total gross profit (pre-disc)  : ${df['gross_profit_before_discount'].sum():>12,.2f}")
    print(f"  Total net profit  (post-disc)  : ${df['net_profit_after_discount'].sum():>12,.2f}")
    print(f"\n  Avg product_price              : ${df['product_price'].mean():>8.2f}")
    print(f"  Avg gross_margin_rate          : {df['gross_margin_rate'].mean():.4f}")
    print(f"  Avg discount_cost (buyers only): ${buyers['discount_cost'].mean():.2f}")

    # ── Net profit by tier (key economics check) ───────────────────────────────
    _section("Avg Net-Profit per Customer by Discount Level")
    for lvl in DISCOUNT_LEVELS:
        sub = df[df["discount_level"] == lvl]
        avg_np = sub["net_profit_after_discount"].mean()
        print(f"  discount_level={lvl:>2} : avg net profit = ${avg_np:.4f}")

    # ── Ground-truth CATE summary ──────────────────────────────────────────────
    _section("Ground-Truth CATE Summary (Potential Outcome Differences)")
    print(f"  Mean tau_10_true  : {df['tau_10_true'].mean():.6f}  ({df['tau_10_true'].mean()*100:.2f} pp)")
    print(f"  Mean tau_20_true  : {df['tau_20_true'].mean():.6f}  ({df['tau_20_true'].mean()*100:.2f} pp)")
    print(f"  Std  tau_10_true  : {df['tau_10_true'].std():.6f}")
    print(f"  Std  tau_20_true  : {df['tau_20_true'].std():.6f}")
    print(f"  Min  tau_10_true  : {df['tau_10_true'].min():.6f}")
    print(f"  Max  tau_10_true  : {df['tau_10_true'].max():.6f}")
    print(f"  Min  tau_20_true  : {df['tau_20_true'].min():.6f}")
    print(f"  Max  tau_20_true  : {df['tau_20_true'].max():.6f}")

    _section("CATE by Segment (tau_10_true vs tau_20_true)")
    for seg in CUSTOMER_SEGMENTS:
        sub = df[df["customer_segment"] == seg]
        m10 = sub["tau_10_true"].mean() * 100
        m20 = sub["tau_20_true"].mean() * 100
        print(f"  {seg:<20}: tau_10={m10:+.2f} pp  tau_20={m20:+.2f} pp")

    # ── Overlap check ──────────────────────────────────────────────────────────
    _section("Overlap Check (treatment proportions within each segment)")
    for seg in CUSTOMER_SEGMENTS:
        sub = df[df["customer_segment"] == seg]
        props = sub["discount_level"].value_counts(normalize=True).sort_index()
        prop_str = "  ".join([f"T={int(k)}: {v*100:.1f}%" for k, v in props.items()])
        print(f"  {seg:<20}: {prop_str}")

    # ── Artifacts written ─────────────────────────────────────────────────────
    _section("Artifacts Written")
    print(f"  CSV  -> {OUTPUT_CSV}")
    print(f"  JSON -> {OUTPUT_DGP_JSON}")
    print(f"\n[DONE] Generation complete. Dataset contains {len(df):,} records.")
    print(f"       The original Week 1 dataset was NOT modified.")


if __name__ == "__main__":
    main()
