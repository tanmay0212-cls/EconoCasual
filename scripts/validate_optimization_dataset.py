"""Validation script for the multi-tier optimization retail dataset.

Checks:
    1.  Required columns present
    2.  No missing values in required columns
    3.  No duplicate customer_id values
    4.  discount_level contains only valid values {0, 10, 20}
    5.  discount_amount consistent with discount_level
    6.  purchase outcome is binary {0, 1}
    7.  Discount cost incurred only on purchase (redemption model integrity)
    8.  Revenue and profit sign/direction consistency
    9.  Numerical ranges for age, loyalty_score, product_price, gross_margin_rate
    10. Treatment group balance and overlap (positivity check)
    11. Potential-outcome ordering (tau_20 >= tau_10 >= 0 for the majority)
    12. Ground-truth CATE reasonableness
    13. Diminishing-returns check (tau_20 > tau_10 but marginal gain < tau_10)

Usage
-----
    python scripts/validate_optimization_dataset.py

Returns exit code 0 on full pass, 1 if any CRITICAL check fails.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.data.optimization_generator import OPT_REQUIRED_COLUMNS, DISCOUNT_LEVELS

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "optimization_retail_data.csv")

VALID_SEGMENTS  = {"budget_seeker", "casual_shopper", "regular_buyer", "vip_loyalist"}
VALID_CHANNELS  = {"email", "push", "sms", "social"}
VALID_QUARTERS  = {"2026-Q1", "2026-Q2", "2026-Q3", "2026-Q4"}
VALID_PURCHASES = {0, 1}
VALID_DISC_LVL  = {0, 10, 20}

MIN_SEGMENT_PROP_PER_TIER = 0.05   # Every segment must have ≥5% in each tier (positivity)
MAX_NULL_RATE             = 0.0    # Zero nulls tolerated in required columns

# ── Helper ─────────────────────────────────────────────────────────────────────
PASS  = "✅ PASS"
FAIL  = "❌ FAIL"
WARN  = "⚠️  WARN"

checks_run:    int = 0
checks_passed: int = 0
checks_failed: int = 0

results: list[dict] = []


def record(
    check_id: str,
    description: str,
    status: str,
    detail: str = "",
    critical: bool = True,
) -> None:
    global checks_run, checks_passed, checks_failed
    checks_run += 1
    ok = status == PASS
    if ok:
        checks_passed += 1
    elif critical:
        checks_failed += 1
    results.append({
        "id": check_id,
        "description": description,
        "status": status,
        "detail": detail,
        "critical": critical,
    })
    marker = "CRIT" if (critical and not ok) else ("WARN" if not ok else "    ")
    print(f"  [{marker}] {check_id:<5} {status}  {description}")
    if detail:
        for line in detail.split("\n"):
            print(f"               {line}")


def section(title: str) -> None:
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")


# ── Load ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  EconoCausal — Optimization Dataset Validation")
print(f"  File: {DATA_PATH}")
print(f"{'='*70}")

if not os.path.exists(DATA_PATH):
    print(f"\n{FAIL} Dataset file not found: {DATA_PATH}")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
N = len(df)
print(f"\n  Loaded {N:,} rows × {len(df.columns)} columns.")

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 1 — Required columns
# ═══════════════════════════════════════════════════════════════════════════════
section("Check 1 — Required Columns")
missing_cols = [c for c in OPT_REQUIRED_COLUMNS if c not in df.columns]
if not missing_cols:
    record("C1", "All required columns present", PASS,
           f"All {len(OPT_REQUIRED_COLUMNS)} required columns found.")
else:
    record("C1", "All required columns present", FAIL,
           f"Missing columns: {missing_cols}")

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 2 — Missing values
# ═══════════════════════════════════════════════════════════════════════════════
section("Check 2 — Missing Values")
avail_req = [c for c in OPT_REQUIRED_COLUMNS if c in df.columns]
null_counts = df[avail_req].isnull().sum()
total_nulls = null_counts.sum()
if total_nulls == 0:
    record("C2", "Zero missing values in required columns", PASS,
           "100% completeness across all required columns.")
else:
    cols_with_nulls = null_counts[null_counts > 0].to_dict()
    record("C2", "Zero missing values in required columns", FAIL,
           f"Null counts: {cols_with_nulls}")

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 3 — Duplicate customer_ids
# ═══════════════════════════════════════════════════════════════════════════════
section("Check 3 — Customer ID Uniqueness")
n_dupes = df["customer_id"].duplicated().sum()
if n_dupes == 0:
    record("C3", "No duplicate customer_id values", PASS, f"All {N:,} IDs are unique.")
else:
    record("C3", "No duplicate customer_id values", FAIL, f"{n_dupes:,} duplicates found.")

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 4 — Valid discount levels
# ═══════════════════════════════════════════════════════════════════════════════
section("Check 4 — Valid Discount Levels")
actual_lvls = set(df["discount_level"].unique())
invalid_lvls = actual_lvls - VALID_DISC_LVL
if not invalid_lvls:
    record("C4", "discount_level ∈ {0, 10, 20}", PASS,
           f"Levels found: {sorted(actual_lvls)}")
else:
    record("C4", "discount_level ∈ {0, 10, 20}", FAIL,
           f"Invalid levels: {invalid_lvls}")

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 5 — discount_amount consistent with discount_level
# ═══════════════════════════════════════════════════════════════════════════════
section("Check 5 — Discount Amount Consistency")
mismatch = (df["discount_amount"] != df["discount_level"].astype(float)).sum()
if mismatch == 0:
    record("C5", "discount_amount == discount_level for every row", PASS)
else:
    record("C5", "discount_amount == discount_level for every row", FAIL,
           f"{mismatch:,} mismatches found.")

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 6 — Binary purchase outcome
# ═══════════════════════════════════════════════════════════════════════════════
section("Check 6 — Binary Purchase Outcome")
actual_purchase_vals = set(df["purchase"].unique())
invalid_purch = actual_purchase_vals - VALID_PURCHASES
if not invalid_purch:
    record("C6", "purchase ∈ {0, 1}", PASS,
           f"Overall conversion rate: {df['purchase'].mean()*100:.2f}%")
else:
    record("C6", "purchase ∈ {0, 1}", FAIL,
           f"Unexpected values: {invalid_purch}")

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 7 — Redemption model integrity (discount cost only on purchase)
# ═══════════════════════════════════════════════════════════════════════════════
section("Check 7 — Redemption Model (Discount Cost Only on Purchase)")
# No purchase ⟹ discount_cost == 0
no_purchase_nonzero_cost = ((df["purchase"] == 0) & (df["discount_cost"] > 0)).sum()
# Purchase ⟹ discount_cost == discount_amount
purchase_cost_mismatch = (
    (df["purchase"] == 1) &
    (df["discount_cost"] != df["discount_amount"])
).sum()

if no_purchase_nonzero_cost == 0 and purchase_cost_mismatch == 0:
    record("C7a", "No discount_cost when purchase==0", PASS)
    record("C7b", "discount_cost == discount_amount when purchase==1", PASS)
else:
    if no_purchase_nonzero_cost > 0:
        record("C7a", "No discount_cost when purchase==0", FAIL,
               f"{no_purchase_nonzero_cost:,} violations.")
    else:
        record("C7a", "No discount_cost when purchase==0", PASS)
    if purchase_cost_mismatch > 0:
        record("C7b", "discount_cost == discount_amount when purchase==1", FAIL,
               f"{purchase_cost_mismatch:,} violations.")
    else:
        record("C7b", "discount_cost == discount_amount when purchase==1", PASS)

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 8 — Revenue and profit consistency
# ═══════════════════════════════════════════════════════════════════════════════
section("Check 8 — Revenue and Profit Consistency")
# gross_revenue = product_price when purchased, else 0
expected_gross = np.where(df["purchase"] == 1, df["product_price"], 0.0)
rev_ok = np.allclose(df["gross_revenue"], expected_gross, atol=0.015)

# net_revenue = gross_revenue - discount_cost
net_rev_ok = np.allclose(
    df["net_revenue"],
    df["gross_revenue"] - df["discount_cost"],
    atol=0.015,
)

# net_profit = gross_revenue * margin - discount_cost
expected_net_profit = df["gross_revenue"] * df["gross_margin_rate"] - df["discount_cost"]
np_ok = np.allclose(df["net_profit_after_discount"], expected_net_profit, atol=0.015)

record("C8a", "gross_revenue = product_price × purchase",
       PASS if rev_ok else FAIL)
record("C8b", "net_revenue = gross_revenue - discount_cost",
       PASS if net_rev_ok else FAIL)
record("C8c", "net_profit_after_discount = gross_profit_before_discount - discount_cost",
       PASS if np_ok else FAIL)

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 9 — Numerical ranges
# ═══════════════════════════════════════════════════════════════════════════════
section("Check 9 — Numerical Ranges")
range_checks = [
    ("age",               18,   75,  True),
    ("loyalty_score",      1,   99,  True),
    ("previous_purchases", 0, None,  True),  # non-negative
    ("product_price",      8,  600,  True),
    ("gross_margin_rate", 0.10, 0.80, True),
]
for col, lo, hi, crit in range_checks:
    if col not in df.columns:
        continue
    vals = df[col]
    lo_ok = (vals >= lo).all() if lo is not None else True
    hi_ok = (vals <= hi).all() if hi is not None else True
    ok = lo_ok and hi_ok
    detail = f"min={vals.min():.2f}  max={vals.max():.2f}"
    if lo is not None:
        detail += f"  (expected [{lo}, {hi if hi is not None else '∞'}])"
    record(f"C9-{col}", f"{col} within expected range", PASS if ok else WARN,
           detail, critical=crit)

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 10 — Treatment balance and overlap (positivity)
# ═══════════════════════════════════════════════════════════════════════════════
section("Check 10 — Treatment Balance and Overlap (Positivity)")
overall_props = df["discount_level"].value_counts(normalize=True).sort_index()
print(f"    Overall distribution: " +
      "  ".join([f"T={int(k)}: {v*100:.1f}%" for k, v in overall_props.items()]))

positivity_ok = True
for seg in sorted(df["customer_segment"].unique()):
    sub = df[df["customer_segment"] == seg]
    props = sub["discount_level"].value_counts(normalize=True)
    for lvl in DISCOUNT_LEVELS:
        prop = props.get(lvl, 0.0)
        if prop < MIN_SEGMENT_PROP_PER_TIER:
            positivity_ok = False
            print(f"    ⚠️  {seg} / T={lvl}: prop={prop:.3f} < {MIN_SEGMENT_PROP_PER_TIER}")

record("C10", "Positivity — every segment has ≥5% in each discount tier",
       PASS if positivity_ok else FAIL,
       "Overlap sufficient for valid causal identification." if positivity_ok else
       "Some segment/tier combinations below positivity threshold.")

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 11 — Potential-outcome ordering (monotonicity)
# ═══════════════════════════════════════════════════════════════════════════════
section("Check 11 — Potential-Outcome Monotonicity")
# y0 ≤ y10 ≤ y20 must hold for the vast majority (DGP guarantees this)
mono_10 = (df["y10_potential"] >= df["y0_potential"]).mean()
mono_20 = (df["y20_potential"] >= df["y10_potential"]).mean()
record("C11a", "y10_potential ≥ y0_potential (conversion monotone in discount)",
       PASS if mono_10 >= 0.99 else WARN,
       f"{mono_10*100:.2f}% of rows satisfy y10 ≥ y0",
       critical=False)
record("C11b", "y20_potential ≥ y10_potential (larger discount ≥ smaller discount, prob scale)",
       PASS if mono_20 >= 0.99 else WARN,
       f"{mono_20*100:.2f}% of rows satisfy y20 ≥ y10",
       critical=False)

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 12 — Ground-truth CATE reasonableness
# ═══════════════════════════════════════════════════════════════════════════════
section("Check 12 — Ground-Truth CATE Reasonableness")
tau10_neg = (df["tau_10_true"] < 0).mean()
tau20_neg = (df["tau_20_true"] < 0).mean()
tau10_huge = (df["tau_10_true"] > 0.60).mean()
tau20_huge = (df["tau_20_true"] > 0.60).mean()

record("C12a", "tau_10_true all non-negative",
       PASS if tau10_neg == 0 else WARN,
       f"{tau10_neg*100:.2f}% of rows have negative tau_10", critical=False)
record("C12b", "tau_20_true all non-negative",
       PASS if tau20_neg == 0 else WARN,
       f"{tau20_neg*100:.2f}% of rows have negative tau_20", critical=False)
record("C12c", "No unrealistically large effects (> 60 pp)",
       PASS if (tau10_huge + tau20_huge) == 0 else WARN,
       f"tau_10 > 60pp: {tau10_huge*100:.2f}%  tau_20 > 60pp: {tau20_huge*100:.2f}%",
       critical=False)

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 13 — Diminishing-returns economics
# ═══════════════════════════════════════════════════════════════════════════════
section("Check 13 — Diminishing Returns (Economics)")
# For budget_seeker segment (product_price ~$35, margin ~0.30):
# A $20 discount should often produce LOWER net_profit than $10
for seg in ["budget_seeker", "casual_shopper"]:
    sub10 = df[(df["customer_segment"] == seg) & (df["discount_level"] == 10)]
    sub20 = df[(df["customer_segment"] == seg) & (df["discount_level"] == 20)]
    if len(sub10) == 0 or len(sub20) == 0:
        continue
    avg_np10 = sub10["net_profit_after_discount"].mean()
    avg_np20 = sub20["net_profit_after_discount"].mean()
    diminishes = avg_np20 < avg_np10
    record(f"C13-{seg}",
           f"{seg}: avg net_profit(T=20) < avg net_profit(T=10)",
           PASS if diminishes else WARN,
           f"avg_net_profit T=10: ${avg_np10:.3f}  T=20: ${avg_np20:.3f}",
           critical=False)

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  VALIDATION SUMMARY")
print(f"{'='*70}")
print(f"  Checks run    : {checks_run}")
print(f"  Passed        : {checks_passed}")
print(f"  Critical fails: {checks_failed}")
warnings_count = checks_run - checks_passed - checks_failed
print(f"  Warnings      : {warnings_count}")
print()

if checks_failed == 0:
    print("  OVERALL STATUS: ✅ ALL CRITICAL CHECKS PASSED")
    print(f"  Dataset is valid and ready for causal estimation.")
    exit_code = 0
else:
    print(f"  OVERALL STATUS: ❌ {checks_failed} CRITICAL CHECK(S) FAILED")
    print("  Dataset requires remediation before causal estimation.")
    exit_code = 1

print(f"{'='*70}\n")
sys.exit(exit_code)

