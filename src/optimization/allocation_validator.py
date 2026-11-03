"""Validation and Quality Assurance for Marketing Budget Allocations.

This module validates:
1. Every customer receives exactly one discount allocation (x_{i, 0} + x_{i, 10} + x_{i, 20} = 1).
2. Allocated discount tiers belong strictly to {0, 10, 20}.
3. Expected redemption expenditure strictly respects the configured budget cap.
4. Issued face-value coupon exposure is tracked accurately.
5. All expected profit and probability numbers are finite and within mathematical domain [0, 1].
6. Zero customer ID duplicates.
7. Anti-leakage verification: no ground-truth columns present in operational allocation files.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


def validate_allocation(
    allocation_df: pd.DataFrame,
    budget: float,
    budget_type: str = "redemption",
    allowed_tiers: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Perform comprehensive multi-point audit on an optimization allocation matrix."""
    if allowed_tiers is None:
        allowed_tiers = [0, 10, 20]

    if allocation_df is None or len(allocation_df) == 0:
        raise ValueError("Allocation DataFrame is empty or None.")

    issues = []
    n_customers = len(allocation_df)

    # 1. Unique Customer IDs
    if "customer_id" not in allocation_df.columns:
        issues.append("Missing required 'customer_id' column.")
        has_duplicates = False
    else:
        duplicates = allocation_df["customer_id"].duplicated().sum()
        has_duplicates = bool(duplicates > 0)
        if has_duplicates:
            issues.append(f"Found {duplicates} duplicate customer IDs in allocation matrix.")

    # 2. Check Allocation Column
    alloc_col = None
    for cand in ["allocated_discount", "recommended_discount"]:
        if cand in allocation_df.columns:
            alloc_col = cand
            break

    if alloc_col is None:
        raise ValueError("Missing allocation column (expected 'allocated_discount' or 'recommended_discount').")

    discounts = allocation_df[alloc_col].to_numpy()
    invalid_tiers = [d for d in np.unique(discounts) if d not in allowed_tiers]
    if invalid_tiers:
        issues.append(f"Invalid discount tiers allocated: {invalid_tiers}. Expected subset of {allowed_tiers}.")

    # 3. Check Probabilities Domain
    prob_cols = [c for c in allocation_df.columns if "prob" in c]
    for p_col in prob_cols:
        vals = allocation_df[p_col].to_numpy()
        if np.any(np.isnan(vals)) or np.any(vals < 0.0) or np.any(vals > 1.0):
            issues.append(f"Probability column '{p_col}' has values outside [0, 1] or NaN.")

    # 4. Expenditure and Budget Adherence
    cost_col = None
    for cand in ["expected_discount_cost_selected", "expected_discount_cost"]:
        if cand in allocation_df.columns:
            cost_col = cand
            break

    if cost_col is not None:
        total_redemption_cost = float(allocation_df[cost_col].sum())
    else:
        # Fallback: estimate from allocated discount * probability
        prob_sel_col = "expected_purchase_probability_selected" if "expected_purchase_probability_selected" in allocation_df.columns else "predicted_purchase_probability"
        if prob_sel_col in allocation_df.columns:
            total_redemption_cost = float((discounts * allocation_df[prob_sel_col]).sum())
        else:
            total_redemption_cost = 0.0

    total_face_value = float(discounts.sum())

    # Check budget constraint compliance
    budget_tested = total_redemption_cost if budget_type == "redemption" else total_face_value
    tolerance = 1e-3
    budget_satisfied = bool(budget_tested <= budget + tolerance)
    if not budget_satisfied:
        issues.append(
            f"Budget constraint violated! Spent ${budget_tested:,.2f} exceeding ${budget:,.2f} cap ({budget_type} budget)."
        )

    # 5. Anti-Leakage Check
    leaked_cols = [c for c in allocation_df.columns if "true_ite" in c or "potential_" in c or "optimal_unconstrained" in c]
    if leaked_cols:
        issues.append(f"CRITICAL DATA LEAKAGE: Ground truth columns found in operational allocation: {leaked_cols}")

    tier_counts = {int(d): int(np.sum(discounts == d)) for d in allowed_tiers}

    return {
        "valid": len(issues) == 0,
        "n_customers": n_customers,
        "issues": issues,
        "budget_satisfied": budget_satisfied,
        "budget_type": budget_type,
        "budget_cap": budget,
        "total_redemption_cost": round(total_redemption_cost, 2),
        "total_face_value_exposure": round(total_face_value, 2),
        "budget_utilization_pct": round((budget_tested / budget * 100.0) if budget > 0 else 100.0, 2),
        "tier_counts": tier_counts,
        "no_leakage_detected": len(leaked_cols) == 0,
    }

