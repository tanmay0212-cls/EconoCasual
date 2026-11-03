"""Multi-tier promotional discount dataset generator for EconoCausal Week 3.

This module generates a synthetic retail customer dataset supporting three
discrete treatment arms corresponding to promotional discount amounts:

    T = 0  →  No discount  ($0)
    T = 10 →  Standard coupon ($10)
    T = 20 →  Premium coupon  ($20)

The dataset is PURPOSE-BUILT for multi-arm causal estimation, budget-constrained
optimization, and counterfactual revenue modelling. It is entirely ADDITIVE and does
NOT touch the original Week 1/2 files:
    - sample_data/synthetic_retail_campaign.csv  (binary DML training data)
    - src/data/generator.py                      (binary campaign DGP)
    - src/data/schema.py                         (binary schema definitions)

LABEL: SYNTHETIC DATASET — For causal inference research, optimisation, and benchmarking.
       Does NOT contain real consumer data or personally identifiable information.

Data-Generating Process (DGP) Summary
======================================
1. Customer pre-treatment features are drawn from documented distributions.
2. Each customer is assigned a ``customer_segment`` that governs their baseline
   purchase propensity and price-sensitivity.
3. Treatment (discount_level) is assigned via a documented multinomial propensity
   mechanism that deliberately correlates with customer segment (marketing selection
   bias) while maintaining adequate overlap across all groups (positivity).
4. Purchase outcome is sampled from segment- and discount-specific logistic models
   so that heterogeneous treatment effects (HTEs) are encoded as ground truth.
5. Five key economic figures are computed per row using a transparent accounting model.
6. Potential-outcome columns (``y0_potential``, ``y10_potential``, ``y20_potential``)
   and ground-truth CATE columns (``tau_10_true``, ``tau_20_true``) are persisted so
   that model estimates can be evaluated against the known generating mechanism.

Economics Conventions
======================
- Discount is incurred ONLY when the customer both receives an offer AND makes a purchase
  (redemption model, not broadcast cost model).
- ``gross_revenue``   = product_price  if purchased == 1, else 0
- ``discount_cost``   = discount_amount if purchased == 1, else 0
- ``net_revenue``     = gross_revenue - discount_cost
- ``gross_profit_before_discount`` = gross_revenue × gross_margin_rate
- ``net_profit_after_discount``    = gross_profit_before_discount - discount_cost
- Cost-of-goods is captured implicitly through ``gross_margin_rate``; no explicit COGS
  column is included, but it equals ``gross_revenue × (1 - gross_margin_rate)``.

Author : EconoCausal Engineering Team (Week 3 extension)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Public constants — treatment level mapping
# ---------------------------------------------------------------------------

DISCOUNT_LEVELS: list[int] = [0, 10, 20]
"""Discrete treatment arms (dollar face-value of discount coupon)."""

CUSTOMER_SEGMENTS: list[str] = [
    "budget_seeker",
    "casual_shopper",
    "regular_buyer",
    "vip_loyalist",
]
"""Four synthetic customer archetypes with distinct elasticity profiles."""

CAMPAIGN_CHANNELS: list[str] = ["email", "push", "sms", "social"]

# Required columns in the output DataFrame (subset used for causal estimation)
OPT_REQUIRED_COLUMNS: list[str] = [
    "customer_id",
    "age",
    "loyalty_score",
    "previous_purchases",
    "customer_segment",
    "campaign_channel",
    "customer_timestamp",
    "product_price",
    "gross_margin_rate",
    "discount_level",
    "discount_amount",
    "purchase",
    "gross_revenue",
    "discount_cost",
    "net_revenue",
    "gross_profit_before_discount",
    "net_profit_after_discount",
    "y0_potential",
    "y10_potential",
    "y20_potential",
    "tau_10_true",
    "tau_20_true",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax for (N, K) logit array."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Documented DGP parameters (exported to JSON)
# ---------------------------------------------------------------------------

DGP_PARAMETERS: Dict[str, Any] = {
    "dataset_name": "optimization_retail_data",
    "version": "1.0.0",
    "purpose": (
        "Multi-arm causal estimation dataset for Week 3 budget-constrained "
        "discount optimisation in EconoCausal."
    ),
    "treatment_levels": DISCOUNT_LEVELS,
    "customer_segments": CUSTOMER_SEGMENTS,
    "campaign_channels": CAMPAIGN_CHANNELS,
    # ── pre-treatment feature distributions ──────────────────────────────────
    "features": {
        "age": {
            "distribution": "Normal(mu=42, sigma=12) clipped to [18, 75]",
            "role": "confounder",
            "note": "Younger customers are more price-sensitive.",
        },
        "loyalty_score": {
            "distribution": "Beta(alpha=2, beta=3) × 100 clipped to [1, 99]",
            "role": "confounder",
            "note": (
                "Right-skewed toward lower values; VIP segment boosted +20 pts. "
                "High loyalty → lower price-sensitivity (already converts)."
            ),
        },
        "previous_purchases": {
            "distribution": "Poisson(lambda varies by segment [2, 4, 7, 14])",
            "role": "confounder",
            "note": "Proxy for brand loyalty; influences both targeting and conversion.",
        },
        "customer_segment": {
            "distribution": "Categorical [0.25, 0.30, 0.30, 0.15]",
            "levels": CUSTOMER_SEGMENTS,
            "note": "Governs baseline conversion, price elasticity, and product tier.",
        },
        "campaign_channel": {
            "distribution": "Categorical [0.38, 0.28, 0.20, 0.14]",
            "levels": CAMPAIGN_CHANNELS,
            "note": "Email and push channels deliver coupons at higher rates.",
        },
        "product_price": {
            "distribution": "LogNormal per segment; means ≈ [$35, $55, $80, $150]",
            "role": "context_variable",
            "note": (
                "Reflects the typical basket value for each customer archetype. "
                "Determines absolute dollar impact of discounts."
            ),
        },
        "gross_margin_rate": {
            "distribution": "Normal per segment; means ≈ [0.30, 0.38, 0.45, 0.55] ± 0.04",
            "role": "context_variable",
            "note": (
                "Budget-seekers buy commodity items (thin margins); "
                "VIPs buy premium goods (thick margins)."
            ),
        },
        "customer_timestamp": {
            "distribution": "Uniform draw from ['2026-Q1','2026-Q2','2026-Q3','2026-Q4']",
            "role": "temporal_batch",
            "note": "Campaign quarter identifier for drift-detection baseline creation.",
        },
    },
    # ── treatment-assignment mechanism ───────────────────────────────────────
    "treatment_assignment": {
        "mechanism": "Multinomial-logistic propensity with segment bias + Gumbel noise",
        "description": (
            "Each customer's probability of receiving each discount tier is determined "
            "by their customer_segment (marketing selection bias) and idiosyncratic noise. "
            "Overlap is guaranteed: no tier has zero probability for any segment."
        ),
        "segment_logit_biases": {
            # [logit_T0, logit_T10, logit_T20]
            "budget_seeker":   [0.0,  0.40, 0.80],
            "casual_shopper":  [0.30, 0.60, 0.30],
            "regular_buyer":   [0.50, 0.50, 0.10],
            "vip_loyalist":    [0.80, 0.20, 0.00],
        },
        "noise_scale": 1.0,
        "positivity_note": (
            "After softmax with noise scale=1.0, every tier receives "
            "at least ~8% probability across all segments."
        ),
    },
    # ── outcome generation ───────────────────────────────────────────────────
    "outcome_generation": {
        "model": "Logistic (Bernoulli) with segment-specific intercepts and HTEs",
        "baseline_logits": {
            "budget_seeker":  -1.80,   # baseline conv ~14%
            "casual_shopper": -0.80,   # baseline conv ~31%
            "regular_buyer":  -0.45,   # baseline conv ~39%
            "vip_loyalist":    0.50,   # baseline conv ~62%
        },
        "covariate_effects": {
            "age_norm":               -0.20,
            "loyalty_norm":            0.55,
            "prev_purchases_norm":     0.35,
        },
        "channel_effects": {
            "email":  0.15,
            "push":   0.25,
            "sms":   -0.05,
            "social":-0.15,
        },
        "treatment_lift_on_logit": {
            "T10_by_segment": {
                "budget_seeker":  0.65,
                "casual_shopper": 0.55,
                "regular_buyer":  0.45,
                "vip_loyalist":   0.12,
            },
            "T20_by_segment": {
                "budget_seeker":  1.10,
                "casual_shopper": 0.85,
                "regular_buyer":  0.62,
                "vip_loyalist":   0.18,
            },
            "price_sensitivity_modifier": {
                "formula": "sensitivity = (1 - loyalty_norm) × max(0, 1 - age_norm)",
                "coefficient": 0.30,
                "note": (
                    "Younger, less-loyal customers receive up to +0.30 additional logit "
                    "lift from discounts, capturing individual-level HTE heterogeneity."
                ),
            },
        },
        "heterogeneity_note": (
            "Treatment effects are non-linear and segment-heterogeneous. "
            "The marginal lift of moving from $10 to $20 is smaller than moving from $0 to $10 "
            "for every segment (diminishing returns in conversion probability). "
            "Combined with thin margins, a $20 discount may generate LESS net profit than $10 "
            "for budget_seeker and casual_shopper segments."
        ),
    },
    # ── economics ────────────────────────────────────────────────────────────
    "economics": {
        "discount_incurrence": "redemption_only",
        "description": (
            "Discount cost is incurred ONLY when (a) the customer receives an offer "
            "AND (b) the customer purchases. A customer who receives a coupon but does "
            "not purchase incurs zero discount cost."
        ),
        "accounting": {
            "gross_revenue":               "product_price  × purchase",
            "discount_cost":               "discount_amount × purchase",
            "net_revenue":                 "gross_revenue  - discount_cost",
            "gross_profit_before_discount":"gross_revenue  × gross_margin_rate",
            "net_profit_after_discount":   "gross_profit_before_discount - discount_cost",
        },
    },
}


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def generate_optimization_dataset(
    n_samples: int = 15_000,
    random_seed: int = 42,
    output_filepath: Optional[str] = None,
    dgp_params_filepath: Optional[str] = None,
) -> pd.DataFrame:
    """Generate the multi-arm optimization dataset.

    Args:
        n_samples:          Number of synthetic customer records (≥ 5,000).
        random_seed:        RNG seed for exact reproducibility.
        output_filepath:    If provided, saves the DataFrame as a CSV.
        dgp_params_filepath:If provided, saves DGP_PARAMETERS as a JSON file.

    Returns:
        pd.DataFrame with all columns listed in OPT_REQUIRED_COLUMNS.

    Raises:
        ValueError: If ``n_samples`` < 5,000.
    """
    if n_samples < 5_000:
        raise ValueError(
            f"n_samples must be at least 5,000 (received {n_samples})."
        )

    rng = np.random.default_rng(seed=random_seed)

    # ── 1. Customer segment assignment ───────────────────────────────────────
    segment_probs = [0.25, 0.30, 0.30, 0.15]
    segments_idx = rng.choice(len(CUSTOMER_SEGMENTS), size=n_samples, p=segment_probs)
    customer_segment = np.array(CUSTOMER_SEGMENTS)[segments_idx]

    # ── 2. Pre-treatment features ────────────────────────────────────────────
    # Age: Normal, segment-adjusted baseline age
    age_raw = rng.normal(loc=42.0, scale=12.0, size=n_samples)
    age = np.clip(np.round(age_raw), 18, 75).astype(int)

    # Loyalty score: Beta-distributed, boosted for VIPs
    loyalty_raw = rng.beta(a=2.0, b=3.0, size=n_samples) * 100.0
    vip_mask = customer_segment == "vip_loyalist"
    loyalty_raw = np.where(vip_mask, np.minimum(loyalty_raw + 20.0, 99.0), loyalty_raw)
    loyalty_score = np.clip(np.round(loyalty_raw, 1), 1.0, 99.0)

    # Previous purchases: Poisson, lambda varies by segment
    segment_lambda_map = {
        "budget_seeker":   2.0,
        "casual_shopper":  4.0,
        "regular_buyer":   7.0,
        "vip_loyalist":   14.0,
    }
    prev_lambda = np.array([segment_lambda_map[s] for s in customer_segment])
    previous_purchases = rng.poisson(lam=prev_lambda)

    # Campaign channel: categorical
    channel_probs = [0.38, 0.28, 0.20, 0.14]
    channel_idx = rng.choice(len(CAMPAIGN_CHANNELS), size=n_samples, p=channel_probs)
    campaign_channel = np.array(CAMPAIGN_CHANNELS)[channel_idx]

    # Customer timestamp (campaign quarter)
    quarters = ["2026-Q1", "2026-Q2", "2026-Q3", "2026-Q4"]
    customer_timestamp = rng.choice(quarters, size=n_samples)

    # Product price: LogNormal, segment-specific
    price_mean_log_by_seg = {
        "budget_seeker":  np.log(35.0),
        "casual_shopper": np.log(55.0),
        "regular_buyer":  np.log(80.0),
        "vip_loyalist":   np.log(150.0),
    }
    price_sigma_log = 0.35
    price_log_mean = np.array([price_mean_log_by_seg[s] for s in customer_segment])
    product_price = np.round(
        np.clip(
            np.exp(rng.normal(price_log_mean, price_sigma_log)),
            8.0, 600.0
        ),
        2,
    )

    # Gross margin rate: Normal, segment-specific
    margin_mean_by_seg = {
        "budget_seeker":  0.30,
        "casual_shopper": 0.38,
        "regular_buyer":  0.45,
        "vip_loyalist":   0.55,
    }
    margin_sigma = 0.04
    margin_mean = np.array([margin_mean_by_seg[s] for s in customer_segment])
    gross_margin_rate = np.round(
        np.clip(rng.normal(margin_mean, margin_sigma), 0.10, 0.80), 4
    )

    # ── 3. Treatment assignment ──────────────────────────────────────────────
    # Multinomial propensity: segment logit biases + Gumbel noise
    seg_biases = {
        "budget_seeker":  [0.00, 0.40, 0.80],
        "casual_shopper": [0.30, 0.60, 0.30],
        "regular_buyer":  [0.50, 0.50, 0.10],
        "vip_loyalist":   [0.80, 0.20, 0.00],
    }
    base_logits = np.array([seg_biases[s] for s in customer_segment])  # (N, 3)
    noise = rng.normal(scale=1.0, size=(n_samples, 3))
    treatment_logits = base_logits + noise
    treatment_probs = _softmax_rows(treatment_logits)  # (N, 3)

    # Sample discount_level ∈ {0, 10, 20}
    discount_level_idx = np.array([
        rng.choice(3, p=treatment_probs[i]) for i in range(n_samples)
    ])
    discount_level = np.array(DISCOUNT_LEVELS)[discount_level_idx]
    discount_amount = discount_level.astype(float)

    # ── 4. Potential-outcome computation (ground truth) ──────────────────────
    # Normalised covariates for logit construction
    age_norm = (age - 42.0) / 12.0
    loyalty_norm = (loyalty_score - 50.0) / 30.0
    prev_norm = np.log1p(previous_purchases) / np.log1p(14.0)

    # Baseline logits by segment
    baseline_logit_map = {
        "budget_seeker":  -1.80,
        "casual_shopper": -0.80,
        "regular_buyer":  -0.45,
        "vip_loyalist":    0.50,
    }
    baseline_logit = np.array([baseline_logit_map[s] for s in customer_segment])

    # Covariate contributions
    covariate_effect = (
        -0.20 * age_norm
        + 0.55 * loyalty_norm
        + 0.35 * prev_norm
    )

    # Channel effects
    channel_effect_map = {"email": 0.15, "push": 0.25, "sms": -0.05, "social": -0.15}
    channel_effect = np.array([channel_effect_map[c] for c in campaign_channel])

    # Price-sensitivity modifier (individual-level HTE heterogeneity)
    price_sensitivity = (
        np.maximum(0.0, 1.0 - loyalty_norm) * np.maximum(0.0, 1.0 - np.maximum(0.0, age_norm))
    )
    sensitivity_modifier = 0.30 * price_sensitivity  # up to +0.30 on logit

    # Treatment lift by segment
    t10_lift_map = {
        "budget_seeker":  0.65,
        "casual_shopper": 0.55,
        "regular_buyer":  0.45,
        "vip_loyalist":   0.12,
    }
    t20_lift_map = {
        "budget_seeker":  1.10,
        "casual_shopper": 0.85,
        "regular_buyer":  0.62,
        "vip_loyalist":   0.18,
    }
    tau10_base = np.array([t10_lift_map[s] for s in customer_segment])
    tau20_base = np.array([t20_lift_map[s] for s in customer_segment])

    # Total treatment lifts (logit scale) including HTE modifier
    tau10_logit = tau10_base + sensitivity_modifier
    tau20_logit = tau20_base + sensitivity_modifier

    # Logit under each treatment arm
    logit_T0  = baseline_logit + covariate_effect + channel_effect
    logit_T10 = logit_T0 + tau10_logit
    logit_T20 = logit_T0 + tau20_logit

    # Potential-outcome probabilities (ground truth, not observed)
    y0_potential  = np.round(_sigmoid(logit_T0),  6)
    y10_potential = np.round(_sigmoid(logit_T10), 6)
    y20_potential = np.round(_sigmoid(logit_T20), 6)

    # Ground-truth CATE (probability-scale lifts)
    tau_10_true = np.round(y10_potential - y0_potential, 6)
    tau_20_true = np.round(y20_potential - y0_potential, 6)

    # ── 5. Realised outcome ──────────────────────────────────────────────────
    # Sample actual purchase from the arm each customer was ASSIGNED to
    assignment_prob = np.where(
        discount_level == 0,  y0_potential,
        np.where(discount_level == 10, y10_potential, y20_potential)
    )
    purchase = rng.binomial(n=1, p=assignment_prob)

    # ── 6. Revenue and profit accounting ────────────────────────────────────
    # Discount ONLY incurred on successful purchase (redemption model)
    gross_revenue    = np.where(purchase == 1, product_price, 0.0)
    discount_cost_   = np.where(purchase == 1, discount_amount, 0.0)
    net_revenue      = gross_revenue - discount_cost_
    gross_profit_bd  = np.round(gross_revenue * gross_margin_rate, 4)
    net_profit_ad    = np.round(gross_profit_bd - discount_cost_, 4)

    gross_revenue   = np.round(gross_revenue, 2)
    discount_cost_  = np.round(discount_cost_, 2)
    net_revenue     = np.round(net_revenue, 2)

    # ── 7. Customer IDs ──────────────────────────────────────────────────────
    customer_id = [f"OPT_{i+1:06d}" for i in range(n_samples)]

    # ── 8. Assemble DataFrame ────────────────────────────────────────────────
    df = pd.DataFrame({
        "customer_id":                  customer_id,
        "age":                          age,
        "loyalty_score":                loyalty_score,
        "previous_purchases":           previous_purchases,
        "customer_segment":             customer_segment,
        "campaign_channel":             campaign_channel,
        "customer_timestamp":           customer_timestamp,
        "product_price":                product_price,
        "gross_margin_rate":            gross_margin_rate,
        "discount_level":               discount_level,
        "discount_amount":              discount_amount,
        "purchase":                     purchase,
        "gross_revenue":                gross_revenue,
        "discount_cost":                discount_cost_,
        "net_revenue":                  net_revenue,
        "gross_profit_before_discount": gross_profit_bd,
        "net_profit_after_discount":    net_profit_ad,
        "y0_potential":                 y0_potential,
        "y10_potential":                y10_potential,
        "y20_potential":                y20_potential,
        "tau_10_true":                  tau_10_true,
        "tau_20_true":                  tau_20_true,
    })

    # Enforce column order
    df = df[OPT_REQUIRED_COLUMNS]

    # ── 9. Save artifacts ────────────────────────────────────────────────────
    if output_filepath:
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        df.to_csv(output_filepath, index=False)

    if dgp_params_filepath:
        os.makedirs(os.path.dirname(os.path.abspath(dgp_params_filepath)), exist_ok=True)
        params_with_meta = dict(DGP_PARAMETERS)
        params_with_meta["n_samples"]   = n_samples
        params_with_meta["random_seed"] = random_seed
        with open(dgp_params_filepath, "w", encoding="utf-8") as fh:
            json.dump(params_with_meta, fh, indent=2)

    return df

