"""Reproducible synthetic retail marketing campaign data generator.

This module generates a realistic synthetic retail marketing dataset with
documented confounding between customer attributes, marketing discount assignment,
and purchase conversion.

LABEL: SYNTHETIC DATASET - For causal inference research, testing, and benchmarking.
"""

from typing import Optional
import os
import numpy as np
import pandas as pd

from .schema import (
    REQUIRED_COLUMNS,
    TREATMENT_COL,
    OUTCOME_COL,
    validate_dataset,
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Standard numerically stable logistic sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))


def generate_synthetic_campaign_data(
    n_samples: int = 2500,
    random_seed: int = 42,
    output_filepath: Optional[str] = None,
) -> pd.DataFrame:
    """Generate reproducible synthetic retail campaign dataset with known confounding.

    Causal Data-Generating Process (DGP):
    1. Pre-treatment customer characteristics (Confounders W):
       - customer_tenure: Months customer has been with the retailer (1 to 60).
       - prior_purchases: Historical order frequency, positively skewed.
       - prior_spend: Historical monetary value, correlated with prior_purchases.
       - engagement_score: Normalized app/email activity metric (0 - 100).
       - campaign_channel: Marketing outreach medium (email, push, sms, social).

    2. Treatment Assignment Propensity P(T=1 | W):
       - Retail marketers disproportionately target engaged customers and loyalty tiers
         with promotional discount codes (marketing selection bias).

    3. Outcome Generation Propensity P(Y=1 | T, W):
       - Purchase conversion is determined by both the discount offer (true causal treatment)
         AND customer baseline propensity (prior purchases, engagement, loyalty).
       - Because W affects both T and Y, naive comparison between treated and untreated groups
         suffers from confounding bias.

    4. Downstream Post-Treatment Variables (Not confounders):
       - discount_amount: Realized dollar discount value (0 if no discount offered).
       - revenue: Final realized basket revenue post-discount (0 if not purchased).

    Args:
        n_samples: Number of customer records to generate (minimum 1,000).
        random_seed: Random seed for exact reproducibility.
        output_filepath: Optional filepath to save the resulting CSV.

    Returns:
        pd.DataFrame containing synthetic campaign records.
    """
    if n_samples < 1000:
        raise ValueError(f"n_samples must be at least 1,000 (received {n_samples}).")

    rng = np.random.default_rng(seed=random_seed)

    # 1. Generate Pre-Treatment Observed Confounders (W)
    # Customer tenure (in months): Gamma distribution skewed towards newer/mid-term customers
    tenure_raw = rng.gamma(shape=2.5, scale=8.0, size=n_samples)
    customer_tenure = np.clip(np.round(tenure_raw + 1), 1, 72).astype(int)

    # Prior purchase count: Poisson distribution conditioned on tenure
    tenure_factor = np.log1p(customer_tenure) / 2.0
    prior_purchases_lambda = np.clip(2.0 + 1.8 * tenure_factor, 1.0, 20.0)
    prior_purchases = rng.poisson(lam=prior_purchases_lambda)

    # Prior spend ($): Lognormal, correlated with purchase count
    spend_base = prior_purchases * 38.0 + rng.normal(loc=20.0, scale=15.0, size=n_samples)
    spend_noise = rng.lognormal(mean=2.8, sigma=0.6, size=n_samples)
    prior_spend = np.round(np.clip(spend_base + spend_noise, 5.0, 5000.0), 2)

    # Engagement score: Truncated normal on 0 - 100 scale
    eng_latent = rng.normal(loc=52.0, scale=18.0, size=n_samples) + (0.5 * prior_purchases)
    engagement_score = np.round(np.clip(eng_latent, 2.0, 99.0), 1)

    # Campaign channel: Categorical marketing channel
    channel_choices = ["email", "push", "sms", "social"]
    channel_probs = [0.40, 0.28, 0.18, 0.14]
    campaign_channel = rng.choice(channel_choices, size=n_samples, p=channel_probs)

    # Channel propensity weights for discount assignment & conversion
    channel_treatment_bias = {
        "email": 0.20,
        "push": 0.35,
        "sms": -0.10,
        "social": -0.25,
    }
    channel_outcome_bias = {
        "email": 0.15,
        "push": 0.25,
        "sms": -0.05,
        "social": -0.15,
    }

    c_treat = np.array([channel_treatment_bias[c] for c in campaign_channel])
    c_out = np.array([channel_outcome_bias[c] for c in campaign_channel])

    # 2. Treatment Assignment Mechanism: P(discount_offered = 1 | W)
    # Marketers target higher-engagement and higher-spend customers (promotional loyalty bias)
    norm_eng = (engagement_score - 50.0) / 20.0
    norm_purchases = (prior_purchases - 5.0) / 4.0
    norm_spend = (np.log1p(prior_spend) - 5.0) / 1.2
    norm_tenure = (customer_tenure - 20.0) / 15.0

    treatment_logit = (
        -0.45
        + 0.55 * norm_eng
        + 0.40 * norm_purchases
        + 0.30 * norm_spend
        + 0.15 * norm_tenure
        + c_treat
    )
    propensity_score = sigmoid(treatment_logit)
    discount_offered = rng.binomial(n=1, p=propensity_score)

    # 3. Post-Treatment Variable: discount_amount ($)
    # Only applicable if discount_offered == 1
    # Higher baseline spenders receive higher tier discount values ($10, $15, $25, $50)
    base_discount_draw = rng.choice(
        [10.0, 15.0, 20.0, 25.0, 35.0],
        size=n_samples,
        p=[0.30, 0.30, 0.20, 0.15, 0.05],
    )
    discount_amount = np.where(discount_offered == 1, base_discount_draw, 0.0)

    # 4. Outcome Generation: P(purchased = 1 | discount_offered, W)
    # Ground truth causal treatment lift: ~ +0.65 on logit scale (~10-15 percentage point lift)
    true_causal_effect = 0.70

    outcome_logit = (
        -1.30
        + (true_causal_effect * discount_offered)
        + 0.60 * norm_eng
        + 0.45 * norm_purchases
        + 0.35 * norm_spend
        + 0.10 * norm_tenure
        + c_out
    )
    purchase_prob = sigmoid(outcome_logit)
    purchased = rng.binomial(n=1, p=purchase_prob)

    # 5. Post-Treatment Outcome: Realized revenue ($)
    # Zero if customer did not purchase
    raw_basket_value = (
        45.0
        + 0.08 * prior_spend
        + 12.0 * np.log1p(prior_purchases)
        + rng.normal(loc=15.0, scale=12.0, size=n_samples)
    )
    realized_basket = np.clip(raw_basket_value - (0.5 * discount_amount), 8.0, 850.0)
    revenue = np.where(purchased == 1, np.round(realized_basket, 2), 0.0)

    # Customer identifiers
    customer_id = [f"CUST_{i+1:05d}" for i in range(n_samples)]

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "prior_purchases": prior_purchases,
            "prior_spend": prior_spend,
            "engagement_score": engagement_score,
            "customer_tenure": customer_tenure,
            "discount_offered": discount_offered,
            "discount_amount": discount_amount,
            "purchased": purchased,
            "revenue": revenue,
            "campaign_channel": campaign_channel,
        }
    )

    # Reorder explicitly to canonical schema
    df = df[REQUIRED_COLUMNS]

    # Validate generated data integrity
    val_report = validate_dataset(df)
    if not val_report["is_valid"]:
        raise ValueError(f"DGP produced invalid dataset: {val_report['errors']}")

    # Save to disk if requested
    if output_filepath:
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        df.to_csv(output_filepath, index=False)

    return df
