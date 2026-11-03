"""Counterfactual Outcome Prediction and Business Value Quantification.

This module provides:
1. Generation of customer-level counterfactual potential outcomes across all treatment tiers.
2. Calibration and validity checks on predicted counterfactual probabilities.
3. Computation of expected business metrics (Gross Margin, Discount Cost, Net Contribution).
4. Unconstrained profit-optimal policy derivation.
"""

from typing import Dict, Any, List, Tuple
import os
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


def generate_synthetic_counterfactuals(
    causal_engine,
    df: pd.DataFrame,
    product_price: float = 95.0,
    gross_margin_rate: float = 0.40,
) -> pd.DataFrame:
    """Generate counterfactual purchase probabilities, treatment effects, and expected profits."""
    # 1. Predict effects and counterfactual probabilities
    effects_df = causal_engine.predict_effects(df)
    probs_df = causal_engine.predict_counterfactual_probabilities(df)

    p0 = probs_df["pred_prob_0"].to_numpy()
    p10 = probs_df["pred_prob_10"].to_numpy()
    p20 = probs_df["pred_prob_20"].to_numpy()

    # Basket expansion elasticity from DGP
    price_0 = product_price
    price_10 = product_price * 1.03
    price_20 = product_price * 1.06

    # 2. Expected Financial Outcomes per Customer under each decision d in {0, 10, 20}
    # Expected discount cost = d * P(Y_d = 1)
    exp_cost_0 = np.zeros(len(df))
    exp_cost_10 = np.round(10.0 * p10, 2)
    exp_cost_20 = np.round(20.0 * p20, 2)

    # Expected gross revenue = (price - d) * P(Y_d = 1)
    exp_rev_0 = np.round(price_0 * p0, 2)
    exp_rev_10 = np.round((price_10 - 10.0) * p10, 2)
    exp_rev_20 = np.round((price_20 - 20.0) * p20, 2)

    # Expected net profit = (price * margin_rate - d) * P(Y_d = 1)
    exp_profit_0 = np.round(price_0 * gross_margin_rate * p0, 2)
    exp_profit_10 = np.round((price_10 * gross_margin_rate - 10.0) * p10, 2)
    exp_profit_20 = np.round((price_20 * gross_margin_rate - 20.0) * p20, 2)

    # 3. Model Recommended Optimal Policy (Unconstrained)
    profits_matrix = np.column_stack([exp_profit_0, exp_profit_10, exp_profit_20])
    best_discount_idx = np.argmax(profits_matrix, axis=1)
    treatment_levels = np.array([0, 10, 20])
    recommended_discount = treatment_levels[best_discount_idx]

    pred_results = pd.DataFrame(
        {
            "customer_id": df["customer_id"].values if "customer_id" in df.columns else np.arange(len(df)),
            # Counterfactual purchase probabilities
            "pred_prob_0": p0,
            "pred_prob_10": p10,
            "pred_prob_20": p20,
            # Estimated CATE contrasts
            "pred_ite_10_vs_0": effects_df["pred_ite_10_vs_0"].values,
            "pred_ite_20_vs_0": effects_df["pred_ite_20_vs_0"].values,
            "pred_ite_20_vs_10": effects_df["pred_ite_20_vs_10"].values,
            # Expected discount liability
            "exp_cost_0": exp_cost_0,
            "exp_cost_10": exp_cost_10,
            "exp_cost_20": exp_cost_20,
            # Expected gross revenue
            "exp_revenue_0": exp_rev_0,
            "exp_revenue_10": exp_rev_10,
            "exp_revenue_20": exp_rev_20,
            # Expected net contribution
            "exp_profit_0": exp_profit_0,
            "exp_profit_10": exp_profit_10,
            "exp_profit_20": exp_profit_20,
            # Recommended policy
            "model_recommended_discount": recommended_discount,
        },
        index=df.index,
    )

    return pred_results


def evaluate_probability_calibration(
    df_with_obs: pd.DataFrame,
    preds_df: pd.DataFrame,
    treatment_col: str = "discount_amount",
    outcome_col: str = "purchase",
) -> Dict[str, Any]:
    """Evaluate calibration of predicted counterfactual probabilities against realized outcomes."""
    treatments = df_with_obs[treatment_col].astype(int).to_numpy()
    actual_y = df_with_obs[outcome_col].astype(int).to_numpy()

    # Assign predicted probability under actually observed treatment
    pred_y_observed = np.where(
        treatments == 0,
        preds_df["pred_prob_0"].to_numpy(),
        np.where(treatments == 10, preds_df["pred_prob_10"].to_numpy(), preds_df["pred_prob_20"].to_numpy())
    )

    brier = float(brier_score_loss(actual_y, pred_y_observed))
    logloss = float(log_loss(actual_y, pred_y_observed))

    tier_calibrations = {}
    for d in [0, 10, 20]:
        mask = (treatments == d)
        actual_rate = float(actual_y[mask].mean())
        mean_pred = float(pred_y_observed[mask].mean())
        tier_calibrations[d] = {
            "n_obs": int(np.sum(mask)),
            "actual_conversion_rate": round(actual_rate, 4),
            "predicted_mean_probability": round(mean_pred, 4),
            "absolute_calibration_error": round(abs(actual_rate - mean_pred), 4),
        }

    return {
        "overall_brier_score": round(brier, 4),
        "overall_log_loss": round(logloss, 4),
        "tier_calibrations": tier_calibrations,
    }


def generate_hillstrom_counterfactuals(
    causal_engine,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate counterfactual predictions for visit, conversion, and spend across Hillstrom arms."""
    visit_cf = causal_engine.predict_counterfactuals(df, outcome="visit")
    conv_cf = causal_engine.predict_counterfactuals(df, outcome="conversion")
    spend_cf = causal_engine.predict_counterfactuals(df, outcome="spend")

    combined = pd.concat([visit_cf, conv_cf, spend_cf], axis=1)
    if "customer_id" in df.columns:
        combined.insert(0, "customer_id", df["customer_id"].values)
    return combined

