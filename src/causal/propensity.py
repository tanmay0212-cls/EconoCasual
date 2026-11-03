"""Treatment Propensity Modeling, Positivity, and Common Support Diagnostics.

This module provides:
1. Multinomial propensity score estimation for multi-treatment settings.
2. Positivity / Common Support overlap evaluation across all treatment tiers.
3. Effective Sample Size (ESS) and Inverse Probability Weighting (IPW) diagnostics.
4. Covariate balance evaluation (Standardized Mean Differences) for RCT benchmarks (Hillstrom).
"""

from typing import Dict, Any, List, Tuple
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class MultiTreatmentPropensityEstimator:
    """Estimates and evaluates generalized propensity scores for discrete multi-treatments."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.numerical_cols = [
            "age",
            "loyalty_score",
            "previous_purchases",
            "average_previous_order_value",
            "customer_activity_score",
        ]
        self.categorical_cols = ["customer_segment"]
        self.treatment_classes_ = None
        self.pipeline = None

    def fit(self, df: pd.DataFrame, treatment_col: str = "discount_amount") -> "MultiTreatmentPropensityEstimator":
        """Fit multinomial logistic regression pipeline for propensity scores."""
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numerical_cols),
                ("cat", OneHotEncoder(drop="first", sparse_output=False), self.categorical_cols),
            ]
        )

        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=self.random_state,
        )

        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])

        X = df[self.numerical_cols + self.categorical_cols]
        y = df[treatment_col].astype(int)
        self.pipeline.fit(X, y)
        self.treatment_classes_ = self.pipeline.named_steps["classifier"].classes_
        return self

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict conditional treatment probabilities P(T = d | X)."""
        X = df[self.numerical_cols + self.categorical_cols]
        probs = self.pipeline.predict_proba(X)
        col_names = [f"propensity_t_{c}" for c in self.treatment_classes_]
        return pd.DataFrame(probs, columns=col_names, index=df.index)

    def evaluate_overlap(
        self,
        df: pd.DataFrame,
        treatment_col: str = "discount_amount"
    ) -> Dict[str, Any]:
        """Compute comprehensive common support, overlap, and Effective Sample Size (ESS) metrics."""
        probs_df = self.predict_proba(df)
        treatments = df[treatment_col].astype(int).to_numpy()

        results = {
            "classes": [int(c) for c in self.treatment_classes_],
            "sample_size": len(df),
            "tier_metrics": {},
        }

        # IPW calculation for overall and per-tier ESS
        ipw_weights = np.zeros(len(df))
        for idx, c in enumerate(self.treatment_classes_):
            col_name = f"propensity_t_{c}"
            tier_probs = probs_df[col_name].to_numpy()
            tier_mask = (treatments == c)

            # Assigned treatment propensity
            ipw_weights[tier_mask] = 1.0 / np.clip(tier_probs[tier_mask], 1e-4, 1.0)

            # Overlap statistics
            p_min = float(tier_probs.min())
            p_max = float(tier_probs.max())
            p_mean = float(tier_probs.mean())
            p_q01 = float(np.percentile(tier_probs, 1))
            p_q99 = float(np.percentile(tier_probs, 99))
            
            # Extreme propensity violations
            below_5pct = int(np.sum(tier_probs < 0.05))
            above_90pct = int(np.sum(tier_probs > 0.90))

            # Tier ESS: (sum w)^2 / sum(w^2)
            tier_w = ipw_weights[tier_mask]
            tier_ess = float((tier_w.sum() ** 2) / ((tier_w ** 2).sum()))

            results["tier_metrics"][int(c)] = {
                "count": int(np.sum(tier_mask)),
                "propensity_min": round(p_min, 4),
                "propensity_max": round(p_max, 4),
                "propensity_mean": round(p_mean, 4),
                "p_1st_percentile": round(p_q01, 4),
                "p_99th_percentile": round(p_q99, 4),
                "count_below_0.05": below_5pct,
                "pct_below_0.05": round(below_5pct / len(df) * 100, 2),
                "count_above_0.90": above_90pct,
                "tier_effective_sample_size": round(tier_ess, 1),
                "ess_ratio": round(tier_ess / np.sum(tier_mask), 3),
            }

        total_ess = float((ipw_weights.sum() ** 2) / ((ipw_weights ** 2).sum()))
        results["overall_effective_sample_size"] = round(total_ess, 1)
        results["overall_ess_ratio"] = round(total_ess / len(df), 3)
        results["sufficient_overlap"] = all(
            m["p_1st_percentile"] >= 0.08 and m["p_99th_percentile"] <= 0.85
            for m in results["tier_metrics"].values()
        )

        return results


def calculate_smd(
    series_treated: pd.Series,
    series_control: pd.Series
) -> float:
    """Calculate Standardized Mean Difference (SMD) between two groups."""
    mean_t = series_treated.mean()
    mean_c = series_control.mean()
    var_t = series_treated.var()
    var_c = series_control.var()
    pooled_std = np.sqrt((var_t + var_c) / 2.0)
    if pooled_std == 0:
        return 0.0
    return float(abs(mean_t - mean_c) / pooled_std)


def evaluate_hillstrom_balance(df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate baseline covariate balance across all 3 randomized arms of Hillstrom dataset."""
    treatment_col = "treatment_group"
    arms = {0: "no_email", 1: "mens_email", 2: "womens_email"}
    
    numerical_features = ["recency", "history", "mens", "womens", "newbie"]
    categorical_features = ["zip_code", "channel"]

    # One-hot encode categorical features for SMD calculation
    df_encoded = pd.get_dummies(df[numerical_features + categorical_features], drop_first=True)
    all_features = list(df_encoded.columns)

    balance_results = {
        "arms": arms,
        "sample_sizes": {arms[a]: int(np.sum(df[treatment_col] == a)) for a in arms},
        "pairwise_smd": {
            "mens_vs_control": {},
            "womens_vs_control": {},
            "womens_vs_mens": {},
        },
        "max_smd_per_contrast": {},
        "all_smd_below_threshold": True,
    }

    g0 = df_encoded[df[treatment_col] == 0]
    g1 = df_encoded[df[treatment_col] == 1]
    g2 = df_encoded[df[treatment_col] == 2]

    for feat in all_features:
        smd_1_0 = calculate_smd(g1[feat], g0[feat])
        smd_2_0 = calculate_smd(g2[feat], g0[feat])
        smd_2_1 = calculate_smd(g2[feat], g1[feat])

        balance_results["pairwise_smd"]["mens_vs_control"][feat] = round(smd_1_0, 4)
        balance_results["pairwise_smd"]["womens_vs_control"][feat] = round(smd_2_0, 4)
        balance_results["pairwise_smd"]["womens_vs_mens"][feat] = round(smd_2_1, 4)

    max_1_0 = max(balance_results["pairwise_smd"]["mens_vs_control"].values())
    max_2_0 = max(balance_results["pairwise_smd"]["womens_vs_control"].values())
    max_2_1 = max(balance_results["pairwise_smd"]["womens_vs_mens"].values())

    balance_results["max_smd_per_contrast"]["mens_vs_control"] = max_1_0
    balance_results["max_smd_per_contrast"]["womens_vs_control"] = max_2_0
    balance_results["max_smd_per_contrast"]["womens_vs_mens"] = max_2_1
    balance_results["all_smd_below_threshold"] = (max(max_1_0, max_2_0, max_2_1) < 0.05)

    return balance_results
