"""Multi-Treatment Causal Estimation Engine for EconoCausal Week 3.

This module implements a production-grade multi-arm causal estimator using EconML's
Doubly Robust Learner (LinearDRLearner). It estimates heterogeneous treatment effects
and predicted purchase probabilities across three discrete promotional tiers:
    - Control (T = 0):  No discount ($0)
    - Tier 10 (T = 10): $10 discount coupon
    - Tier 20 (T = 20): $20 discount coupon

METHODOLOGICAL GUARDRAILS:
1. Multi-Valued Discrete Treatment:
   Treatments T ∈ {0, 10, 20} are modeled using Doubly Robust learning (Robins et al.,
   1994; Chernozhukov et al., 2018). Propensity scores e_d(W) = P(T = d | W) are fit
   via multinomial classification, and outcome baselines g_d(W) = E[Y | W, T = d] are
   fit via regression with K-fold cross-fitting.
2. Anti-Leakage & Quarantining:
   - Identifiers ('customer_id') and outcomes ('purchase') are strictly excluded from features.
   - Post-treatment economics ('gross_revenue', 'discount_cost', 'net_revenue',
     'gross_profit_before_discount', 'net_profit_after_discount') and context attributes
     ('product_price', 'gross_margin_rate') are quarantined.
   - Synthetic ground-truth counterfactuals ('y0_potential', 'y10_potential', 'y20_potential',
     'tau_10_true', 'tau_20_true') are strictly quarantined for evaluation only.
3. Separation of CATE and Purchase Probabilities:
   - CATE tau_d(X) = E[Y(d) - Y(0) | X] is the treatment lift relative to no discount.
   - Predicted purchase probabilities P(Y = 1 | X, T = d) are constructed by combining a
     calibrated baseline outcome model P(Y = 1 | X, T = 0) with the causal CATE lifts,
     clipped to the valid probability simplex [0, 1].
"""

from __future__ import annotations

import os
import json
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import joblib
import sklearn
import econml
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.stats import spearmanr
from econml.dr import LinearDRLearner

# Canonical feature and treatment definitions for multi-tier causal estimation
MULTI_TREATMENT_LEVELS: List[int] = [0, 10, 20]
DEFAULT_TREATMENT_COL: str = "discount_level"
DEFAULT_OUTCOME_COL: str = "purchase"

DEFAULT_PRE_TREATMENT_CONFOUNDERS: List[str] = [
    "age",
    "loyalty_score",
    "previous_purchases",
    "customer_segment",
    "campaign_channel",
]

NUMERICAL_CONFOUNDERS: List[str] = [
    "age",
    "loyalty_score",
    "previous_purchases",
]

CATEGORICAL_CONFOUNDERS: List[str] = [
    "customer_segment",
    "campaign_channel",
]

PROHIBITED_LEAKAGE_COLUMNS: List[str] = [
    "customer_id",
    "purchase",
    "discount_amount",
    "gross_revenue",
    "discount_cost",
    "net_revenue",
    "gross_profit_before_discount",
    "net_profit_after_discount",
    "product_price",
    "gross_margin_rate",
    "y0_potential",
    "y10_potential",
    "y20_potential",
    "tau_10_true",
    "tau_20_true",
]


class MultiTierCausalDML:
    """Production-grade Multi-Treatment Doubly Robust Causal Estimator.

    Wraps EconML's LinearDRLearner with automated pre-treatment feature isolation,
    cross-fitting, multi-arm propensity diagnostics, baseline outcome modeling,
    and asymptotic Wald confidence intervals.

    Attributes:
        cv: Number of cross-fitting folds (default: 5).
        random_state: Reproducible random seed.
        preprocessor: Fitted ColumnTransformer for feature scaling and one-hot encoding.
        estimator: The fitted EconML LinearDRLearner instance.
        baseline_model: Calibrated classifier estimating P(purchase = 1 | X, T = 0).
        is_fitted_: Boolean indicating whether the model is trained.
    """

    def __init__(
        self,
        n_estimators_nuisance: int = 100,
        max_depth_nuisance: int = 5,
        min_samples_leaf: int = 10,
        cv: int = 5,
        random_state: int = 42,
    ) -> None:
        """Initialize MultiTierCausalDML configuration.

        Args:
            n_estimators_nuisance: Number of trees for nuisance models.
            max_depth_nuisance: Maximum tree depth for nuisance models.
            min_samples_leaf: Minimum samples per leaf for regularization.
            cv: Number of cross-fitting folds for residualization (default: 5).
            random_state: Random seed for exact reproducibility.
        """
        self.n_estimators_nuisance = n_estimators_nuisance
        self.max_depth_nuisance = max_depth_nuisance
        self.min_samples_leaf = min_samples_leaf
        self.cv = cv
        self.random_state = random_state

        self.preprocessor: Optional[ColumnTransformer] = None
        self.estimator: Optional[LinearDRLearner] = None
        self.baseline_model: Optional[GradientBoostingClassifier] = None
        self.feature_names_in_: Optional[List[str]] = None
        self.preprocessed_feature_names_: Optional[List[str]] = None
        self.treatment_col_: Optional[str] = None
        self.outcome_col_: Optional[str] = None
        self.treatment_levels_: List[int] = list(MULTI_TREATMENT_LEVELS)
        self.is_fitted_: bool = False

    def validate_inputs(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        feature_cols: List[str],
    ) -> None:
        """Validate causal assumptions, anti-leakage rules, and data integrity.

        Raises:
            ValueError: If any validation checks fail.
        """
        if df is None or len(df) == 0:
            raise ValueError("Input DataFrame is empty or None.")

        if len(df) < 300:
            raise ValueError(f"Insufficient data for multi-treatment DML: minimum 300 rows required, received {len(df)}.")

        # 1. Anti-leakage guardrails: strictly forbid post-treatment and ground-truth columns
        leaky_vars = [var for var in PROHIBITED_LEAKAGE_COLUMNS if var in feature_cols]
        if leaky_vars:
            raise ValueError(
                f"Data leakage detected! The following post-treatment, context, or ground-truth columns "
                f"were passed as features: {leaky_vars}. Conditioning on these violates causal validity."
            )

        if outcome_col in feature_cols:
            raise ValueError(f"Data leakage: outcome column '{outcome_col}' cannot be included in feature set.")

        if treatment_col in feature_cols:
            raise ValueError(f"Treatment column '{treatment_col}' cannot be included in feature set.")

        # 2. Check treatment values
        t_vals = set(df[treatment_col].dropna().unique())
        valid_t = set(MULTI_TREATMENT_LEVELS)
        if not t_vals.issubset(valid_t):
            raise ValueError(
                f"Treatment column '{treatment_col}' contains invalid levels: {t_vals - valid_t}. "
                f"Expected subset of {valid_t}."
            )
        if len(t_vals) < 3:
            raise ValueError(
                f"Treatment column '{treatment_col}' must contain all three levels {valid_t}, found: {t_vals}."
            )

        # 3. Check binary domain of outcome
        y_vals = set(df[outcome_col].dropna().unique())
        if not y_vals.issubset({0, 1}):
            raise ValueError(f"Outcome column '{outcome_col}' must be strictly binary (0 or 1), found: {y_vals}.")

        # 4. Null checks
        null_counts = df[[treatment_col, outcome_col] + feature_cols].isnull().sum()
        if null_counts.sum() > 0:
            cols_with_null = null_counts[null_counts > 0].to_dict()
            raise ValueError(f"Null values detected in training columns: {cols_with_null}.")

        # 5. Non-finite checks (inf / -inf)
        num_cols = [c for c in feature_cols if c in NUMERICAL_CONFOUNDERS or (c in df.columns and pd.api.types.is_numeric_dtype(df[c]))]
        inf_cols = {}
        for c in num_cols:
            if np.isinf(df[c].to_numpy(dtype=float)).any():
                inf_cols[c] = int(np.isinf(df[c].to_numpy(dtype=float)).sum())
        if inf_cols:
            raise ValueError(f"Non-finite (inf/-inf) values detected in feature columns: {inf_cols}.")

    def _build_preprocessor(self, feature_cols: List[str]) -> ColumnTransformer:
        """Construct ColumnTransformer for numerical scaling and one-hot encoding."""
        num_cols = [c for c in feature_cols if c in NUMERICAL_CONFOUNDERS]
        cat_cols = [c for c in feature_cols if c in CATEGORICAL_CONFOUNDERS]

        remaining = [c for c in feature_cols if c not in num_cols and c not in cat_cols]
        for c in remaining:
            num_cols.append(c)

        transformers = []
        if num_cols:
            transformers.append(("num", StandardScaler(), num_cols))
        if cat_cols:
            transformers.append(
                (
                    "cat",
                    OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                    cat_cols,
                )
            )

        return ColumnTransformer(transformers=transformers)

    def fit(
        self,
        df_train: pd.DataFrame,
        treatment_col: str = DEFAULT_TREATMENT_COL,
        outcome_col: str = DEFAULT_OUTCOME_COL,
        feature_cols: Optional[List[str]] = None,
    ) -> "MultiTierCausalDML":
        """Fit the Multi-Treatment Doubly Robust Causal Estimator on training data.

        Learns preprocessor transformations, nuisance models, and baseline outcome
        model strictly on the training partition to prevent data leakage.

        Args:
            df_train: Training DataFrame.
            treatment_col: Name of treatment column (default: 'discount_level').
            outcome_col: Name of outcome column (default: 'purchase').
            feature_cols: Optional list of pre-treatment confounder feature names.

        Returns:
            Fitted MultiTierCausalDML instance.
        """
        f_cols = list(feature_cols) if feature_cols is not None else list(DEFAULT_PRE_TREATMENT_CONFOUNDERS)
        self.validate_inputs(df_train, treatment_col, outcome_col, f_cols)

        self.treatment_col_ = treatment_col
        self.outcome_col_ = outcome_col
        self.feature_names_in_ = f_cols

        # 1. Fit preprocessor strictly on training features
        self.preprocessor = self._build_preprocessor(f_cols)
        X_train_proc = self.preprocessor.fit_transform(df_train[f_cols])

        # Retrieve transformed feature names
        proc_names: List[str] = []
        for name, transformer, cols in self.preprocessor.transformers_:
            if hasattr(transformer, "get_feature_names_out"):
                proc_names.extend(list(transformer.get_feature_names_out(cols)))
            else:
                proc_names.extend(cols)
        self.preprocessed_feature_names_ = proc_names

        # 2. Extract targets
        T_train = df_train[treatment_col].to_numpy().astype(int)
        Y_train = df_train[outcome_col].to_numpy().astype(float)

        # 3. Instantiate nuisance models
        # Treatment model: multinomial propensity classifier P(T=d | X)
        model_propensity = RandomForestClassifier(
            n_estimators=self.n_estimators_nuisance,
            max_depth=self.max_depth_nuisance,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )
        # Outcome model: conditional regression E[Y | X, T=d]
        model_regression = RandomForestRegressor(
            n_estimators=self.n_estimators_nuisance,
            max_depth=self.max_depth_nuisance,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )

        # 4. Instantiate and fit LinearDRLearner
        self.estimator = LinearDRLearner(
            model_propensity=model_propensity,
            model_regression=model_regression,
            cv=self.cv,
            random_state=self.random_state,
        )
        self.estimator.fit(Y=Y_train, T=T_train, X=X_train_proc)

        # 5. Fit baseline outcome model P(Y = 1 | X, T = 0) strictly on control group
        mask_control = T_train == 0
        X_control_proc = X_train_proc[mask_control]
        Y_control = Y_train[mask_control]

        self.baseline_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.08,
            random_state=self.random_state,
        )
        self.baseline_model.fit(X_control_proc, Y_control)

        self.is_fitted_ = True
        return self

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Validate and transform inference features using the fitted preprocessor."""
        if not self.is_fitted_ or self.preprocessor is None:
            raise RuntimeError("Estimator has not been fitted yet. Call fit() before inference.")

        if self.feature_names_in_ is None:
            raise RuntimeError("feature_names_in_ is undefined.")

        if df is None or len(df) == 0:
            raise ValueError("Input DataFrame is empty or None.")

        missing = [col for col in self.feature_names_in_ if col not in df.columns]
        if missing:
            raise ValueError(f"Input DataFrame is missing required features: {missing}. Expected: {self.feature_names_in_}")

        null_counts = df[self.feature_names_in_].isnull().sum()
        if null_counts.sum() > 0:
            cols_with_null = null_counts[null_counts > 0].to_dict()
            raise ValueError(f"Null values detected in inference features: {cols_with_null}")

        num_cols = [c for c in self.feature_names_in_ if c in NUMERICAL_CONFOUNDERS or (c in df.columns and pd.api.types.is_numeric_dtype(df[c]))]
        inf_cols = {}
        for c in num_cols:
            if np.isinf(df[c].to_numpy(dtype=float)).any():
                inf_cols[c] = int(np.isinf(df[c].to_numpy(dtype=float)).sum())
        if inf_cols:
            raise ValueError(f"Non-finite (inf/-inf) values detected in inference features: {inf_cols}")

        return self.preprocessor.transform(df[self.feature_names_in_])

    def effect(
        self,
        df: pd.DataFrame,
        T0: int = 0,
        T1: int = 10,
    ) -> np.ndarray:
        """Estimate Conditional Average Treatment Effect (CATE) for T1 versus T0.

        Args:
            df: DataFrame containing customer pre-treatment features.
            T0: Baseline treatment level (default: 0).
            T1: Target treatment level (default: 10).

        Returns:
            1D array of estimated treatment effects (percentage point lift on purchase).
        """
        if T0 not in MULTI_TREATMENT_LEVELS or T1 not in MULTI_TREATMENT_LEVELS:
            raise ValueError(f"T0 and T1 must be in {MULTI_TREATMENT_LEVELS}, received T0={T0}, T1={T1}.")

        X_proc = self._prepare_features(df)
        effects = self.estimator.effect(X_proc, T0=T0, T1=T1)
        return np.asarray(effects).flatten()

    def effect_interval(
        self,
        df: pd.DataFrame,
        T0: int = 0,
        T1: int = 10,
        alpha: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate Wald confidence intervals for customer-level treatment effects.

        Args:
            df: DataFrame containing customer pre-treatment features.
            T0: Baseline treatment level (default: 0).
            T1: Target treatment level (default: 10).
            alpha: Significance level (default: 0.05 for 95% CI).

        Returns:
            Tuple of (lower_bounds, upper_bounds) 1D arrays.
        """
        if T0 not in MULTI_TREATMENT_LEVELS or T1 not in MULTI_TREATMENT_LEVELS:
            raise ValueError(f"T0 and T1 must be in {MULTI_TREATMENT_LEVELS}, received T0={T0}, T1={T1}.")

        X_proc = self._prepare_features(df)
        lb, ub = self.estimator.effect_interval(X_proc, T0=T0, T1=T1, alpha=alpha)
        return np.asarray(lb).flatten(), np.asarray(ub).flatten()

    def ate(
        self,
        df: pd.DataFrame,
        T0: int = 0,
        T1: int = 10,
    ) -> float:
        """Estimate population Average Treatment Effect (ATE) for T1 versus T0.

        Args:
            df: DataFrame containing evaluation customers.
            T0: Baseline treatment level (default: 0).
            T1: Target treatment level (default: 10).

        Returns:
            Scalar ATE point estimate.
        """
        if T0 not in MULTI_TREATMENT_LEVELS or T1 not in MULTI_TREATMENT_LEVELS:
            raise ValueError(f"T0 and T1 must be in {MULTI_TREATMENT_LEVELS}, received T0={T0}, T1={T1}.")

        X_proc = self._prepare_features(df)
        ate_val = self.estimator.ate(X_proc, T0=T0, T1=T1)
        return float(ate_val)

    def ate_interval(
        self,
        df: pd.DataFrame,
        T0: int = 0,
        T1: int = 10,
        alpha: float = 0.05,
    ) -> Tuple[float, float]:
        """Estimate confidence interval for the Average Treatment Effect (ATE).

        Args:
            df: DataFrame containing evaluation customers.
            T0: Baseline treatment level (default: 0).
            T1: Target treatment level (default: 10).
            alpha: Significance level (default: 0.05 for 95% CI).

        Returns:
            Tuple of (lower_bound, upper_bound).
        """
        if T0 not in MULTI_TREATMENT_LEVELS or T1 not in MULTI_TREATMENT_LEVELS:
            raise ValueError(f"T0 and T1 must be in {MULTI_TREATMENT_LEVELS}, received T0={T0}, T1={T1}.")

        X_proc = self._prepare_features(df)
        lb, ub = self.estimator.ate_interval(X_proc, T0=T0, T1=T1, alpha=alpha)
        return float(lb), float(ub)

    def predict_probabilities(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Predict purchase probabilities under all three treatment options: $0, $10, and $20.

        Methodology:
        1. Predicts baseline conversion probability P(Y = 1 | X, T = 0) using the
           calibrated outcome model fit on control observations.
        2. Estimates CATE lifts tau_10(X) and tau_20(X) relative to control using LinearDRLearner.
        3. Constructs P(Y = 1 | X, T = d) = clip(P(Y = 1 | X, T = 0) + tau_d(X), 0.0, 1.0).

        Args:
            df: Customer DataFrame containing pre-treatment features.

        Returns:
            DataFrame containing customer_id (if present), predicted probabilities,
            and estimated CATE lifts.
        """
        X_proc = self._prepare_features(df)

        # Baseline probability P(Y = 1 | X, T = 0)
        p0 = self.baseline_model.predict_proba(X_proc)[:, 1]

        # CATE lifts relative to T = 0
        tau_10 = self.effect(df, T0=0, T1=10)
        tau_20 = self.effect(df, T0=0, T1=20)

        # Counterfactual probabilities clipped to probability simplex [0, 1]
        p10 = np.clip(p0 + tau_10, 0.0, 1.0)
        p20 = np.clip(p0 + tau_20, 0.0, 1.0)

        result_dict = {
            "p_purchase_t0": np.round(p0, 6),
            "p_purchase_t10": np.round(p10, 6),
            "p_purchase_t20": np.round(p20, 6),
            "cate_lift_t10": np.round(tau_10, 6),
            "cate_lift_t20": np.round(tau_20, 6),
        }

        if "customer_id" in df.columns:
            result_dict = {"customer_id": df["customer_id"].astype(str).values, **result_dict}

        return pd.DataFrame(result_dict)

    def check_overlap(
        self,
        df: pd.DataFrame,
        propensity_threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """Examine propensity distribution and report unsupported treatment-feature regions.

        Positivity requires 0 < P(T = d | X) < 1 across all treatment arms and customer profiles.

        Args:
            df: DataFrame containing features and treatment column.
            propensity_threshold: Minimum expected propensity below which a unit is flagged (default: 0.05).

        Returns:
            Dictionary containing propensity statistics, overlap violations, and segment distributions.
        """
        t_col = self.treatment_col_ or DEFAULT_TREATMENT_COL
        if t_col not in df.columns:
            raise ValueError(f"Treatment column '{t_col}' not found in DataFrame.")

        # Multinomial propensity classifier fit on preprocessed features
        X_proc = self._prepare_features(df)
        prop_clf = RandomForestClassifier(
            n_estimators=50,
            max_depth=4,
            random_state=self.random_state,
            n_jobs=-1,
        )
        prop_clf.fit(X_proc, df[t_col].to_numpy().astype(int))
        probs = prop_clf.predict_proba(X_proc)  # (N, 3)

        classes = list(prop_clf.classes_)
        prop_stats = {}
        unsupported_counts = {}

        for idx, lvl in enumerate(classes):
            p_arm = probs[:, idx]
            unsupported = int(np.sum(p_arm < propensity_threshold))
            unsupported_counts[f"T={lvl}"] = unsupported
            prop_stats[f"T={lvl}"] = {
                "min": round(float(np.min(p_arm)), 4),
                "q01": round(float(np.percentile(p_arm, 1)), 4),
                "q25": round(float(np.percentile(p_arm, 25)), 4),
                "median": round(float(np.median(p_arm)), 4),
                "mean": round(float(np.mean(p_arm)), 4),
                "q75": round(float(np.percentile(p_arm, 75)), 4),
                "q99": round(float(np.percentile(p_arm, 99)), 4),
                "max": round(float(np.max(p_arm)), 4),
                "unsupported_count": unsupported,
                "unsupported_pct": round((unsupported / len(df)) * 100.0, 2),
            }

        # Cross-tabulation by customer_segment if present
        segment_overlap = {}
        if "customer_segment" in df.columns:
            for seg in df["customer_segment"].unique():
                seg_mask = df["customer_segment"] == seg
                seg_df = df[seg_mask]
                t_dist = seg_df[t_col].value_counts(normalize=True).to_dict()
                segment_overlap[seg] = {
                    f"T={int(k)}": round(float(v), 4) for k, v in t_dist.items()
                }

        has_unsupported_regions = sum(unsupported_counts.values()) > 0
        min_prop = min(v["min"] for v in prop_stats.values())

        return {
            "sample_size": len(df),
            "propensity_threshold": propensity_threshold,
            "min_propensity_observed": min_prop,
            "has_unsupported_regions": has_unsupported_regions,
            "propensity_distribution": prop_stats,
            "segment_empirical_proportions": segment_overlap,
            "positivity_verdict": "ADEQUATE OVERLAP" if not has_unsupported_regions else "PARTIAL SUPPORT DEFICIT",
        }

    def evaluate(
        self,
        df_eval: pd.DataFrame,
        ground_truth: bool = True,
    ) -> Dict[str, Any]:
        """Compute comprehensive causal estimation benchmarks on held-out evaluation data.

        Evaluates:
        1. Model-estimated ATEs vs Ground-Truth ATEs for $10 vs $0 and $20 vs $0.
        2. 95% Wald Confidence Intervals and coverage verification.
        3. Naive observational differences and confounding bias removed.
        4. Customer-level CATE error metrics: MAE, RMSE, Spearman rank correlation.
        5. Predicted probability fidelity against true potential-outcome probabilities.

        Args:
            df_eval: Held-out evaluation DataFrame.
            ground_truth: Whether to compute ground-truth benchmark metrics.

        Returns:
            Dictionary of metrics and causal validation results.
        """
        t_col = self.treatment_col_ or DEFAULT_TREATMENT_COL
        y_col = self.outcome_col_ or DEFAULT_OUTCOME_COL

        # 1. CATE and ATE estimates
        ate10_est = self.ate(df_eval, T0=0, T1=10)
        ate10_ci = self.ate_interval(df_eval, T0=0, T1=10, alpha=0.05)
        cate10_est = self.effect(df_eval, T0=0, T1=10)

        ate20_est = self.ate(df_eval, T0=0, T1=20)
        ate20_ci = self.ate_interval(df_eval, T0=0, T1=20, alpha=0.05)
        cate20_est = self.effect(df_eval, T0=0, T1=20)

        # 2. Naive observational differences
        mean_y0 = float(df_eval[df_eval[t_col] == 0][y_col].mean())
        mean_y10 = float(df_eval[df_eval[t_col] == 10][y_col].mean())
        mean_y20 = float(df_eval[df_eval[t_col] == 20][y_col].mean())

        naive_diff_10 = mean_y10 - mean_y0
        naive_diff_20 = mean_y20 - mean_y0

        # 3. Predicted probabilities
        prob_df = self.predict_probabilities(df_eval)

        metrics: Dict[str, Any] = {
            "eval_sample_size": len(df_eval),
            "tier_10_vs_0": {
                "ate_estimate": round(ate10_est, 6),
                "ate_estimate_percentage_points": round(ate10_est * 100.0, 2),
                "ate_ci_95": [round(ate10_ci[0], 6), round(ate10_ci[1], 6)],
                "ate_ci_95_percentage_points": [round(ate10_ci[0] * 100.0, 2), round(ate10_ci[1] * 100.0, 2)],
                "naive_observational_diff": round(naive_diff_10, 6),
                "naive_diff_percentage_points": round(naive_diff_10 * 100.0, 2),
                "cate_distribution": {
                    "mean": round(float(np.mean(cate10_est)), 4),
                    "std": round(float(np.std(cate10_est)), 4),
                    "min": round(float(np.min(cate10_est)), 4),
                    "q25": round(float(np.percentile(cate10_est, 25)), 4),
                    "median": round(float(np.median(cate10_est)), 4),
                    "q75": round(float(np.percentile(cate10_est, 75)), 4),
                    "max": round(float(np.max(cate10_est)), 4),
                },
            },
            "tier_20_vs_0": {
                "ate_estimate": round(ate20_est, 6),
                "ate_estimate_percentage_points": round(ate20_est * 100.0, 2),
                "ate_ci_95": [round(ate20_ci[0], 6), round(ate20_ci[1], 6)],
                "ate_ci_95_percentage_points": [round(ate20_ci[0] * 100.0, 2), round(ate20_ci[1] * 100.0, 2)],
                "naive_observational_diff": round(naive_diff_20, 6),
                "naive_diff_percentage_points": round(naive_diff_20 * 100.0, 2),
                "cate_distribution": {
                    "mean": round(float(np.mean(cate20_est)), 4),
                    "std": round(float(np.std(cate20_est)), 4),
                    "min": round(float(np.min(cate20_est)), 4),
                    "q25": round(float(np.percentile(cate20_est, 25)), 4),
                    "median": round(float(np.median(cate20_est)), 4),
                    "q75": round(float(np.percentile(cate20_est, 75)), 4),
                    "max": round(float(np.max(cate20_est)), 4),
                },
            },
            "predicted_probabilities_summary": {
                "mean_p0": round(float(prob_df["p_purchase_t0"].mean()), 4),
                "mean_p10": round(float(prob_df["p_purchase_t10"].mean()), 4),
                "mean_p20": round(float(prob_df["p_purchase_t20"].mean()), 4),
            },
        }

        # 4. Ground-truth benchmark comparisons (if columns present)
        if ground_truth and "tau_10_true" in df_eval.columns and "tau_20_true" in df_eval.columns:
            true_tau10 = df_eval["tau_10_true"].to_numpy()
            true_tau20 = df_eval["tau_20_true"].to_numpy()
            true_ate10 = float(np.mean(true_tau10))
            true_ate20 = float(np.mean(true_tau20))

            bias10_in_naive = naive_diff_10 - true_ate10
            bias20_in_naive = naive_diff_20 - true_ate20

            spearman_10 = float(spearmanr(cate10_est, true_tau10).statistic)
            spearman_20 = float(spearmanr(cate20_est, true_tau20).statistic)

            mae_10 = float(np.mean(np.abs(cate10_est - true_tau10)))
            mae_20 = float(np.mean(np.abs(cate20_est - true_tau20)))

            rmse_10 = float(np.sqrt(np.mean((cate10_est - true_tau10) ** 2)))
            rmse_20 = float(np.sqrt(np.mean((cate20_est - true_tau20) ** 2)))

            covers_10 = bool(ate10_ci[0] <= true_ate10 <= ate10_ci[1])
            covers_20 = bool(ate20_ci[0] <= true_ate20 <= ate20_ci[1])

            gt_metrics = {
                "true_sample_ate_t10": round(true_ate10, 6),
                "true_sample_ate_t20": round(true_ate20, 6),
                "ate_error_t10": round(ate10_est - true_ate10, 6),
                "ate_error_t20": round(ate20_est - true_ate20, 6),
                "ci_covers_true_ate_t10": covers_10,
                "ci_covers_true_ate_t20": covers_20,
                "confounding_bias_in_naive_t10": round(bias10_in_naive, 6),
                "confounding_bias_in_naive_t20": round(bias20_in_naive, 6),
                "cate_spearman_rank_corr_t10": round(spearman_10, 4),
                "cate_spearman_rank_corr_t20": round(spearman_20, 4),
                "cate_mae_t10": round(mae_10, 4),
                "cate_mae_t20": round(mae_20, 4),
                "cate_rmse_t10": round(rmse_10, 4),
                "cate_rmse_t20": round(rmse_20, 4),
            }

            if "y0_potential" in df_eval.columns:
                y0_true = df_eval["y0_potential"].to_numpy()
                y10_true = df_eval["y10_potential"].to_numpy()
                y20_true = df_eval["y20_potential"].to_numpy()

                gt_metrics["prob_mae_p0"] = round(float(np.mean(np.abs(prob_df["p_purchase_t0"].to_numpy() - y0_true))), 4)
                gt_metrics["prob_mae_p10"] = round(float(np.mean(np.abs(prob_df["p_purchase_t10"].to_numpy() - y10_true))), 4)
                gt_metrics["prob_mae_p20"] = round(float(np.mean(np.abs(prob_df["p_purchase_t20"].to_numpy() - y20_true))), 4)

            metrics["ground_truth_benchmarks"] = gt_metrics

        return metrics

    def save(
        self,
        model_path: str,
        metadata_path: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Safely serialize the multi-treatment pipeline and metadata to disk.

        Args:
            model_path: Filepath for the .joblib artifact.
            metadata_path: Optional filepath for configuration JSON.
            extra_metadata: Optional additional dictionary fields to store.
        """
        if not self.is_fitted_:
            raise RuntimeError("Cannot save unfitted estimator.")

        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)

        bundle = {
            "estimator_class": "LinearDRLearner",
            "n_estimators_nuisance": self.n_estimators_nuisance,
            "max_depth_nuisance": self.max_depth_nuisance,
            "min_samples_leaf": self.min_samples_leaf,
            "cv": self.cv,
            "random_state": self.random_state,
            "treatment_levels": self.treatment_levels_,
            "preprocessor": self.preprocessor,
            "estimator": self.estimator,
            "baseline_model": self.baseline_model,
            "feature_names_in": self.feature_names_in_,
            "preprocessed_feature_names": self.preprocessed_feature_names_,
            "treatment_col": self.treatment_col_,
            "outcome_col": self.outcome_col_,
        }
        joblib.dump(bundle, model_path)

        if metadata_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(metadata_path)), exist_ok=True)
            meta = {
                "model_class": "LinearDRLearner",
                "estimator_framework": "EconML",
                "nuisance_treatment_model": "RandomForestClassifier",
                "nuisance_outcome_model": "RandomForestRegressor",
                "baseline_outcome_model": "GradientBoostingClassifier",
                "cv_folds": self.cv,
                "random_state": self.random_state,
                "treatment_col": self.treatment_col_,
                "outcome_col": self.outcome_col_,
                "treatment_levels": self.treatment_levels_,
                "features": self.feature_names_in_,
                "preprocessed_features": self.preprocessed_feature_names_,
                "packages": {
                    "econml": econml.__version__,
                    "scikit-learn": sklearn.__version__,
                    "joblib": joblib.__version__,
                },
                "methodological_assumptions": [
                    "Conditional Unconfoundedness: Treatment assignment is conditionally independent of potential outcomes given pre-treatment confounders.",
                    "Positivity (Overlap): 0 < P(T = d | W) < 1 for all d in {0, 10, 20} across all customer strata.",
                    "No Post-Treatment Confounders: Post-treatment economics and outcomes are strictly quarantined.",
                    "Doubly Robust Orthogonality: Estimator is robust to misspecification of either the propensity model or outcome regression.",
                ],
            }
            if extra_metadata:
                meta.update(extra_metadata)

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, model_path: str) -> "MultiTierCausalDML":
        """Load a serialized MultiTierCausalDML bundle from disk.

        Args:
            model_path: Filepath to the .joblib artifact.

        Returns:
            Restored MultiTierCausalDML instance.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        bundle = joblib.load(model_path)
        instance = cls(
            n_estimators_nuisance=bundle["n_estimators_nuisance"],
            max_depth_nuisance=bundle["max_depth_nuisance"],
            min_samples_leaf=bundle["min_samples_leaf"],
            cv=bundle["cv"],
            random_state=bundle["random_state"],
        )
        instance.preprocessor = bundle["preprocessor"]
        instance.estimator = bundle["estimator"]
        instance.baseline_model = bundle["baseline_model"]
        instance.treatment_levels_ = bundle["treatment_levels"]
        instance.feature_names_in_ = bundle["feature_names_in"]
        instance.preprocessed_feature_names_ = bundle["preprocessed_feature_names"]
        instance.treatment_col_ = bundle["treatment_col"]
        instance.outcome_col_ = bundle["outcome_col"]
        instance.is_fitted_ = True
        return instance

