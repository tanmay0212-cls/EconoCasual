"""Double Machine Learning (DML) Causal Estimator for EconoCausal.

This module implements the Double Machine Learning (Chernozhukov et al., 2018)
causal-effect estimator using EconML. It estimates the causal effect of promotional
discounts on customer purchase conversion using orthogonalized cross-fitting.

METHODOLOGICAL GUARDRAILS:
1. Confounder Adjustment Set:
   Only verified pre-treatment covariates ('prior_purchases', 'prior_spend',
   'engagement_score', 'customer_tenure', 'campaign_channel') are adjusted for.
2. Anti-Leakage Protection:
   Post-treatment variables ('discount_amount', 'order_value', 'revenue') and
   identifiers ('customer_id') are strictly prohibited from the feature matrix.
3. Linear Treatment Effect Assumption:
   Under LinearDML, the Conditional Average Treatment Effect (CATE) is modeled
   as tau(X) = <theta, X> + alpha, assuming treatment effect varies linearly with
   customer features. This provides asymptotic normality and valid confidence intervals.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import os
import json
import numpy as np
import pandas as pd
import joblib
import sklearn
import econml
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from econml.dml import LinearDML, CausalForestDML

# Canonical variable and alias definitions
DEFAULT_TREATMENT_NAMES: List[str] = ["discount_received", "discount_offered"]
DEFAULT_OUTCOME_NAMES: List[str] = ["purchase", "purchased"]
DEFAULT_CONFOUNDERS: List[str] = [
    "prior_purchases",
    "prior_spend",
    "engagement_score",
    "customer_tenure",
    "campaign_channel",
]
NUMERICAL_CONFOUNDERS: List[str] = [
    "prior_purchases",
    "prior_spend",
    "engagement_score",
    "customer_tenure",
]
CATEGORICAL_CONFOUNDERS: List[str] = [
    "campaign_channel",
]
PROHIBITED_LEAKAGE_VARS: List[str] = [
    "discount_amount",
    "order_value",
    "revenue",
    "customer_id",
]

# Ground truth benchmarks established in Week 1/Week 2 audit
SYNTHETIC_SAMPLE_ATE: float = 0.1446
SYNTHETIC_SPEC_BENCHMARK_ATE: float = 0.1369


class EconoCausalDML:
    """Production-grade Double Machine Learning estimator for promotional dynamic pricing.

    Wraps EconML's LinearDML (or CausalForestDML) with automated preprocessing,
    anti-leakage validation, cross-fitting, and causal inference diagnostics.

    Attributes:
        model_type: DML architecture ('linear' for LinearDML, 'forest' for CausalForestDML).
        base_estimator: Underlying nuisance models ('random_forest' or 'lightgbm').
        cv: Number of cross-fitting folds (default: 5).
        random_state: Reproducible random seed.
        preprocessor: Fitted ColumnTransformer for feature scaling and encoding.
        estimator: The fitted EconML DML instance.
        is_fitted_: Boolean indicating whether estimator is trained.
    """

    def __init__(
        self,
        model_type: str = "linear",
        base_estimator: str = "random_forest",
        n_estimators: int = 100,
        max_depth: int = 5,
        cv: int = 5,
        random_state: int = 42,
    ) -> None:
        """Initialize EconoCausalDML configuration.

        Args:
            model_type: 'linear' (default, LinearDML) or 'forest' (CausalForestDML).
            base_estimator: 'random_forest' (default) or 'lightgbm'.
            n_estimators: Number of trees for nuisance models.
            max_depth: Maximum tree depth for nuisance models.
            cv: Number of cross-fitting splits for residualization.
            random_state: Random seed for exact reproducibility.
        """
        if model_type not in ["linear", "forest"]:
            raise ValueError(f"model_type must be 'linear' or 'forest', received: '{model_type}'")
        if base_estimator not in ["random_forest", "lightgbm"]:
            raise ValueError(f"base_estimator must be 'random_forest' or 'lightgbm', received: '{base_estimator}'")

        self.model_type = model_type
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.cv = cv
        self.random_state = random_state

        self.preprocessor: Optional[ColumnTransformer] = None
        self.estimator: Optional[Union[LinearDML, CausalForestDML]] = None
        self.feature_names_in_: Optional[List[str]] = None
        self.preprocessed_feature_names_: Optional[List[str]] = None
        self.treatment_col_: Optional[str] = None
        self.outcome_col_: Optional[str] = None
        self.is_fitted_: bool = False

    @staticmethod
    def resolve_columns(
        df: pd.DataFrame,
        treatment_col: Optional[str] = None,
        outcome_col: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> Tuple[str, str, List[str]]:
        """Resolve and validate column names across canonical naming conventions.

        Args:
            df: Input DataFrame.
            treatment_col: Optional explicit treatment column name.
            outcome_col: Optional explicit outcome column name.
            feature_cols: Optional explicit list of feature column names.

        Returns:
            Tuple of (resolved_treatment_col, resolved_outcome_col, resolved_feature_cols).
        """
        if df is None or len(df) == 0:
            raise ValueError("Input DataFrame is empty or None.")

        # Resolve treatment
        if treatment_col is not None:
            if treatment_col not in df.columns:
                raise ValueError(f"Specified treatment column '{treatment_col}' not found in DataFrame.")
            resolved_treatment = treatment_col
        else:
            matches = [col for col in DEFAULT_TREATMENT_NAMES if col in df.columns]
            if not matches:
                raise ValueError(
                    f"Treatment column not found. Expected one of {DEFAULT_TREATMENT_NAMES}, "
                    f"available columns: {list(df.columns)}"
                )
            resolved_treatment = matches[0]

        # Resolve outcome
        if outcome_col is not None:
            if outcome_col not in df.columns:
                raise ValueError(f"Specified outcome column '{outcome_col}' not found in DataFrame.")
            resolved_outcome = outcome_col
        else:
            matches = [col for col in DEFAULT_OUTCOME_NAMES if col in df.columns]
            if not matches:
                raise ValueError(
                    f"Outcome column not found. Expected one of {DEFAULT_OUTCOME_NAMES}, "
                    f"available columns: {list(df.columns)}"
                )
            resolved_outcome = matches[0]

        # Resolve features
        if feature_cols is not None:
            missing = [col for col in feature_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Specified feature columns missing from DataFrame: {missing}")
            resolved_features = list(feature_cols)
        else:
            missing = [col for col in DEFAULT_CONFOUNDERS if col not in df.columns]
            if missing:
                raise ValueError(f"Default confounder features missing from DataFrame: {missing}")
            resolved_features = list(DEFAULT_CONFOUNDERS)

        return resolved_treatment, resolved_outcome, resolved_features

    def validate_inputs(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        feature_cols: List[str],
    ) -> None:
        """Validate causal assumptions, anti-leakage rules, and data integrity.

        Raises:
            ValueError: If validation checks fail.
        """
        if df is None or len(df) == 0:
            raise ValueError("Input DataFrame is empty or None.")

        if len(df) < 100:
            raise ValueError(f"Insufficient data for DML: minimum 100 rows required, received {len(df)}.")

        # Anti-leakage guardrail: strictly forbid post-treatment and identifier variables
        leaky_vars = [var for var in PROHIBITED_LEAKAGE_VARS if var in feature_cols]
        if leaky_vars:
            raise ValueError(
                f"Data leakage detected! The following post-treatment or identifier columns "
                f"were passed as features: {leaky_vars}. Conditioning on these induces collider/mediator bias."
            )

        if outcome_col in feature_cols:
            raise ValueError(f"Data leakage: outcome column '{outcome_col}' cannot be included in feature set.")

        if treatment_col in feature_cols:
            raise ValueError(f"Treatment column '{treatment_col}' cannot be included in feature set.")

        # Check binary domain of treatment
        t_vals = set(df[treatment_col].dropna().unique())
        if not t_vals.issubset({0, 1}):
            raise ValueError(f"Treatment column '{treatment_col}' must be strictly binary (0 or 1), found: {t_vals}")
        if len(t_vals) < 2:
            raise ValueError(f"Treatment column '{treatment_col}' must contain both 0 and 1, found only: {t_vals}")

        # Check binary domain of outcome
        y_vals = set(df[outcome_col].dropna().unique())
        if not y_vals.issubset({0, 1}):
            raise ValueError(f"Outcome column '{outcome_col}' must be strictly binary (0 or 1), found: {y_vals}")

        # Null checks
        null_counts = df[[treatment_col, outcome_col] + feature_cols].isnull().sum()
        if null_counts.sum() > 0:
            cols_with_null = null_counts[null_counts > 0].to_dict()
            raise ValueError(f"Null values detected in training columns: {cols_with_null}")

        # Non-finite checks (inf / -inf)
        num_cols = [c for c in feature_cols if c in NUMERICAL_CONFOUNDERS or (c in df.columns and pd.api.types.is_numeric_dtype(df[c]))]
        inf_cols = {}
        for c in num_cols:
            if np.isinf(df[c].to_numpy(dtype=float)).any():
                inf_cols[c] = int(np.isinf(df[c].to_numpy(dtype=float)).sum())
        if inf_cols:
            raise ValueError(f"Non-finite (inf/-inf) values detected in training columns: {inf_cols}")

    def _build_nuisance_models(self) -> Tuple[Any, Any]:
        """Construct scikit-learn base estimators for outcome and treatment models."""
        if self.base_estimator == "random_forest":
            model_y = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )
            model_t = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.base_estimator == "lightgbm":
            from lightgbm import LGBMRegressor, LGBMClassifier
            model_y = LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                verbose=-1,
            )
            model_t = LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                verbose=-1,
            )
        else:
            raise ValueError(f"Unknown base estimator: {self.base_estimator}")

        return model_y, model_t

    def _build_preprocessor(self, feature_cols: List[str]) -> ColumnTransformer:
        """Construct preprocessor for scaling numeric features and one-hot encoding categories."""
        num_cols = [c for c in feature_cols if c in NUMERICAL_CONFOUNDERS]
        cat_cols = [c for c in feature_cols if c in CATEGORICAL_CONFOUNDERS]

        # In case user passes custom columns not in canonical list
        remaining_cols = [c for c in feature_cols if c not in num_cols and c not in cat_cols]
        for c in remaining_cols:
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
        treatment_col: Optional[str] = None,
        outcome_col: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> "EconoCausalDML":
        """Fit the Double Machine Learning estimator on training data.

        Learns preprocessor transformations strictly on the training partition
        to prevent data leakage.

        Args:
            df_train: Training DataFrame.
            treatment_col: Optional treatment column name (auto-resolved if None).
            outcome_col: Optional outcome column name (auto-resolved if None).
            feature_cols: Optional feature columns list (defaults to approved confounders).

        Returns:
            Fitted EconoCausalDML instance.
        """
        t_col, y_col, f_cols = self.resolve_columns(df_train, treatment_col, outcome_col, feature_cols)
        self.validate_inputs(df_train, t_col, y_col, f_cols)

        self.treatment_col_ = t_col
        self.outcome_col_ = y_col
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
        T_train = df_train[t_col].to_numpy().astype(int)
        Y_train = df_train[y_col].to_numpy().astype(float)

        # 3. Instantiate nuisance models
        model_y, model_t = self._build_nuisance_models()

        # 4. Instantiate EconML DML estimator
        if self.model_type == "linear":
            self.estimator = LinearDML(
                model_y=model_y,
                model_t=model_t,
                discrete_treatment=True,
                cv=self.cv,
                random_state=self.random_state,
            )
        elif self.model_type == "forest":
            self.estimator = CausalForestDML(
                model_y=model_y,
                model_t=model_t,
                discrete_treatment=True,
                cv=self.cv,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                verbose=0,
            )

        # 5. Fit EconML DML
        self.estimator.fit(Y=Y_train, T=T_train, X=X_train_proc, W=None)
        self.is_fitted_ = True
        return self

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Transform test/evaluation features using the fitted preprocessor."""
        if not self.is_fitted_ or self.preprocessor is None:
            raise RuntimeError("Estimator has not been fitted yet. Call fit() before inference.")

        if self.feature_names_in_ is None:
            raise RuntimeError("feature_names_in_ is undefined.")

        if df is None or len(df) == 0:
            raise ValueError("Input DataFrame is empty or None.")

        missing = [col for col in self.feature_names_in_ if col not in df.columns]
        if missing:
            raise ValueError(f"Input DataFrame is missing required features: {missing}")

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

    def effect(self, df: pd.DataFrame) -> np.ndarray:
        """Estimate Conditional Average Treatment Effect (CATE) for each customer.

        Args:
            df: DataFrame containing pre-treatment customer features.

        Returns:
            1D array of estimated treatment effects (percentage point lift on purchase).
        """
        X_proc = self._prepare_features(df)
        effects = self.estimator.effect(X_proc)
        return np.asarray(effects).flatten()

    def effect_interval(self, df: pd.DataFrame, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate confidence intervals for individual customer treatment effects.

        Args:
            df: DataFrame containing pre-treatment customer features.
            alpha: Significance level (default: 0.05 for 95% CI).

        Returns:
            Tuple of (lower_bounds, upper_bounds).
        """
        X_proc = self._prepare_features(df)
        lb, ub = self.estimator.effect_interval(X_proc, alpha=alpha)
        return np.asarray(lb).flatten(), np.asarray(ub).flatten()

    def ate(self, df: pd.DataFrame) -> float:
        """Estimate Average Treatment Effect (ATE) across the population in df.

        Args:
            df: DataFrame containing evaluation customers.

        Returns:
            Scalar ATE point estimate.
        """
        X_proc = self._prepare_features(df)
        ate_val = self.estimator.ate(X_proc)
        return float(ate_val)

    def ate_interval(self, df: pd.DataFrame, alpha: float = 0.05) -> Tuple[float, float]:
        """Estimate confidence interval for the Average Treatment Effect (ATE).

        Args:
            df: DataFrame containing evaluation customers.
            alpha: Significance level (default: 0.05 for 95% CI).

        Returns:
            Tuple of (ate_lower_bound, ate_upper_bound).
        """
        X_proc = self._prepare_features(df)
        lb, ub = self.estimator.ate_interval(X_proc, alpha=alpha)
        return float(lb), float(ub)

    def summary(self) -> str:
        """Return formatted summary table from the EconML estimator."""
        if not self.is_fitted_ or self.estimator is None:
            raise RuntimeError("Estimator has not been fitted yet.")
        if hasattr(self.estimator, "summary"):
            return str(self.estimator.summary())
        return "Model summary is not supported by the selected estimator type."

    def evaluate(
        self,
        df_eval: pd.DataFrame,
        ground_truth_sample_ate: float = SYNTHETIC_SAMPLE_ATE,
        ground_truth_spec_ate: float = SYNTHETIC_SPEC_BENCHMARK_ATE,
    ) -> Dict[str, Any]:
        """Compute comprehensive causal inference metrics and effect distributions on evaluation data.

        Args:
            df_eval: Held-out evaluation DataFrame.
            ground_truth_sample_ate: Known sample ATE derived from generator.
            ground_truth_spec_ate: Benchmark ATE from specification.

        Returns:
            Dictionary with ATE, CATE distribution statistics, naive comparison, and ground truth delta.
        """
        t_col, y_col, _ = self.resolve_columns(df_eval, self.treatment_col_, self.outcome_col_, self.feature_names_in_)

        # Point estimates and intervals
        ate_est = self.ate(df_eval)
        ate_ci_lower, ate_ci_upper = self.ate_interval(df_eval, alpha=0.05)
        cate_estimates = self.effect(df_eval)

        # Naive observational difference
        treated_rate = float(df_eval[df_eval[t_col] == 1][y_col].mean())
        control_rate = float(df_eval[df_eval[t_col] == 0][y_col].mean())
        naive_diff = treated_rate - control_rate

        # CATE distribution summary
        n_eval = len(cate_estimates)
        n_neg = int(np.sum(cate_estimates < 0))
        n_near_zero = int(np.sum(np.abs(cate_estimates) <= 0.02))
        n_pos = int(np.sum(cate_estimates > 0))

        metrics = {
            "evaluation_sample_size": n_eval,
            "ate_estimate": round(ate_est, 4),
            "ate_ci_95": [round(ate_ci_lower, 4), round(ate_ci_upper, 4)],
            "ground_truth_sample_ate": round(ground_truth_sample_ate, 4),
            "ground_truth_spec_ate": round(ground_truth_spec_ate, 4),
            "error_vs_sample_ground_truth": round(ate_est - ground_truth_sample_ate, 4),
            "error_vs_spec_benchmark": round(ate_est - ground_truth_spec_ate, 4),
            "covers_sample_ground_truth": bool(ate_ci_lower <= ground_truth_sample_ate <= ate_ci_upper),
            "naive_observational_diff": round(naive_diff, 4),
            "confounding_bias_in_naive": round(naive_diff - ground_truth_sample_ate, 4),
            "cate_summary": {
                "mean": round(float(np.mean(cate_estimates)), 4),
                "std": round(float(np.std(cate_estimates)), 4),
                "median": round(float(np.median(cate_estimates)), 4),
                "min": round(float(np.min(cate_estimates)), 4),
                "q25": round(float(np.percentile(cate_estimates, 25)), 4),
                "q75": round(float(np.percentile(cate_estimates, 75)), 4),
                "max": round(float(np.max(cate_estimates)), 4),
            },
            "cate_distribution_counts": {
                "negative_count": n_neg,
                "negative_pct": round((n_neg / n_eval) * 100, 2),
                "near_zero_count": n_near_zero,
                "near_zero_pct": round((n_near_zero / n_eval) * 100, 2),
                "positive_count": n_pos,
                "positive_pct": round((n_pos / n_eval) * 100, 2),
            },
        }
        return metrics

    def save(
        self,
        model_path: str,
        metadata_path: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Safely serialize the trained DML pipeline and metadata to disk.

        Args:
            model_path: Filepath to save the .joblib artifact.
            metadata_path: Optional filepath to save configuration JSON.
            extra_metadata: Optional additional dictionary fields to store.
        """
        if not self.is_fitted_:
            raise RuntimeError("Cannot save unfitted estimator.")

        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)

        bundle = {
            "model_type": self.model_type,
            "base_estimator": self.base_estimator,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "cv": self.cv,
            "random_state": self.random_state,
            "preprocessor": self.preprocessor,
            "estimator": self.estimator,
            "feature_names_in": self.feature_names_in_,
            "preprocessed_feature_names": self.preprocessed_feature_names_,
            "treatment_col": self.treatment_col_,
            "outcome_col": self.outcome_col_,
        }
        joblib.dump(bundle, model_path)

        if metadata_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(metadata_path)), exist_ok=True)
            meta = {
                "model_class": self.estimator.__class__.__name__,
                "model_type": self.model_type,
                "base_estimator": self.base_estimator,
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "cv": self.cv,
                "random_state": self.random_state,
                "treatment_col": self.treatment_col_,
                "outcome_col": self.outcome_col_,
                "features": self.feature_names_in_,
                "preprocessed_features": self.preprocessed_feature_names_,
                "packages": {
                    "econml": econml.__version__,
                    "scikit-learn": sklearn.__version__,
                    "joblib": joblib.__version__,
                },
                "methodological_assumptions": [
                    "Conditional Unconfoundedness: treatment assignment is independent of potential outcomes given pre-treatment confounders W.",
                    "Positivity: propensity score 0 < P(T=1 | W) < 1 for all customer segments.",
                    "No Post-Treatment Confounders: discount_amount and revenue/order_value are strictly quarantined.",
                    "Linear Treatment Effect: CATE tau(X) is parameterized as a linear function of customer features.",
                ],
            }
            if extra_metadata:
                meta.update(extra_metadata)

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, model_path: str) -> "EconoCausalDML":
        """Load a serialized EconoCausalDML bundle from disk.

        Args:
            model_path: Filepath to the .joblib artifact.

        Returns:
            Restored EconoCausalDML instance.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        bundle = joblib.load(model_path)
        instance = cls(
            model_type=bundle["model_type"],
            base_estimator=bundle["base_estimator"],
            n_estimators=bundle["n_estimators"],
            max_depth=bundle["max_depth"],
            cv=bundle["cv"],
            random_state=bundle["random_state"],
        )
        instance.preprocessor = bundle["preprocessor"]
        instance.estimator = bundle["estimator"]
        instance.feature_names_in_ = bundle["feature_names_in"]
        instance.preprocessed_feature_names_ = bundle["preprocessed_feature_names"]
        instance.treatment_col_ = bundle["treatment_col"]
        instance.outcome_col_ = bundle["outcome_col"]
        instance.is_fitted_ = True
        return instance
