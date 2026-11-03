"""Multi-Treatment Causal Estimation Engine using EconML CausalForestDML.

This module provides causal estimation for retail promotional pricing with three discrete treatment options:
- T0 = $0 (Control / No discount)
- T1 = $10 (Moderate promotional discount)
- T2 = $20 (Deep promotional discount)

Architecture:
- Estimator: EconML CausalForestDML with discrete_treatment=True
- Nuisance models: Cross-fitted gradient boosted regressors and calibrated logistic propensities
- Neyman-orthogonal double machine learning to eliminate confounding bias from customer activity & loyalty
- Heterogeneous Individual Treatment Effect (CATE) estimation for all pairwise treatment contrasts:
  * tau_10_vs_0(X) = E[Y(10) - Y(0) | X]
  * tau_20_vs_0(X) = E[Y(20) - Y(0) | X]
  * tau_20_vs_10(X) = E[Y(20) - Y(10) | X]
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMRegressor, LGBMClassifier
from econml.dml import CausalForestDML


class SyntheticMultiTreatmentDML:
    """EconML Causal Forest Double Machine Learning engine for multi-treatment discounts."""

    def __init__(
        self,
        n_estimators: int = 120,
        subforest_size: int = 4,
        min_samples_leaf: int = 15,
        max_depth: Optional[int] = 8,
        cv: int = 3,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.subforest_size = subforest_size
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.numerical_cols = [
            "age",
            "loyalty_score",
            "previous_purchases",
            "average_previous_order_value",
            "customer_activity_score",
        ]
        self.categorical_cols = ["customer_segment"]
        self.feature_names = None
        self.preprocessor = None
        self.causal_model = None
        self.baseline_outcome_model = None
        self.treatment_classes = [0, 10, 20]

    def _build_preprocessor(self) -> ColumnTransformer:
        """Construct scikit-learn ColumnTransformer for pre-treatment covariates."""
        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numerical_cols),
                ("cat", OneHotEncoder(drop="first", sparse_output=False), self.categorical_cols),
            ]
        )

    def fit(self, df_train: pd.DataFrame, treatment_col: str = "discount_amount", outcome_col: str = "purchase") -> "SyntheticMultiTreatmentDML":
        """Fit Double Machine Learning Causal Forest and Baseline Outcome Model."""
        # 1. Feature Preprocessing
        self.preprocessor = self._build_preprocessor()
        X_df = df_train[self.numerical_cols + self.categorical_cols]
        X = self.preprocessor.fit_transform(X_df)
        
        # Extract transformed feature names
        num_names = self.numerical_cols
        cat_names = list(self.preprocessor.named_transformers_["cat"].get_feature_names_out(self.categorical_cols))
        self.feature_names = num_names + cat_names

        T = df_train[treatment_col].astype(int).to_numpy()
        Y = df_train[outcome_col].astype(float).to_numpy()

        # 2. First-Stage Nuisance Models
        # model_t: multinomial propensity model P(T=d | X)
        model_t = LogisticRegression(
            solver="lbfgs",
            max_iter=500,
            random_state=self.random_state,
        )
        
        # model_y: conditional expectation E[Y | X] (probability in [0, 1])
        model_y = LGBMRegressor(
            n_estimators=60,
            max_depth=5,
            learning_rate=0.08,
            random_state=self.random_state,
            verbose=-1,
        )

        # 3. EconML CausalForestDML with discrete treatment
        self.causal_model = CausalForestDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=True,
            categories=self.treatment_classes,
            n_estimators=self.n_estimators,
            subforest_size=self.subforest_size,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            cv=self.cv,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.causal_model.fit(Y, T, X=X)

        # 4. Baseline Outcome Model mu_0(X) = P(Y=1 | T=0, X) for Counterfactual Probabilities
        control_mask = (T == 0)
        X_control = X[control_mask]
        Y_control = Y[control_mask].astype(int)

        base_lgbm = LGBMClassifier(
            n_estimators=75,
            max_depth=5,
            learning_rate=0.05,
            random_state=self.random_state,
            verbose=-1,
        )
        self.baseline_outcome_model = CalibratedClassifierCV(
            estimator=base_lgbm,
            method="sigmoid",
            cv=3,
        )
        self.baseline_outcome_model.fit(X_control, Y_control)

        return self

    def predict_effects(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate customer-level conditional treatment effects (CATE / ITE)."""
        X_df = df[self.numerical_cols + self.categorical_cols]
        X = self.preprocessor.transform(X_df)

        # Pairwise contrasts relative to control T0 = 0
        tau_10_vs_0 = self.causal_model.effect(X, T0=0, T1=10)
        tau_20_vs_0 = self.causal_model.effect(X, T0=0, T1=20)
        tau_20_vs_10 = self.causal_model.effect(X, T0=10, T1=20)

        # Flatten arrays if necessary
        tau_10_vs_0 = np.asarray(tau_10_vs_0).ravel()
        tau_20_vs_0 = np.asarray(tau_20_vs_0).ravel()
        tau_20_vs_10 = np.asarray(tau_20_vs_10).ravel()

        results_df = pd.DataFrame(
            {
                "pred_ite_10_vs_0": np.round(tau_10_vs_0, 4),
                "pred_ite_20_vs_0": np.round(tau_20_vs_0, 4),
                "pred_ite_20_vs_10": np.round(tau_20_vs_10, 4),
            },
            index=df.index,
        )
        if "customer_id" in df.columns:
            results_df.insert(0, "customer_id", df["customer_id"].values)
        return results_df

    def predict_counterfactual_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict calibrated counterfactual purchase probabilities under all three discount options."""
        X_df = df[self.numerical_cols + self.categorical_cols]
        X = self.preprocessor.transform(X_df)

        # Baseline organic purchase probability P(Y=1 | T=0, X)
        prob_0 = self.baseline_outcome_model.predict_proba(X)[:, 1]

        # Add estimated CATE to baseline
        effects_df = self.predict_effects(df)
        ite_10 = effects_df["pred_ite_10_vs_0"].to_numpy()
        ite_20 = effects_df["pred_ite_20_vs_0"].to_numpy()

        prob_10 = np.clip(prob_0 + ite_10, 0.001, 0.999)
        prob_20 = np.clip(prob_0 + ite_20, 0.001, 0.999)

        probs_df = pd.DataFrame(
            {
                "pred_prob_0": np.round(prob_0, 4),
                "pred_prob_10": np.round(prob_10, 4),
                "pred_prob_20": np.round(prob_20, 4),
            },
            index=df.index,
        )
        if "customer_id" in df.columns:
            probs_df.insert(0, "customer_id", df["customer_id"].values)
        return probs_df

    def save(self, model_dir: str = "models/synthetic_retail"):
        """Save model artifacts, preprocessor, and metadata."""
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.preprocessor, os.path.join(model_dir, "preprocessor.joblib"))
        joblib.dump(self.causal_model, os.path.join(model_dir, "causal_forest_dml.joblib"))
        joblib.dump(self.baseline_outcome_model, os.path.join(model_dir, "baseline_outcome_model.joblib"))

        metadata = {
            "model_type": "CausalForestDML",
            "treatment_classes": self.treatment_classes,
            "numerical_features": self.numerical_cols,
            "categorical_features": self.categorical_cols,
            "transformed_features": self.feature_names,
            "hyperparameters": {
                "n_estimators": self.n_estimators,
                "subforest_size": self.subforest_size,
                "min_samples_leaf": self.min_samples_leaf,
                "max_depth": self.max_depth,
                "cv": self.cv,
                "random_state": self.random_state,
            },
        }
        with open(os.path.join(model_dir, "model_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  [Artifacts] Multi-treatment causal model saved to: {model_dir}")

    @classmethod
    def load(cls, model_dir: str = "models/synthetic_retail") -> "SyntheticMultiTreatmentDML":
        """Load trained causal engine from disk."""
        with open(os.path.join(model_dir, "model_metadata.json"), "r") as f:
            metadata = json.load(f)

        instance = cls(
            n_estimators=metadata["hyperparameters"]["n_estimators"],
            subforest_size=metadata["hyperparameters"]["subforest_size"],
            min_samples_leaf=metadata["hyperparameters"]["min_samples_leaf"],
            max_depth=metadata["hyperparameters"]["max_depth"],
            cv=metadata["hyperparameters"]["cv"],
            random_state=metadata["hyperparameters"]["random_state"],
        )
        instance.preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.joblib"))
        instance.causal_model = joblib.load(os.path.join(model_dir, "causal_forest_dml.joblib"))
        instance.baseline_outcome_model = joblib.load(os.path.join(model_dir, "baseline_outcome_model.joblib"))
        instance.feature_names = metadata["transformed_features"]
        instance.treatment_classes = metadata["treatment_classes"]
        return instance

