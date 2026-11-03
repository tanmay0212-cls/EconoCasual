"""Hillstrom MineThatData 3-Arm Randomized Benchmark Causal Engine.

This module provides causal estimation and benchmark evaluation for the Hillstrom
MineThatData Email Marketing experiment:
- Arm 0: No E-Mail (Control, N ~ 21,306)
- Arm 1: Men's E-Mail (N ~ 21,307)
- Arm 2: Women's E-Mail (N ~ 21,387)

Evaluates three business funnel outcomes separately:
1. `visit` (binary immediate engagement)
2. `conversion` (binary purchase decision)
3. `spend` (continuous revenue realization)
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from lightgbm import LGBMRegressor, LGBMClassifier
from econml.dml import LinearDML, CausalForestDML


class HillstromCausalEngine:
    """Multi-outcome, multi-treatment causal estimation engine for Hillstrom RCT."""

    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.numerical_cols = ["recency", "history", "mens", "womens", "newbie"]
        self.categorical_cols = ["zip_code", "channel", "history_segment"]
        
        self.preprocessor = None
        self.feature_names = None
        self.treatment_classes = [0, 1, 2]
        self.treatment_labels = {0: "no_email", 1: "mens_email", 2: "womens_email"}
        
        # Dedicated estimators per outcome
        self.models = {}
        self.baseline_models = {}

    def _build_preprocessor(self) -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numerical_cols),
                ("cat", OneHotEncoder(drop="first", sparse_output=False), self.categorical_cols),
            ]
        )

    def fit(self, df_train: pd.DataFrame, treatment_col: str = "treatment_group") -> "HillstromCausalEngine":
        """Fit causal models for visit, conversion, and spend outcomes."""
        self.preprocessor = self._build_preprocessor()
        X_df = df_train[self.numerical_cols + self.categorical_cols]
        X = self.preprocessor.fit_transform(X_df)

        num_names = self.numerical_cols
        cat_names = list(self.preprocessor.named_transformers_["cat"].get_feature_names_out(self.categorical_cols))
        self.feature_names = num_names + cat_names

        T = df_train[treatment_col].astype(int).to_numpy()

        # Fit models across all three distinct outcomes
        for outcome in ["visit", "conversion", "spend"]:
            Y = df_train[outcome].astype(float).to_numpy()

            # Orthogonalized DML with discrete treatment support
            model_t = LogisticRegression(solver="lbfgs", max_iter=500, random_state=self.random_state)
            
            if outcome in ["visit", "conversion"]:
                model_y = LGBMRegressor(n_estimators=50, max_depth=4, learning_rate=0.08, random_state=self.random_state, verbose=-1)
            else:
                model_y = Ridge(alpha=10.0, random_state=self.random_state)

            dml_model = LinearDML(
                model_y=model_y,
                model_t=model_t,
                discrete_treatment=True,
                categories=self.treatment_classes,
                cv=2,
                random_state=self.random_state,
            )
            dml_model.fit(Y, T, X=X)
            self.models[outcome] = dml_model

            # Fit baseline control group outcome model for counterfactual prediction
            control_mask = (T == 0)
            X_control = X[control_mask]
            Y_control = Y[control_mask]

            if outcome in ["visit", "conversion"]:
                base_clf = LGBMClassifier(n_estimators=50, max_depth=4, learning_rate=0.05, random_state=self.random_state, verbose=-1)
                base_clf.fit(X_control, Y_control.astype(int))
                self.baseline_models[outcome] = base_clf
            else:
                base_reg = Ridge(alpha=10.0, random_state=self.random_state)
                base_reg.fit(X_control, Y_control)
                self.baseline_models[outcome] = base_reg

        return self

    def predict_effects(self, df: pd.DataFrame, outcome: str = "conversion") -> pd.DataFrame:
        """Estimate causal treatment effects for a specific outcome."""
        if outcome not in self.models:
            raise ValueError(f"Outcome '{outcome}' not trained. Choose from {list(self.models.keys())}")

        X_df = df[self.numerical_cols + self.categorical_cols]
        X = self.preprocessor.transform(X_df)
        model = self.models[outcome]

        tau_1_vs_0 = np.asarray(model.effect(X, T0=0, T1=1)).ravel()
        tau_2_vs_0 = np.asarray(model.effect(X, T0=0, T1=2)).ravel()
        tau_2_vs_1 = np.asarray(model.effect(X, T0=1, T1=2)).ravel()

        return pd.DataFrame({
            f"ite_{outcome}_mens_vs_control": np.round(tau_1_vs_0, 5),
            f"ite_{outcome}_womens_vs_control": np.round(tau_2_vs_0, 5),
            f"ite_{outcome}_womens_vs_mens": np.round(tau_2_vs_1, 5),
        }, index=df.index)

    def predict_counterfactuals(self, df: pd.DataFrame, outcome: str = "conversion") -> pd.DataFrame:
        """Predict expected counterfactual outcomes under all three email arms."""
        X_df = df[self.numerical_cols + self.categorical_cols]
        X = self.preprocessor.transform(X_df)
        base_model = self.baseline_models[outcome]
        effects_df = self.predict_effects(df, outcome=outcome)

        ite_1 = effects_df[f"ite_{outcome}_mens_vs_control"].to_numpy()
        ite_2 = effects_df[f"ite_{outcome}_womens_vs_control"].to_numpy()

        if outcome in ["visit", "conversion"]:
            y0 = base_model.predict_proba(X)[:, 1]
            y1 = np.clip(y0 + ite_1, 0.0001, 0.9999)
            y2 = np.clip(y0 + ite_2, 0.0001, 0.9999)
        else: # spend
            y0 = np.maximum(0.0, base_model.predict(X))
            y1 = np.maximum(0.0, y0 + ite_1)
            y2 = np.maximum(0.0, y0 + ite_2)

        return pd.DataFrame({
            f"pred_{outcome}_no_email": np.round(y0, 5),
            f"pred_{outcome}_mens_email": np.round(y1, 5),
            f"pred_{outcome}_womens_email": np.round(y2, 5),
        }, index=df.index)

    def save(self, model_dir: str = "models/hillstrom"):
        """Save Hillstrom causal model artifacts."""
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.preprocessor, os.path.join(model_dir, "preprocessor.joblib"))
        joblib.dump(self.models, os.path.join(model_dir, "hillstrom_dml_models.joblib"))
        joblib.dump(self.baseline_models, os.path.join(model_dir, "hillstrom_baseline_models.joblib"))

        metadata = {
            "model_type": "LinearDML_MultiOutcome",
            "treatment_classes": self.treatment_classes,
            "treatment_labels": self.treatment_labels,
            "outcomes": list(self.models.keys()),
            "numerical_features": self.numerical_cols,
            "categorical_features": self.categorical_cols,
            "transformed_features": self.feature_names,
        }
        with open(os.path.join(model_dir, "model_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  [Artifacts] Hillstrom causal model saved to: {model_dir}")

    @classmethod
    def load(cls, model_dir: str = "models/hillstrom") -> "HillstromCausalEngine":
        """Load trained Hillstrom causal engine."""
        with open(os.path.join(model_dir, "model_metadata.json"), "r") as f:
            metadata = json.load(f)
        instance = cls()
        instance.preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.joblib"))
        instance.models = joblib.load(os.path.join(model_dir, "hillstrom_dml_models.joblib"))
        instance.baseline_models = joblib.load(os.path.join(model_dir, "hillstrom_baseline_models.joblib"))
        instance.feature_names = metadata["transformed_features"]
        return instance

