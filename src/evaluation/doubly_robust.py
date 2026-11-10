"""Doubly Robust (DR) Off-Policy Evaluation Engine for Multi-Treatment Pricing Policies.

Combines an outcome regression model mu_d(X) and generalized propensity score model e_d(X)
to achieve doubly robust policy value estimation: unbiased if either the outcome model
or the propensity model is correctly specified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

from .ipw import InverseProbabilityWeighting, IPWResult


@dataclass
class DRResult:
    """Encapsulates the result of Doubly Robust policy value estimation."""
    policy_name: str
    dr_policy_value: float
    direct_method_value: float
    ipw_policy_value: float
    variance: float
    standard_error: float
    effective_sample_size: float
    sample_size: int
    propensity_clipped_pct: float
    influence_scores: np.ndarray = field(repr=False)
    outcome_type: str = "conversion"
    ci_95_normal: Tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to JSON-serializable dictionary."""
        return {
            "policy_name": self.policy_name,
            "dr_policy_value": float(self.dr_policy_value),
            "direct_method_value": float(self.direct_method_value),
            "ipw_policy_value": float(self.ipw_policy_value),
            "variance": float(self.variance),
            "standard_error": float(self.standard_error),
            "ci_95_normal": [float(self.ci_95_normal[0]), float(self.ci_95_normal[1])],
            "effective_sample_size": float(self.effective_sample_size),
            "sample_size": int(self.sample_size),
            "propensity_clipped_pct": float(self.propensity_clipped_pct),
            "outcome_type": self.outcome_type,
        }


class DoublyRobustEstimator:
    """Multi-treatment Doubly Robust (DR) Policy Value Estimator.

    Formalized for discrete multi-treatment arms D in {0, 10, 20}:
        psi_i(pi) = sum_{d in D} [ pi(d | X_i) * mu_d(X_i) +
                                   (1(T_i = d) * pi(d | X_i) / e_d(X_i)) * (Y_i - mu_d(X_i)) ]
        V_DR(pi) = (1 / N) * sum_{i=1}^N psi_i(pi)
    """

    def __init__(
        self,
        treatment_classes: Optional[List[int]] = None,
        min_propensity: float = 0.01,
        max_weight: Optional[float] = 100.0,
    ):
        self.treatment_classes = treatment_classes if treatment_classes is not None else [0, 10, 20]
        self.ipw = InverseProbabilityWeighting(
            treatment_classes=self.treatment_classes,
            min_propensity=min_propensity,
            max_weight=max_weight,
        )
        self.min_propensity = self.ipw.min_propensity

    def evaluate_policy(
        self,
        y: np.ndarray,
        t: np.ndarray,
        propensities: np.ndarray,
        mu_hat: np.ndarray,
        policy_actions: Union[np.ndarray, Dict[int, np.ndarray]],
        policy_name: str = "evaluated_policy",
        outcome_type: str = "conversion",
    ) -> DRResult:
        """Evaluate a policy using the Doubly Robust estimator.

        Args:
            y: Observed outcomes (binary conversion Y in {0, 1} or continuous profit R).
            t: Observed treatments T in {0, 10, 20}.
            propensities: Matrix of shape (N, K) containing P(T = d_k | X).
            mu_hat: Matrix of shape (N, K) containing predicted outcomes E[Y | X, T = d_k].
            policy_actions: Assigned treatment tiers (1D array) or stochastic policy dict.
            policy_name: Descriptive name for reporting.
            outcome_type: "conversion" or "profit".

        Returns:
            DRResult containing DR value, DM value, IPW value, SE, and influence scores.
        """
        y_arr, t_arr, prop_arr, pol_arr = self.ipw._validate_inputs(y, t, propensities, policy_actions)
        n = len(y_arr)
        k = len(self.treatment_classes)

        mu_arr = np.asarray(mu_hat, dtype=float)
        if mu_arr.shape != (n, k):
            raise ValueError(
                f"mu_hat shape mismatch: expected ({n}, {k}), got {mu_arr.shape}"
            )

        # Map observed treatment values to index
        class_to_idx = {c: i for i, c in enumerate(self.treatment_classes)}
        t_indices = np.array([class_to_idx[val] for val in t_arr], dtype=int)

        # Clip propensities for observed treatments
        observed_prop = prop_arr[np.arange(n), t_indices]
        clipped_mask = observed_prop < self.min_propensity
        clipped_pct = float(np.mean(clipped_mask) * 100.0)
        clipped_prop = np.maximum(observed_prop, self.min_propensity)

        # 1. Direct Method Component: sum_d pi(d | X_i) * mu_d(X_i)
        dm_scores = np.sum(pol_arr * mu_arr, axis=1)
        dm_value = float(np.mean(dm_scores))

        # 2. IPW Correction Component:
        # For the observed arm T_i:
        # correction_i = (pi(T_i | X_i) / e_{T_i}(X_i)) * (Y_i - mu_{T_i}(X_i))
        observed_pol_prob = pol_arr[np.arange(n), t_indices]
        observed_mu = mu_arr[np.arange(n), t_indices]
        residuals = y_arr - observed_mu

        weights = observed_pol_prob / clipped_prop
        if self.ipw.max_weight is not None:
            weights = np.minimum(weights, self.ipw.max_weight)

        ipw_correction = weights * residuals

        # Individual Doubly Robust Influence Scores
        psi = dm_scores + ipw_correction
        dr_value = float(np.mean(psi))

        # Pure IPW value for triangulation
        ipw_scores = weights * y_arr
        ipw_value = float(np.mean(ipw_scores))

        # Asymptotic variance & SE
        variance = float(np.var(psi, ddof=1) / n) if n > 1 else 0.0
        se = float(np.sqrt(variance))
        ci_95 = (dr_value - 1.96 * se, dr_value + 1.96 * se)

        # Effective Sample Size
        sum_w = np.sum(weights)
        sum_sq_w = np.sum(weights ** 2)
        ess = float((sum_w ** 2) / sum_sq_w) if sum_sq_w > 0 else 0.0

        return DRResult(
            policy_name=policy_name,
            dr_policy_value=dr_value,
            direct_method_value=dm_value,
            ipw_policy_value=ipw_value,
            variance=variance,
            standard_error=se,
            effective_sample_size=ess,
            sample_size=n,
            propensity_clipped_pct=clipped_pct,
            influence_scores=psi,
            outcome_type=outcome_type,
            ci_95_normal=ci_95,
        )

    def cross_fit_nuisances(
        self,
        X: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        n_splits: int = 5,
        random_state: int = 42,
        base_regressor: Optional[BaseEstimator] = None,
    ) -> np.ndarray:
        """Estimate outcome regressions mu_d(X) via K-fold cross-fitting.

        Ensures that mu_d(X_i) for observation i is generated by a model
        trained on folds that exclude observation i.
        """
        n = len(y)
        k = len(self.treatment_classes)
        mu_hat = np.zeros((n, k), dtype=float)

        from sklearn.base import clone
        from sklearn.model_selection import StratifiedKFold
        from sklearn.ensemble import GradientBoostingRegressor

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        for train_idx, val_idx in skf.split(X, t):
            X_train, y_train, t_train = X[train_idx], y[train_idx], t[train_idx]
            X_val = X[val_idx]

            for idx, arm in enumerate(self.treatment_classes):
                arm_mask = (t_train == arm)
                if np.sum(arm_mask) < 5:
                    # Fallback to mean if arm is rare
                    mu_hat[val_idx, idx] = np.mean(y_train) if len(y_train) > 0 else 0.0
                    continue

                if base_regressor is None:
                    # Default: GradientBoostingRegressor with moderate depth
                    model = GradientBoostingRegressor(
                        n_estimators=50,
                        max_depth=3,
                        learning_rate=0.08,
                        random_state=random_state,
                    )
                else:
                    model = clone(base_regressor)

                model.fit(X_train[arm_mask], y_train[arm_mask])
                mu_hat[val_idx, idx] = model.predict(X_val)

        return mu_hat
