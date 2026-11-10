"""Multi-Treatment Inverse Probability Weighting (IPW) Policy Evaluation Engine.

Provides Horvitz-Thompson and Self-Normalized (Hajek) IPW policy value estimation
for discrete multi-treatment policies with positivity overlap diagnostics and
propensity clipping analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class IPWResult:
    """Encapsulates the result of Inverse Probability Weighting policy evaluation."""
    policy_name: str
    policy_value: float
    normalized_policy_value: float
    variance: float
    standard_error: float
    effective_sample_size: float
    sample_size: int
    propensity_clipped_count: int
    propensity_clipped_pct: float
    min_observed_propensity: float
    max_importance_weight: float
    mean_importance_weight: float
    weights: np.ndarray = field(repr=False)
    clipping_sensitivity: Dict[float, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to JSON-serializable dictionary."""
        return {
            "policy_name": self.policy_name,
            "policy_value": float(self.policy_value),
            "normalized_policy_value": float(self.normalized_policy_value),
            "variance": float(self.variance),
            "standard_error": float(self.standard_error),
            "effective_sample_size": float(self.effective_sample_size),
            "sample_size": int(self.sample_size),
            "propensity_clipped_count": int(self.propensity_clipped_count),
            "propensity_clipped_pct": float(self.propensity_clipped_pct),
            "min_observed_propensity": float(self.min_observed_propensity),
            "max_importance_weight": float(self.max_importance_weight),
            "mean_importance_weight": float(self.mean_importance_weight),
            "clipping_sensitivity": {str(k): float(v) for k, v in self.clipping_sensitivity.items()},
        }


class InverseProbabilityWeighting:
    """Multi-treatment Inverse Probability Weighting (IPW) Policy Evaluator.

    Formalized for discrete multi-treatment arms D in {0, 10, 20} under the
    generalized propensity framework:
        w_i = pi(T_i | X_i) / e_{T_i}(X_i)
        V_IPW = (1 / N) * sum_{i=1}^N w_i * Y_i
        V_SNIPW = (sum_{i=1}^N w_i * Y_i) / (sum_{i=1}^N w_i)
    """

    def __init__(
        self,
        treatment_classes: Optional[List[int]] = None,
        min_propensity: float = 0.01,
        max_weight: Optional[float] = 100.0,
    ):
        """Initialize the IPW evaluator.

        Args:
            treatment_classes: Discrete treatment classes, e.g. [0, 10, 20]. Defaults to [0, 10, 20].
            min_propensity: Lower bound threshold for propensity clipping to prevent division by zero.
            max_weight: Upper bound threshold for importance weight truncation.
        """
        self.treatment_classes = treatment_classes if treatment_classes is not None else [0, 10, 20]
        if min_propensity <= 0.0 or min_propensity >= 1.0:
            raise ValueError(f"min_propensity must be in (0, 1), got {min_propensity}")
        self.min_propensity = float(min_propensity)
        self.max_weight = float(max_weight) if max_weight is not None else None

    def _validate_inputs(
        self,
        y: np.ndarray,
        t: np.ndarray,
        propensities: np.ndarray,
        policy_actions: Union[np.ndarray, Dict[int, np.ndarray]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Validate input arrays and formats."""
        y_arr = np.asarray(y, dtype=float).ravel()
        t_arr = np.asarray(t, dtype=int).ravel()
        n = len(y_arr)

        if len(t_arr) != n:
            raise ValueError(f"Length mismatch: y has {n} rows, t has {len(t_arr)} rows.")

        if n == 0:
            raise ValueError("Input dataset is empty (N=0).")

        prop_arr = np.asarray(propensities, dtype=float)
        if prop_arr.ndim != 2:
            raise ValueError(f"Propensities must be a 2D array of shape (N, K), got ndim={prop_arr.ndim}")
        if prop_arr.shape[0] != n:
            raise ValueError(f"Propensities rows {prop_arr.shape[0]} != dataset rows {n}")
        if prop_arr.shape[1] != len(self.treatment_classes):
            raise ValueError(
                f"Propensities columns {prop_arr.shape[1]} != treatment classes {len(self.treatment_classes)}"
            )

        # Check valid treatment values
        invalid_t = set(np.unique(t_arr)) - set(self.treatment_classes)
        if invalid_t:
            raise ValueError(f"Observed treatments contain invalid classes: {invalid_t}. Expected {self.treatment_classes}")

        # Check policy actions
        if isinstance(policy_actions, dict):
            # Stochastic policy: Dict[treatment_arm, array_of_probabilities]
            pol_matrix = np.zeros((n, len(self.treatment_classes)), dtype=float)
            for idx, arm in enumerate(self.treatment_classes):
                if arm not in policy_actions:
                    raise ValueError(f"Policy action distribution missing treatment arm {arm}")
                pol_matrix[:, idx] = np.asarray(policy_actions[arm], dtype=float).ravel()
            # Normalize row sums to 1
            row_sums = pol_matrix.sum(axis=1, keepdims=True)
            if np.any(row_sums <= 0):
                raise ValueError("Policy assignment probabilities sum to <= 0 for some rows.")
            pol_arr = pol_matrix / row_sums
        elif isinstance(policy_actions, np.ndarray) and policy_actions.ndim == 2:
            # Stochastic policy passed as 2D matrix (N, K)
            if policy_actions.shape != (n, len(self.treatment_classes)):
                raise ValueError(
                    f"Policy probability matrix shape {policy_actions.shape} != expected ({n}, {len(self.treatment_classes)})"
                )
            row_sums = policy_actions.sum(axis=1, keepdims=True)
            if np.any(row_sums <= 0):
                raise ValueError("Policy assignment probabilities sum to <= 0 for some rows.")
            pol_arr = policy_actions / row_sums
        else:
            # Deterministic policy: 1D array of assigned treatments
            actions_1d = np.asarray(policy_actions, dtype=int).ravel()
            if len(actions_1d) != n:
                raise ValueError(f"Policy actions length {len(actions_1d)} != dataset rows {n}")
            invalid_pol = set(np.unique(actions_1d)) - set(self.treatment_classes)
            if invalid_pol:
                raise ValueError(f"Policy actions contain invalid classes: {invalid_pol}. Expected {self.treatment_classes}")
            # Convert deterministic actions to indicator matrix
            pol_arr = np.zeros((n, len(self.treatment_classes)), dtype=float)
            for idx, arm in enumerate(self.treatment_classes):
                pol_arr[:, idx] = (actions_1d == arm).astype(float)

        return y_arr, t_arr, prop_arr, pol_arr

    def compute_weights(
        self,
        t: np.ndarray,
        propensities: np.ndarray,
        policy_prob_matrix: np.ndarray,
        min_propensity: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute generalized importance weights with propensity clipping.

        Returns:
            weights: 1D array of individual importance weights w_i.
            diagnostics: Dictionary of clipping and overlap diagnostics.
        """
        eff_clip = min_propensity if min_propensity is not None else self.min_propensity
        n = len(t)
        weights = np.zeros(n, dtype=float)

        # Map treatment value to column index
        class_to_idx = {c: i for i, c in enumerate(self.treatment_classes)}
        t_indices = np.array([class_to_idx[val] for val in t], dtype=int)

        # Extract propensity for observed treatment
        observed_propensities = propensities[np.arange(n), t_indices]
        min_obs = float(np.min(observed_propensities))

        # Check how many propensities are below threshold
        clipped_mask = observed_propensities < eff_clip
        clipped_count = int(np.sum(clipped_mask))
        clipped_pct = float((clipped_count / n) * 100.0)

        clipped_propensities = np.maximum(observed_propensities, eff_clip)

        # Extract policy probability of assigning observed treatment
        policy_prob_observed = policy_prob_matrix[np.arange(n), t_indices]

        weights = policy_prob_observed / clipped_propensities

        if self.max_weight is not None:
            weights = np.minimum(weights, self.max_weight)

        # Effective Sample Size (Kish's ESS)
        sum_w = np.sum(weights)
        sum_sq_w = np.sum(weights ** 2)
        ess = float((sum_w ** 2) / sum_sq_w) if sum_sq_w > 0 else 0.0

        diagnostics = {
            "propensity_clipped_count": clipped_count,
            "propensity_clipped_pct": clipped_pct,
            "min_observed_propensity": min_obs,
            "max_importance_weight": float(np.max(weights)),
            "mean_importance_weight": float(np.mean(weights)),
            "effective_sample_size": ess,
        }
        return weights, diagnostics

    def evaluate_policy(
        self,
        y: np.ndarray,
        t: np.ndarray,
        propensities: np.ndarray,
        policy_actions: Union[np.ndarray, Dict[int, np.ndarray]],
        policy_name: str = "evaluated_policy",
        evaluate_clipping_grid: bool = True,
    ) -> IPWResult:
        """Evaluate policy value using Inverse Probability Weighting."""
        y_arr, t_arr, prop_arr, pol_arr = self._validate_inputs(y, t, propensities, policy_actions)
        n = len(y_arr)

        weights, diag = self.compute_weights(t_arr, prop_arr, pol_arr, self.min_propensity)

        # Unnormalized Horvitz-Thompson value
        weighted_outcomes = weights * y_arr
        policy_value = float(np.mean(weighted_outcomes))

        # Self-Normalized (Hajek) value
        sum_weights = np.sum(weights)
        if sum_weights > 0:
            normalized_value = float(np.sum(weighted_outcomes) / sum_weights)
        else:
            normalized_value = 0.0

        # Sample variance of the estimator
        var_est = float(np.var(weighted_outcomes, ddof=1) / n) if n > 1 else 0.0
        se_est = float(np.sqrt(var_est))

        # Sensitivity analysis across clipping grid
        sensitivity: Dict[float, float] = {}
        if evaluate_clipping_grid:
            grid = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
            for clip_val in grid:
                w_grid, _ = self.compute_weights(t_arr, prop_arr, pol_arr, min_propensity=clip_val)
                val_grid = float(np.mean(w_grid * y_arr))
                sensitivity[clip_val] = val_grid

        return IPWResult(
            policy_name=policy_name,
            policy_value=policy_value,
            normalized_policy_value=normalized_value,
            variance=var_est,
            standard_error=se_est,
            effective_sample_size=diag["effective_sample_size"],
            sample_size=n,
            propensity_clipped_count=diag["propensity_clipped_count"],
            propensity_clipped_pct=diag["propensity_clipped_pct"],
            min_observed_propensity=diag["min_observed_propensity"],
            max_importance_weight=diag["max_importance_weight"],
            mean_importance_weight=diag["mean_importance_weight"],
            weights=weights,
            clipping_sensitivity=sensitivity,
        )
