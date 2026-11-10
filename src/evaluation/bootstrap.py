"""Non-Parametric Bootstrap Confidence Interval Engine for Policy Evaluation.

Provides reproducible percentile and normal-approximation bootstrap intervals
for individual policy values and paired policy contrasts (incremental value).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np


@dataclass
class BootstrapResult:
    """Encapsulates non-parametric bootstrap estimation results."""
    point_estimate: float
    standard_error: float
    ci_lower_percentile: float
    ci_upper_percentile: float
    ci_lower_normal: float
    ci_upper_normal: float
    n_bootstraps: int
    confidence_level: float
    bootstrap_distribution: np.ndarray = field(repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to JSON-serializable dictionary."""
        return {
            "point_estimate": float(self.point_estimate),
            "standard_error": float(self.standard_error),
            "ci_lower_percentile": float(self.ci_lower_percentile),
            "ci_upper_percentile": float(self.ci_upper_percentile),
            "ci_lower_normal": float(self.ci_lower_normal),
            "ci_upper_normal": float(self.ci_upper_normal),
            "n_bootstraps": int(self.n_bootstraps),
            "confidence_level": float(self.confidence_level),
        }


class PolicyEvaluatorBootstrap:
    """Bootstrap inference engine for Off-Policy Evaluation."""

    def __init__(
        self,
        n_bootstraps: int = 500,
        confidence_level: float = 0.95,
        random_state: int = 42,
    ):
        if n_bootstraps < 10:
            raise ValueError(f"n_bootstraps must be >= 10, got {n_bootstraps}")
        if confidence_level <= 0.0 or confidence_level >= 1.0:
            raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")

        self.n_bootstraps = n_bootstraps
        self.confidence_level = confidence_level
        self.random_state = random_state

    def bootstrap_influence_scores(
        self,
        scores: np.ndarray,
        point_estimate: Optional[float] = None,
    ) -> BootstrapResult:
        """Compute bootstrap CI from pre-computed individual influence scores (e.g. DR psi_i).

        By the asymptotic linearity of the Doubly Robust estimator:
            V_DR = (1 / N) * sum_{i=1}^N psi_i
        A non-parametric resample of observations corresponds directly to
        resampling the influence score vector psi.
        """
        arr = np.asarray(scores, dtype=float).ravel()
        n = len(arr)
        if n == 0:
            raise ValueError("Input scores array is empty.")

        pe = point_estimate if point_estimate is not None else float(np.mean(arr))

        rng = np.random.default_rng(self.random_state)
        # Vectorized resampling: draw (n_bootstraps, n) indices with replacement
        boot_indices = rng.integers(0, n, size=(self.n_bootstraps, n))
        boot_means = np.mean(arr[boot_indices], axis=1)

        # Standard error
        se = float(np.std(boot_means, ddof=1))

        # Alpha percentiles
        alpha = 1.0 - self.confidence_level
        lower_pct = (alpha / 2.0) * 100.0
        upper_pct = (1.0 - alpha / 2.0) * 100.0

        ci_lower_perc = float(np.percentile(boot_means, lower_pct))
        ci_upper_perc = float(np.percentile(boot_means, upper_pct))

        # Normal approximation
        z_crit = 1.96 if abs(self.confidence_level - 0.95) < 1e-4 else 2.576
        ci_lower_norm = pe - z_crit * se
        ci_upper_norm = pe + z_crit * se

        return BootstrapResult(
            point_estimate=pe,
            standard_error=se,
            ci_lower_percentile=ci_lower_perc,
            ci_upper_percentile=ci_upper_perc,
            ci_lower_normal=ci_lower_norm,
            ci_upper_normal=ci_upper_norm,
            n_bootstraps=self.n_bootstraps,
            confidence_level=self.confidence_level,
            bootstrap_distribution=boot_means,
        )

    def bootstrap_paired_contrast(
        self,
        scores_policy: np.ndarray,
        scores_baseline: np.ndarray,
    ) -> BootstrapResult:
        """Compute bootstrap CI for the incremental contrast between two policies:
            Delta V = V(pi_policy) - V(pi_baseline)

        Uses paired resampling on the same bootstrap samples to preserve cross-policy covariance.
        """
        sp = np.asarray(scores_policy, dtype=float).ravel()
        sb = np.asarray(scores_baseline, dtype=float).ravel()

        if len(sp) != len(sb):
            raise ValueError(f"Length mismatch: policy scores ({len(sp)}) != baseline scores ({len(sb)})")

        diff_scores = sp - sb
        return self.bootstrap_influence_scores(diff_scores)
