"""Comprehensive Multi-Treatment Policy Comparison and Off-Policy Evaluation Engine.

Compares candidate promotional policies against baseline business strategies:
1. No-Discount Policy ($0 for all)
2. Flat $10 Discount Policy ($10 for all)
3. Random Discount Policy (uniform stochastic assignment)
4. Existing DML + MILP Policy (Prescriptive allocation under $5,000 budget)
5. Future Uncertainty-Aware Policy Placeholder (marked unavailable)

Evaluates both conversion rates and economic profits using Doubly Robust estimation,
IPW, Direct Method, non-parametric paired bootstrap, and ground-truth validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd

from .ipw import InverseProbabilityWeighting, IPWResult
from .doubly_robust import DoublyRobustEstimator, DRResult
from .bootstrap import PolicyEvaluatorBootstrap, BootstrapResult
from src.optimization.profit_calculator import ProfitCalculator


@dataclass
class PolicyEvaluationSummary:
    """Encapsulates off-policy evaluation metrics for a single promotional policy."""
    policy_name: str
    status: str  # "evaluated" or "unavailable"
    description: str
    is_available: bool = True

    # Conversion Metrics (Y in [0, 1])
    dr_conversion_rate: float = 0.0
    dr_conversion_ci_95: Tuple[float, float] = (0.0, 0.0)
    ipw_conversion_rate: float = 0.0
    dm_conversion_rate: float = 0.0
    incremental_conversion_vs_no_discount: float = 0.0
    incremental_conversion_ci_95: Tuple[float, float] = (0.0, 0.0)

    # Economic Metrics (Dollars)
    dr_expected_profit: float = 0.0
    dr_profit_ci_95: Tuple[float, float] = (0.0, 0.0)
    model_expected_profit: float = 0.0
    ground_truth_expected_profit: Optional[float] = None
    realized_observed_profit: Optional[float] = None
    expected_promotional_cost: float = 0.0
    incremental_profit_vs_no_discount: float = 0.0
    incremental_profit_ci_95: Tuple[float, float] = (0.0, 0.0)
    incremental_profit_to_cost_ratio: Optional[float] = None

    # Diagnostics
    effective_sample_size: float = 0.0
    propensity_clipped_pct: float = 0.0
    tier_distribution: Dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "policy_name": self.policy_name,
            "status": self.status,
            "is_available": self.is_available,
            "description": self.description,
            "conversion_metrics": {
                "dr_conversion_rate": float(self.dr_conversion_rate),
                "dr_conversion_ci_95": [float(self.dr_conversion_ci_95[0]), float(self.dr_conversion_ci_95[1])],
                "ipw_conversion_rate": float(self.ipw_conversion_rate),
                "dm_conversion_rate": float(self.dm_conversion_rate),
                "incremental_conversion_vs_no_discount": float(self.incremental_conversion_vs_no_discount),
                "incremental_conversion_ci_95": [float(self.incremental_conversion_ci_95[0]), float(self.incremental_conversion_ci_95[1])],
            },
            "economic_metrics": {
                "dr_expected_profit": float(self.dr_expected_profit),
                "dr_profit_ci_95": [float(self.dr_profit_ci_95[0]), float(self.dr_profit_ci_95[1])],
                "model_expected_profit": float(self.model_expected_profit),
                "ground_truth_expected_profit": float(self.ground_truth_expected_profit) if self.ground_truth_expected_profit is not None else None,
                "realized_observed_profit": float(self.realized_observed_profit) if self.realized_observed_profit is not None else None,
                "expected_promotional_cost": float(self.expected_promotional_cost),
                "incremental_profit_vs_no_discount": float(self.incremental_profit_vs_no_discount),
                "incremental_profit_ci_95": [float(self.incremental_profit_ci_95[0]), float(self.incremental_profit_ci_95[1])],
                "incremental_profit_to_cost_ratio": float(self.incremental_profit_to_cost_ratio) if self.incremental_profit_to_cost_ratio is not None else None,
            },
            "diagnostics": {
                "effective_sample_size": float(self.effective_sample_size),
                "propensity_clipped_pct": float(self.propensity_clipped_pct),
                "tier_distribution": {str(k): int(v) for k, v in self.tier_distribution.items()},
            },
        }


class MultiTreatmentPolicyComparator:
    """Orchestrates comprehensive multi-policy comparison using Doubly Robust OPE."""

    def __init__(
        self,
        treatment_classes: Optional[List[int]] = None,
        min_propensity: float = 0.01,
        n_bootstraps: int = 500,
        random_state: int = 42,
    ):
        self.treatment_classes = treatment_classes if treatment_classes is not None else [0, 10, 20]
        self.min_propensity = min_propensity
        self.n_bootstraps = n_bootstraps
        self.random_state = random_state

        self.ipw_evaluator = InverseProbabilityWeighting(
            treatment_classes=self.treatment_classes,
            min_propensity=min_propensity,
        )
        self.dr_evaluator = DoublyRobustEstimator(
            treatment_classes=self.treatment_classes,
            min_propensity=min_propensity,
        )
        self.bootstrap_engine = PolicyEvaluatorBootstrap(
            n_bootstraps=n_bootstraps,
            confidence_level=0.95,
            random_state=random_state,
        )
        self.profit_calculator = ProfitCalculator()

    def calculate_individual_unit_profits(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate unit economics for each treatment tier from customer pre-treatment features.

        Returns:
            unit_profit_0: Array of net profit per conversion for $0 discount.
            unit_profit_10: Array of net profit per conversion for $10 discount.
            unit_profit_20: Array of net profit per conversion for $20 discount.
        """
        aov = df["average_previous_order_value"].to_numpy(dtype=float)
        seg = df["customer_segment"].to_numpy(dtype=str)
        margin_rate = self.profit_calculator.estimate_gross_margin_rate(aov, seg)

        order_val_0 = aov * 1.00
        order_val_10 = aov * 1.03
        order_val_20 = aov * 1.06

        # Net profit when customer converts: OrderValue * MarginRate - Discount
        unit_prof_0 = order_val_0 * margin_rate - 0.0
        unit_prof_10 = order_val_10 * margin_rate - 10.0
        unit_prof_20 = order_val_20 * margin_rate - 20.0

        return unit_prof_0, unit_prof_10, unit_prof_20

    def evaluate_all_policies(
        self,
        eval_df: pd.DataFrame,
        propensities: np.ndarray,
        pred_probs: np.ndarray,
        dml_milp_actions: np.ndarray,
        ground_truth_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, PolicyEvaluationSummary]:
        """Run complete off-policy evaluation across all 5 candidate policies.

        Args:
            eval_df: Held-out evaluation DataFrame containing observed 'discount_amount' and 'purchase'.
            propensities: Matrix of shape (N, 3) containing logging propensity scores P(T=d | X).
            pred_probs: Matrix of shape (N, 3) containing model predicted conversion probabilities P(Y=1 | X, d).
            dml_milp_actions: 1D array of assigned treatments from the DML + MILP optimizer.
            ground_truth_df: Optional DataFrame with true synthetic counterfactuals for validation.

        Returns:
            Dictionary mapping policy name to PolicyEvaluationSummary.
        """
        n = len(eval_df)
        t_obs = eval_df["discount_amount"].to_numpy(dtype=int)
        y_obs = eval_df["purchase"].to_numpy(dtype=float)

        # 1. Economic Unit Profits
        up0, up10, up20 = self.calculate_individual_unit_profits(eval_df)
        unit_profits = np.column_stack([up0, up10, up20])

        # Predicted profit nuisance matrix: mu_profit_{i, d} = pred_probs_{i, d} * unit_profit_{i, d}
        mu_profit = pred_probs * unit_profits

        # Observed individual profit: RealizedProfit_i = y_obs_i * unit_profit_{i, T_i}
        class_to_idx = {c: i for i, c in enumerate(self.treatment_classes)}
        t_indices = np.array([class_to_idx[val] for val in t_obs], dtype=int)
        realized_profit_obs = y_obs * unit_profits[np.arange(n), t_indices]

        # Define Candidate Policies
        policies = {}

        # Policy 1: No-Discount Policy
        policies["no_discount"] = {
            "name": "No-Discount Policy",
            "description": "Blanket $0 promotion to all customers (organic baseline).",
            "actions": np.zeros(n, dtype=int),
            "is_available": True,
        }

        # Policy 2: Flat $10 Discount Policy
        policies["flat_10"] = {
            "name": "Flat $10 Discount Policy",
            "description": "Blanket $10 promotion to all customers regardless of individual elasticity.",
            "actions": np.full(n, 10, dtype=int),
            "is_available": True,
        }

        # Policy 3: Random Discount Policy
        rng = np.random.default_rng(self.random_state)
        random_actions = rng.choice(self.treatment_classes, size=n, p=[1/3, 1/3, 1/3])
        policies["random_discount"] = {
            "name": "Random Discount Policy",
            "description": "Stochastic exploration assigning equal probability (1/3) across $0, $10, $20.",
            "actions": random_actions,
            "is_available": True,
        }

        # Policy 4: Existing DML + MILP Prescriptive Policy
        policies["dml_milp"] = {
            "name": "DML + MILP Prescriptive Policy",
            "description": "Optimal budget-constrained allocation ($5,000 budget cap) via HiGHS MILP.",
            "actions": np.asarray(dml_milp_actions, dtype=int).ravel(),
            "is_available": True,
        }

        # First evaluate No-Discount baseline for paired contrast
        no_disc_conv_res = self.dr_evaluator.evaluate_policy(
            y=y_obs, t=t_obs, propensities=propensities, mu_hat=pred_probs,
            policy_actions=policies["no_discount"]["actions"], policy_name="no_discount_conv",
        )
        no_disc_prof_res = self.dr_evaluator.evaluate_policy(
            y=realized_profit_obs, t=t_obs, propensities=propensities, mu_hat=mu_profit,
            policy_actions=policies["no_discount"]["actions"], policy_name="no_discount_prof",
            outcome_type="profit",
        )

        results: Dict[str, PolicyEvaluationSummary] = {}

        # Evaluate available policies
        for pol_key, pol_spec in policies.items():
            acts = pol_spec["actions"]

            # Tier distribution
            unique_tiers, counts = np.unique(acts, return_counts=True)
            tier_dist = {int(k): int(v) for k, v in zip(unique_tiers, counts)}

            # A. Conversion Evaluation (Binary Purchase Y)
            conv_dr = self.dr_evaluator.evaluate_policy(
                y=y_obs, t=t_obs, propensities=propensities, mu_hat=pred_probs,
                policy_actions=acts, policy_name=pol_spec["name"], outcome_type="conversion",
            )
            conv_boot = self.bootstrap_engine.bootstrap_influence_scores(conv_dr.influence_scores)

            # Paired contrast vs No-Discount for conversion
            conv_contrast_boot = self.bootstrap_engine.bootstrap_paired_contrast(
                conv_dr.influence_scores, no_disc_conv_res.influence_scores
            )

            # B. Profit Evaluation (Monetary Contribution R)
            prof_dr = self.dr_evaluator.evaluate_policy(
                y=realized_profit_obs, t=t_obs, propensities=propensities, mu_hat=mu_profit,
                policy_actions=acts, policy_name=pol_spec["name"], outcome_type="profit",
            )
            prof_boot = self.bootstrap_engine.bootstrap_influence_scores(prof_dr.influence_scores)

            # Paired contrast vs No-Discount for profit
            prof_contrast_boot = self.bootstrap_engine.bootstrap_paired_contrast(
                prof_dr.influence_scores, no_disc_prof_res.influence_scores
            )

            # Expected promotional cost
            act_indices = np.array([class_to_idx[val] for val in acts], dtype=int)
            pol_probs = pred_probs[np.arange(n), act_indices]
            exp_cost = float(np.sum(acts * pol_probs))

            # Model expected profit: sum_i P(Y=1 | X_i, a_i) * unit_profit_{i, a_i}
            model_exp_prof = float(np.sum(mu_profit[np.arange(n), act_indices]))

            # Ground truth expected profit if ground truth counterfactuals are provided
            gt_profit = None
            if ground_truth_df is not None:
                # Align ground truth records
                gt_cols = {0: "expected_net_profit_0", 10: "expected_net_profit_10", 20: "expected_net_profit_20"}
                if all(col in ground_truth_df.columns for col in gt_cols.values()):
                    gt_aligned = ground_truth_df.loc[eval_df.index] if hasattr(eval_df, "index") else ground_truth_df.head(n)
                    gt_arr = np.zeros(n, dtype=float)
                    for d_val, col_name in gt_cols.items():
                        mask = (acts == d_val)
                        gt_arr[mask] = gt_aligned.loc[mask, col_name].values
                    gt_profit = float(np.sum(gt_arr))

            # Realized observed profit on records where observed treatment matches policy
            match_mask = (t_obs == acts)
            realized_prof = float(np.sum(realized_profit_obs[match_mask])) if np.any(match_mask) else 0.0

            # Incremental profit to promotional cost ratio
            inc_prof_total = prof_contrast_boot.point_estimate * n
            inc_ratio = (inc_prof_total / exp_cost * 100.0) if exp_cost > 0 else None

            results[pol_key] = PolicyEvaluationSummary(
                policy_name=pol_spec["name"],
                status="evaluated",
                description=pol_spec["description"],
                is_available=True,
                dr_conversion_rate=conv_dr.dr_policy_value,
                dr_conversion_ci_95=(conv_boot.ci_lower_percentile, conv_boot.ci_upper_percentile),
                ipw_conversion_rate=conv_dr.ipw_policy_value,
                dm_conversion_rate=conv_dr.direct_method_value,
                incremental_conversion_vs_no_discount=conv_contrast_boot.point_estimate,
                incremental_conversion_ci_95=(conv_contrast_boot.ci_lower_percentile, conv_contrast_boot.ci_upper_percentile),
                dr_expected_profit=prof_dr.dr_policy_value * n,
                dr_profit_ci_95=(prof_boot.ci_lower_percentile * n, prof_boot.ci_upper_percentile * n),
                model_expected_profit=model_exp_prof,
                ground_truth_expected_profit=gt_profit,
                realized_observed_profit=realized_prof,
                expected_promotional_cost=exp_cost,
                incremental_profit_vs_no_discount=inc_prof_total,
                incremental_profit_ci_95=(prof_contrast_boot.ci_lower_percentile * n, prof_contrast_boot.ci_upper_percentile * n),
                incremental_profit_to_cost_ratio=inc_ratio,
                effective_sample_size=prof_dr.effective_sample_size,
                propensity_clipped_pct=prof_dr.propensity_clipped_pct,
                tier_distribution=tier_dist,
            )

        # Policy 5: Future Uncertainty-Aware Policy Placeholder
        results["uncertainty_aware_placeholder"] = PolicyEvaluationSummary(
            policy_name="Uncertainty-Aware Causal Policy (Future Phase)",
            status="unavailable",
            description="Thompson sampling / robust Bayesian posterior policy exploring high-uncertainty elasticity strata. Marked unavailable until Phase 3/4 implementation.",
            is_available=False,
        )

        return results
