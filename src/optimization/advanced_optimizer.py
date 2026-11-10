"""Uncertainty-Aware and Fairness-Constrained Mixed-Integer Linear Programming (MILP) Optimizer.

Advanced prescriptive optimization engine extending SciPy MILP for retail discount allocation:
1. Multi-Objective Decision Modes:
   - "baseline": Classical nominal expected profit maximization.
   - "conservative_lcb": Conservative Lower-Confidence-Bound (LCB) optimization.
   - "risk_adjusted": Mean-Variance risk-penalized expected profit optimization:
     Maximize [ E[Profit] - lambda * SE(Profit) ].
2. Business-Defined Fairness Constraints:
   - Demographic Parity & Allocation Floor constraints across business segments (budget, regular, premium, vip).
   - Configurable parity ratios and minimum segment allocation thresholds.
   - Strictly excludes sensitive personal attributes (age, race, gender, etc.).
3. Rigorous Operational Guardrails:
   - Exactly one discount tier per customer.
   - Strict adherence to ex-ante expected redemption budget ceiling.
   - Explicit solver status, infeasibility diagnostics, and cash-budget disclaimers.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

from src.optimization.profit_calculator import ProfitCalculator


@dataclass
class SegmentFairnessDiagnostic:
    """Detailed allocation, uplift, and economic metrics for a customer segment."""
    segment_name: str
    sample_size: int
    allocation_count: int
    allocation_rate_pct: float
    average_discount_amount: float
    mean_incremental_conversion_pct: float
    mean_incremental_profit: float
    parity_ratio_vs_overall: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "segment_name": self.segment_name,
            "sample_size": int(self.sample_size),
            "allocation_count": int(self.allocation_count),
            "allocation_rate_pct": round(float(self.allocation_rate_pct), 2),
            "average_discount_amount": round(float(self.average_discount_amount), 2),
            "mean_incremental_conversion_pct": round(float(self.mean_incremental_conversion_pct), 2),
            "mean_incremental_profit": round(float(self.mean_incremental_profit), 2),
            "parity_ratio_vs_overall": round(float(self.parity_ratio_vs_overall), 3),
        }


@dataclass
class AdvancedOptimizationResult:
    """Encapsulates the complete result of an advanced MILP optimization run."""
    success: bool
    status_code: int
    status_message: str
    optimization_mode: str
    risk_aversion: float
    budget: float
    budget_type: str
    total_expected_cost: float
    budget_utilization_pct: float
    nominal_expected_profit: float
    conservative_lcb_profit: float
    risk_adjusted_objective_value: float
    tier_counts: Dict[int, int]
    selected_customers_mean_uncertainty: float
    selected_customers_reliability_distribution: Dict[str, int]
    fairness_metrics: Dict[str, SegmentFairnessDiagnostic]
    allocation_df: pd.DataFrame
    runtime_seconds: float
    cash_budget_disclaimer: str = (
        "Expected promotional cost is an ex-ante mathematical expectation calculated from "
        "predicted redemption probabilities. Realized redemption expenditure depends on actual "
        "customer purchases and may exhibit finite-sample variance around this expectation."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert result summary to JSON-serializable dictionary."""
        return {
            "success": bool(self.success),
            "status_code": int(self.status_code),
            "status_message": str(self.status_message),
            "optimization_mode": str(self.optimization_mode),
            "risk_aversion": float(self.risk_aversion),
            "budget": float(self.budget),
            "budget_type": str(self.budget_type),
            "financial_kpis": {
                "total_expected_cost": round(float(self.total_expected_cost), 2),
                "budget_utilization_pct": round(float(self.budget_utilization_pct), 2),
                "nominal_expected_profit": round(float(self.nominal_expected_profit), 2),
                "conservative_lcb_profit": round(float(self.conservative_lcb_profit), 2),
                "risk_adjusted_objective_value": round(float(self.risk_adjusted_objective_value), 2),
            },
            "tier_counts": {str(k): int(v) for k, v in self.tier_counts.items()},
            "uncertainty_kpis": {
                "selected_customers_mean_uncertainty": round(float(self.selected_customers_mean_uncertainty), 4),
                "selected_customers_reliability_distribution": {
                    str(k): int(v) for k, v in self.selected_customers_reliability_distribution.items()
                },
            },
            "fairness_metrics": {
                seg: diag.to_dict() for seg, diag in self.fairness_metrics.items()
            },
            "runtime_seconds": round(float(self.runtime_seconds), 4),
            "cash_budget_disclaimer": str(self.cash_budget_disclaimer),
        }


class AdvancedMILPOptimizer:
    """Production-grade uncertainty-aware and fairness-constrained MILP optimizer."""

    def __init__(
        self,
        budget: float = 5000.0,
        budget_type: str = "redemption",
        optimization_mode: str = "baseline",
        risk_aversion: float = 1.0,
        confidence_level: float = 0.95,
        fairness_enabled: bool = False,
        fairness_mode: str = "min_allocation_rate",
        min_segment_rate: Optional[Dict[str, float]] = None,
        parity_ratio_floor: float = 0.60,
        discount_tiers: Optional[List[int]] = None,
        segment_col: str = "customer_segment",
    ):
        if budget < 0.0:
            raise ValueError(f"Budget must be non-negative, got {budget}")
        if budget_type not in ["redemption", "face_value"]:
            raise ValueError(f"Unknown budget_type '{budget_type}'. Expected 'redemption' or 'face_value'.")
        if optimization_mode not in ["baseline", "risk_adjusted", "conservative_lcb"]:
            raise ValueError(
                f"Unknown optimization_mode '{optimization_mode}'. Expected 'baseline', 'risk_adjusted', or 'conservative_lcb'."
            )
        if risk_aversion < 0.0:
            raise ValueError(f"risk_aversion must be non-negative, got {risk_aversion}")
        if not (0.50 < confidence_level < 1.0):
            raise ValueError(f"confidence_level must be in (0.50, 1.0), got {confidence_level}")
        if fairness_mode not in ["min_allocation_rate", "parity_ratio_floor"]:
            raise ValueError(
                f"Unknown fairness_mode '{fairness_mode}'. Expected 'min_allocation_rate' or 'parity_ratio_floor'."
            )

        self.budget = float(budget)
        self.budget_type = budget_type
        self.optimization_mode = optimization_mode
        self.risk_aversion = float(risk_aversion)
        self.confidence_level = float(confidence_level)
        self.fairness_enabled = bool(fairness_enabled)
        self.fairness_mode = fairness_mode
        self.min_segment_rate = min_segment_rate if min_segment_rate is not None else {}
        self.parity_ratio_floor = float(parity_ratio_floor)
        self.discount_tiers = discount_tiers if discount_tiers is not None else [0, 10, 20]
        self.segment_col = segment_col
        self.profit_calculator = ProfitCalculator()

        # Compute z critical value for LCB mode
        alpha = 1.0 - confidence_level
        self.z_crit = 1.95996 if np.isclose(confidence_level, 0.95) else 1.64485

    def compute_customer_economics(
        self,
        df_customers: pd.DataFrame,
        df_preds: pd.DataFrame,
        uncertainty_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, np.ndarray]:
        """Compute customer-level economics, unit profit, and uncertainty metrics."""
        n = len(df_customers)
        prices = df_customers["average_previous_order_value"].to_numpy(dtype=float)
        segments = df_customers[self.segment_col].to_numpy(dtype=str) if self.segment_col in df_customers.columns else np.full(n, "regular")
        margins = self.profit_calculator.estimate_gross_margin_rate(prices, segments)

        # Baseline and counterfactual purchase probabilities
        p0 = np.clip(df_preds["pred_prob_0"].to_numpy(dtype=float), 0.0, 1.0)
        p10 = np.clip(df_preds["pred_prob_10"].to_numpy(dtype=float), 0.0, 1.0)
        p20 = np.clip(df_preds["pred_prob_20"].to_numpy(dtype=float), 0.0, 1.0)

        # DGP Basket Expansion
        order_val_0 = prices * 1.00
        order_val_10 = prices * 1.03
        order_val_20 = prices * 1.06

        # Unit Margin Contribution per conversion: OrderValue * MarginRate - Discount
        unit_prof_0 = order_val_0 * margins - 0.0
        unit_prof_10 = order_val_10 * margins - 10.0
        unit_prof_20 = order_val_20 * margins - 20.0

        # Nominal Expected Net Profit: P(purchase) * unit_profit
        net_prof_0 = p0 * unit_prof_0
        net_prof_10 = p10 * unit_prof_10
        net_prof_20 = p20 * unit_prof_20

        # Expected Promotional Cost
        if self.budget_type == "redemption":
            cost_0 = np.zeros(n, dtype=float)
            cost_10 = 10.0 * p10
            cost_20 = 20.0 * p20
        else:  # face_value
            cost_0 = np.zeros(n, dtype=float)
            cost_10 = np.full(n, 10.0, dtype=float)
            cost_20 = np.full(n, 20.0, dtype=float)

        # Treatment Effect Lifts
        tau_10 = p10 - p0
        tau_20 = p20 - p0

        # Extract or calibrate standard errors and reliability
        if uncertainty_df is not None:
            # Pivot or filter uncertainty records by treatment
            u10 = uncertainty_df[uncertainty_df["treatment"] == 10]
            u20 = uncertainty_df[uncertainty_df["treatment"] == 20]

            # Merge by customer_id if available to preserve exact ordering
            if "customer_id" in df_customers.columns and "customer_id" in uncertainty_df.columns:
                m10 = df_customers[["customer_id"]].merge(u10, on="customer_id", how="left")
                m20 = df_customers[["customer_id"]].merge(u20, on="customer_id", how="left")
                se_10 = m10["standard_error"].fillna(0.0335).to_numpy(dtype=float)
                se_20 = m20["standard_error"].fillna(0.0485).to_numpy(dtype=float)
                rel_10 = m10["reliability_status"].fillna("reliable").to_numpy(dtype=object)
                rel_20 = m20["reliability_status"].fillna("reliable").to_numpy(dtype=object)
            else:
                se_10 = u10["standard_error"].to_numpy(dtype=float)[:n] if len(u10) >= n else np.full(n, 0.0335)
                se_20 = u20["standard_error"].to_numpy(dtype=float)[:n] if len(u20) >= n else np.full(n, 0.0485)
                rel_10 = u10["reliability_status"].to_numpy(dtype=object)[:n] if len(u10) >= n else np.full(n, "reliable")
                rel_20 = u20["reliability_status"].to_numpy(dtype=object)[:n] if len(u20) >= n else np.full(n, "reliable")
        else:
            # Default calibrated standard error from CausalForestDML
            se_10 = np.full(n, 0.0335, dtype=float)
            se_20 = np.full(n, 0.0485, dtype=float)
            rel_10 = np.full(n, "reliable", dtype=object)
            rel_20 = np.full(n, "reliable", dtype=object)

        # Standard error in incremental monetary profit: SE(Delta Profit) = SE(tau) * unit_profit
        # When customer converts, unit_profit is the margin contribution
        se_profit_0 = np.zeros(n, dtype=float)
        se_profit_10 = se_10 * np.maximum(unit_prof_10, 0.0)
        se_profit_20 = se_20 * np.maximum(unit_prof_20, 0.0)

        # Conservative LCB Probabilities and Profits
        p10_lcb = np.clip(p0 + tau_10 - self.z_crit * se_10, 0.0, 1.0)
        p20_lcb = np.clip(p0 + tau_20 - self.z_crit * se_20, 0.0, 1.0)

        net_prof_10_lcb = p10_lcb * unit_prof_10
        net_prof_20_lcb = p20_lcb * unit_prof_20

        return {
            "n": n,
            "prices": prices,
            "segments": segments,
            "p0": p0,
            "p10": p10,
            "p20": p20,
            "tau_10": tau_10,
            "tau_20": tau_20,
            "cost_0": cost_0,
            "cost_10": cost_10,
            "cost_20": cost_20,
            "net_prof_0": net_prof_0,
            "net_prof_10": net_prof_10,
            "net_prof_20": net_prof_20,
            "net_prof_10_lcb": net_prof_10_lcb,
            "net_prof_20_lcb": net_prof_20_lcb,
            "se_10": se_10,
            "se_20": se_20,
            "se_profit_0": se_profit_0,
            "se_profit_10": se_profit_10,
            "se_profit_20": se_profit_20,
            "rel_10": rel_10,
            "rel_20": rel_20,
        }

    def build_optimization_problem(
        self,
        econ: Dict[str, Any],
    ) -> Tuple[np.ndarray, LinearConstraint, Bounds, np.ndarray]:
        """Construct the objective vector, linear constraints, bounds, and integrality."""
        n = econ["n"]
        k = len(self.discount_tiers)  # 3 tiers: 0, 10, 20

        # ---------------------------------------------------------------------
        # 1. Objective Vector (SciPy minimizes: c = -values)
        # ---------------------------------------------------------------------
        if self.optimization_mode == "baseline":
            # Nominal expected profit
            v0 = econ["net_prof_0"]
            v10 = econ["net_prof_10"]
            v20 = econ["net_prof_20"]
        elif self.optimization_mode == "conservative_lcb":
            # Conservative Lower-Confidence-Bound profit
            v0 = econ["net_prof_0"]
            v10 = econ["net_prof_10_lcb"]
            v20 = econ["net_prof_20_lcb"]
        elif self.optimization_mode == "risk_adjusted":
            # Mean-Variance Risk-Adjusted: E[Profit] - lambda * SE(Delta Profit)
            v0 = econ["net_prof_0"]
            v10 = econ["net_prof_10"] - self.risk_aversion * econ["se_profit_10"]
            v20 = econ["net_prof_20"] - self.risk_aversion * econ["se_profit_20"]
        else:
            raise ValueError(f"Unknown optimization mode: {self.optimization_mode}")

        obj_values = np.empty(k * n, dtype=float)
        obj_values[0::3] = v0
        obj_values[1::3] = v10
        obj_values[2::3] = v20
        c = -obj_values  # Minimization of negative value

        # ---------------------------------------------------------------------
        # 2. Exact Assignment Constraint: sum_d z_{i, d} = 1 for each customer
        # ---------------------------------------------------------------------
        rows_eq = np.repeat(np.arange(n), k)
        cols_eq = np.arange(k * n)
        data_eq = np.ones(k * n, dtype=float)
        a_eq = sp.csc_matrix((data_eq, (rows_eq, cols_eq)), shape=(n, k * n))
        lb_list = [np.ones(n, dtype=float)]
        ub_list = [np.ones(n, dtype=float)]
        matrix_blocks = [a_eq]

        # ---------------------------------------------------------------------
        # 3. Budget Constraint: sum_i sum_{d in {10, 20}} Cost_{i, d} * z_{i, d} <= Budget
        # ---------------------------------------------------------------------
        costs = np.empty(k * n, dtype=float)
        costs[0::3] = econ["cost_0"]
        costs[1::3] = econ["cost_10"]
        costs[2::3] = econ["cost_20"]

        a_budget = sp.csc_matrix(costs.reshape(1, -1))
        matrix_blocks.append(a_budget)
        lb_list.append(np.array([-np.inf]))
        ub_list.append(np.array([self.budget]))

        # ---------------------------------------------------------------------
        # 4. Optional Fairness Constraints
        # ---------------------------------------------------------------------
        if self.fairness_enabled:
            segments = econ["segments"]
            unique_segments = np.unique(segments)

            if self.fairness_mode == "min_allocation_rate":
                # Enforce minimum allocation floor per segment: sum_{i in s} (z_{i, 10} + z_{i, 20}) >= floor_s
                fair_rows, fair_cols, fair_data = [], [], []
                fair_lb, fair_ub = [], []
                r_idx = 0

                for seg in unique_segments:
                    seg_mask = (segments == seg)
                    n_seg = np.sum(seg_mask)
                    # Configurable rate or default floor
                    rate_floor = self.min_segment_rate.get(seg, 0.15)
                    floor_count = int(np.ceil(n_seg * rate_floor))

                    seg_cust_indices = np.where(seg_mask)[0]
                    for ci in seg_cust_indices:
                        # Non-zero on tier 10 and tier 20
                        fair_rows.extend([r_idx, r_idx])
                        fair_cols.extend([ci * 3 + 1, ci * 3 + 2])
                        fair_data.extend([1.0, 1.0])

                    fair_lb.append(float(floor_count))
                    fair_ub.append(np.inf)
                    r_idx += 1

                if r_idx > 0:
                    a_fair = sp.csc_matrix(
                        (fair_data, (fair_rows, fair_cols)), shape=(r_idx, k * n)
                    )
                    matrix_blocks.append(a_fair)
                    lb_list.append(np.array(fair_lb, dtype=float))
                    ub_list.append(np.array(fair_ub, dtype=float))

            elif self.fairness_mode == "parity_ratio_floor":
                # Relative Demographic Parity: (K_s / N_s) >= parity_ratio_floor * (K_all / N)
                # Linear: sum_{i in s} (z_{10} + z_{20}) - (parity_ratio_floor * N_s / N) * sum_i (z_{10} + z_{20}) >= 0
                fair_rows, fair_cols, fair_data = [], [], []
                fair_lb, fair_ub = [], []
                r_idx = 0

                for seg in unique_segments:
                    seg_mask = (segments == seg)
                    n_seg = np.sum(seg_mask)
                    mult = self.parity_ratio_floor * (n_seg / n)

                    for ci in range(n):
                        in_seg = seg_mask[ci]
                        # For tier 10 and tier 20
                        coeff = (1.0 - mult) if in_seg else (-mult)
                        if coeff != 0.0:
                            fair_rows.extend([r_idx, r_idx])
                            fair_cols.extend([ci * 3 + 1, ci * 3 + 2])
                            fair_data.extend([coeff, coeff])

                    fair_lb.append(0.0)
                    fair_ub.append(np.inf)
                    r_idx += 1

                if r_idx > 0:
                    a_fair = sp.csc_matrix(
                        (fair_data, (fair_rows, fair_cols)), shape=(r_idx, k * n)
                    )
                    matrix_blocks.append(a_fair)
                    lb_list.append(np.array(fair_lb, dtype=float))
                    ub_list.append(np.array(fair_ub, dtype=float))

        # Stack constraint blocks
        a_stacked = sp.vstack(matrix_blocks, format="csc")
        lb_stacked = np.concatenate(lb_list)
        ub_stacked = np.concatenate(ub_list)

        constraints = LinearConstraint(a_stacked, lb=lb_stacked, ub=ub_stacked)
        bounds = Bounds(0, 1)
        integrality = np.ones(k * n)  # Strict binary variables

        return c, constraints, bounds, integrality

    def compute_fairness_diagnostics(
        self,
        econ: Dict[str, Any],
        allocated_tiers: np.ndarray,
    ) -> Dict[str, SegmentFairnessDiagnostic]:
        """Compute post-allocation fairness, uplift, and economic metrics by customer segment."""
        segments = econ["segments"]
        unique_segments = np.unique(segments)
        n_total = len(allocated_tiers)
        total_discounted = np.sum(allocated_tiers > 0)
        overall_rate = (total_discounted / n_total * 100.0) if n_total > 0 else 0.0

        diagnostics = {}
        for seg in sorted(unique_segments):
            mask = (segments == seg)
            n_seg = int(np.sum(mask))
            if n_seg == 0:
                continue

            seg_tiers = allocated_tiers[mask]
            seg_alloc_count = int(np.sum(seg_tiers > 0))
            seg_rate = (seg_alloc_count / n_seg * 100.0)
            avg_discount = float(np.mean(seg_tiers))

            # Incremental conversions and incremental profit under selected policy
            tau_10_seg = econ["tau_10"][mask]
            tau_20_seg = econ["tau_20"][mask]
            inc_conv = np.zeros(n_seg, dtype=float)
            inc_conv[seg_tiers == 10] = tau_10_seg[seg_tiers == 10]
            inc_conv[seg_tiers == 20] = tau_20_seg[seg_tiers == 20]
            mean_inc_conv = float(np.mean(inc_conv) * 100.0)

            inc_prof = np.zeros(n_seg, dtype=float)
            inc_prof[seg_tiers == 10] = econ["net_prof_10"][mask][seg_tiers == 10] - econ["net_prof_0"][mask][seg_tiers == 10]
            inc_prof[seg_tiers == 20] = econ["net_prof_20"][mask][seg_tiers == 20] - econ["net_prof_0"][mask][seg_tiers == 20]
            mean_inc_prof = float(np.mean(inc_prof))

            parity_ratio = (seg_rate / overall_rate) if overall_rate > 0 else 1.0

            diagnostics[str(seg)] = SegmentFairnessDiagnostic(
                segment_name=str(seg),
                sample_size=n_seg,
                allocation_count=seg_alloc_count,
                allocation_rate_pct=seg_rate,
                average_discount_amount=avg_discount,
                mean_incremental_conversion_pct=mean_inc_conv,
                mean_incremental_profit=mean_inc_prof,
                parity_ratio_vs_overall=parity_ratio,
            )

        return diagnostics

    def optimize(
        self,
        df_customers: pd.DataFrame,
        df_preds: pd.DataFrame,
        uncertainty_df: Optional[pd.DataFrame] = None,
    ) -> AdvancedOptimizationResult:
        """Run complete uncertainty-aware and fairness-constrained MILP optimization."""
        t0 = time.time()
        n = len(df_customers)

        # 1. Compute customer economics
        econ = self.compute_customer_economics(df_customers, df_preds, uncertainty_df)

        # Edge case: Budget is 0.0 -> assign everyone $0 control
        if np.isclose(self.budget, 0.0):
            alloc_tiers = np.zeros(n, dtype=int)
            tier_counts = {0: n, 10: 0, 20: 0}
            fairness = self.compute_fairness_diagnostics(econ, alloc_tiers)
            runtime = time.time() - t0

            df_alloc = pd.DataFrame({
                "customer_id": df_customers["customer_id"] if "customer_id" in df_customers.columns else [f"C_{i}" for i in range(n)],
                "allocated_discount": alloc_tiers,
                "expected_discount_cost": np.zeros(n),
                "nominal_expected_profit": econ["net_prof_0"],
                "conservative_lcb_profit": econ["net_prof_0"],
                "reliability_status": np.full(n, "reliable"),
            })

            return AdvancedOptimizationResult(
                success=True,
                status_code=0,
                status_message="Optimal solution found (Zero budget: all control).",
                optimization_mode=self.optimization_mode,
                risk_aversion=self.risk_aversion,
                budget=self.budget,
                budget_type=self.budget_type,
                total_expected_cost=0.0,
                budget_utilization_pct=0.0,
                nominal_expected_profit=float(np.sum(econ["net_prof_0"])),
                conservative_lcb_profit=float(np.sum(econ["net_prof_0"])),
                risk_adjusted_objective_value=float(np.sum(econ["net_prof_0"])),
                tier_counts=tier_counts,
                selected_customers_mean_uncertainty=0.0,
                selected_customers_reliability_distribution={"reliable": n},
                fairness_metrics=fairness,
                allocation_df=df_alloc,
                runtime_seconds=runtime,
            )

        # 2. Build optimization problem
        c, constraints, bounds, integrality = self.build_optimization_problem(econ)

        # 3. Solve MILP via SciPy HiGHS
        sol = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
        runtime = time.time() - t0

        if not sol.success or sol.status != 0:
            # Handle infeasibility gracefully
            tier_counts = {0: n, 10: 0, 20: 0}
            msg = f"MILP Solver Failed. Status: {sol.status} ({sol.message})."
            if sol.status == 2:
                msg += " Problem is mathematically infeasible under current budget and fairness constraints."

            return AdvancedOptimizationResult(
                success=False,
                status_code=int(sol.status),
                status_message=msg,
                optimization_mode=self.optimization_mode,
                risk_aversion=self.risk_aversion,
                budget=self.budget,
                budget_type=self.budget_type,
                total_expected_cost=0.0,
                budget_utilization_pct=0.0,
                nominal_expected_profit=0.0,
                conservative_lcb_profit=0.0,
                risk_adjusted_objective_value=0.0,
                tier_counts=tier_counts,
                selected_customers_mean_uncertainty=0.0,
                selected_customers_reliability_distribution={},
                fairness_metrics={},
                allocation_df=pd.DataFrame(),
                runtime_seconds=runtime,
            )

        # 4. Extract solution decisions
        z_solution = sol.x.reshape(n, 3)
        tier_indices = np.argmax(z_solution, axis=1)
        tier_map = np.array(self.discount_tiers)
        allocated_tiers = tier_map[tier_indices]

        unique_t, counts_t = np.unique(allocated_tiers, return_counts=True)
        tier_counts = {int(k): int(v) for k, v in zip(unique_t, counts_t)}
        for t in self.discount_tiers:
            if t not in tier_counts:
                tier_counts[t] = 0

        # 5. Extract KPI values
        cost_arr = np.empty(n, dtype=float)
        cost_arr[allocated_tiers == 0] = econ["cost_0"][allocated_tiers == 0]
        cost_arr[allocated_tiers == 10] = econ["cost_10"][allocated_tiers == 10]
        cost_arr[allocated_tiers == 20] = econ["cost_20"][allocated_tiers == 20]
        total_cost = float(np.sum(cost_arr))
        budget_util = (total_cost / self.budget * 100.0) if self.budget > 0 else 0.0

        nom_prof_arr = np.empty(n, dtype=float)
        nom_prof_arr[allocated_tiers == 0] = econ["net_prof_0"][allocated_tiers == 0]
        nom_prof_arr[allocated_tiers == 10] = econ["net_prof_10"][allocated_tiers == 10]
        nom_prof_arr[allocated_tiers == 20] = econ["net_prof_20"][allocated_tiers == 20]
        nominal_profit = float(np.sum(nom_prof_arr))

        lcb_prof_arr = np.empty(n, dtype=float)
        lcb_prof_arr[allocated_tiers == 0] = econ["net_prof_0"][allocated_tiers == 0]
        lcb_prof_arr[allocated_tiers == 10] = econ["net_prof_10_lcb"][allocated_tiers == 10]
        lcb_prof_arr[allocated_tiers == 20] = econ["net_prof_20_lcb"][allocated_tiers == 20]
        conservative_profit = float(np.sum(lcb_prof_arr))

        risk_adj_obj_val = float(-sol.fun)

        # Selected customers uncertainty metrics
        positive_discount_mask = (allocated_tiers > 0)
        if np.any(positive_discount_mask):
            selected_se = np.where(
                allocated_tiers == 10, econ["se_10"],
                np.where(allocated_tiers == 20, econ["se_20"], 0.0)
            )[positive_discount_mask]
            mean_uncertainty = float(np.mean(selected_se))

            selected_rel = np.where(
                allocated_tiers == 10, econ["rel_10"],
                np.where(allocated_tiers == 20, econ["rel_20"], "reliable")
            )[positive_discount_mask]
            rel_u, rel_c = np.unique(selected_rel, return_counts=True)
            reliability_dist = {str(k): int(v) for k, v in zip(rel_u, rel_c)}
        else:
            mean_uncertainty = 0.0
            reliability_dist = {}

        # 6. Post-allocation fairness diagnostics
        fairness_diag = self.compute_fairness_diagnostics(econ, allocated_tiers)

        # 7. Customer allocation DataFrame
        df_alloc = pd.DataFrame({
            "customer_id": df_customers["customer_id"] if "customer_id" in df_customers.columns else [f"C_{i}" for i in range(n)],
            "customer_segment": econ["segments"],
            "allocated_discount": allocated_tiers,
            "expected_discount_cost": np.round(cost_arr, 2),
            "nominal_expected_profit": np.round(nom_prof_arr, 2),
            "conservative_lcb_profit": np.round(lcb_prof_arr, 2),
            "assigned_uncertainty_se": np.round(
                np.where(allocated_tiers == 10, econ["se_10"], np.where(allocated_tiers == 20, econ["se_20"], 0.0)), 4
            ),
            "reliability_status": np.where(
                allocated_tiers == 10, econ["rel_10"], np.where(allocated_tiers == 20, econ["rel_20"], "organic_control")
            ),
        })

        return AdvancedOptimizationResult(
            success=True,
            status_code=0,
            status_message="Optimization terminated successfully (HiGHS Optimal).",
            optimization_mode=self.optimization_mode,
            risk_aversion=self.risk_aversion,
            budget=self.budget,
            budget_type=self.budget_type,
            total_expected_cost=total_cost,
            budget_utilization_pct=budget_util,
            nominal_expected_profit=nominal_profit,
            conservative_lcb_profit=conservative_profit,
            risk_adjusted_objective_value=risk_adj_obj_val,
            tier_counts=tier_counts,
            selected_customers_mean_uncertainty=mean_uncertainty,
            selected_customers_reliability_distribution=reliability_dist,
            fairness_metrics=fairness_diag,
            allocation_df=df_alloc,
            runtime_seconds=runtime,
        )

