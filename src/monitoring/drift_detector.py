"""Automated Data Drift and Distributional Shift Detection Engine for EconoCausal.

This module detects whether incoming customer feature distributions have shifted
compared to the baseline reference dataset used to develop the causal pricing models.

STATISTICAL TESTS & METRICS:
1. Numerical Features:
   - Two-Sample Kolmogorov-Smirnov (KS) Test (scipy.stats.ks_2samp):
     Non-parametric test comparing the empirical cumulative distributions:
     D = sup_x |F_ref(x) - F_cur(x)|.
     Flags statistical significance when p-value < alpha (default: 0.05).
   - Population Stability Index (PSI):
     Quantifies the degree of divergence in quantile bin proportions:
     PSI = sum_{k=1}^K (P_k - Q_k) * ln(P_k / Q_k)
     Thresholds:
       * PSI < 0.10: No significant shift (Stable)
       * 0.10 <= PSI < 0.25: Moderate shift (Monitor)
       * PSI >= 0.25: Significant / severe shift (Action required)
   - Wasserstein Distance (Earth Mover's Distance):
     Measures the minimum work required to transform one distribution into the other.

2. Categorical Features:
   - Chi-Square Goodness-of-Fit Test (scipy.stats.chisquare):
     Tests whether observed categorical frequencies in the current batch match the
     reference distribution's multinomial proportions.
   - Total Variation Distance (TVD):
     TVD = 0.5 * sum_c |P(c) - Q(c)|.
   - Category Emergence / Disappearance:
     Detects newly appearing categories (unseen in training) and disappearing categories.

3. Model-Input Validation (Schema Integrity vs Statistical Drift):
   - Keeps schema errors (missing columns, type mismatches, out-of-range bounds)
     strictly distinct from statistical distributional drift.

4. Causal & Methodological Distinctions:
   - Data Drift (Covariate Shift): Shift in P(X).
   - Concept Drift: Shift in the conditional outcome mechanism P(Y | X, T).
   - Treatment Effect Drift: Shift in the CATE function tau(X) = E[Y(1) - Y(0) | X].
   Input drift in P(X) does NOT automatically imply that the causal treatment effect model
   has become inaccurate. Retraining is not triggered automatically; actionable alerts
   and diagnostics are generated instead.
"""

from __future__ import annotations

import os
import enum
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

NUMERICAL_FEATURES: List[str] = [
    "age",
    "loyalty_score",
    "previous_purchases",
    "product_price",
    "gross_margin_rate",
    "average_previous_order_value",
]

CATEGORICAL_FEATURES: List[str] = [
    "customer_segment",
    "campaign_channel",
]

PREDICTION_FEATURES: List[str] = [
    "predicted_purchase_probability_0",
    "predicted_purchase_probability_10",
    "predicted_purchase_probability_20",
]

REQUIRED_MODEL_FEATURES: List[str] = [
    "age",
    "loyalty_score",
    "previous_purchases",
    "customer_segment",
]

NUMERICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "age": (18.0, 105.0),
    "loyalty_score": (0.0, 100.0),
    "previous_purchases": (0.0, 1000.0),
    "product_price": (0.0, 5000.0),
    "gross_margin_rate": (0.0, 1.0),
    "average_previous_order_value": (0.0, 50000.0),
}


class DriftSeverity(str, enum.Enum):
    """Severity classification for an individual feature's drift."""
    NONE = "none"
    MODERATE = "moderate"
    HIGH = "high"


class OverallDriftStatus(str, enum.Enum):
    """Overall multi-feature drift assessment status."""
    NO_DRIFT = "No significant drift detected"
    POTENTIAL_DRIFT = "Potential drift detected"
    SEVERE_DRIFT = "Severe or widespread drift detected"


@dataclass
class SchemaValidationResult:
    """Results of pre-drift model input and schema integrity verification."""
    is_valid: bool
    missing_required_features: List[str] = field(default_factory=list)
    unexpected_columns: List[str] = field(default_factory=list)
    type_errors: List[str] = field(default_factory=list)
    missing_value_rates: Dict[str, float] = field(default_factory=dict)
    out_of_range_features: Dict[str, List[str]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class FeatureDriftResult:
    """Detailed statistical drift test results for a single feature."""
    feature_name: str
    feature_type: str  # 'numeric' | 'categorical'
    drift_detected: bool
    severity: DriftSeverity
    test_name: str  # 'kolmogorov_smirnov' | 'chi_square'
    test_statistic: float
    p_value: float
    psi: Optional[float] = None
    wasserstein_distance: Optional[float] = None
    tvd: Optional[float] = None
    reference_summary: Dict[str, Any] = field(default_factory=dict)
    current_summary: Dict[str, Any] = field(default_factory=dict)
    new_categories: List[str] = field(default_factory=list)
    missing_categories: List[str] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)
    recommended_action: str = ""


@dataclass
class DriftReport:
    """Comprehensive data drift and schema integrity report."""
    overall_status: OverallDriftStatus
    drift_detected: bool
    total_features_evaluated: int
    drifted_features_count: int
    severe_drift_count: int
    reference_sample_size: int
    current_sample_size: int
    schema_validation: SchemaValidationResult
    features: List[FeatureDriftResult]
    summary_message: str
    recommended_next_action: str
    prediction_drift: List[FeatureDriftResult] = field(default_factory=list)
    methodological_notice: str = (
        "Methodological Notice: Covariate drift (P(X)) measures shift in customer features. "
        "It does NOT automatically imply concept drift (P(Y | X, T)) or invalidity of causal treatment effects. "
        "Do not automatically retrain models without empirical policy verification."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass report to serializable dictionary."""
        d = asdict(self)
        d["overall_status"] = self.overall_status.value
        for feat in d["features"]:
            feat["severity"] = feat["severity"].value if isinstance(feat["severity"], DriftSeverity) else str(feat["severity"])
        for feat in d.get("prediction_drift", []):
            feat["severity"] = feat["severity"].value if isinstance(feat["severity"], DriftSeverity) else str(feat["severity"])
        return d


class EconoCausalDriftDetector:
    """Production data drift detector for causal retail pricing features."""

    def __init__(
        self,
        reference_data: pd.DataFrame,
        ks_alpha: float = 0.05,
        psi_moderate_threshold: float = 0.10,
        psi_high_threshold: float = 0.25,
        num_psi_bins: int = 10,
        min_sample_size: int = 20,
    ) -> None:
        """Initialize the drift detector with baseline reference data and statistical thresholds.

        Args:
            reference_data: Approved baseline training/reference customer DataFrame.
            ks_alpha: Significance level for Kolmogorov-Smirnov test (default: 0.05).
            psi_moderate_threshold: Lower PSI threshold for moderate drift (default: 0.10).
            psi_high_threshold: Upper PSI threshold for severe drift (default: 0.25).
            num_psi_bins: Number of reference quantile bins for PSI computation (default: 10).
            min_sample_size: Minimum sample size required to perform statistical tests (default: 20).
        """
        if reference_data is None or len(reference_data) == 0:
            raise ValueError("Reference dataset is empty or None.")

        self.reference_data = reference_data.copy()
        self.ks_alpha = float(ks_alpha)
        self.psi_moderate_threshold = float(psi_moderate_threshold)
        self.psi_high_threshold = float(psi_high_threshold)
        self.num_psi_bins = int(num_psi_bins)
        self.min_sample_size = int(min_sample_size)

        # Precompute reference statistics for fast repeated comparisons
        self._precompute_reference_stats()

    def _precompute_reference_stats(self) -> None:
        """Precompute reference distribution summaries, quantile bin edges, and category frequencies."""
        self.ref_numeric_summaries: Dict[str, Dict[str, float]] = {}
        self.ref_psi_bins: Dict[str, np.ndarray] = {}
        self.ref_categorical_props: Dict[str, Dict[str, float]] = {}

        # 1. Numerical reference features
        for feat in NUMERICAL_FEATURES:
            if feat in self.reference_data.columns:
                vals = pd.to_numeric(self.reference_data[feat], errors="coerce").dropna().to_numpy(dtype=float)
                if len(vals) > 0:
                    self.ref_numeric_summaries[feat] = {
                        "mean": round(float(np.mean(vals)), 2),
                        "std": round(float(np.std(vals)), 2),
                        "min": round(float(np.min(vals)), 2),
                        "p25": round(float(np.percentile(vals, 25)), 2),
                        "p50": round(float(np.percentile(vals, 50)), 2),
                        "p75": round(float(np.percentile(vals, 75)), 2),
                        "max": round(float(np.max(vals)), 2),
                    }
                    # Quantile bin edges for PSI
                    quantiles = np.linspace(0, 100, self.num_psi_bins + 1)
                    bins = np.percentile(vals, quantiles)
                    bins[0] = -np.inf
                    bins[-1] = np.inf
                    self.ref_psi_bins[feat] = np.unique(bins)

        # 2. Categorical reference features
        for feat in CATEGORICAL_FEATURES:
            if feat in self.reference_data.columns:
                series = self.reference_data[feat].dropna().astype(str)
                props = series.value_counts(normalize=True).to_dict()
                self.ref_categorical_props[feat] = {k: round(float(v), 4) for k, v in props.items()}

    def validate_schema(self, current_data: pd.DataFrame) -> SchemaValidationResult:
        """Verify model-input schema integrity, column presence, types, and range bounds.

        Keeps schema and type violations strictly distinct from statistical drift.
        """
        errors: List[str] = []
        warnings: List[str] = []
        missing_required: List[str] = []
        unexpected_cols: List[str] = []
        type_errors: List[str] = []
        null_rates: Dict[str, float] = {}
        out_of_range: Dict[str, List[str]] = {}

        if current_data is None or len(current_data) == 0:
            return SchemaValidationResult(
                is_valid=False,
                errors=["Current evaluation batch is empty or None."],
            )

        cur_cols = list(current_data.columns)

        # 1. Check required features
        for req in REQUIRED_MODEL_FEATURES:
            if req not in cur_cols:
                missing_required.append(req)
                errors.append(f"Missing required model feature: '{req}'")

        # 2. Check for unexpected / extraneous columns
        known_cols = set(self.reference_data.columns).union(
            {"customer_id", "product_price", "gross_margin_rate", "campaign_channel", "discount_level", "purchase"}
        )
        for col in cur_cols:
            if col not in known_cols:
                unexpected_cols.append(col)
                warnings.append(f"Unexpected column detected: '{col}' (not present in reference schema)")

        # 3. Check null value rates
        for col in cur_cols:
            null_count = current_data[col].isnull().sum()
            rate = round(float(null_count) / len(current_data), 4)
            if rate > 0:
                null_rates[col] = rate
                if col in REQUIRED_MODEL_FEATURES:
                    errors.append(f"Missing values detected in required feature '{col}': {rate * 100:.1f}%")
                else:
                    warnings.append(f"Missing values in column '{col}': {rate * 100:.1f}%")

        # 4. Numerical range bounds checks
        for feat, (min_val, max_val) in NUMERICAL_BOUNDS.items():
            if feat in cur_cols:
                num_series = pd.to_numeric(current_data[feat], errors="coerce")
                non_num_count = num_series.isnull().sum() - current_data[feat].isnull().sum()
                if non_num_count > 0:
                    type_errors.append(f"Non-numeric values in numeric column '{feat}' ({non_num_count} rows)")
                    errors.append(f"Column '{feat}' expected float/int, found non-numeric entries.")

                valid_nums = num_series.dropna()
                below = (valid_nums < min_val).sum()
                above = (valid_nums > max_val).sum()
                if below > 0 or above > 0:
                    out_of_range[feat] = []
                    if below > 0:
                        out_of_range[feat].append(f"{below} rows < {min_val}")
                    if above > 0:
                        out_of_range[feat].append(f"{above} rows > {max_val}")
                    warnings.append(f"Out-of-range values in '{feat}': {out_of_range[feat]}")

        is_valid = len(errors) == 0
        return SchemaValidationResult(
            is_valid=is_valid,
            missing_required_features=missing_required,
            unexpected_columns=unexpected_cols,
            type_errors=type_errors,
            missing_value_rates=null_rates,
            out_of_range_features=out_of_range,
            errors=errors,
            warnings=warnings,
        )

    def calculate_psi(
        self,
        ref_vals: np.ndarray,
        cur_vals: np.ndarray,
        bins: Optional[np.ndarray] = None,
        epsilon: float = 1e-4,
    ) -> float:
        """Calculate Population Stability Index (PSI) using reference quantile bins."""
        if len(ref_vals) == 0 or len(cur_vals) == 0:
            return 0.0

        if bins is None or len(bins) < 3:
            quantiles = np.linspace(0, 100, self.num_psi_bins + 1)
            bins = np.percentile(ref_vals, quantiles)
            bins[0] = -np.inf
            bins[-1] = np.inf
            bins = np.unique(bins)

        if len(bins) < 3:
            return 0.0

        ref_counts, _ = np.histogram(ref_vals, bins=bins)
        cur_counts, _ = np.histogram(cur_vals, bins=bins)

        ref_pct = (ref_counts + epsilon) / (len(ref_vals) + epsilon * len(ref_counts))
        cur_pct = (cur_counts + epsilon) / (len(cur_vals) + epsilon * len(cur_counts))

        psi_val = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(max(0.0, psi_val))

    def evaluate_numerical_feature(
        self,
        feature_name: str,
        current_data: pd.DataFrame,
    ) -> FeatureDriftResult:
        """Run two-sample Kolmogorov-Smirnov test, PSI, and Wasserstein distance on numerical feature."""
        ref_vals = pd.to_numeric(self.reference_data[feature_name], errors="coerce").dropna().to_numpy(dtype=float)
        cur_vals = pd.to_numeric(current_data[feature_name], errors="coerce").dropna().to_numpy(dtype=float)

        # Current summary statistics
        cur_summary = {
            "mean": round(float(np.mean(cur_vals)), 2) if len(cur_vals) > 0 else 0.0,
            "std": round(float(np.std(cur_vals)), 2) if len(cur_vals) > 0 else 0.0,
            "min": round(float(np.min(cur_vals)), 2) if len(cur_vals) > 0 else 0.0,
            "p25": round(float(np.percentile(cur_vals, 25)), 2) if len(cur_vals) > 0 else 0.0,
            "p50": round(float(np.percentile(cur_vals, 50)), 2) if len(cur_vals) > 0 else 0.0,
            "p75": round(float(np.percentile(cur_vals, 75)), 2) if len(cur_vals) > 0 else 0.0,
            "max": round(float(np.max(cur_vals)), 2) if len(cur_vals) > 0 else 0.0,
        }
        ref_summary = self.ref_numeric_summaries.get(feature_name, {})

        # Handle small sample sizes or identical data
        if len(cur_vals) < 5 or len(ref_vals) < 5:
            return FeatureDriftResult(
                feature_name=feature_name,
                feature_type="numeric",
                drift_detected=False,
                severity=DriftSeverity.NONE,
                test_name="kolmogorov_smirnov",
                test_statistic=0.0,
                p_value=1.0,
                psi=0.0,
                wasserstein_distance=0.0,
                reference_summary=ref_summary,
                current_summary=cur_summary,
                thresholds={"ks_alpha": self.ks_alpha, "psi_high": self.psi_high_threshold},
                recommended_action="Sample size too small for statistical testing; collect more observations.",
            )

        # 1. Kolmogorov-Smirnov Two-Sample Test
        ks_res = stats.ks_2samp(ref_vals, cur_vals)
        ks_stat = float(ks_res.statistic)
        p_val = float(ks_res.pvalue)

        # 2. Population Stability Index
        bins = self.ref_psi_bins.get(feature_name)
        psi_val = self.calculate_psi(ref_vals, cur_vals, bins=bins)

        # 3. Wasserstein Distance
        w_dist = float(stats.wasserstein_distance(ref_vals, cur_vals))

        # Determine drift detection & severity:
        # A feature is flagged as drifted if KS p-value < alpha OR PSI >= moderate threshold
        drift_detected = (p_val < self.ks_alpha) or (psi_val >= self.psi_moderate_threshold)

        if psi_val >= self.psi_high_threshold or (p_val < 0.0001 and psi_val >= 0.15):
            severity = DriftSeverity.HIGH
            action = f"Significant distributional divergence in '{feature_name}'. Investigate acquisition shift and re-calibrate propensity models."
        elif drift_detected:
            severity = DriftSeverity.MODERATE
            action = f"Moderate shift in '{feature_name}'. Monitor downstream conversion and policy elasticity."
        else:
            severity = DriftSeverity.NONE
            action = f"Feature '{feature_name}' is stable and consistent with the reference distribution."

        return FeatureDriftResult(
            feature_name=feature_name,
            feature_type="numeric",
            drift_detected=drift_detected,
            severity=severity,
            test_name="kolmogorov_smirnov",
            test_statistic=round(ks_stat, 4),
            p_value=round(p_val, 6),
            psi=round(psi_val, 4),
            wasserstein_distance=round(w_dist, 4),
            reference_summary=ref_summary,
            current_summary=cur_summary,
            thresholds={"ks_alpha": self.ks_alpha, "psi_moderate": self.psi_moderate_threshold, "psi_high": self.psi_high_threshold},
            recommended_action=action,
        )

    def evaluate_categorical_feature(
        self,
        feature_name: str,
        current_data: pd.DataFrame,
    ) -> FeatureDriftResult:
        """Run Chi-Square test, Total Variation Distance, and category emergence checks on categorical feature."""
        ref_series = self.reference_data[feature_name].dropna().astype(str)
        cur_series = current_data[feature_name].dropna().astype(str)

        ref_counts = ref_series.value_counts()
        cur_counts = cur_series.value_counts()

        ref_cats = set(ref_counts.index)
        cur_cats = set(cur_counts.index)

        new_categories = sorted(list(cur_cats - ref_cats))
        missing_categories = sorted(list(ref_cats - cur_cats))

        ref_props = {k: round(float(v), 4) for k, v in ref_series.value_counts(normalize=True).items()}
        cur_props = {k: round(float(v), 4) for k, v in cur_series.value_counts(normalize=True).items()}

        all_unique = sorted(list(ref_cats.union(cur_cats)))

        # Total Variation Distance (TVD)
        tvd = 0.5 * sum(abs(cur_props.get(c, 0.0) - ref_props.get(c, 0.0)) for c in all_unique)

        # Chi-Square Goodness-of-Fit
        if len(cur_series) < 10:
            return FeatureDriftResult(
                feature_name=feature_name,
                feature_type="categorical",
                drift_detected=False,
                severity=DriftSeverity.NONE,
                test_name="chi_square",
                test_statistic=0.0,
                p_value=1.0,
                tvd=round(tvd, 4),
                reference_summary={"proportions": ref_props},
                current_summary={"proportions": cur_props},
                new_categories=new_categories,
                missing_categories=missing_categories,
                thresholds={"alpha": self.ks_alpha, "tvd_threshold": 0.15},
                recommended_action="Sample size too small for Chi-Square test; collect more customer records.",
            )

        cur_freq_aligned = np.array([cur_counts.get(c, 0) for c in all_unique], dtype=float)
        # Expected frequencies based on reference probabilities
        ref_probs_aligned = np.array([ref_props.get(c, 1e-4) for c in all_unique], dtype=float)
        ref_probs_aligned /= ref_probs_aligned.sum()
        expected_freq = ref_probs_aligned * len(cur_series)

        try:
            chi2_res = stats.chisquare(f_obs=cur_freq_aligned, f_exp=expected_freq)
            chi2_stat = float(chi2_res.statistic)
            p_val = float(chi2_res.pvalue)
        except Exception:
            chi2_stat = 0.0
            p_val = 1.0

        # Drift conditions: Chi2 p-val < alpha OR TVD >= 0.15 OR new categories appeared
        drift_detected = (p_val < self.ks_alpha) or (tvd >= 0.15) or (len(new_categories) > 0)

        if len(new_categories) > 0 or p_val < 0.001 or tvd >= 0.25:
            severity = DriftSeverity.HIGH
            action = f"Severe distribution shift in '{feature_name}' (TVD={tvd:.2f}, new categories={new_categories}). Review feature taxonomy."
        elif drift_detected:
            severity = DriftSeverity.MODERATE
            action = f"Moderate segment frequency shift in '{feature_name}'. Monitor category conversion dynamics."
        else:
            severity = DriftSeverity.NONE
            action = f"Categorical feature '{feature_name}' frequencies align closely with the baseline reference."

        return FeatureDriftResult(
            feature_name=feature_name,
            feature_type="categorical",
            drift_detected=drift_detected,
            severity=severity,
            test_name="chi_square",
            test_statistic=round(chi2_stat, 4),
            p_value=round(p_val, 6),
            tvd=round(tvd, 4),
            reference_summary={"proportions": ref_props},
            current_summary={"proportions": cur_props},
            new_categories=new_categories,
            missing_categories=missing_categories,
            thresholds={"alpha": self.ks_alpha, "tvd_threshold": 0.15},
            recommended_action=action,
        )

    def detect_drift(self, current_data: pd.DataFrame) -> DriftReport:
        """Execute full data drift detection workflow across schema, numerical, and categorical features.

        Args:
            current_data: Current evaluation DataFrame.

        Returns:
            Comprehensive DriftReport.
        """
        # 1. Model-input & schema integrity check
        schema_res = self.validate_schema(current_data)

        if not schema_res.is_valid:
            return DriftReport(
                overall_status=OverallDriftStatus.SEVERE_DRIFT,
                drift_detected=True,
                total_features_evaluated=0,
                drifted_features_count=0,
                severe_drift_count=0,
                reference_sample_size=len(self.reference_data),
                current_sample_size=len(current_data) if current_data is not None else 0,
                schema_validation=schema_res,
                features=[],
                summary_message=f"Model input schema validation failed: {schema_res.errors}",
                recommended_next_action="Resolve schema errors (missing columns or invalid types) before performing statistical drift tests.",
            )

        # 2. Evaluate all features
        feature_results: List[FeatureDriftResult] = []
        drifted_count = 0
        severe_count = 0

        # Numerical features
        for feat in NUMERICAL_FEATURES:
            if feat in current_data.columns and feat in self.reference_data.columns:
                res = self.evaluate_numerical_feature(feat, current_data)
                feature_results.append(res)
                if res.drift_detected:
                    drifted_count += 1
                if res.severity == DriftSeverity.HIGH:
                    severe_count += 1

        # Categorical features
        for feat in CATEGORICAL_FEATURES:
            if feat in current_data.columns and feat in self.reference_data.columns:
                res = self.evaluate_categorical_feature(feat, current_data)
                feature_results.append(res)
                if res.drift_detected:
                    drifted_count += 1
                if res.severity == DriftSeverity.HIGH:
                    severe_count += 1

        # 3. Formulate Overall Drift Status
        total_feats = len(feature_results)
        if severe_count >= 2 or (total_feats > 0 and drifted_count / total_feats >= 0.5):
            overall_status = OverallDriftStatus.SEVERE_DRIFT
            summary = (
                f"Severe or widespread drift detected across {drifted_count} of {total_feats} evaluated features "
                f"({severe_count} features with high severity). Significant covariate shift in customer population."
            )
            next_action = (
                "Review marketing channel acquisition shifts. Inspect whether unconfoundedness and positivity overlap "
                "still hold in the drifted feature space. Re-estimate propensity scores and re-run policy optimization."
            )
        elif drifted_count > 0:
            overall_status = OverallDriftStatus.POTENTIAL_DRIFT
            summary = (
                f"Potential drift detected in {drifted_count} of {total_feats} features. "
                f"Moderate distributional changes observed."
            )
            next_action = (
                "Monitor incoming campaign cohorts over the next cycle. Verify that customer conversion elasticity "
                "remains consistent with CATE predictions before considering model adjustment."
            )
        else:
            overall_status = OverallDriftStatus.NO_DRIFT
            summary = f"No significant drift detected across all {total_feats} evaluated features. Distribution is stable."
            next_action = "Continue standard promotional policy deployment. Normal monitoring interval applies."

        # Prediction drift check (if predictions exist in columns)
        prediction_results = self.evaluate_prediction_drift(current_data)

        return DriftReport(
            overall_status=overall_status,
            drift_detected=drifted_count > 0,
            total_features_evaluated=total_feats,
            drifted_features_count=drifted_count,
            severe_drift_count=severe_count,
            reference_sample_size=len(self.reference_data),
            current_sample_size=len(current_data),
            schema_validation=schema_res,
            features=feature_results,
            prediction_drift=prediction_results,
            summary_message=summary,
            recommended_next_action=next_action,
        )

    def evaluate_prediction_drift(
        self,
        current_data: pd.DataFrame,
        reference_data: Optional[pd.DataFrame] = None,
    ) -> List[FeatureDriftResult]:
        """Run statistical KS and PSI tests on counterfactual purchase probabilities (P0, P10, P20)."""
        ref_df = reference_data if reference_data is not None else self.reference_data
        pred_results: List[FeatureDriftResult] = []

        if current_data is None or ref_df is None:
            return pred_results

        for feat in PREDICTION_FEATURES:
            if feat in current_data.columns and feat in ref_df.columns:
                ref_vals = pd.to_numeric(ref_df[feat], errors="coerce").dropna().to_numpy(dtype=float)
                cur_vals = pd.to_numeric(current_data[feat], errors="coerce").dropna().to_numpy(dtype=float)

                if len(cur_vals) < 5 or len(ref_vals) < 5:
                    continue

                ks_res = stats.ks_2samp(ref_vals, cur_vals)
                ks_stat = float(ks_res.statistic)
                p_val = float(ks_res.pvalue)
                psi_val = self.calculate_psi(ref_vals, cur_vals)
                w_dist = float(stats.wasserstein_distance(ref_vals, cur_vals))

                drift_detected = (p_val < self.ks_alpha) or (psi_val >= self.psi_moderate_threshold)
                if psi_val >= self.psi_high_threshold or (p_val < 0.0001 and psi_val >= 0.15):
                    severity = DriftSeverity.HIGH
                    action = f"Significant prediction drift in '{feat}'. Conversion behavior shifted; review pricing elasticity."
                elif drift_detected:
                    severity = DriftSeverity.MODERATE
                    action = f"Moderate prediction shift in '{feat}'. Monitor conversion calibration."
                else:
                    severity = DriftSeverity.NONE
                    action = f"Prediction distribution for '{feat}' is stable relative to baseline reference."

                pred_results.append(
                    FeatureDriftResult(
                        feature_name=feat,
                        feature_type="prediction",
                        drift_detected=drift_detected,
                        severity=severity,
                        test_name="kolmogorov_smirnov",
                        test_statistic=round(ks_stat, 4),
                        p_value=round(p_val, 6),
                        psi=round(psi_val, 4),
                        wasserstein_distance=round(w_dist, 4),
                        reference_summary={"mean": round(float(np.mean(ref_vals)), 4), "std": round(float(np.std(ref_vals)), 4)},
                        current_summary={"mean": round(float(np.mean(cur_vals)), 4), "std": round(float(np.std(cur_vals)), 4)},
                        thresholds={"ks_alpha": self.ks_alpha, "psi_high": self.psi_high_threshold},
                        recommended_action=action,
                    )
                )

        return pred_results


