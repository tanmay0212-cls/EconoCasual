"""Automated data drift detection service using statistical divergence metrics."""

import os
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy import stats
from fastapi import HTTPException, status

from src.api.config import settings
from src.api.schemas.drift import (
    DriftCheckRequest,
    DriftCheckResponse,
    FeatureDriftDetail,
)

logger = logging.getLogger("econocausal.drift_service")

NUMERICAL_DRIFT_FEATURES = [
    "age",
    "loyalty_score",
    "previous_purchases",
    "product_price",
    "gross_margin_rate",
]

CATEGORICAL_DRIFT_FEATURES = [
    "customer_segment",
]


class DriftService:
    """Service to compute statistical drift tests between incoming batches and reference data."""

    _reference_df: Optional[pd.DataFrame] = None

    @classmethod
    def get_reference_data(cls) -> pd.DataFrame:
        """Load and cache the approved baseline reference dataset."""
        if cls._reference_df is None:
            ref_path = settings.reference_data_path
            if not os.path.exists(ref_path):
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Approved reference dataset not found at: {ref_path}",
                )
            try:
                cls._reference_df = pd.read_csv(ref_path)
                logger.info("Loaded reference drift dataset with %d rows", len(cls._reference_df))
            except Exception as exc:
                logger.error("Failed to read reference dataset: %s", exc)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Could not load baseline reference dataset: {str(exc)}",
                )
        return cls._reference_df

    @classmethod
    def calculate_psi(
        cls,
        reference: np.ndarray,
        current: np.ndarray,
        num_bins: int = 10,
        epsilon: float = 1e-4,
    ) -> float:
        """Compute Population Stability Index (PSI) using quantile binning."""
        try:
            # Generate bin edges based on reference quantiles
            quantiles = np.linspace(0, 100, num_bins + 1)
            bins = np.percentile(reference, quantiles)
            bins[0] = -np.inf
            bins[-1] = np.inf
            # Ensure strictly increasing bins
            bins = np.unique(bins)
            if len(bins) < 3:
                return 0.0

            ref_counts, _ = np.histogram(reference, bins=bins)
            cur_counts, _ = np.histogram(current, bins=bins)

            ref_pct = (ref_counts + epsilon) / (len(reference) + epsilon * len(ref_counts))
            cur_pct = (cur_counts + epsilon) / (len(current) + epsilon * len(cur_counts))

            psi_val = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
            return float(max(0.0, psi_val))
        except Exception:
            return 0.0

    @classmethod
    def check_drift(
        cls,
        request: DriftCheckRequest,
        data_service_ref_resolver,
    ) -> DriftCheckResponse:
        """Evaluate incoming batch for covariate drift against baseline distribution."""
        ref_df = cls.get_reference_data()

        # 1. Resolve current batch
        if request.records is not None and len(request.records) > 0:
            current_df = pd.DataFrame([r.model_dump() for r in request.records])
        elif request.dataset_reference:
            current_df = data_service_ref_resolver(request.dataset_reference)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'records' or an approved 'dataset_reference' must be provided.",
            )

        if len(current_df) < 5:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Incoming batch size ({len(current_df)}) is too small for statistical drift tests (minimum 5).",
            )

        feature_reports: List[FeatureDriftDetail] = []
        drifted_count = 0

        # 2. Evaluate Numerical Features
        for feat in NUMERICAL_DRIFT_FEATURES:
            if feat not in current_df.columns or feat not in ref_df.columns:
                continue

            ref_vals = pd.to_numeric(ref_df[feat], errors="coerce").dropna().to_numpy(dtype=float)
            cur_vals = pd.to_numeric(current_df[feat], errors="coerce").dropna().to_numpy(dtype=float)

            if len(ref_vals) < 5 or len(cur_vals) < 5:
                continue

            # Two-sample KS test
            ks_res = stats.ks_2samp(ref_vals, cur_vals)
            stat = float(ks_res.statistic)
            p_val = float(ks_res.pvalue)

            # PSI & Wasserstein
            psi = cls.calculate_psi(ref_vals, cur_vals)
            wasserstein = float(stats.wasserstein_distance(ref_vals, cur_vals))

            # Determine drift severity
            is_drift = (p_val < request.ks_p_value_threshold) or (psi >= request.psi_threshold)
            if is_drift:
                drifted_count += 1
                severity = "high" if (psi >= 0.25 or p_val < 0.001) else "moderate"
            else:
                severity = "none"

            feature_reports.append(
                FeatureDriftDetail(
                    feature_name=feat,
                    feature_type="numeric",
                    drift_detected=is_drift,
                    drift_severity=severity,
                    test_name="kolmogorov_smirnov",
                    test_statistic=round(stat, 4),
                    p_value=round(p_val, 6),
                    psi=round(psi, 4),
                    wasserstein_distance=round(wasserstein, 4),
                    reference_mean=round(float(np.mean(ref_vals)), 2),
                    current_mean=round(float(np.mean(cur_vals)), 2),
                    reference_std=round(float(np.std(ref_vals)), 2),
                    current_std=round(float(np.std(cur_vals)), 2),
                )
            )

        # 3. Evaluate Categorical Features
        for feat in CATEGORICAL_DRIFT_FEATURES:
            if feat not in current_df.columns or feat not in ref_df.columns:
                continue

            ref_cats = ref_df[feat].astype(str)
            cur_cats = current_df[feat].astype(str)

            all_unique = sorted(list(set(ref_cats.unique()).union(set(cur_cats.unique()))))
            ref_freq = ref_cats.value_counts().reindex(all_unique, fill_value=0).to_numpy()
            cur_freq = cur_cats.value_counts().reindex(all_unique, fill_value=0).to_numpy()

            ref_prob = ref_freq / ref_freq.sum()
            expected_cur_freq = ref_prob * cur_freq.sum()

            # Chi-square test
            try:
                chi2_res = stats.chisquare(f_obs=cur_freq, f_exp=expected_cur_freq)
                stat = float(chi2_res.statistic)
                p_val = float(chi2_res.pvalue)
            except Exception:
                stat = 0.0
                p_val = 1.0

            is_drift = p_val < request.ks_p_value_threshold
            if is_drift:
                drifted_count += 1
                severity = "high" if p_val < 0.001 else "moderate"
            else:
                severity = "none"

            feature_reports.append(
                FeatureDriftDetail(
                    feature_name=feat,
                    feature_type="categorical",
                    drift_detected=is_drift,
                    drift_severity=severity,
                    test_name="chi_square",
                    test_statistic=round(stat, 4),
                    p_value=round(p_val, 6),
                    psi=None,
                    wasserstein_distance=None,
                    reference_mean=None,
                    current_mean=None,
                    reference_std=None,
                    current_std=None,
                )
            )

        overall_drift = drifted_count > 0
        summary_msg = (
            f"Drift detected in {drifted_count} of {len(feature_reports)} features. "
            f"Model retraining or policy re-calibration recommended."
            if overall_drift else
            f"No statistically significant data drift detected across {len(feature_reports)} evaluated features."
        )

        return DriftCheckResponse(
            drift_detected=overall_drift,
            total_features_evaluated=len(feature_reports),
            drifted_features_count=drifted_count,
            reference_sample_size=len(ref_df),
            current_sample_size=len(current_df),
            features=feature_reports,
            summary_message=summary_msg,
        )
