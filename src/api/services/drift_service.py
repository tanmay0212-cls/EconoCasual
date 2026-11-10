"""Automated data drift detection service using statistical divergence metrics."""

import os
import json
import uuid
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from fastapi import HTTPException, status

from src.api.config import settings
from src.monitoring.drift_detector import (
    EconoCausalDriftDetector,
    DriftReport,
    DriftSeverity,
    OverallDriftStatus,
)
from src.api.schemas.drift import (
    DriftCheckRequest,
    DriftCheckResponse,
    FeatureDriftDetail,
    DriftHistoryItem,
    DriftHistoryResponse,
)

logger = logging.getLogger("econocausal.drift_service")


class DriftService:
    """Service to execute statistical drift tests between incoming batches and reference data."""

    _detector: Optional[EconoCausalDriftDetector] = None
    _history_lock: threading.Lock = threading.Lock()

    @classmethod
    def get_detector(cls) -> EconoCausalDriftDetector:
        """Load and cache the baseline reference dataset and instantiate detector."""
        if cls._detector is None:
            ref_path = settings.reference_data_path

            if not os.path.exists(ref_path):
                logger.error("Approved reference dataset not found at configured path: %s", ref_path)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Approved baseline reference dataset is not available on server.",
                )

            try:
                ref_df = pd.read_csv(ref_path)
                logger.info("Initializing EconoCausalDriftDetector with %d baseline rows", len(ref_df))
                cls._detector = EconoCausalDriftDetector(reference_data=ref_df)
            except ValueError as val_err:
                logger.error("Reference dataset validation error: %s", val_err)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Baseline reference dataset validation failed: {str(val_err)}",
                )
            except Exception as exc:
                logger.error("Failed to load reference dataset: %s", exc)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Could not load baseline reference dataset due to an internal server error.",
                )
        return cls._detector

    @classmethod
    def get_reference_data(cls) -> pd.DataFrame:
        """Retrieve reference data for API health and warmups."""
        return cls.get_detector().reference_data

    @classmethod
    def check_drift(
        cls,
        request: DriftCheckRequest,
        data_service_ref_resolver,
    ) -> DriftCheckResponse:
        """Evaluate incoming batch for covariate drift against baseline distribution."""
        detector = cls.get_detector()

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

        # 2. Retrieve causal model if available for prediction drift scoring
        loaded_model = None
        try:
            from src.api.services.model_loader import model_registry
            loaded_model = model_registry.multi_tier_model
        except Exception:
            pass

        # 3. Execute drift detection with isolated request thresholds (no singleton mutation)
        report: DriftReport = detector.detect_drift(
            current_data=current_df,
            ks_alpha=request.ks_p_value_threshold,
            psi_high_threshold=request.psi_threshold,
            min_sample_size=request.min_sample_size,
            model=loaded_model,
        )

        # 4. Map report to API schema
        feature_details: List[FeatureDriftDetail] = []
        for feat in report.features:
            ref_sum = feat.reference_summary
            cur_sum = feat.current_summary
            feature_details.append(
                FeatureDriftDetail(
                    feature_name=feat.feature_name,
                    feature_type=feat.feature_type,
                    drift_detected=feat.drift_detected,
                    drift_severity=feat.severity.value if isinstance(feat.severity, DriftSeverity) else str(feat.severity),
                    test_name=feat.test_name,
                    test_statistic=feat.test_statistic,
                    p_value=feat.p_value,
                    psi=feat.psi,
                    wasserstein_distance=feat.wasserstein_distance,
                    tvd=feat.tvd,
                    reference_mean=ref_sum.get("mean"),
                    current_mean=cur_sum.get("mean"),
                    reference_std=ref_sum.get("std"),
                    current_std=cur_sum.get("std"),
                    reference_summary=ref_sum,
                    current_summary=cur_sum,
                    new_categories=feat.new_categories,
                    missing_categories=feat.missing_categories,
                    recommended_action=feat.recommended_action,
                )
            )

        # 5. Map prediction drift
        prediction_details: List[FeatureDriftDetail] = []
        for feat in getattr(report, "prediction_drift", []):
            ref_sum = feat.reference_summary
            cur_sum = feat.current_summary
            prediction_details.append(
                FeatureDriftDetail(
                    feature_name=feat.feature_name,
                    feature_type="prediction",
                    drift_detected=feat.drift_detected,
                    drift_severity=feat.severity.value if isinstance(feat.severity, DriftSeverity) else str(feat.severity),
                    test_name=feat.test_name,
                    test_statistic=feat.test_statistic,
                    p_value=feat.p_value,
                    psi=feat.psi,
                    wasserstein_distance=feat.wasserstein_distance,
                    tvd=feat.tvd,
                    reference_mean=ref_sum.get("mean"),
                    current_mean=cur_sum.get("mean"),
                    reference_std=ref_sum.get("std"),
                    current_std=cur_sum.get("std"),
                    reference_summary=ref_sum,
                    current_summary=cur_sum,
                    new_categories=feat.new_categories,
                    missing_categories=feat.missing_categories,
                    recommended_action=feat.recommended_action,
                )
            )

        # 6. Compute empirical distribution comparisons per feature (isolated per-feature failure)
        distributions_map: Dict[str, Dict[str, Any]] = {}
        ref_df = detector.reference_data
        for feat in report.features:
            fname = feat.feature_name
            try:
                if fname in ref_df.columns and fname in current_df.columns:
                    if feat.feature_type == "numeric":
                        ref_series = pd.to_numeric(ref_df[fname], errors="coerce").dropna()
                        curr_series = pd.to_numeric(current_df[fname], errors="coerce").dropna()
                        if len(ref_series) > 0 and len(curr_series) > 0:
                            combined = pd.concat([ref_series, curr_series])
                            min_val, max_val = float(combined.min()), float(combined.max())
                            if min_val == max_val:
                                min_val -= 0.5
                                max_val += 0.5
                            bins = np.linspace(min_val, max_val, 9)
                            ref_counts, _ = np.histogram(ref_series, bins=bins)
                            curr_counts, _ = np.histogram(curr_series, bins=bins)
                            ref_pct = (ref_counts / len(ref_series) * 100).round(2).tolist()
                            curr_pct = (curr_counts / len(curr_series) * 100).round(2).tolist()
                            labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]
                            distributions_map[fname] = {
                                "labels": labels,
                                "ref_pct": ref_pct,
                                "curr_pct": curr_pct,
                            }
                        else:
                            distributions_map[fname] = {
                                "labels": [],
                                "ref_pct": [],
                                "curr_pct": [],
                                "error": "Insufficient valid numeric observations.",
                            }
                    elif feat.feature_type == "categorical":
                        ref_series = ref_df[fname].dropna().astype(str)
                        curr_series = current_df[fname].dropna().astype(str)
                        if len(ref_series) > 0 and len(curr_series) > 0:
                            categories = sorted(list(set(ref_series.unique()) | set(curr_series.unique())))
                            ref_counts = ref_series.value_counts(normalize=True) * 100
                            curr_counts = curr_series.value_counts(normalize=True) * 100
                            distributions_map[fname] = {
                                "labels": [str(c) for c in categories],
                                "ref_pct": [round(float(ref_counts.get(cat, 0.0)), 2) for cat in categories],
                                "curr_pct": [round(float(curr_counts.get(cat, 0.0)), 2) for cat in categories],
                            }
                        else:
                            distributions_map[fname] = {
                                "labels": [],
                                "ref_pct": [],
                                "curr_pct": [],
                                "error": "Insufficient valid categorical observations.",
                            }
            except Exception as exc:
                logger.warning("Could not compute distribution for feature '%s': %s", fname, exc)
                distributions_map[fname] = {
                    "labels": [],
                    "ref_pct": [],
                    "curr_pct": [],
                    "error": f"Distribution calculation failed: {str(exc)}",
                }

        dataset_label = request.dataset_reference or "uploaded_batch"
        overall_status_val = report.overall_status.value if isinstance(report.overall_status, OverallDriftStatus) else str(report.overall_status)
        feature_summary_map = {feat.feature_name: feat.drift_severity for feat in feature_details}
        run_type_val = getattr(request, "run_type", None) or "operational"

        # 7. Append to historical drift log under lock
        cls._append_drift_history(
            dataset_name=dataset_label,
            overall_status=overall_status_val,
            drift_detected=report.drift_detected,
            drifted_count=report.drifted_features_count,
            severe_count=report.severe_drift_count,
            total_features=report.total_features_evaluated,
            run_type=run_type_val,
            feature_summary=feature_summary_map,
        )

        return DriftCheckResponse(
            overall_status=overall_status_val,
            drift_detected=report.drift_detected,
            total_features_evaluated=report.total_features_evaluated,
            drifted_features_count=report.drifted_features_count,
            severe_drift_count=report.severe_drift_count,
            reference_sample_size=report.reference_sample_size,
            current_sample_size=report.current_sample_size,
            features=feature_details,
            prediction_drift=prediction_details,
            prediction_drift_status=getattr(report, "prediction_drift_status", "available"),
            schema_validation_passed=report.schema_validation.is_valid,
            distributions=distributions_map,
            summary_message=report.summary_message,
            recommended_next_action=report.recommended_next_action,
            methodological_notice=report.methodological_notice,
        )

    @classmethod
    def _get_history_file_path(cls) -> str:
        runs_dir = settings.runs_dir
        os.makedirs(runs_dir, exist_ok=True)
        return os.path.join(runs_dir, "drift_history.json")

    @classmethod
    def _append_drift_history(
        cls,
        dataset_name: str,
        overall_status: str,
        drift_detected: bool,
        drifted_count: int,
        severe_count: int,
        total_features: int,
        run_type: str = "operational",
        feature_summary: Optional[Dict[str, str]] = None,
    ) -> None:
        """Persist a drift evaluation record into the historical log with retry deduplication and thread safety."""
        with cls._history_lock:
            history_file = cls._get_history_file_path()
            history_data: List[Dict[str, Any]] = []

            if os.path.exists(history_file):
                try:
                    with open(history_file, "r", encoding="utf-8") as f:
                        history_data = json.load(f)
                    if not isinstance(history_data, list):
                        history_data = []
                except Exception:
                    history_data = []

            now_utc = datetime.now(timezone.utc)

            # Retry Deduplication: Skip recording if identical run was appended within the last 5 seconds
            if history_data:
                last_entry = history_data[-1]
                try:
                    last_ts = datetime.fromisoformat(last_entry.get("timestamp", ""))
                    diff_sec = abs((now_utc - last_ts).total_seconds())
                    if (
                        diff_sec < 5.0
                        and last_entry.get("dataset_name") == dataset_name
                        and last_entry.get("overall_status") == overall_status
                        and last_entry.get("drifted_features_count") == drifted_count
                    ):
                        logger.info("Skipping duplicate drift history log within 5s window (deduplication applied).")
                        return
                except Exception:
                    pass

            new_entry = {
                "timestamp": now_utc.isoformat(),
                "run_id": str(uuid.uuid4()),
                "dataset_name": dataset_name,
                "overall_status": overall_status,
                "drift_detected": drift_detected,
                "drifted_features_count": drifted_count,
                "severe_drift_count": severe_count,
                "total_features_evaluated": total_features,
                "run_type": run_type,
                "feature_summary": feature_summary or {},
            }
            history_data.append(new_entry)

            # Keep last 50 entries
            if len(history_data) > 50:
                history_data = history_data[-50:]

            try:
                with open(history_file, "w", encoding="utf-8") as f:
                    json.dump(history_data, f, indent=2)
            except Exception as exc:
                logger.warning("Could not persist drift history log: %s", exc)

    @classmethod
    def get_drift_history(cls) -> DriftHistoryResponse:
        """Retrieve recent historical drift evaluation runs with corruption detection."""
        with cls._history_lock:
            history_file = cls._get_history_file_path()
            if os.path.exists(history_file):
                try:
                    with open(history_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if not isinstance(data, list):
                        raise ValueError("Corrupted drift history format: expected a JSON list.")
                    items = [DriftHistoryItem.model_validate(item) for item in data]
                    return DriftHistoryResponse(history=items)
                except Exception as exc:
                    logger.error("Failed to read drift history file: %s", exc)
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Historical drift surveillance logs are corrupted or unreadable.",
                    )

            return DriftHistoryResponse(history=[])
