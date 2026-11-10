"""Service for customer-level CATE estimation and counterfactual purchase probability prediction."""

import logging
from typing import List, Any
import pandas as pd
import numpy as np
from fastapi import HTTPException, status

from src.api.config import settings
try:
    from src.causal.multi_tier_dml import MultiTierCausalDML
except Exception:
    MultiTierCausalDML = Any  # type: ignore
from src.api.schemas.predict import (
    CustomerFeatureRecord,
    PredictEffectsResponse,
    CustomerEffectPrediction,
)

logger = logging.getLogger("econocausal.predict_service")


class PredictService:
    """Service to predict treatment effects and counterfactual probabilities using preloaded DML model."""

    @staticmethod
    def predict_effects(
        records: List[CustomerFeatureRecord],
        model: MultiTierCausalDML,
    ) -> PredictEffectsResponse:
        """Run batch inference for pre-treatment customer records."""
        if not records:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty customer record list provided.",
            )

        if len(records) > settings.max_predict_batch_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"Requested batch size ({len(records)}) exceeds maximum allowed "
                    f"limit ({settings.max_predict_batch_size} records)."
                ),
            )

        # Convert records to DataFrame
        data_dicts = [r.model_dump() for r in records]
        df = pd.DataFrame(data_dicts)
        if "campaign_channel" not in df.columns:
            df["campaign_channel"] = "email"

        try:
            # Predict counterfactual probabilities and CATE lifts using MultiTierCausalDML
            pred_df = model.predict_probabilities(df)
        except Exception as exc:
            logger.error("Error during model prediction: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference engine failure: {str(exc)}",
            )

        predictions: List[CustomerEffectPrediction] = []
        for i, row in pred_df.iterrows():
            cust_id = str(row.get("customer_id", f"CUST_{i:04d}"))
            predictions.append(
                CustomerEffectPrediction(
                    customer_id=cust_id,
                    predicted_effect_10_vs_0=round(float(row["cate_lift_t10"]), 6),
                    predicted_effect_20_vs_0=round(float(row["cate_lift_t20"]), 6),
                    predicted_purchase_probability_0=round(float(row["p_purchase_t0"]), 6),
                    predicted_purchase_probability_10=round(float(row["p_purchase_t10"]), 6),
                    predicted_purchase_probability_20=round(float(row["p_purchase_t20"]), 6),
                )
            )

        meta = getattr(model, "metadata", {}) or {}
        return PredictEffectsResponse(
            model_version=str(meta.get("model_version", "1.0.0")),
            estimator_class="LinearDRLearner (EconML) + CalibratedGBM",
            record_count=len(predictions),
            predictions=predictions,
        )
