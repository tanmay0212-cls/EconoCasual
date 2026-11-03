"""Singleton model registry and lifespan loader for EconoCausal API."""

import os
import json
import logging
from typing import Optional, Dict, Any
from src.causal.multi_tier_dml import MultiTierCausalDML
from src.api.config import settings
from src.api.schemas.health import ModelStatus

logger = logging.getLogger("econocausal.model_loader")


class ModelRegistry:
    """Thread-safe singleton registry that caches loaded causal models."""

    _instance: Optional["ModelRegistry"] = None
    _multi_tier_model: Optional[MultiTierCausalDML] = None
    _metadata: Optional[Dict[str, Any]] = None

    def __new__(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_models(self) -> None:
        """Load trained causal models from disk during application lifespan startup."""
        model_path = settings.model_path
        metadata_path = settings.model_metadata_path

        if not os.path.exists(model_path):
            logger.warning("Multi-treatment model file not found at: %s", model_path)
            self._multi_tier_model = None
            return

        try:
            logger.info("Loading multi-treatment causal model from: %s", model_path)
            self._multi_tier_model = MultiTierCausalDML.load(model_path)
            logger.info("Multi-treatment model successfully loaded.")
        except Exception as exc:
            logger.error("Failed to load multi-treatment model: %s", exc, exc_info=True)
            self._multi_tier_model = None

        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self._metadata = json.load(f)
            except Exception as exc:
                logger.warning("Failed to parse model metadata: %s", exc)
                self._metadata = None

    @property
    def multi_tier_model(self) -> Optional[MultiTierCausalDML]:
        return self._multi_tier_model

    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        return self._metadata

    def get_status(self) -> Dict[str, ModelStatus]:
        """Return diagnostic status of registered models."""
        is_loaded = self._multi_tier_model is not None and getattr(self._multi_tier_model, "is_fitted_", False)
        file_size_mb = None
        if os.path.exists(settings.model_path):
            file_size_mb = round(os.path.getsize(settings.model_path) / (1024 * 1024), 2)

        meta = self._metadata or {}
        model_status = ModelStatus(
            model_name="multi_tier_dml",
            estimator_class="LinearDRLearner (EconML) + CalibratedGBM",
            is_loaded=is_loaded,
            version=meta.get("model_version", "1.0.0"),
            treatment_levels=meta.get("treatment_levels", [0, 10, 20]),
            feature_names=meta.get("features", ["age", "loyalty_score", "previous_purchases", "customer_segment"]),
            model_file_size_mb=file_size_mb,
        )

        return {"multi_tier_dml": model_status}


model_registry = ModelRegistry()


def get_model() -> MultiTierCausalDML:
    """Dependency provider returning the loaded MultiTierCausalDML model or raising 503."""
    model = model_registry.multi_tier_model
    if model is None or not getattr(model, "is_fitted_", False):
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Multi-treatment causal model is not loaded or unavailable on server.",
        )
    return model
