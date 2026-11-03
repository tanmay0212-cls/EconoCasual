"""API Schemas package."""

from src.api.schemas.common import APIResponse, APIErrorDetail
from src.api.schemas.health import HealthResponse, ModelStatus
from src.api.schemas.data import DatasetValidationResponse, DatasetValidationSummary, DatasetReferenceRequest
from src.api.schemas.predict import CustomerFeatureRecord, PredictEffectsRequest, PredictEffectsResponse, CustomerEffectPrediction
from src.api.schemas.optimize import OptimizationRequest, OptimizationResponse, OptimizationKPIs, CustomerAllocationRecord
from src.api.schemas.drift import DriftCheckRequest, DriftCheckResponse, FeatureDriftDetail

__all__ = [
    "APIResponse",
    "APIErrorDetail",
    "HealthResponse",
    "ModelStatus",
    "DatasetValidationResponse",
    "DatasetValidationSummary",
    "DatasetReferenceRequest",
    "CustomerFeatureRecord",
    "PredictEffectsRequest",
    "PredictEffectsResponse",
    "CustomerEffectPrediction",
    "OptimizationRequest",
    "OptimizationResponse",
    "OptimizationKPIs",
    "CustomerAllocationRecord",
    "DriftCheckRequest",
    "DriftCheckResponse",
    "FeatureDriftDetail",
]

