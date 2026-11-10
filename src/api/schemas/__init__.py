"""API Schemas package."""

from src.api.schemas.common import APIResponse, APIErrorDetail
from src.api.schemas.health import HealthResponse, ModelStatus
from src.api.schemas.data import (
    DatasetValidationResponse,
    DatasetValidationSummary,
    DatasetReferenceRequest,
    UploadSummaryResponse,
)
from src.api.schemas.predict import (
    CustomerFeatureRecord,
    PredictEffectsRequest,
    PredictEffectsResponse,
    CustomerEffectPrediction,
)
from src.api.schemas.optimize import (
    OptimizationRequest,
    OptimizationResponse,
    OptimizationKPIs,
    CustomerAllocationRecord,
    MetricsResponse,
)
from src.api.schemas.drift import (
    DriftCheckRequest,
    DriftCheckResponse,
    FeatureDriftDetail,
    DriftHistoryItem,
    DriftHistoryResponse,
)
from src.api.schemas.v2 import (
    PolicyEvaluationSummaryResponse,
    PolicyEvaluateRequest,
    UncertaintySummaryResponse,
    AdvancedOptimizationRequest,
    AdvancedOptimizationResponse,
    AdvancedBenchmarksResponse,
    FairnessDiagnosticsResponse,
    ExperimentHistoryResponse,
    ExperimentDetailResponse,
)

__all__ = [
    "APIResponse",
    "APIErrorDetail",
    "HealthResponse",
    "ModelStatus",
    "DatasetValidationResponse",
    "DatasetValidationSummary",
    "DatasetReferenceRequest",
    "UploadSummaryResponse",
    "CustomerFeatureRecord",
    "PredictEffectsRequest",
    "PredictEffectsResponse",
    "CustomerEffectPrediction",
    "OptimizationRequest",
    "OptimizationResponse",
    "OptimizationKPIs",
    "CustomerAllocationRecord",
    "MetricsResponse",
    "DriftCheckRequest",
    "DriftCheckResponse",
    "FeatureDriftDetail",
    "DriftHistoryItem",
    "DriftHistoryResponse",
]


