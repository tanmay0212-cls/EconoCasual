"""Schemas for health check and system diagnostic endpoints."""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class ModelStatus(BaseModel):
    """Status details of a loaded causal ML model artifact."""
    model_name: str
    estimator_class: str
    is_loaded: bool
    version: Optional[str] = None
    treatment_levels: list[int] = Field(default_factory=lambda: [0, 10, 20])
    feature_names: list[str] = Field(default_factory=list)
    model_file_size_mb: Optional[float] = None


class HealthResponse(BaseModel):
    """System health and diagnostic response schema."""
    status: str = Field(..., description="'healthy' | 'degraded' | 'unhealthy'")
    app_version: str = Field(..., description="API semantic version")
    environment: str = Field("production", description="Runtime environment")
    models: Dict[str, ModelStatus] = Field(..., description="Availability status of registered models")
    reference_data_available: bool = Field(..., description="Indicates if baseline drift dataset is available")
    timestamp: str = Field(..., description="Current ISO-8601 UTC timestamp")
