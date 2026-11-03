"""Schemas for automated data drift detection and distributional divergence checks."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from src.api.schemas.predict import CustomerFeatureRecord


class DriftCheckRequest(BaseModel):
    """Payload to trigger data drift check."""
    records: Optional[List[CustomerFeatureRecord]] = Field(
        None,
        description="Incoming customer records to compare against reference distribution."
    )
    dataset_reference: Optional[str] = Field(
        None,
        description="Approved dataset reference to compare against baseline (e.g. 'optimization_retail_data')",
        examples=["optimization_retail_data"]
    )
    ks_p_value_threshold: float = Field(
        default=0.05,
        ge=0.001,
        le=0.20,
        description="P-value threshold for two-sample Kolmogorov-Smirnov test (drift flagged if p < threshold)"
    )
    psi_threshold: float = Field(
        default=0.25,
        ge=0.05,
        le=0.50,
        description="Population Stability Index (PSI) threshold for significant drift"
    )


class FeatureDriftDetail(BaseModel):
    """Distributional drift metrics for an individual feature."""
    feature_name: str
    feature_type: str = Field(..., description="'numeric' | 'categorical'")
    drift_detected: bool
    drift_severity: str = Field(..., description="'none' | 'moderate' | 'high'")
    test_name: str = Field(..., description="'kolmogorov_smirnov' | 'chi_square'")
    test_statistic: float
    p_value: Optional[float] = None
    psi: Optional[float] = None
    wasserstein_distance: Optional[float] = None
    reference_mean: Optional[float] = None
    current_mean: Optional[float] = None
    reference_std: Optional[float] = None
    current_std: Optional[float] = None


class DriftCheckResponse(BaseModel):
    """Comprehensive data drift report."""
    drift_detected: bool = Field(..., description="True if any evaluated feature has significant drift")
    total_features_evaluated: int
    drifted_features_count: int
    reference_sample_size: int
    current_sample_size: int
    features: List[FeatureDriftDetail]
    summary_message: str
