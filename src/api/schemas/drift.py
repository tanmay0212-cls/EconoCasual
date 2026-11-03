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
        description="Approved dataset reference to compare against baseline (e.g. 'optimization_retail_data', 'drift_reference', 'drift_current')",
        examples=["drift_current"]
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
    tvd: Optional[float] = None
    reference_mean: Optional[float] = None
    current_mean: Optional[float] = None
    reference_std: Optional[float] = None
    current_std: Optional[float] = None
    reference_summary: Dict[str, Any] = Field(default_factory=dict)
    current_summary: Dict[str, Any] = Field(default_factory=dict)
    new_categories: List[str] = Field(default_factory=list)
    missing_categories: List[str] = Field(default_factory=list)
    recommended_action: str = ""


class DriftCheckResponse(BaseModel):
    """Comprehensive data drift report."""
    overall_status: str = Field("No significant drift detected", description="Overall assessment")
    drift_detected: bool = Field(..., description="True if any evaluated feature has significant drift")
    total_features_evaluated: int
    drifted_features_count: int
    severe_drift_count: int = 0
    reference_sample_size: int
    current_sample_size: int
    features: List[FeatureDriftDetail]
    summary_message: str
    recommended_next_action: str = ""
    methodological_notice: str = (
        "Methodological Notice: Covariate drift (P(X)) measures shift in customer features. "
        "It does NOT automatically imply concept drift (P(Y | X, T)) or invalidity of causal treatment effects. "
        "Do not automatically retrain models without empirical policy verification."
    )
