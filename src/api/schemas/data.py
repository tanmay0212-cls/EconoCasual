"""Schemas for dataset upload, validation, and schema compatibility checks."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class DatasetReferenceRequest(BaseModel):
    """Request body when validating an approved server-side dataset reference."""
    dataset_name: str = Field(
        ...,
        description="Approved dataset identifier (e.g. 'optimization_retail_data' or 'mock_retail_data')",
        examples=["optimization_retail_data"]
    )


class DatasetValidationSummary(BaseModel):
    """Summary metrics of the validated dataset."""
    total_rows: int
    valid_rows: int
    columns: List[str]
    column_count: int
    missing_value_counts: Dict[str, int]
    treatment_distribution: Dict[str, int] = Field(default_factory=dict)
    conversion_rate: Optional[float] = None
    average_order_value: Optional[float] = None


class DatasetValidationResponse(BaseModel):
    """Response schema for dataset validation endpoint."""
    is_valid: bool = Field(..., description="Whether the dataset passed all schema and causal rules")
    dataset_name: str = Field(..., description="Name or identifier of the evaluated dataset")
    total_rows: int
    valid_rows: int
    errors: List[str] = Field(default_factory=list, description="Critical validation errors")
    warnings: List[str] = Field(default_factory=list, description="Advisory schema warnings")
    summary: DatasetValidationSummary


class UploadSummaryResponse(BaseModel):
    """Upload summary response schema for POST /api/upload."""
    status: str = Field(..., description="'success' | 'warning' | 'error'")
    filename: str
    total_rows: int
    valid_rows: int
    invalid_rows: int
    columns_detected: List[str]
    missing_values_by_column: Dict[str, int]
    duplicate_customer_ids: int
    data_types: Dict[str, str]
    validation_passed: bool
    quarantined_columns: List[str] = Field(default_factory=list, description="Ground truth or leakage columns safely quarantined")
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    preview: List[Dict[str, Any]] = Field(default_factory=list)


