"""Schemas for prescriptive budget-constrained discount optimization and allocation retrieval."""

from typing import List, Optional, Literal, Dict
from pydantic import BaseModel, Field
from src.api.schemas.predict import CustomerFeatureRecord


class OptimizationRequest(BaseModel):
    """Payload to trigger budget-constrained discount policy optimization."""
    budget: float = Field(
        ...,
        gt=0.0,
        description="Marketing campaign budget ceiling in USD",
        examples=[5000.0]
    )
    budget_type: Literal["redemption", "face_value"] = Field(
        default="redemption",
        description="Budget accounting mode: 'redemption' (expected redeemed cost) or 'face_value' (issued coupon exposure)"
    )
    discount_tiers: List[int] = Field(
        default=[0, 10, 20],
        description="Supported discount tiers (must include control tier 0)",
        examples=[[0, 10, 20]]
    )
    objective_type: Literal["net_profit", "net_revenue", "gross_revenue"] = Field(
        default="net_profit",
        description="Business objective: 'net_profit' (default contribution margin) | 'net_revenue' | 'gross_revenue'"
    )
    records: Optional[List[CustomerFeatureRecord]] = Field(
        None,
        description="Explicit customer feature records to optimize. If omitted, dataset_reference must be provided."
    )
    dataset_reference: Optional[str] = Field(
        None,
        description="Approved server-side dataset identifier (e.g. 'optimization_retail_data')",
        examples=["optimization_retail_data"]
    )


class CustomerAllocationRecord(BaseModel):
    """Individual customer optimal allocation record."""
    customer_id: str
    recommended_discount: int = Field(..., description="Prescribed discount tier (0, 10, or 20)")
    predicted_purchase_probability: float = Field(..., ge=0.0, le=1.0)
    expected_revenue: float
    expected_discount_cost: float
    expected_net_profit: float


class OptimizationKPIs(BaseModel):
    """Aggregate campaign performance metrics under the optimal policy."""
    total_customers: int
    count_discount_0: int
    count_discount_10: int
    count_discount_20: int
    expected_revenue: float
    expected_discount_expenditure: float
    expected_net_profit: float
    budget_cap: float
    remaining_budget: float
    budget_utilization_pct: float
    conversion_rate: Optional[float] = Field(default=None, description="Average predicted conversion under policy")


class OptimizationResponse(BaseModel):
    """Result of budget optimization run."""
    run_id: str = Field(..., description="Unique UUID identifying this optimization run")
    solver_status: str = Field(..., description="HiGHS solver exit status (e.g. 'OPTIMAL (Status 7)')")
    objective_type: str
    budget_type: str = Field(default="redemption", description="Budget accounting mode applied")
    solve_time_seconds: float
    kpis: OptimizationKPIs
    allocation: List[CustomerAllocationRecord]


class MetricsResponse(BaseModel):
    """Comprehensive campaign performance metrics returned by GET /api/metrics."""
    run_id: str = Field(..., description="Unique UUID of the evaluated optimization run")
    solver_status: str
    objective_type: str
    budget_type: str
    total_customers: int
    expected_profit: float
    expected_expenditure: float
    expected_revenue: float
    conversion_rate: float
    budget_cap: float
    remaining_budget: float
    budget_utilization_pct: float
    discount_distribution: Dict[str, int]


