"""Pydantic schemas for EconoCasual 2.0 (Phase 4: Policy Evaluation, Uncertainty, & Fairness-Constrained Optimization)."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


# ==============================================================================
# 1. Policy Evaluation & Comparison Schemas (Phase 1)
# ==============================================================================

class PolicyConversionMetrics(BaseModel):
    """Conversion rate metrics and 95% bootstrap confidence intervals for a policy."""
    dr_conversion_rate: float = Field(..., description="Doubly Robust conversion rate estimate")
    dr_conversion_ci_95: List[float] = Field(..., description="[lower, upper] 95% bootstrap CI")
    ipw_conversion_rate: float = Field(..., description="Inverse Probability Weighted conversion estimate")
    dm_conversion_rate: float = Field(..., description="Direct Method (outcome model) conversion estimate")
    incremental_conversion_vs_no_discount: float = Field(..., description="Incremental conversion vs organic baseline")
    incremental_conversion_ci_95: List[float] = Field(..., description="[lower, upper] 95% bootstrap CI for incremental conversion")


class PolicyEconomicMetrics(BaseModel):
    """Monetary profit metrics, costs, and 95% bootstrap confidence intervals."""
    dr_expected_profit: float = Field(..., description="Doubly Robust expected total profit in USD")
    dr_profit_ci_95: List[float] = Field(..., description="[lower, upper] 95% bootstrap CI for DR profit")
    model_expected_profit: float = Field(..., description="Model-predicted expected profit")
    ground_truth_expected_profit: Optional[float] = Field(None, description="Synthetic DGP ground truth profit")
    realized_observed_profit: Optional[float] = Field(None, description="Observed historical profit in held-out data")
    expected_promotional_cost: float = Field(..., description="Ex-ante mathematical expectation of promotional budget cost")
    incremental_profit_vs_no_discount: float = Field(..., description="Incremental profit relative to no-discount baseline")
    incremental_profit_ci_95: List[float] = Field(..., description="[lower, upper] 95% bootstrap CI for incremental profit")
    incremental_profit_to_cost_ratio: Optional[float] = Field(None, description="Incremental profit / cost ratio")


class PolicyDiagnostics(BaseModel):
    """Overlap and sample-size diagnostics for off-policy evaluation."""
    effective_sample_size: float = Field(..., description="Kish's effective sample size (ESS)")
    propensity_clipped_pct: float = Field(..., description="Percentage of propensity weights clipped")
    tier_distribution: Dict[str, int] = Field(..., description="Count of customers allocated to each discount tier")


class PolicySummaryDetail(BaseModel):
    """Full evaluation record for an individual marketing policy."""
    policy_name: str = Field(..., description="Human-readable policy title")
    status: str = Field(..., description="Evaluation status ('evaluated' or 'placeholder')")
    is_available: bool = Field(..., description="Flag indicating if policy is available for deployment")
    description: str = Field(..., description="Methodological description of the policy")
    conversion_metrics: Optional[PolicyConversionMetrics] = None
    economic_metrics: Optional[PolicyEconomicMetrics] = None
    diagnostics: Optional[PolicyDiagnostics] = None


class PolicyEvaluationSummaryResponse(BaseModel):
    """Payload returned by GET /api/v2/evaluation/summary."""
    success: bool = True
    metadata: Dict[str, Any] = Field(..., description="Evaluation experiment metadata and hyperparameters")
    policies: Dict[str, PolicySummaryDetail] = Field(..., description="Dictionary of evaluated policies")
    canonical_milestone3_benchmarks: Optional[Dict[str, Any]] = Field(None, description="Canonical Milestone 3 reference metrics")
    execution_time_ms: float = Field(..., description="Endpoint processing latency in milliseconds")


class PolicyEvaluateRequest(BaseModel):
    """Payload for POST /api/v2/evaluation/evaluate."""
    n_bootstraps: int = Field(100, ge=10, le=1000, description="Number of bootstrap replications for CI estimation")
    min_propensity_clip: float = Field(0.01, ge=0.001, le=0.2, description="Lower bound for propensity score clipping")
    random_seed: int = Field(42, description="Reproducible random seed")
    dataset_reference: Optional[str] = Field("sample_retail_campaign", description="Dataset identifier")


# ==============================================================================
# 2. Uncertainty Quantification Schemas (Phase 2)
# ==============================================================================

class ContrastUncertaintyMetrics(BaseModel):
    """Statistical uncertainty evaluation metrics for a specific treatment contrast."""
    cate_mae: Optional[float] = Field(None, description="Mean Absolute Error of CATE vs ground truth")
    cate_rmse: Optional[float] = Field(None, description="Root Mean Squared Error of CATE vs ground truth")
    ate_bias: Optional[float] = Field(None, description="ATE estimation bias")
    coverage_rate_pct: Optional[float] = Field(None, description="Empirical 95% CI coverage rate against synthetic DGP")
    mean_interval_width: Optional[float] = Field(None, description="Average 95% confidence interval width")
    reliability_counts: Optional[Dict[str, int]] = Field(None, description="Breakdown of customer estimates by reliability category")
    segment_diagnostics: Optional[Dict[str, Any]] = Field(None, description="Coverage and width diagnostics by customer segment")


class UncertaintyModelSummary(BaseModel):
    """Model summary record for uncertainty quantification approaches."""
    description: str = Field(..., description="Technical model description")
    runtime_seconds: float = Field(..., description="Inference runtime in seconds")
    contrasts: Dict[str, ContrastUncertaintyMetrics] = Field(..., description="Contrasts (e.g. 10_vs_0, 20_vs_0)")


class UncertaintySummaryResponse(BaseModel):
    """Payload returned by GET /api/v2/uncertainty/summary."""
    success: bool = True
    metadata: Dict[str, Any] = Field(..., description="Uncertainty quantification metadata")
    models: Dict[str, UncertaintyModelSummary] = Field(..., description="Comparative models (point estimate, asymptotic, bootstrap)")
    distribution_summary: Optional[Dict[str, Any]] = Field(None, description="Aggregated distribution statistics for standard errors")
    execution_time_ms: float = Field(..., description="Endpoint processing latency in milliseconds")


# ==============================================================================
# 3. Advanced Optimization & Fairness Schemas (Phase 3)
# ==============================================================================

class SegmentFairnessMetric(BaseModel):
    """Fairness and demographic parity diagnostic metrics for a customer segment."""
    segment_name: str = Field(..., description="Customer segment identifier ('budget', 'regular', 'premium', 'vip')")
    sample_size: int = Field(..., description="Total customers in this segment")
    allocation_count: int = Field(..., description="Number of customers allocated a positive discount ($10 or $20)")
    allocation_rate_pct: float = Field(..., description="Percentage of segment receiving a promotional discount")
    average_discount_amount: float = Field(..., description="Average discount dollars awarded per customer in segment")
    mean_incremental_conversion_pct: float = Field(..., description="Average predicted incremental conversion lift (%)")
    mean_incremental_profit: float = Field(..., description="Average predicted incremental profit ($)")
    parity_ratio_vs_overall: float = Field(..., description="Segment allocation rate divided by cohort-wide allocation rate")
    is_floor_satisfied: Optional[bool] = Field(None, description="Whether segment meets configured minimum allocation floor")
    is_parity_satisfied: Optional[bool] = Field(None, description="Whether segment meets minimum parity ratio")


class AdvancedOptimizationRequest(BaseModel):
    """Request payload for POST /api/v2/optimize/advanced."""
    budget: float = Field(5000.0, gt=0.0, le=10000000.0, description="Promotional campaign budget cap in USD")
    budget_type: str = Field("redemption", description="Budget constraint type: 'redemption' (expected cost) or 'allocated' (face value)")
    optimization_mode: str = Field("conservative_lcb", description="Objective mode: 'baseline', 'conservative_lcb', or 'risk_adjusted'")
    risk_aversion: float = Field(1.0, ge=0.0, le=10.0, description="Risk aversion parameter lambda >= 0")
    segment_budget_floor: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional minimum allocation rate floor for each segment")
    min_parity_ratio: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional minimum demographic parity ratio vs cohort")
    max_customers: Optional[int] = Field(None, gt=0, le=50000, description="Optional customer cohort sample limit")
    customer_data_reference: Optional[str] = Field("uncertainty_aware_counterfactuals", description="Dataset identifier")


class AdvancedOptimizationResponse(BaseModel):
    """Response payload for POST /api/v2/optimize/advanced."""
    success: bool = True
    status_code: int = Field(..., description="HiGHS solver return code (0 = Optimal)")
    status_message: str = Field(..., description="HiGHS solver termination message")
    optimization_mode: str = Field(..., description="Active optimization mode")
    risk_aversion: float = Field(..., description="Risk aversion parameter used")
    budget: float = Field(..., description="Configured budget cap in USD")
    budget_type: str = Field(..., description="Configured budget type")
    financial_kpis: Dict[str, float] = Field(..., description="Financial KPI dictionary (expected cost, nominal profit, LCB profit)")
    tier_counts: Dict[str, int] = Field(..., description="Count of customers assigned to $0, $10, $20")
    uncertainty_kpis: Dict[str, Any] = Field(..., description="Uncertainty diagnostics for selected recipients")
    fairness_metrics: Dict[str, SegmentFairnessMetric] = Field(..., description="Segment-level fairness and parity diagnostics")
    cash_budget_disclaimer: str = Field(..., description="Explicit cash budget expectation disclaimer")
    runtime_seconds: float = Field(..., description="Optimization solve duration in seconds")
    allocation_preview: Optional[List[Dict[str, Any]]] = Field(None, description="Preview of first 20 customer allocations")


class AdvancedBenchmarksResponse(BaseModel):
    """Payload returned by GET /api/v2/optimize/benchmarks."""
    success: bool = True
    metadata: Dict[str, Any] = Field(..., description="Benchmark experiment metadata")
    core_regimes: Dict[str, Any] = Field(..., description="Comparative 4 core regimes at $5,000 budget cap")
    risk_aversion_sweep: Optional[List[Dict[str, Any]]] = Field(None, description="Risk aversion parameter sensitivity sweep")
    fairness_tradeoff_sweep: Optional[List[Dict[str, Any]]] = Field(None, description="Fairness constraint budget floor tradeoff sweep")
    execution_time_ms: float = Field(..., description="Endpoint processing latency in milliseconds")


# ==============================================================================
# 4. Fairness Diagnostics Schemas (Phase 3 Part B)
# ==============================================================================

class FairnessDiagnosticsResponse(BaseModel):
    """Payload returned by GET /api/v2/fairness/diagnostics."""
    success: bool = True
    total_customers: int = Field(..., description="Total customers in analyzed cohort")
    overall_allocation_rate_pct: float = Field(..., description="Overall percentage receiving promotional discount")
    segments: Dict[str, SegmentFairnessMetric] = Field(..., description="Per-segment allocation and parity metrics")
    demographic_parity_gap_pct: float = Field(..., description="Max difference between segment allocation rates")
    min_parity_ratio: float = Field(..., description="Lowest parity ratio among all segments")
    active_constraints_status: Dict[str, Any] = Field(..., description="Status of active business fairness constraints")
    cash_budget_disclaimer: str = Field(..., description="Cash budget disclaimer")
    execution_time_ms: float = Field(..., description="Endpoint processing latency in milliseconds")


# ==============================================================================
# 5. Experiment History Schemas
# ==============================================================================

class ExperimentRunMeta(BaseModel):
    """Metadata summary of a reproducible experiment run."""
    experiment_id: str = Field(..., description="Unique experiment slug")
    phase: str = Field(..., description="EconoCasual 2.0 phase (e.g. 'Phase 1', 'Phase 2', 'Phase 3')")
    title: str = Field(..., description="Human-readable experiment title")
    timestamp: str = Field(..., description="ISO 8601 execution timestamp")
    summary: str = Field(..., description="Concise narrative summary of findings")
    status: str = Field(..., description="Completion status ('completed' or 'pending')")
    artifacts: List[str] = Field(..., description="List of generated artifact filepaths")


class ExperimentHistoryResponse(BaseModel):
    """Payload returned by GET /api/v2/experiments/history."""
    success: bool = True
    total_experiments: int = Field(..., description="Total count of historical experiment runs")
    experiments: List[ExperimentRunMeta] = Field(..., description="List of experiment metadata records")
    execution_time_ms: float = Field(..., description="Endpoint processing latency in milliseconds")


class ExperimentDetailResponse(BaseModel):
    """Payload returned by GET /api/v2/experiments/{experiment_id}."""
    success: bool = True
    experiment_id: str = Field(..., description="Unique experiment identifier")
    phase: str = Field(..., description="Phase identifier")
    timestamp: str = Field(..., description="Execution timestamp")
    data: Dict[str, Any] = Field(..., description="Full experiment raw JSON payload")
    execution_time_ms: float = Field(..., description="Endpoint processing latency in milliseconds")
