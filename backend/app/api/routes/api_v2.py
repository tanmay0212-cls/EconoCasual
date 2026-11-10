"""FastAPI Router for EconoCasual 2.0 (Phase 4).

Exposes validated endpoints for:
- Doubly Robust off-policy evaluation & policy comparison
- CATE uncertainty estimation & standard error distributions
- Risk-aware & conservative lower-confidence-bound optimization
- Business customer segment fairness diagnostics & constraints
- Experiment tracking and reproducible artifact retrieval
"""

import os
from fastapi import APIRouter, Path, Query, Response, status
from fastapi.responses import FileResponse

from src.api.services.v2_service import V2Service
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

router = APIRouter(prefix="/api/v2", tags=["EconoCasual 2.0"])


# ==============================================================================
# 1. Policy Evaluation Endpoints (Phase 1)
# ==============================================================================

@router.get(
    "/evaluation/summary",
    response_model=PolicyEvaluationSummaryResponse,
    summary="Retrieve Phase 1 Doubly Robust policy evaluation summary",
    description=(
        "Returns off-policy evaluation metrics (Doubly Robust, IPW, Direct Method) and 95% bootstrap CIs "
        "comparing No-Discount, Flat $10, Random, and DML+MILP policies. Clearly marks the future uncertainty-aware "
        "policy placeholder as unavailable."
    ),
)
async def get_policy_evaluation_summary() -> PolicyEvaluationSummaryResponse:
    """Retrieve comprehensive OPE results comparing marketing policies."""
    return V2Service.get_policy_evaluation_summary()


@router.post(
    "/evaluation/evaluate",
    response_model=PolicyEvaluationSummaryResponse,
    summary="Evaluate marketing policies under custom bootstrap and clipping parameters",
    description="Executes policy evaluation with requested bootstrap replications and propensity clipping bounds.",
)
async def evaluate_policies(request: PolicyEvaluateRequest) -> PolicyEvaluationSummaryResponse:
    """Run off-policy evaluation with custom hyperparameters."""
    return V2Service.run_policy_evaluation(request)


# ==============================================================================
# 2. Uncertainty Quantification Endpoints (Phase 2)
# ==============================================================================

@router.get(
    "/uncertainty/summary",
    response_model=UncertaintySummaryResponse,
    summary="Retrieve CATE uncertainty quantification and confidence interval coverage",
    description=(
        "Returns empirical 95% CI coverage against synthetic DGP, mean interval widths, and reliability breakdown "
        "for $10 vs $0 and $20 vs $0 treatment contrasts."
    ),
)
async def get_uncertainty_summary() -> UncertaintySummaryResponse:
    """Retrieve uncertainty estimation summary and standard error distributions."""
    return V2Service.get_uncertainty_summary()


@router.get(
    "/uncertainty/export",
    summary="Export customer CATE uncertainty and counterfactual dataset",
    description="Downloads the full CSV containing customer CATE point estimates, standard errors, 95% CIs, and reliability indicators.",
    response_class=FileResponse,
)
async def export_uncertainty_counterfactuals() -> FileResponse:
    """Stream uncertainty-aware customer counterfactuals CSV."""
    fpath = V2Service.get_counterfactuals_csv_path()
    return FileResponse(
        path=fpath,
        media_type="text/csv",
        filename=os.path.basename(fpath),
    )


# ==============================================================================
# 3. Advanced Optimization & Benchmarks Endpoints (Phase 3)
# ==============================================================================

@router.post(
    "/optimize/advanced",
    response_model=AdvancedOptimizationResponse,
    summary="Run uncertainty-aware and fairness-constrained MILP optimization",
    description=(
        "Executes SciPy HiGHS MILP under configurable objective modes: 'baseline' (nominal profit), "
        "'conservative_lcb' (lower confidence bound), or 'risk_adjusted' (mean minus risk-weighted variance). "
        "Supports optional segment budget floors and demographic parity constraints."
    ),
)
async def run_advanced_optimization(request: AdvancedOptimizationRequest) -> AdvancedOptimizationResponse:
    """Solve multiple-choice knapsack allocation with risk aversion and business fairness constraints."""
    return V2Service.run_advanced_optimization(request)


@router.get(
    "/optimize/benchmarks",
    response_model=AdvancedBenchmarksResponse,
    summary="Retrieve canonical optimization benchmarks across 4 core regimes",
    description=(
        "Returns benchmark metrics comparing Regime 1 (Original MILP), Regime 2 (Risk-Adjusted), "
        "Regime 3 (Fairness-Constrained), and Regime 4 (Combined Risk & Fairness) at $5,000 budget cap."
    ),
)
async def get_optimization_benchmarks() -> AdvancedBenchmarksResponse:
    """Retrieve multi-regime optimization benchmarks and sensitivity sweeps."""
    return V2Service.get_optimization_benchmarks()


@router.get(
    "/optimize/allocation/export",
    summary="Export advanced customer discount allocations",
    description="Downloads the assigned discount tiers and expected costs for the benchmarked multi-regime allocation.",
    response_class=FileResponse,
)
async def export_advanced_allocations() -> FileResponse:
    """Stream advanced customer allocations CSV."""
    fpath = V2Service.get_allocations_csv_path()
    return FileResponse(
        path=fpath,
        media_type="text/csv",
        filename=os.path.basename(fpath),
    )


# ==============================================================================
# 4. Fairness Diagnostics Endpoints (Phase 3 Part B)
# ==============================================================================

@router.get(
    "/fairness/diagnostics",
    response_model=FairnessDiagnosticsResponse,
    summary="Retrieve customer segment fairness and demographic parity diagnostics",
    description=(
        "Computes allocation rates, average discount dollars, and parity ratios across business-defined customer "
        "segments (budget, regular, premium, vip). Strictly excludes sensitive personal demographic attributes."
    ),
)
async def get_fairness_diagnostics() -> FairnessDiagnosticsResponse:
    """Compute segment-level fairness and demographic parity metrics."""
    return V2Service.get_fairness_diagnostics()


# ==============================================================================
# 5. Experiment History & Run Retrieval Endpoints
# ==============================================================================

@router.get(
    "/experiments/history",
    response_model=ExperimentHistoryResponse,
    summary="List historical EconoCasual 2.0 reproducible experiment runs",
    description="Returns metadata catalog of completed Phase 1, Phase 2, and Phase 3 research experiments.",
)
async def get_experiment_history() -> ExperimentHistoryResponse:
    """List historical experiment runs and artifact locations."""
    return V2Service.get_experiment_history()


@router.get(
    "/experiments/{experiment_id}",
    response_model=ExperimentDetailResponse,
    summary="Retrieve full raw JSON payload for a specific experiment",
    description="Fetches raw artifact JSON for 'phase1_doubly_robust_policy_evaluation', 'phase2_uncertainty_quantification', or 'phase3_uncertainty_and_fairness_optimization'.",
)
async def get_experiment_detail(
    experiment_id: str = Path(..., description="Unique experiment slug"),
) -> ExperimentDetailResponse:
    """Retrieve experiment artifact JSON by ID."""
    return V2Service.get_experiment_detail(experiment_id)
