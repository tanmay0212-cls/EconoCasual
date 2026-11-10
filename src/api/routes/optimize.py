"""Budget-constrained discount optimization and allocation retrieval routes."""

from typing import Any
from fastapi import APIRouter, Depends, Path
try:
    from src.causal.multi_tier_dml import MultiTierCausalDML
except Exception:
    MultiTierCausalDML = Any  # type: ignore
from src.api.services.model_loader import get_model
from src.api.services.data_service import DataService
from src.api.services.optimize_service import OptimizeService
from src.api.schemas.optimize import OptimizationRequest, OptimizationResponse

router = APIRouter(tags=["Optimization & Allocation"])


@router.post(
    "/optimize",
    response_model=OptimizationResponse,
    summary="Optimize personalized discount allocations under marketing budget",
    description=(
        "Executes a SciPy Mixed-Integer Linear Program (MILP / HiGHS) to allocate personalized "
        "promotional discounts ($0, $10, $20) maximizing expected net profit or revenue subject to a strict budget ceiling."
    ),
)
async def optimize_budget(
    request: OptimizationRequest,
    model: MultiTierCausalDML = Depends(get_model),
) -> OptimizationResponse:
    """Solve multiple-choice knapsack allocation under campaign budget constraints."""
    return OptimizeService.run_optimization(
        request=request,
        model=model,
        data_service_ref_resolver=DataService.resolve_dataset_reference,
    )


@router.get(
    "/allocation/{run_id}",
    response_model=OptimizationResponse,
    summary="Retrieve saved optimization allocation results",
    description="Returns the full allocation policy, customer recommendations, and KPIs for a previous optimization run.",
)
async def get_allocation_results(
    run_id: str = Path(..., description="Unique UUID identifying the optimization run"),
) -> OptimizationResponse:
    """Look up a saved optimization run by run_id."""
    return OptimizeService.get_allocation_run(run_id=run_id)

