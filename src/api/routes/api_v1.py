"""Canonical Milestone 4 API v1 router (/api/*).

Provides the unified, production REST API endpoints for EconoCausal:
- GET  /api/health            : System status, version, and model readiness
- POST /api/upload            : Customer dataset CSV upload and causal schema validation
- POST /api/predict           : CATE estimation & counterfactual purchase probabilities
- POST /api/optimize          : SciPy MILP budget-constrained discount optimization
- GET  /api/metrics           : Campaign KPIs, conversion rates, and discount distributions
- GET  /api/allocation/export : Optimized customer allocation CSV export
- POST /api/drift/check       : Statistical data, covariate, and prediction drift monitoring
- GET  /api/drift/history     : Historical drift monitoring run logs
"""

import os
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Depends, Query, UploadFile, File, Response, HTTPException, status

from src.api.config import settings
from src.causal.multi_tier_dml import MultiTierCausalDML
from src.api.services.model_loader import model_registry, get_model
from src.api.services.data_service import DataService
from src.api.services.predict_service import PredictService
from src.api.services.optimize_service import OptimizeService
from src.api.services.drift_service import DriftService

from src.api.schemas.health import HealthResponse
from src.api.schemas.data import UploadSummaryResponse
from src.api.schemas.predict import PredictEffectsRequest, PredictEffectsResponse
from src.api.schemas.optimize import (
    OptimizationRequest,
    OptimizationResponse,
    MetricsResponse,
)
from src.api.schemas.drift import (
    DriftCheckRequest,
    DriftCheckResponse,
    DriftHistoryResponse,
)

router = APIRouter(prefix="/api", tags=["EconoCausal Production API"])


# ── 1. GET /api/health ─────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check backend operational status and model readiness",
    description="Returns backend health, app version, causal model load state, and reference data availability.",
)
async def api_health() -> HealthResponse:
    """Check API status and model availability."""
    models_status = model_registry.get_status()
    all_loaded = all(m.is_loaded for m in models_status.values())
    ref_exists = os.path.exists(settings.reference_data_path)

    status_str = "healthy" if (all_loaded and ref_exists) else "degraded" if all_loaded else "unhealthy"

    return HealthResponse(
        status=status_str,
        app_version=settings.version,
        environment="production",
        models=models_status,
        reference_data_available=ref_exists,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ── 2. POST /api/upload ────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=UploadSummaryResponse,
    summary="Upload and validate customer CSV file",
    description=(
        "Accepts a customer campaign CSV file via multipart/form-data. "
        "Validates schema columns, data types, null counts, duplicate customer IDs, "
        "and quarantines potential-outcome or ground-truth leakage columns."
    ),
)
async def api_upload(
    file: UploadFile = File(..., description="Customer CSV dataset to upload and validate"),
) -> UploadSummaryResponse:
    """Accept and validate uploaded customer CSV dataset."""
    df = await DataService.read_uploaded_csv(file)
    return DataService.process_upload(df, filename=file.filename or "uploaded_dataset.csv")


# ── 3. POST /api/predict ───────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictEffectsResponse,
    summary="Predict counterfactual purchase probabilities and CATE lifts",
    description=(
        "Generates individual customer counterfactual conversion probabilities for discount tiers $0, $10, and $20 "
        "using the pre-trained Double Machine Learning model without retraining on operational data."
    ),
)
async def api_predict(
    request: PredictEffectsRequest,
    model: MultiTierCausalDML = Depends(get_model),
) -> PredictEffectsResponse:
    """Predict customer treatment effects and counterfactual conversion probabilities."""
    return PredictService.predict_effects(records=request.records, model=model)


# ── 4. POST /api/optimize ──────────────────────────────────────────────────────

@router.post(
    "/optimize",
    response_model=OptimizationResponse,
    summary="Optimize personalized discount allocations under marketing budget",
    description=(
        "Solves the budget-constrained Multiple-Choice Knapsack problem using SciPy MILP (HiGHS). "
        "Allocates discounts ($0, $10, $20) to maximize expected profit while strictly satisfying the budget constraint."
    ),
)
async def api_optimize(
    request: OptimizationRequest,
    model: MultiTierCausalDML = Depends(get_model),
) -> OptimizationResponse:
    """Execute budget-constrained discount optimization."""
    return OptimizeService.run_optimization(
        request=request,
        model=model,
        data_service_ref_resolver=DataService.resolve_dataset_reference,
    )


# ── 5. GET /api/metrics ────────────────────────────────────────────────────────

@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Retrieve campaign performance and allocation KPIs",
    description=(
        "Returns expected net profit, promotional expenditure, gross revenue, conversion rate, "
        "budget utilization, and customer discount distribution for the specified or latest optimization run."
    ),
)
async def api_metrics(
    run_id: Optional[str] = Query(None, description="Optional UUID of specific optimization run; defaults to latest"),
) -> MetricsResponse:
    """Retrieve aggregate optimization metrics and discount distribution."""
    run = OptimizeService.get_latest_or_default_run(run_id)
    kpis = run.kpis

    avg_conversion = kpis.conversion_rate
    if avg_conversion is None:
        probs = [r.predicted_purchase_probability for r in run.allocation]
        avg_conversion = round(float(sum(probs) / len(probs)), 4) if probs else 0.0

    return MetricsResponse(
        run_id=run.run_id,
        solver_status=run.solver_status,
        objective_type=run.objective_type,
        budget_type=getattr(run, "budget_type", "redemption"),
        total_customers=kpis.total_customers,
        expected_profit=kpis.expected_net_profit,
        expected_expenditure=kpis.expected_discount_expenditure,
        expected_revenue=kpis.expected_revenue,
        conversion_rate=avg_conversion,
        budget_cap=kpis.budget_cap,
        remaining_budget=kpis.remaining_budget,
        budget_utilization_pct=kpis.budget_utilization_pct,
        discount_distribution={
            "0": kpis.count_discount_0,
            "10": kpis.count_discount_10,
            "20": kpis.count_discount_20,
        },
    )


# ── 6. GET /api/allocation/export ──────────────────────────────────────────────

@router.get(
    "/allocation/export",
    summary="Export optimized customer discount allocations as CSV",
    description="Streams a downloadable CSV file containing customer-level discount allocations and economic metrics.",
    responses={
        200: {
            "content": {"text/csv": {}},
            "description": "Downloadable CSV of customer discount allocations.",
        }
    },
)
async def api_export_allocation(
    run_id: Optional[str] = Query(None, description="Optional UUID of specific optimization run; defaults to latest"),
) -> Response:
    """Stream customer allocation results as a downloadable CSV file."""
    csv_str, filename = OptimizeService.export_allocation_csv(run_id)
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-cache",
        },
    )


# ── 7. POST /api/drift/check ───────────────────────────────────────────────────

@router.post(
    "/drift/check",
    response_model=DriftCheckResponse,
    summary="Evaluate customer feature and prediction drift",
    description=(
        "Performs statistical tests (two-sample KS, PSI, Chi-Square, TVD) comparing incoming batches "
        "against the baseline training distribution across customer features and model predictions."
    ),
)
async def api_drift_check(
    request: DriftCheckRequest,
) -> DriftCheckResponse:
    """Run data and prediction drift surveillance tests."""
    return DriftService.check_drift(
        request=request,
        data_service_ref_resolver=DataService.resolve_dataset_reference,
    )


# ── 8. GET /api/drift/history ──────────────────────────────────────────────────

@router.get(
    "/drift/history",
    response_model=DriftHistoryResponse,
    summary="Retrieve historical data drift monitoring runs",
    description="Returns the history of statistical drift checks performed during operational monitoring.",
)
async def api_drift_history() -> DriftHistoryResponse:
    """Retrieve historical drift check logs."""
    return DriftService.get_drift_history()
