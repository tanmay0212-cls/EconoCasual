"""Automated data drift detection routes."""

from fastapi import APIRouter
from src.api.services.data_service import DataService
from src.api.services.drift_service import DriftService
from src.api.schemas.drift import DriftCheckRequest, DriftCheckResponse

router = APIRouter(prefix="/drift", tags=["Data Drift Detection"])


@router.post(
    "/check",
    response_model=DriftCheckResponse,
    summary="Detect covariate and distributional data drift",
    description=(
        "Compares an incoming customer batch against the approved baseline reference distribution "
        "using two-sample Kolmogorov-Smirnov tests, Population Stability Index (PSI), and Chi-square tests."
    ),
)
async def check_data_drift(request: DriftCheckRequest) -> DriftCheckResponse:
    """Evaluate distributional shift between incoming data and reference dataset."""
    return DriftService.check_drift(
        request=request,
        data_service_ref_resolver=DataService.resolve_dataset_reference,
    )

