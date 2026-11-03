"""Health check and diagnostic routes."""

from datetime import datetime, timezone
from fastapi import APIRouter
from src.api.config import settings
from src.api.services.model_loader import model_registry
from src.api.schemas.health import HealthResponse

router = APIRouter(tags=["Health & Diagnostics"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check API status and model availability",
    description="Returns the operational status of the service, application version, and model readiness.",
)
async def get_health() -> HealthResponse:
    """Return system health status and registered model availability."""
    models_status = model_registry.get_status()
    all_loaded = all(m.is_loaded for m in models_status.values())
    import os
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
