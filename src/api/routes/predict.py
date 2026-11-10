"""Treatment-effect and counterfactual probability prediction routes."""

from typing import Any
from fastapi import APIRouter, Depends
try:
    from src.causal.multi_tier_dml import MultiTierCausalDML
except Exception:
    MultiTierCausalDML = Any  # type: ignore
from src.api.services.model_loader import get_model
from src.api.services.predict_service import PredictService
from src.api.schemas.predict import PredictEffectsRequest, PredictEffectsResponse

router = APIRouter(prefix="/predict", tags=["Causal Predictions"])


@router.post(
    "/effects",
    response_model=PredictEffectsResponse,
    summary="Predict customer treatment effects and purchase probabilities",
    description=(
        "Accepts pre-treatment customer features and returns estimated causal lifts "
        "($10 vs $0, $20 vs $0) along with counterfactual purchase probabilities for all three discount tiers."
    ),
)
async def predict_effects(
    request: PredictEffectsRequest,
    model: MultiTierCausalDML = Depends(get_model),
) -> PredictEffectsResponse:
    """Compute individual customer CATEs and counterfactual conversion probabilities."""
    return PredictService.predict_effects(records=request.records, model=model)

