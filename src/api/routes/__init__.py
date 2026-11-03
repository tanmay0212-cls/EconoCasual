"""API Routes package."""

from src.api.routes.health import router as health_router
from src.api.routes.data import router as data_router
from src.api.routes.predict import router as predict_router
from src.api.routes.optimize import router as optimize_router
from src.api.routes.drift import router as drift_router

__all__ = [
    "health_router",
    "data_router",
    "predict_router",
    "optimize_router",
    "drift_router",
]

