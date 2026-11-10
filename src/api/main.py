"""FastAPI application entrypoint for EconoCausal."""

import time
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.config import settings
from src.api.services.model_loader import model_registry
from src.api.services.drift_service import DriftService
from src.api.routes import (
    health_router,
    data_router,
    predict_router,
    optimize_router,
    drift_router,
    api_v1_router,
    api_v2_router,
)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("econocausal.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager: loads models and warms caches once on startup."""
    logger.info("Starting EconoCausal API (version %s)...", settings.version)

    # 1. Load multi-treatment causal model into singleton memory
    try:
        model_registry.load_models()
    except Exception as exc:
        logger.error("Error during startup model loading: %s", exc)

    # 2. Pre-warm drift baseline dataset
    try:
        DriftService.get_reference_data()
    except Exception as exc:
        logger.warning("Reference drift data pre-warming notice: %s", exc)

    logger.info("EconoCausal API startup complete. Ready to receive requests.")
    yield
    logger.info("Shutting down EconoCausal API.")


def create_app() -> FastAPI:
    """Application factory for the EconoCausal FastAPI application."""
    app = FastAPI(
        title=settings.title,
        version=settings.version,
        description=settings.description,
        docs_url=settings.docs_url,
        redoc_url=settings.redoc_url,
        openapi_url=settings.openapi_url,
        lifespan=lifespan,
        openapi_tags=[
            {"name": "Health & Diagnostics", "description": "System health and model readiness checks."},
            {"name": "Dataset Validation", "description": "Schema, type, and causal domain validation for retail campaign data."},
            {"name": "Causal Predictions", "description": "Double Machine Learning CATE and counterfactual purchase probabilities."},
            {"name": "Optimization & Allocation", "description": "Budget-constrained Multiple-Choice Knapsack optimization (SciPy MILP)."},
            {"name": "Data Drift Detection", "description": "Two-sample KS, PSI, and Chi-square statistical divergence testing."},
        ],
    )

    # 1. Configure CORS for the React frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # 2. Performance and security logging middleware (sanitizes sensitive customer attributes)
    @app.middleware("http")
    async def log_requests_middleware(request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Log request metadata without payload body to preserve customer privacy
        logger.info(
            "%s %s -> Status %d (Duration: %0.2fms)",
            request.method,
            request.url.path,
            response.status_code,
            process_time_ms,
        )
        response.headers["X-Process-Time-Ms"] = str(process_time_ms)
        return response

    # 3. Custom exception handlers
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "code": f"HTTP_{exc.status_code}",
                    "message": exc.detail,
                },
                "path": request.url.path,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        formatted_errors = []
        for err in exc.errors():
            loc = " -> ".join(str(item) for item in err.get("loc", []))
            formatted_errors.append({
                "field": loc,
                "message": err.get("msg", "Invalid value"),
                "type": err.get("type", "value_error"),
            })

        logger.warning("Validation error on %s %s: %s", request.method, request.url.path, formatted_errors)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Input validation failed. Please check field requirements and bounds.",
                    "details": formatted_errors,
                },
                "path": request.url.path,
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled internal server exception on %s: %s", request.url.path, exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected server error occurred while processing the request.",
                },
                "path": request.url.path,
            },
        )

    # 4. Register API Routers
    app.include_router(api_v2_router)
    app.include_router(api_v1_router)
    app.include_router(health_router)
    app.include_router(data_router)
    app.include_router(predict_router)
    app.include_router(optimize_router)
    app.include_router(drift_router)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host=settings.host, port=settings.port, reload=True)

