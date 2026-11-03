"""Configuration settings for the EconoCausal REST API."""

import os
from typing import List, Dict
from pydantic import BaseModel, Field

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class APISettings(BaseModel):
    """API Configuration settings with strict security and operational guardrails."""

    title: str = "EconoCausal REST API"
    version: str = "1.0.0"
    description: str = (
        "Enterprise Causal Machine Learning & Prescriptive Pricing Optimization API. "
        "Exposes Doubly Robust Causal Estimation (EconML), SciPy MILP Knapsack Optimizer, "
        "and Automated Data Drift Detection."
    )
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"

    # Host & Port
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))

    # CORS settings: explicitly allowed origins for React frontend
    allowed_origins: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # File and Model Paths
    project_root: str = PROJECT_ROOT
    model_path: str = os.path.join(PROJECT_ROOT, "models", "multitreatment_model.joblib")
    model_metadata_path: str = os.path.join(PROJECT_ROOT, "models", "multitreatment_metadata.json")
    reference_data_path: str = os.path.join(PROJECT_ROOT, "data", "optimization_retail_data.csv")
    runs_dir: str = os.path.join(PROJECT_ROOT, "models", "runs")

    # Allowed server-side dataset references (strictly whitelisted; arbitrary paths rejected)
    allowed_dataset_references: Dict[str, str] = {
        "optimization_retail_data": os.path.join(PROJECT_ROOT, "data", "optimization_retail_data.csv"),
        "mock_retail_data": os.path.join(PROJECT_ROOT, "data", "mock_retail_data.csv"),
    }

    # Computational Guardrails
    max_predict_batch_size: int = 5000
    max_optimize_customers: int = 10000
    max_upload_size_bytes: int = 15 * 1024 * 1024  # 15 MB cap
    default_budget: float = 5000.0


settings = APISettings()
