"""Unit and integration tests for EconoCasual 2.0 API (Phase 4).

Tests endpoints for:
- Doubly Robust policy evaluation and policy comparison
- CATE uncertainty quantification and export
- Uncertainty-aware and fairness-constrained MILP optimization
- Customer segment fairness diagnostics
- Reproducible experiment tracking and run retrieval
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Module-level TestClient for FastAPI."""
    return TestClient(app)


# ==============================================================================
# 1. OpenAPI & Documentation Tests
# ==============================================================================

def test_v2_openapi_routes(client: TestClient):
    """Verify that all Phase 4 v2 routes are present in the OpenAPI specification."""
    res = client.get("/openapi.json")
    assert res.status_code == 200
    schema = res.json()
    paths = schema["paths"]

    expected_routes = [
        "/api/v2/evaluation/summary",
        "/api/v2/evaluation/evaluate",
        "/api/v2/uncertainty/summary",
        "/api/v2/uncertainty/export",
        "/api/v2/optimize/advanced",
        "/api/v2/optimize/benchmarks",
        "/api/v2/optimize/allocation/export",
        "/api/v2/fairness/diagnostics",
        "/api/v2/experiments/history",
        "/api/v2/experiments/{experiment_id}",
    ]
    for route in expected_routes:
        assert route in paths, f"Route '{route}' missing from OpenAPI schema"


# ==============================================================================
# 2. Policy Evaluation Endpoints (Phase 1)
# ==============================================================================

def test_policy_evaluation_summary(client: TestClient):
    """GET /api/v2/evaluation/summary returns 5 evaluated policies and valid CIs."""
    res = client.get("/api/v2/evaluation/summary")
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert "metadata" in body
    assert "policies" in body

    policies = body["policies"]
    assert "no_discount" in policies
    assert "flat_10" in policies
    assert "random_discount" in policies
    assert "dml_milp" in policies
    assert "uncertainty_aware_placeholder" in policies

    # Check DML + MILP policy metrics
    dml = policies["dml_milp"]
    assert dml["status"] == "evaluated"
    assert dml["is_available"] is True
    assert dml["economic_metrics"]["dr_expected_profit"] > 95000.0
    assert len(dml["economic_metrics"]["dr_profit_ci_95"]) == 2
    assert dml["economic_metrics"]["dr_profit_ci_95"][0] < dml["economic_metrics"]["dr_profit_ci_95"][1]
    assert dml["economic_metrics"]["incremental_profit_to_cost_ratio"] is not None

    # Verify that future placeholder policy is explicitly marked unavailable
    future_p = policies["uncertainty_aware_placeholder"]
    assert future_p["status"] == "unavailable"
    assert future_p["is_available"] is False
    assert future_p["conversion_metrics"]["dr_conversion_rate"] == 0.0


def test_policy_evaluation_evaluate_post(client: TestClient):
    """POST /api/v2/evaluation/evaluate accepts valid hyperparameters."""
    payload = {
        "n_bootstraps": 150,
        "min_propensity_clip": 0.02,
        "random_seed": 123,
    }
    res = client.post("/api/v2/evaluation/evaluate", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert body["metadata"]["requested_n_bootstraps"] == 150
    assert body["metadata"]["requested_min_propensity_clip"] == 0.02


def test_policy_evaluation_validation_error(client: TestClient):
    """POST /api/v2/evaluation/evaluate rejects invalid parameter bounds with 422."""
    payload = {"n_bootstraps": 5}  # ge=10 constraint violated
    res = client.post("/api/v2/evaluation/evaluate", json=payload)
    assert res.status_code == 422
    body = res.json()
    assert body["success"] is False
    assert body["error"]["code"] == "VALIDATION_ERROR"


# ==============================================================================
# 3. Uncertainty Quantification Endpoints (Phase 2)
# ==============================================================================

def test_uncertainty_summary(client: TestClient):
    """GET /api/v2/uncertainty/summary returns coverage rate and reliability breakdown."""
    res = client.get("/api/v2/uncertainty/summary")
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert "models" in body
    models = body["models"]

    assert "point_estimate_baseline" in models
    assert "asymptotic_subforest_model" in models
    assert "bootstrap_residual_model" in models

    # Verify coverage against synthetic DGP is close to 95%
    asymp_10 = models["asymptotic_subforest_model"]["contrasts"]["10_vs_0"]
    assert asymp_10["coverage_rate_pct"] > 90.0
    assert asymp_10["mean_interval_width"] > 0.0
    assert "reliable" in asymp_10["reliability_counts"]


def test_uncertainty_export_csv(client: TestClient):
    """GET /api/v2/uncertainty/export streams CSV with uncertainty columns."""
    res = client.get("/api/v2/uncertainty/export")
    assert res.status_code == 200
    assert "text/csv" in res.headers["content-type"]
    content = res.text
    lines = content.strip().split("\n")
    header = [c.strip() for c in lines[0].split(",")]

    assert "customer_id" in header
    assert "standard_error" in header
    assert "lower_bound" in header
    assert "upper_bound" in header
    assert "reliability_status" in header
    assert len(lines) > 100


# ==============================================================================
# 4. Advanced Optimization & Benchmarks Endpoints (Phase 3)
# ==============================================================================

def test_advanced_benchmarks(client: TestClient):
    """GET /api/v2/optimize/benchmarks returns 4 core regimes and parameter sweeps."""
    res = client.get("/api/v2/optimize/benchmarks")
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert "core_regimes" in body

    regimes = body["core_regimes"]
    assert "regime1_original_milp" in regimes
    assert "regime2_uncertainty_aware_milp" in regimes
    assert "regime3_fairness_constrained_milp" in regimes
    assert "regime4_combined_milp" in regimes

    r1 = regimes["regime1_original_milp"]
    assert r1["financial_kpis"]["nominal_expected_profit"] > 100000.0
    assert r1["financial_kpis"]["total_expected_cost"] <= 5000.0


def test_advanced_optimization_conservative_lcb(client: TestClient):
    """POST /api/v2/optimize/advanced solves conservative LCB optimization."""
    payload = {
        "budget": 2000.0,
        "budget_type": "redemption",
        "optimization_mode": "conservative_lcb",
        "max_customers": 100,
    }
    res = client.post("/api/v2/optimize/advanced", json=payload)
    assert res.status_code == 200
    body = res.json()

    assert body["success"] is True
    assert body["status_code"] == 0
    assert body["optimization_mode"] == "conservative_lcb"
    assert body["financial_kpis"]["total_expected_cost"] <= 2000.0001
    assert "0" in body["tier_counts"]
    assert "10" in body["tier_counts"]
    assert "20" in body["tier_counts"]
    assert len(body["fairness_metrics"]) == 4
    assert len(body["allocation_preview"]) > 0


def test_advanced_optimization_risk_adjusted(client: TestClient):
    """POST /api/v2/optimize/advanced solves risk-adjusted optimization with lambda=2.0."""
    payload = {
        "budget": 3000.0,
        "budget_type": "redemption",
        "optimization_mode": "risk_adjusted",
        "risk_aversion": 2.0,
        "max_customers": 100,
    }
    res = client.post("/api/v2/optimize/advanced", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert body["risk_aversion"] == 2.0
    assert body["financial_kpis"]["total_expected_cost"] <= 3000.0001


def test_advanced_optimization_fairness_constraints(client: TestClient):
    """POST /api/v2/optimize/advanced honors segment floor constraints."""
    payload = {
        "budget": 4000.0,
        "budget_type": "redemption",
        "optimization_mode": "baseline",
        "segment_budget_floor": 0.10,
        "max_customers": 100,
    }
    res = client.post("/api/v2/optimize/advanced", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert body["fairness_metrics"]["budget"]["is_floor_satisfied"] is True


def test_advanced_optimization_validation_failure(client: TestClient):
    """POST /api/v2/optimize/advanced rejects negative budget with 422."""
    payload = {"budget": -100.0}
    res = client.post("/api/v2/optimize/advanced", json=payload)
    assert res.status_code == 422
    body = res.json()
    assert body["success"] is False
    assert body["error"]["code"] == "VALIDATION_ERROR"


def test_advanced_allocations_export_csv(client: TestClient):
    """GET /api/v2/optimize/allocation/export streams CSV with customer allocations."""
    res = client.get("/api/v2/optimize/allocation/export")
    assert res.status_code == 200
    assert "text/csv" in res.headers["content-type"]
    content = res.text
    assert "customer_id" in content
    assert "regime1_original_milp_tier" in content


# ==============================================================================
# 5. Fairness Diagnostics Endpoints (Phase 3 Part B)
# ==============================================================================

def test_fairness_diagnostics(client: TestClient):
    """GET /api/v2/fairness/diagnostics computes business segment parity metrics."""
    res = client.get("/api/v2/fairness/diagnostics")
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert body["total_customers"] > 0
    assert "budget" in body["segments"]
    assert "vip" in body["segments"]

    budget_seg = body["segments"]["budget"]
    assert budget_seg["sample_size"] > 0
    assert budget_seg["parity_ratio_vs_overall"] > 0.0
    assert "cash_budget_disclaimer" in body
    assert "disadvantaged_segment" in body["active_constraints_status"]


# ==============================================================================
# 6. Experiment History & Detail Endpoints
# ==============================================================================

def test_experiment_history(client: TestClient):
    """GET /api/v2/experiments/history returns catalog of Phase 1-3 experiments."""
    res = client.get("/api/v2/experiments/history")
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert body["total_experiments"] >= 3
    exp_ids = [e["experiment_id"] for e in body["experiments"]]
    assert "phase1_doubly_robust_policy_evaluation" in exp_ids
    assert "phase2_uncertainty_quantification" in exp_ids
    assert "phase3_uncertainty_and_fairness_optimization" in exp_ids


def test_experiment_detail_success(client: TestClient):
    """GET /api/v2/experiments/{id} returns raw JSON payload for valid experiment."""
    res = client.get("/api/v2/experiments/phase1_doubly_robust_policy_evaluation")
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert body["experiment_id"] == "phase1_doubly_robust_policy_evaluation"
    assert "policies" in body["data"]


def test_experiment_detail_not_found(client: TestClient):
    """GET /api/v2/experiments/{invalid_id} returns 404."""
    res = client.get("/api/v2/experiments/nonexistent_experiment_xyz")
    assert res.status_code == 404
    body = res.json()
    assert body["success"] is False
