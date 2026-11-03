"""Test suite for Milestone 4 API v1 (/api/*) endpoints."""

import io
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    """Module-scoped TestClient."""
    return TestClient(app)


# ── 1. GET /api/health ─────────────────────────────────────────────────────────

def test_api_health_endpoint(client: TestClient):
    """GET /api/health returns operational status and model readiness."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in {"healthy", "degraded"}
    assert "app_version" in data
    assert "models" in data
    assert "multi_tier_dml" in data["models"]
    assert data["models"]["multi_tier_dml"]["is_loaded"] is True
    assert data["reference_data_available"] is True


# ── 2. POST /api/upload ────────────────────────────────────────────────────────

def test_api_upload_valid_csv(client: TestClient):
    """POST /api/upload accepts valid customer CSV and returns upload summary."""
    csv_content = (
        "customer_id,age,loyalty_score,previous_purchases,customer_segment,product_price,gross_margin_rate\n"
        "CUST_001,34,65.5,4,Gold,85.0,0.40\n"
        "CUST_002,52,80.0,9,Platinum,120.0,0.45\n"
        "CUST_003,28,40.0,1,Silver,50.0,0.35\n"
    )
    files = {"file": ("test_customers.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    res = client.post("/api/upload", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["validation_passed"] is True
    assert data["total_rows"] == 3
    assert data["valid_rows"] == 3
    assert data["invalid_rows"] == 0
    assert len(data["errors"]) == 0
    assert len(data["preview"]) == 3


def test_api_upload_quarantines_leakage_columns(client: TestClient):
    """POST /api/upload detects and quarantines ground-truth leakage columns."""
    csv_content = (
        "customer_id,age,loyalty_score,previous_purchases,customer_segment,potential_outcome_10,counterfactual_spend\n"
        "CUST_001,34,65.5,4,Gold,1,95.5\n"
        "CUST_002,52,80.0,9,Platinum,0,0.0\n"
    )
    files = {"file": ("leakage_data.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    res = client.post("/api/upload", files=files)
    assert res.status_code == 200
    data = res.json()
    assert "potential_outcome_10" in data["quarantined_columns"]
    assert "counterfactual_spend" in data["quarantined_columns"]


def test_api_upload_rejects_non_csv(client: TestClient):
    """POST /api/upload rejects non-CSV files."""
    files = {"file": ("data.txt", io.BytesIO(b"some,text,data"), "text/plain")}
    res = client.post("/api/upload", files=files)
    assert res.status_code == 400


def test_api_upload_detects_duplicate_customer_ids(client: TestClient):
    """POST /api/upload flags duplicate customer IDs in warnings."""
    csv_content = (
        "customer_id,age,loyalty_score,previous_purchases,customer_segment\n"
        "CUST_001,34,65.5,4,Gold\n"
        "CUST_001,34,65.5,4,Gold\n"
    )
    files = {"file": ("dups.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    res = client.post("/api/upload", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["duplicate_customer_ids"] == 1


# ── 3. POST /api/predict ───────────────────────────────────────────────────────

def test_api_predict_effects_success(client: TestClient):
    """POST /api/predict returns CATE lifts and valid counterfactual probabilities."""
    payload = {
        "records": [
            {
                "customer_id": "PRED_01",
                "age": 35.0,
                "loyalty_score": 75.0,
                "previous_purchases": 5,
                "customer_segment": "Gold",
            },
            {
                "customer_id": "PRED_02",
                "age": 55.0,
                "loyalty_score": 30.0,
                "previous_purchases": 1,
                "customer_segment": "Bronze",
            },
        ]
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert len(data["predictions"]) == 2

    for pred in data["predictions"]:
        p0 = pred["predicted_purchase_probability_0"]
        p10 = pred["predicted_purchase_probability_10"]
        p20 = pred["predicted_purchase_probability_20"]
        assert 0.0 <= p0 <= 1.0
        assert 0.0 <= p10 <= 1.0
        assert 0.0 <= p20 <= 1.0
        assert "predicted_effect_10_vs_0" in pred
        assert "predicted_effect_20_vs_0" in pred


# ── 4. POST /api/optimize ──────────────────────────────────────────────────────

def test_api_optimize_budget_redemption(client: TestClient):
    """POST /api/optimize runs SciPy MILP optimizer with redemption budget."""
    payload = {
        "budget": 200.0,
        "budget_type": "redemption",
        "dataset_reference": "optimization_retail_data",
        "objective_type": "net_profit",
    }
    res = client.post("/api/optimize", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "run_id" in data
    assert "HiGHS" in data["solver_status"] or "Optimal" in data["solver_status"]
    assert data["budget_type"] == "redemption"
    kpis = data["kpis"]
    assert kpis["expected_discount_expenditure"] <= 200.0 + 1e-4
    assert kpis["conversion_rate"] is not None
    assert 0.0 <= kpis["conversion_rate"] <= 1.0


def test_api_optimize_budget_face_value(client: TestClient):
    """POST /api/optimize runs SciPy MILP optimizer with face_value budget."""
    payload = {
        "budget": 500.0,
        "budget_type": "face_value",
        "dataset_reference": "optimization_retail_data",
        "objective_type": "net_profit",
    }
    res = client.post("/api/optimize", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["budget_type"] == "face_value"


# ── 5. GET /api/metrics ────────────────────────────────────────────────────────

def test_api_metrics_endpoint(client: TestClient):
    """GET /api/metrics returns KPIs and discount tier distribution."""
    # First ensure a run exists
    client.post("/api/optimize", json={"budget": 100.0, "dataset_reference": "optimization_retail_data"})

    res = client.get("/api/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "expected_profit" in data
    assert "expected_expenditure" in data
    assert "conversion_rate" in data
    assert "budget_utilization_pct" in data
    assert "discount_distribution" in data
    assert set(data["discount_distribution"].keys()) == {"0", "10", "20"}
    assert data["budget_cap"] > 0


# ── 6. GET /api/allocation/export ──────────────────────────────────────────────

def test_api_allocation_export_csv(client: TestClient):
    """GET /api/allocation/export streams a valid CSV file download."""
    res = client.get("/api/allocation/export")
    assert res.status_code == 200
    assert "text/csv" in res.headers.get("content-type", "")
    assert "attachment; filename=" in res.headers.get("content-disposition", "")

    # Parse exported content with pandas
    df = pd.read_csv(io.StringIO(res.text))
    assert len(df) > 0
    assert "customer_id" in df.columns
    assert "recommended_discount" in df.columns
    assert "predicted_purchase_probability" in df.columns
    assert "expected_net_profit" in df.columns


# ── 7. POST /api/drift/check & GET /api/drift/history ─────────────────────────

def test_api_drift_check_and_history(client: TestClient):
    """POST /api/drift/check evaluates drift and logs to GET /api/drift/history."""
    res_clean = client.post("/api/drift/check", json={"dataset_reference": "drift_reference"})
    assert res_clean.status_code == 200
    clean_data = res_clean.json()
    assert "features" in clean_data
    assert "overall_status" in clean_data

    # Check shifted cohort
    res_shifted = client.post("/api/drift/check", json={"dataset_reference": "drift_current"})
    assert res_shifted.status_code == 200
    shifted_data = res_shifted.json()
    assert shifted_data["drift_detected"] is True

    # Check history
    res_hist = client.get("/api/drift/history")
    assert res_hist.status_code == 200
    hist_data = res_hist.json()
    assert len(hist_data["history"]) >= 2
    latest = hist_data["history"][-1]
    assert "timestamp" in latest
    assert "run_id" in latest
    assert "overall_status" in latest
