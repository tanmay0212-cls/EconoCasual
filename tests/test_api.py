"""Integration test suite for the EconoCausal FastAPI REST application."""

import io
import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    """Module-scoped TestClient."""
    return TestClient(app)


# ── 1. Health & Diagnostics Endpoints ─────────────────────────────────────────

def test_health_endpoint(client: TestClient):
    """GET /health must return 200, healthy status, and loaded multi-treatment model."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] in {"healthy", "degraded"}
    assert "app_version" in data
    assert "models" in data
    assert "multi_tier_dml" in data["models"]
    assert data["models"]["multi_tier_dml"]["is_loaded"] is True
    assert data["reference_data_available"] is True


def test_openapi_documentation(client: TestClient):
    """Verify OpenAPI JSON schema and Swagger docs are accessible."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "EconoCausal REST API"
    assert "/health" in schema["paths"]
    assert "/data/validate" in schema["paths"]
    assert "/predict/effects" in schema["paths"]
    assert "/optimize" in schema["paths"]
    assert "/allocation/{run_id}" in schema["paths"]
    assert "/drift/check" in schema["paths"]


# ── 2. Dataset Validation Endpoints ───────────────────────────────────────────

def test_validate_approved_dataset_reference(client: TestClient):
    """POST /data/validate with approved server-side dataset reference."""
    payload = {"dataset_name": "optimization_retail_data"}
    response = client.post("/data/validate", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["is_valid"] is True
    assert data["dataset_name"] == "optimization_retail_data"
    assert data["total_rows"] >= 15000
    assert len(data["errors"]) == 0
    assert data["summary"]["column_count"] >= 9


def test_validate_rejects_unapproved_dataset_reference(client: TestClient):
    """POST /data/validate must reject unknown dataset reference with HTTP 400."""
    payload = {"dataset_name": "malicious_or_unknown_dataset"}
    response = client.post("/data/validate", json=payload)
    assert response.status_code == 400
    assert "Invalid dataset reference" in response.json()["error"]["message"]


def test_validate_rejects_path_traversal(client: TestClient):
    """POST /data/validate must strictly reject path traversal strings."""
    payload = {"dataset_name": "../../../etc/passwd"}
    response = client.post("/data/validate", json=payload)
    assert response.status_code == 400
    assert "Invalid dataset reference" in response.json()["error"]["message"]


def test_validate_uploaded_csv_valid(client: TestClient):
    """POST /data/validate with uploaded multipart CSV file."""
    csv_content = (
        "customer_id,age,loyalty_score,previous_purchases,customer_segment,discount_level,purchase,product_price,gross_margin_rate\n"
        "CUST_001,35.0,75.0,5,vip_loyalist,10,1,90.0,0.45\n"
        "CUST_002,42.0,50.0,2,regular_buyer,0,0,85.0,0.40\n"
    )
    files = {"file": ("test_campaign.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    response = client.post("/data/validate", files=files)
    assert response.status_code == 200

    data = response.json()
    assert data["is_valid"] is True
    assert data["total_rows"] == 2
    assert len(data["errors"]) == 0


def test_validate_uploaded_csv_invalid_domains(client: TestClient):
    """POST /data/validate must flag schema violations (e.g. invalid discount_level)."""
    csv_content = (
        "customer_id,age,loyalty_score,previous_purchases,customer_segment,discount_level,purchase,product_price,gross_margin_rate\n"
        "CUST_001,15.0,150.0,5,vip_loyalist,50,1,90.0,0.45\n"  # age < 18, loyalty > 100, discount_level 50
    )
    files = {"file": ("bad_campaign.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    response = client.post("/data/validate", files=files)
    assert response.status_code == 200

    data = response.json()
    assert data["is_valid"] is False
    assert len(data["errors"]) > 0


# ── 3. Treatment-Effect Prediction Endpoints ──────────────────────────────────

def test_predict_effects_success(client: TestClient):
    """POST /predict/effects computes CATE and probabilities for valid customers."""
    payload = {
        "records": [
            {
                "customer_id": "TEST_001",
                "age": 34.0,
                "loyalty_score": 65.0,
                "previous_purchases": 4,
                "customer_segment": "regular_buyer",
                "campaign_channel": "email",
                "product_price": 85.0,
                "gross_margin_rate": 0.40,
            },
            {
                "customer_id": "TEST_002",
                "age": 28.0,
                "loyalty_score": 25.0,
                "previous_purchases": 1,
                "customer_segment": "budget_seeker",
                "campaign_channel": "push",
                "product_price": 85.0,
                "gross_margin_rate": 0.40,
            },
        ]
    }
    response = client.post("/predict/effects", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["record_count"] == 2
    assert len(data["predictions"]) == 2

    for pred in data["predictions"]:
        assert "predicted_effect_10_vs_0" in pred
        assert "predicted_effect_20_vs_0" in pred
        # Probabilities must be strictly bounded in [0, 1]
        assert 0.0 <= pred["predicted_purchase_probability_0"] <= 1.0
        assert 0.0 <= pred["predicted_purchase_probability_10"] <= 1.0
        assert 0.0 <= pred["predicted_purchase_probability_20"] <= 1.0


def test_predict_effects_validation_error(client: TestClient):
    """POST /predict/effects rejects invalid field values (e.g. age < 18) with HTTP 422."""
    payload = {
        "records": [
            {
                "customer_id": "TEST_BAD",
                "age": 10.0,  # Invalid: age < 18
                "loyalty_score": 50.0,
                "previous_purchases": 0,
                "customer_segment": "casual_shopper",
            }
        ]
    }
    response = client.post("/predict/effects", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


# ── 4. Budget Optimization Endpoints ──────────────────────────────────────────

def test_optimize_budget_with_records(client: TestClient):
    """POST /optimize solves MILP knapsack problem for customer list."""
    payload = {
        "budget": 200.0,
        "discount_tiers": [0, 10, 20],
        "objective_type": "net_profit",
        "records": [
            {
                "customer_id": f"OPT_CUST_{i:03d}",
                "age": 30.0 + (i % 20),
                "loyalty_score": 40.0 + (i % 50),
                "previous_purchases": i % 8,
                "customer_segment": "regular_buyer",
                "campaign_channel": "email",
                "product_price": 85.0,
                "gross_margin_rate": 0.40,
            }
            for i in range(15)
        ],
    }
    response = client.post("/optimize", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "run_id" in data
    assert "solver_status" in data
    assert data["kpis"]["total_customers"] == 15
    # Strict budget constraint check
    assert data["kpis"]["expected_discount_expenditure"] <= payload["budget"] + 1e-4
    assert len(data["allocation"]) == 15

    for alloc in data["allocation"]:
        assert alloc["recommended_discount"] in {0, 10, 20}
        assert alloc["expected_net_profit"] is not None


def test_optimize_negative_budget_rejected(client: TestClient):
    """POST /optimize must reject non-positive budget ceiling with HTTP 422."""
    payload = {
        "budget": -500.0,
        "discount_tiers": [0, 10, 20],
        "records": [
            {
                "customer_id": "CUST_01",
                "age": 30.0,
                "loyalty_score": 50.0,
                "previous_purchases": 2,
                "customer_segment": "regular_buyer",
            }
        ],
    }
    response = client.post("/optimize", json=payload)
    assert response.status_code == 422


def test_optimize_missing_records_and_reference_rejected(client: TestClient):
    """POST /optimize must reject request when neither records nor reference is provided."""
    payload = {
        "budget": 5000.0,
        "discount_tiers": [0, 10, 20],
    }
    response = client.post("/optimize", json=payload)
    assert response.status_code == 400


# ── 5. Allocation Retrieval Endpoints ─────────────────────────────────────────

def test_get_allocation_by_run_id(client: TestClient):
    """GET /allocation/{run_id} returns saved results of a previously executed run."""
    # 1. Execute an optimization run
    opt_payload = {
        "budget": 100.0,
        "records": [
            {
                "customer_id": f"HIST_CUST_{i}",
                "age": 35.0,
                "loyalty_score": 60.0,
                "previous_purchases": 3,
                "customer_segment": "vip_loyalist",
                "campaign_channel": "email",
                "product_price": 90.0,
                "gross_margin_rate": 0.45,
            }
            for i in range(5)
        ],
    }
    opt_resp = client.post("/optimize", json=opt_payload)
    assert opt_resp.status_code == 200
    run_id = opt_resp.json()["run_id"]

    # 2. Retrieve the run by run_id
    get_resp = client.get(f"/allocation/{run_id}")
    assert get_resp.status_code == 200
    saved_data = get_resp.json()
    assert saved_data["run_id"] == run_id
    assert saved_data["kpis"]["total_customers"] == 5


def test_get_allocation_unknown_run_id_returns_404(client: TestClient):
    """GET /allocation/{unknown_run_id} must return HTTP 404."""
    response = client.get("/allocation/non_existent_run_uuid_99999")
    assert response.status_code == 404
    assert "was not found" in response.json()["error"]["message"]


# ── 6. Data Drift Detection Endpoints ─────────────────────────────────────────

def test_drift_check_baseline_comparison(client: TestClient):
    """POST /drift/check comparing baseline to itself must report no drift."""
    payload = {"dataset_reference": "optimization_retail_data"}
    response = client.post("/drift/check", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["drift_detected"] is False
    assert data["drifted_features_count"] == 0
    assert len(data["features"]) >= 5


def test_drift_check_detects_distributional_shift(client: TestClient):
    """POST /drift/check with heavily shifted incoming records must flag drift."""
    # Drastically shifted population: age 90, loyalty 99, 45 purchases
    shifted_records = [
        {
            "customer_id": f"SHIFT_{i}",
            "age": 90.0,
            "loyalty_score": 99.0,
            "previous_purchases": 45,
            "customer_segment": "vip_loyalist",
            "campaign_channel": "push",
            "product_price": 250.0,
            "gross_margin_rate": 0.85,
        }
        for i in range(40)
    ]
    payload = {"records": shifted_records}
    response = client.post("/drift/check", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["drift_detected"] is True
    assert data["drifted_features_count"] > 0
    assert "drift detected" in data["summary_message"].lower()


# ── 7. CORS and Security Headers ──────────────────────────────────────────────

def test_cors_preflight_headers(client: TestClient):
    """OPTIONS request with allowed origin returns CORS headers."""
    headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type",
    }
    response = client.options("/predict/effects", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
