# EconoCasual REST API Reference

**Service:** EconoCasual Causal Estimation & Prescriptive Optimization Engine  
**Framework:** FastAPI 0.115+ / Uvicorn  
**Default Base URL:** `http://localhost:8000`  
**Interactive Docs:** `http://localhost:8000/docs` (Swagger UI) | `http://localhost:8000/redoc` (ReDoc)

---

## 1. Overview of Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | System health status, model availability, and baseline ready state |
| `POST` | `/data/validate` | Schema validation, type checks, and domain bounds for uploaded or referenced CSVs |
| `POST` | `/predict/effects` | Predicts causal treatment effects and counterfactual probabilities for customer records |
| `POST` | `/optimize` | Solves the SciPy MILP knapsack problem to allocate optimal discounts under budget |
| `GET` | `/allocation/{run_id}` | Retrieves full allocation policy, customer decisions, and KPIs by unique run UUID |
| `POST` | `/drift/check` | Compares customer distribution against reference baseline using two-sample KS and PSI |

---

## 2. Endpoint Specifications & Examples

### 2.1 Health Check: `GET /health`

Returns operational status, model availability, and environment metadata.

#### Example Request:
```bash
curl -X GET "http://localhost:8000/health"
```

#### Example Response (HTTP 200 OK):
```json
{
  "status": "healthy",
  "app_version": "1.0.0",
  "environment": "production",
  "models": {
    "multi_tier_dml": {
      "model_name": "multi_tier_dml",
      "estimator_class": "LinearDRLearner (EconML) + CalibratedGBM",
      "is_loaded": true,
      "version": "1.0.0",
      "treatment_levels": [0, 10, 20],
      "feature_names": ["age", "loyalty_score", "previous_purchases", "customer_segment", "campaign_channel"],
      "model_file_size_mb": 4.84
    }
  },
  "reference_data_available": true,
  "timestamp": "2026-11-03T10:00:00.000Z"
}
```

---

### 2.2 Dataset Validation: `POST /data/validate`

Validates CSV schema, data types, missing values, and domain boundaries. Clients can provide either an approved `dataset_name` or upload a multipart CSV file.

#### Security Constraint:
Arbitrary file paths and path traversal sequences (`../../`) are rejected with `HTTP 400 Bad Request`.

#### Example Request (Approved Reference):
```bash
curl -X POST "http://localhost:8000/data/validate" \
     -H "Content-Type: application/json" \
     -d '{"dataset_name": "optimization_retail_data"}'
```

#### Example Response (HTTP 200 OK):
```json
{
  "is_valid": true,
  "dataset_name": "optimization_retail_data",
  "total_rows": 15000,
  "errors": [],
  "warnings": [],
  "summary": {
    "row_count": 15000,
    "column_count": 12,
    "columns_present": ["customer_id", "age", "loyalty_score", "previous_purchases", "customer_segment", "campaign_channel", "discount_level", "purchase", "order_value", "product_price", "gross_margin_rate", "campaign_period"]
  }
}
```

---

### 2.3 Treatment-Effect Prediction: `POST /predict/effects`

Accepts customer feature records and returns predicted causal treatment effects ($\tau_{10}$, $\tau_{20}$) and bounded purchase probabilities ($P_0, P_{10}, P_{20}$).

#### Example Request:
```bash
curl -X POST "http://localhost:8000/predict/effects" \
     -H "Content-Type: application/json" \
     -d '{
       "records": [
         {
           "customer_id": "CUST_1001",
           "age": 42.0,
           "loyalty_score": 68.5,
           "previous_purchases": 5,
           "customer_segment": "regular_buyer",
           "campaign_channel": "email"
         }
       ]
     }'
```

#### Example Response (HTTP 200 OK):
```json
{
  "record_count": 1,
  "predictions": [
    {
      "customer_id": "CUST_1001",
      "predicted_effect_10_vs_0": 0.1524,
      "predicted_effect_20_vs_0": 0.2085,
      "predicted_purchase_probability_0": 0.3680,
      "predicted_purchase_probability_10": 0.5204,
      "predicted_purchase_probability_20": 0.5765
    }
  ]
}
```

---

### 2.4 Budget Optimization: `POST /optimize`

Solves the Mixed-Integer Linear Programming (MILP) knapsack problem via SciPy HiGHS to find optimal discount depths satisfying the campaign budget.

#### Example Request:
```bash
curl -X POST "http://localhost:8000/optimize" \
     -H "Content-Type: application/json" \
     -d '{
       "budget": 5000.0,
       "objective_type": "net_profit",
       "discount_tiers": [0, 10, 20],
       "dataset_reference": "optimization_retail_data"
     }'
```

#### Example Response (HTTP 200 OK):
```json
{
  "run_id": "c4183cb8-0ebd-4ce7-9490-074aceedc15e",
  "solver_status": "Optimization terminated successfully. (HiGHS Status 7: Optimal)",
  "objective_type": "net_profit",
  "solve_time_seconds": 0.081,
  "kpis": {
    "total_customers": 3000,
    "count_discount_0": 2213,
    "count_discount_10": 775,
    "count_discount_20": 12,
    "expected_revenue": 121840.50,
    "expected_discount_expenditure": 4327.86,
    "expected_net_profit": 53855.38,
    "budget_cap": 5000.0,
    "remaining_budget": 672.14,
    "budget_utilization_pct": 86.6
  },
  "allocation": [
    {
      "customer_id": "OPT_000001",
      "recommended_discount": 10,
      "predicted_purchase_probability": 0.3693,
      "expected_revenue": 27.70,
      "expected_discount_cost": 3.69,
      "expected_net_profit": 7.39
    }
  ]
}
```

---

### 2.5 Allocation Retrieval: `GET /allocation/{run_id}`

Retrieves previously persisted optimization run decisions and KPIs.

#### Example Request:
```bash
curl -X GET "http://localhost:8000/allocation/c4183cb8-0ebd-4ce7-9490-074aceedc15e"
```

---

### 2.6 Data Drift Check: `POST /drift/check`

Performs continuous statistical surveillance over incoming customer batches compared against the baseline reference population.

#### Example Request:
```bash
curl -X POST "http://localhost:8000/drift/check" \
     -H "Content-Type: application/json" \
     -d '{"dataset_reference": "drift_current"}'
```

#### Example Response (HTTP 200 OK):
```json
{
  "overall_status": "Severe or widespread drift detected",
  "drift_detected": true,
  "total_features_evaluated": 7,
  "drifted_features_count": 4,
  "severe_drift_count": 4,
  "reference_sample_size": 15000,
  "current_sample_size": 3500,
  "summary_message": "4 of 7 features exhibit statistically significant drift (4 severe).",
  "recommended_next_action": "Review marketing channel acquisition shifts. Inspect common support overlap.",
  "methodological_notice": "Methodological Notice: Covariate drift (P(X)) measures shift in customer features. It does NOT automatically imply concept drift (P(Y | X, T)) or invalidity of causal treatment effects."
}
```

