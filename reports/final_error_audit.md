# EconoCasual — Final Repository Error Audit & System Verification Report

**Project:** EconoCasual – Dynamic Pricing via Double Machine Learning  
**Repository:** [https://github.com/tanmay0212-cls/EconoCasual](https://github.com/tanmay0212-cls/EconoCasual)  
**Branch:** `main`  
**Auditor:** Senior Causal ML Engineer & Software Quality Assurance Architect  
**Date:** 2026-11-03  
**Status:** **AUDITED, VERIFIED, & 100% PASSING (141/141 TESTS)**

---

## 1. Executive Summary

This report documents the exhaustive repository audit, error verification, documentation fix, and end-to-end integration validation for **EconoCasual**. All components across the causal estimation pipeline, SciPy MILP knapsack optimizer, FastAPI REST layer, React 19 analytics dashboard, automated statistical data drift surveillance, and technical documentation have been verified against the current `main` branch.

```
+---------------------------------------------------------------------------------------------------------+
|                                    AUDIT & VERIFICATION SCORECARD                                       |
+------------------------------+---------------------------+----------------------------------------------+
| Dimension                    | Result                    | Status / Notes                               |
+------------------------------+---------------------------+----------------------------------------------+
| Backend Pytest Suite         | 94 / 94 Passed (100%)     | Execution time: 25.85s (All 12 modules)      |
| Frontend Vitest Suite        | 47 / 47 Passed (100%)     | Execution time: 11.28s (All 8 test files)    |
| Production Build (Vite)      | Success (3.06s)           | Zero bundle warnings; gzip optimized         |
| End-to-End API Integration   | 7 / 7 Passed (100%)       | Validated via verify_system_integration.py   |
| API Health & Latency         | 200 OK (<270 ms)          | Multi-tier DML model pre-warmed & ready      |
| Security & Secrets Audit     | 0 Critical Leaks          | CORS restricted, path traversal blocked      |
| Milestone 3 Baseline Profit  | $90,586.38 (Unified)      | Exact $85.19 root cause resolved             |
| Financial Terminology        | Fully Reconciled          | Expected vs Realized, Overrun, Ratio         |
+------------------------------+---------------------------+----------------------------------------------+
```

---

## 2. Issues & Errors Found, Severity, and Fixes Applied

### Issue 1: Incomplete Schema Columns in End-to-End Integration Test Payload
- **File Path:** [`scripts/verify_system_integration.py`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/scripts/verify_system_integration.py#L77) (Line 77)
- **Severity:** Low (Integration Test Fixture)
- **Error Description:** The mock CSV payload used to test the multipart `/data/validate` upload endpoint omitted the required columns `"discount_level"` and `"purchase"`. Although the API successfully returned HTTP 200, it marked `is_valid: False` with schema domain errors.
- **Fix Applied:** Updated the test CSV payload to supply all required multi-tier columns (`customer_id`, `age`, `loyalty_score`, `previous_purchases`, `customer_segment`, `discount_level`, `purchase`, `product_price`, `gross_margin_rate`) with valid domain values. Added assertion `assert upload_valid is True`.

### Issue 2: False-Positive Secret Leakage in Static Audit Script
- **File Path:** [`scripts/verify_system_integration.py`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/scripts/verify_system_integration.py#L261) (Line 261)
- **Severity:** Low (Audit Script Pattern Matching)
- **Error Description:** The static security audit script searched all `.json` files in the repository for sensitive terms (e.g. `password`, `secret_key`, `api_key`). It matched its own previous benchmark artifact (`reports/system_integration_benchmark_data.json`), which recorded the audit pattern array itself, producing 20 false-positive alerts.
- **Fix Applied:** Excluded `reports` and nested development directories from the static security regex scan. Verified zero hardcoded credentials, API keys, or tokens in production code.

### Issue 3: Incomplete Pipeline Guide & Inconsistent Virtual Environment Activation in README
- **File Path:** [`README.md`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/README.md#L231) (Lines 231, 314-370)
- **Severity:** Medium (Documentation / Developer Onboarding)
- **Error Description:**
  1. The pipeline execution guide only documented legacy Week 1/2 offline scripts. It lacked instructions for starting the FastAPI server, navigating to the Swagger UI (`/docs`), running the React frontend (`npm run dev`), testing CORS connections, and running data drift checks.
  2. The distinction between dataset scales (2,500-customer sample vs 15,000-customer optimization vs 3,000-customer display limit) was not documented.
  3. The automated test count badge reflected 129 tests instead of the actual 141 tests.
- **Fix Applied:**
  1. Updated the Windows virtual environment activation command explicitly to `.\.venv\Scripts\Activate.ps1`.
  2. Documented the complete 6-stage operational pipeline (FastAPI startup, Swagger docs at `http://127.0.0.1:8000/docs`, React frontend at `http://localhost:5173`, CSV validation, CATE prediction, SciPy MILP knapsack optimization, allocation export, drift monitoring, and executive summary synthesis).
  3. Clarified the roles of the 2,500 sample dataset, 15,000 optimization dataset, and 3,000 React DOM table virtualization limit.
  4. Updated test badges and test breakdown to 94 Python tests + 47 JS tests (141 tests passing).

### Issue 4: Baseline Blanket $0 Profit Variance ($90,501.19 vs $90,586.38)
- **File Path:** [`src/optimization/profit_calculator.py`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/src/optimization/profit_calculator.py#L73-L80) (Lines 73–80)
- **Severity:** Medium (Data Consistency & Imputation Logic)
- **Error Description:** In `ProfitCalculator.calculate_customer_economics`, gross margins were unconditionally recalculated from customer segment alone ($0.49 for budget segment), discarding recorded `gross_margin_rate` ($0.495) present in `synthetic_retail_test.csv`. This caused an exact $85.1928 variance against `SciPyBudgetOptimizer.compute_unit_economics`.
- **Fix Applied:** Updated `ProfitCalculator` to check for `gross_margin_rate` in the input DataFrame before falling back to segment heuristics. Re-ran `scripts/run_policy_comparison.py`. All reports, CSV exports, and metrics artifacts now uniformly reflect the canonical value: **$90,586.38** ($15.09/customer).

### Issue 5: Financial Terminology and Causal Evaluation Labeling
- **File Path:** [`milestone_3_final_reconciliation.md`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/milestone_3_final_reconciliation.md), [`milestone_3_independent_review.md`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/milestone_3_independent_review.md), [`milestone_3_recalculated_metrics.json`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/milestone_3_recalculated_metrics.json)
- **Severity:** Medium (Methodological Rigor & Financial Reporting)
- **Error Description:**
  1. Preliminary drafts called $102,969.42 "Realized Synthetic Profit" and $4,976.96 "Realized Cost". Because these are calculated using counterfactual potential outcomes ($P^*_d$) rather than actual observed binary conversions ($Y_i \in \{0, 1\}$), calling them "realized" was inaccurate.
  2. The $2,183.17 difference between model estimates and ground truth was described as "proving model conservatism" without subgroup calibration proof.
  3. Unconstrained Oracle promotional spend ($34,669.80) was labeled as a "693.4% budget overrun".
  4. The 204.0% metric was called "GAAP ROMS".
- **Fix Applied:**
  1. Formally renamed to **Ground-Truth Expected Profit (Synthetic DGP): $102,969.42** and **Ground-Truth Expected Promotional Cost (Synthetic DGP): $4,976.96**. Preserved **Model-Expected Profit (Ex-Ante): $100,786.25**.
  2. Formally designated the benchmark as **Ground-Truth Expected Profit Gap Against the Budget-Constrained Oracle (0.986%)**.
  3. Described the $2,183.17 delta strictly as the difference between model-estimated expected profit and ground-truth expected profit under the synthetic DGP.
  4. Rephrased unconstrained spend: **693.4% of the budget (593.4% above budget)**.
  5. Renamed metric to **Incremental Net-Profit-to-Promotion-Cost Ratio (2.0405 / 204.05%)**, clarifying it as an expected liability managerial decision model, not GAAP reporting.

---

## 3. Backend Verification Results (`pytest -v`)

Executed via `.venv\Scripts\pytest.exe -v`:
- **Total Tests Collected:** 94
- **Passed:** 94
- **Failed:** 0
- **Execution Time:** 25.85 seconds

```text
tests/test_allocation_validator.py::test_valid_allocation PASSED                   [  1%]
tests/test_allocation_validator.py::test_budget_violation_caught PASSED            [  2%]
tests/test_allocation_validator.py::test_invalid_tier_caught PASSED                [  3%]
tests/test_allocation_validator.py::test_duplicate_customers_caught PASSED        [  4%]
tests/test_allocation_validator.py::test_ground_truth_leakage_caught PASSED       [  5%]
tests/test_api.py::test_health_endpoint PASSED                                     [  6%]
tests/test_api.py::test_openapi_documentation PASSED                               [  7%]
tests/test_api.py::test_validate_approved_dataset_reference PASSED                 [  8%]
tests/test_api.py::test_validate_rejects_unapproved_dataset_reference PASSED       [  9%]
tests/test_api.py::test_validate_rejects_path_traversal PASSED                     [ 10%]
tests/test_api.py::test_validate_uploaded_csv_valid PASSED                         [ 11%]
tests/test_api.py::test_validate_uploaded_csv_invalid_domains PASSED               [ 12%]
tests/test_api.py::test_predict_effects_success PASSED                             [ 13%]
tests/test_api.py::test_predict_effects_validation_error PASSED                    [ 14%]
tests/test_api.py::test_optimize_budget_with_records PASSED                        [ 15%]
tests/test_api.py::test_optimize_negative_budget_rejected PASSED                   [ 17%]
tests/test_api.py::test_optimize_missing_records_and_reference_rejected PASSED   [ 18%]
tests/test_api.py::test_get_allocation_by_run_id PASSED                            [ 19%]
tests/test_api.py::test_get_allocation_unknown_run_id_returns_404 PASSED           [ 20%]
tests/test_api.py::test_drift_check_baseline_comparison PASSED                     [ 21%]
tests/test_api.py::test_drift_check_detects_distributional_shift PASSED            [ 22%]
tests/test_api.py::test_cors_preflight_headers PASSED                              [ 23%]
tests/test_budget_optimizer.py::test_toy_example_verified_by_inspection PASSED   [ 24%]
tests/test_budget_optimizer.py::test_independent_objective_agreement PASSED       [ 25%]
tests/test_budget_optimizer.py::test_budget_monotonicity PASSED                    [ 26%]
tests/test_budget_optimizer.py::test_face_value_budget_mode PASSED                 [ 27%]
tests/test_causal_dag.py::test_dag_metadata_structure PASSED                       [ 28%]
tests/test_causal_dag.py::test_no_post_treatment_confounders PASSED                [ 29%]
tests/test_causal_dag.py::test_dag_is_strictly_acyclic PASSED                      [ 30%]
tests/test_causal_dag.py::test_dowhy_causal_model_initialization PASSED            [ 31%]
tests/test_causal_dag.py::test_dowhy_backdoor_identification PASSED                [ 32%]
tests/test_causal_dag.py::test_causal_model_rejects_corrupted_data PASSED         [ 34%]
tests/test_data_generation.py::test_reproducible_generation PASSED                 [ 35%]
tests/test_data_generation.py::test_minimum_sample_size_requirement PASSED        [ 36%]
tests/test_data_generation.py::test_required_columns_present PASSED                [ 37%]
tests/test_data_generation.py::test_no_null_values PASSED                          [ 38%]
tests/test_data_generation.py::test_binary_variable_domains PASSED                 [ 39%]
tests/test_data_generation.py::test_post_treatment_logic_consistency PASSED       [ 40%]
tests/test_data_generation.py::test_schema_validator_catches_errors PASSED       [ 41%]
tests/test_dml.py::test_dml_fit_and_effect_shapes PASSED                           [ 42%]
tests/test_dml.py::test_anti_leakage_guards PASSED                                 [ 43%]
tests/test_dml.py::test_binary_domain_validation PASSED                            [ 44%]
tests/test_dml.py::test_missing_features_validation PASSED                         [ 45%]
tests/test_dml.py::test_insufficient_sample_size_validation PASSED                 [ 46%]
tests/test_dml.py::test_model_serialization_and_reload PASSED                      [ 47%]
tests/test_dml.py::test_out_of_sample_inference_without_outcomes PASSED            [ 48%]
tests/test_dml.py::test_dml_random_seed_reproducibility PASSED                     [ 50%]
tests/test_dml.py::test_non_finite_features_validation PASSED                      [ 51%]
tests/test_dml.py::test_empty_dataframe_validation PASSED                          [ 52%]
tests/test_drift.py::test_identical_distributions PASSED                           [ 53%]
tests/test_drift.py::test_deliberately_shifted_numerical_feature PASSED            [ 54%]
tests/test_drift.py::test_changed_categorical_distribution PASSED                  [ 55%]
tests/test_drift.py::test_missing_required_columns PASSED                          [ 56%]
tests/test_drift.py::test_new_and_disappearing_categorical_values PASSED           [ 57%]
tests/test_drift.py::test_small_sample_size_handling PASSED                        [ 58%]
tests/test_drift.py::test_missing_values_and_type_integrity PASSED                 [ 59%]
tests/test_drift.py::test_out_of_range_numerical_values PASSED                     [ 60%]
tests/test_drift.py::test_methodological_notice_included PASSED                    [ 61%]
tests/test_drift.py::test_report_dict_serialization PASSED                         [ 62%]
tests/test_evaluation.py::test_ranking_table_sorting_and_percentiles PASSED       [ 63%]
tests/test_evaluation.py::test_uplift_and_qini_metrics_computation PASSED         [ 64%]
tests/test_evaluation.py::test_all_treated_or_all_control_edge_cases PASSED       [ 65%]
tests/test_evaluation.py::test_small_sample_size_edge_case PASSED                  [ 67%]
tests/test_evaluation.py::test_non_binary_values_edge_case PASSED                  [ 68%]
tests/test_evaluation.py::test_empty_dataframe_and_non_finite_effects PASSED      [ 69%]
tests/test_evaluation.py::test_tied_effects_and_custom_column_names PASSED        [ 70%]
tests/test_evaluation.py::test_plotly_figures_export PASSED                        [ 71%]
tests/test_interpreter.py::test_customer_effects_structure PASSED                  [ 72%]
tests/test_interpreter.py::test_row_to_customer_alignment PASSED                   [ 73%]
tests/test_interpreter.py::test_customer_id_quarantined_from_model PASSED         [ 74%]
tests/test_interpreter.py::test_missing_features_validation PASSED                 [ 75%]
tests/test_interpreter.py::test_null_features_validation PASSED                    [ 76%]
tests/test_interpreter.py::test_non_finite_features_and_empty_input PASSED        [ 77%]
tests/test_interpreter.py::test_configurable_threshold PASSED                      [ 78%]
tests/test_interpreter.py::test_export_results_to_csv_and_markdown PASSED       [ 79%]
tests/test_multitreatment.py::test_multitreatment_fit_and_shapes PASSED            [ 80%]
tests/test_multitreatment.py::test_predicted_probabilities_bounds PASSED           [ 81%]
tests/test_multitreatment.py::test_model_serialization_and_reload PASSED        [ 82%]
tests/test_multitreatment.py::test_anti_leakage_rejection PASSED                   [ 84%]
tests/test_multitreatment.py::test_missing_and_non_finite_features PASSED         [ 85%]
tests/test_multitreatment.py::test_unsupported_treatment_levels PASSED             [ 86%]
tests/test_multitreatment.py::test_seed_reproducibility PASSED                     [ 87%]
tests/test_multitreatment.py::test_overlap_diagnostics PASSED                      [ 88%]
tests/test_optimization.py::test_budget_constraint_strictly_satisfied PASSED      [ 89%]
tests/test_optimization.py::test_mutually_exclusive_single_action PASSED           [ 90%]
tests/test_optimization.py::test_zero_budget_edge_case PASSED                      [ 91%]
tests/test_optimization.py::test_tiny_budget_insufficient_for_all PASSED           [ 92%]
tests/test_optimization.py::test_unconstrained_large_budget PASSED                 [ 93%]
tests/test_optimization.py::test_strategy_comparison_table PASSED                  [ 94%]
tests/test_optimization.py::test_reproducibility PASSED                            [ 95%]
tests/test_optimization.py::test_invalid_parameters_raise_errors PASSED            [ 96%]
tests/test_profit_calculator.py::test_margin_rate_estimation PASSED                [ 97%]
tests/test_profit_calculator.py::test_zero_discount_cost PASSED                    [ 98%]
tests/test_profit_calculator.py::test_profit_accounting_identity PASSED            [100%]
============================== 94 passed in 25.85s ==============================
```

---

## 4. Frontend Verification Results (`npm test` & `npm run build`)

Executed in `frontend/`:
- **Vitest Test Files:** 8 passed (8)
- **Total Tests:** 47 passed (47)
- **Execution Time:** 11.28 seconds
- **Production Build:** Vite v8.3.4 build completed in 3.06s with zero errors.

```text
 ✓ src/tests/csvValidator.test.js (5 tests)
 ✓ src/tests/summaryService.test.js (15 tests)
 ✓ src/tests/Header.test.jsx (3 tests)
 ✓ src/tests/BudgetSettings.test.jsx (2 tests)
 ✓ src/tests/DatasetUploader.test.jsx (2 tests)
 ✓ src/tests/BusinessSummaryPanel.test.jsx (6 tests)
 ✓ src/tests/DataDriftViewer.test.jsx (6 tests)
 ✓ src/tests/AllocationMatrix.test.jsx (8 tests)
 Test Files  8 passed (8)
      Tests  47 passed (47)
```

**Production Bundle Footprint:**
- `dist/index.html`: 1.15 kB (gzip: 0.57 kB)
- `dist/assets/index-bUCK_CAG.css`: 36.71 kB (gzip: 6.97 kB)
- `dist/assets/index-B-5FjW_x.js`: 105.70 kB (gzip: 25.95 kB)
- `dist/assets/vendor-C4hX2lO4.js`: 243.37 kB (gzip: 77.08 kB)
- `dist/assets/plotly-Carj7g2X.js`: 4,699.99 kB (gzip: 1,432.43 kB)

---

## 5. End-to-End Workflow Verification Results

Executed via `python scripts/verify_system_integration.py` against live in-process endpoints using real data:

1. **API Health & Model Loading (`GET /health`):**
   - Latency: 266.07 ms
   - Status: `healthy`
   - Multi-Tier Model: `LinearDRLearner (EconML) + CalibratedGBM` (is_loaded: True, size: 4.84 MB)
   - Reference Data: Ready (`True`)

2. **Dataset Validation & Security Safeguards (`POST /data/validate`):**
   - Server-side reference validation (`optimization_retail_data`): 15,000 rows validated in 83.93 ms.
   - Path traversal attack (`../../etc/passwd`): Blocked with HTTP 400 Bad Request.
   - Unapproved dataset reference (`random_unapproved_name`): Blocked with HTTP 400 Bad Request.
   - Safe multipart CSV upload: Validated with HTTP 200 OK (`is_valid: True`).

3. **CATE & Purchase Probability Prediction Benchmark (`POST /predict/effects`):**
   - Batch size 10: 52.36 ms mean latency
   - Batch size 100: 46.30 ms mean latency
   - Batch size 500: 93.04 ms mean latency
   - Batch size 1,000: 121.29 ms mean latency
   - Outputs: Predicted probabilities strictly bounded in $[0, 1]$. Example `OPT_000001`: $P(0)=0.1822$, $P(10)=0.3693$, $P(20)=0.4268$.

4. **SciPy HiGHS Optimization Engine & Budget Enforcement (`POST /optimize`):**
   - Population 100: Cost = $197.64 / $5,000.00 (4.0%), Profit = $1,507.53, Solver time = 0.022s.
   - Population 500: Cost = $1,002.99 / $5,000.00 (20.1%), Profit = $8,886.21, Solver time = 0.044s.
   - Population 1,500: Cost = $3,073.57 / $5,000.00 (61.5%), Profit = $27,757.62, Solver time = 0.083s.
   - Population 3,000: Cost = $4,997.16 / $5,000.00 (99.9%), Profit = $54,657.36, Solver time = 0.111s.
   - **Zero Budget Violation:** For all populations, expenditure is strictly $\le \$5,000.00$.

5. **Allocation Retrieval by Run ID (`GET /allocation/{run_id}`):**
   - Retrieval latency: 19.76 ms for 3,000 customer records.

6. **Automated Data Drift Detection Endpoint (`POST /drift/check`):**
   - Clean reference baseline: Evaluated in 171.95 ms.
   - Shifted cohort (`data/drift_current.csv`): Flagged 7/7 drifted features (6 severe) in 80.01 ms.
   - Actionable causal recommendation generated automatically.

7. **Concurrency & Thread Safety:**
   - 30 concurrent requests across 8 worker threads completed in 616.16 ms with 100.0% success rate (All HTTP 200 OK).

8. **Security & Codebase Leak Audit:**
   - Zero hardcoded credentials, secret keys, or private tokens found.

---

## 6. Remaining Limitations & Architectural Boundaries

1. **Observational Overlap (Positivity):** In non-randomized retail campaign data, propensity scores in extreme feature tails can approach zero ($P(T=d \mid X) \approx 0$). While bounded in $[0.18, 0.82]$ in the synthetic benchmark, production real-world deployments require monitoring extreme propensities to avoid high-variance CATE extrapolation.
2. **Fixed Basket Expansion Elasticity:** Promotional basket expansion multipliers ($1.00$ for $0, $1.03$ for $10, $1.06$ for $20) are fixed economic parameters in `ProfitCalculator`. In future enterprise extensions, a secondary continuous causal estimator can be trained to personalize basket expansion.
3. **Integer Programming Scaling for Mega-Cohorts:** The SciPy HiGHS solver solves knapsack problems for cohorts up to 15,000 customers in $<0.5$ seconds. For enterprise cohorts exceeding 500,000 customers, Dantzig's greedy knapsack heuristic ($O(N \log N)$ ranking by marginal profit-to-cost ratio) should be deployed as a near-optimal fallback.

---

## 7. Exact Commands to Run the Complete Project on Windows

Open two separate PowerShell terminal windows on your Windows laptop:

### Terminal 1: Start the FastAPI Backend Server
```powershell
# Navigate to the repository root
cd c:\Users\tanma\OneDrive\Desktop\EconoCasual

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Start FastAPI with Uvicorn
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```
* **Backend API Status:** Available at `http://127.0.0.1:8000/health`
* **Interactive Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **ReDoc Documentation:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

### Terminal 2: Start the React Frontend Dashboard
```powershell
# Navigate to the frontend directory
cd c:\Users\tanma\OneDrive\Desktop\EconoCasual\frontend

# Install dependencies (if not already installed)
npm install

# Start the Vite development server
npm run dev
```
* **Frontend Web Dashboard:** [http://localhost:5173](http://localhost:5173)

---

### Terminal 3 (Optional): Run the Full Automated Test Suite
```powershell
cd c:\Users\tanma\OneDrive\Desktop\EconoCasual
.\.venv\Scripts\Activate.ps1

# 1. Run all 94 backend tests
pytest -v

# 2. Run all 47 frontend unit tests
cd frontend
npm test
cd ..

# 3. Run end-to-end integration benchmark
python scripts/verify_system_integration.py
```
*(All 141 tests will execute and pass in ~35 seconds total).*
