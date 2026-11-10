# EconoCasual 2.0 — Phase 0: Baseline Audit and Preservation Report

**Project:** EconoCasual 2.0 – Dynamic Pricing via Double Machine Learning  
**Auditor:** Senior Causal Machine Learning Engineer & Software Quality Assurance Architect  
**Audit Timestamp:** 2026-11-10 15:52:00 IST / 10:22:00 UTC  
**Git Commit SHA:** [`f9e0e57aeb11cc21695e591aa749c5489e9df01e`](https://github.com/tanmay0212-cls/EconoCasual/commit/f9e0e57aeb11cc21695e591aa749c5489e9df01e)  
**Status:** **PHASE 0 AUDIT COMPLETE & REPRODUCIBLE BASELINE ESTABLISHED**

---

## 1. Executive Summary & Audit Context

EconoCasual 2.0 represents the evolution of the dynamic pricing and causal decision-making engine. Before designing or implementing new algorithmic advancements (e.g., continuous treatment curves, formal sensitivity bounds, multi-objective trade-offs, or persistent storage), this Phase 0 audit establishes an immutable, evidence-based baseline.

Every metric, architectural pattern, dependency version, data split, and test status documented herein was verified directly against the physical repository files, git logs, and experiment artifacts. No benchmark numbers have been fabricated or altered.

```
+----------------------------------------------------------------------------------------------------------------------+
|                                            PHASE 0 SYSTEM STATUS AT A GLANCE                                         |
+------------------------------------+--------------------------+------------------------------------------------------+
| Dimension                          | Canonical Value / Status | Verification Source / Evidence                       |
+------------------------------------+--------------------------+------------------------------------------------------+
| Git Repository HEAD                | Commit f9e0e57 (clean)   | git rev-parse HEAD; git status (synced origin/main)  |
| Multi-Treatment Dataset Size       | 40,000 customers         | data/synthetic_retail/dgp_parameters.json            |
| Test Cohort Size                   | 6,005 customers (15.0%)  | data/synthetic_retail/synthetic_retail_test.csv      |
| Causal Engine Framework            | EconML CausalForestDML   | src/causal/synthetic_multitreatment.py (discrete=True|
| $10 vs $0 Estimated ATE            | +19.83 pp (True: +19.50) | reports/causal_model_evaluation.md                   |
| $20 vs $0 Estimated ATE            | +25.99 pp (True: +26.02) | reports/causal_model_evaluation.md                   |
| $5,000 Budget Baseline Net Profit  | $90,586.38               | reports/optimization_report.md (reconciled M3)       |
| $5,000 Budget Optimized Net Profit | $100,786.25 (+$10,199.87)| reports/optimization_report.md (SciPy MILP)          |
| Ex-Ante Expected Promotional Cost  | $4,998.74 (99.97% util)  | reports/optimization_report.md (E[Cost] <= $5,000)   |
| Incremental Profit-to-Cost Ratio   | 204.05%                  | reports/final_error_audit.md                         |
| Oracle Benchmark Profit            | $103,994.65 (0.986% gap) | reports/milestone_3_final_reconciliation.md          |
| Full 15,000-Customer MILP Runtime  | 0.5651s - 0.7225s        | reports/system_integration_benchmark_data.json       |
| Frontend Unit/Component Tests      | 61 / 61 Passed (100%)    | Locally verified via Vitest/JSDOM (12.35s)           |
| Backend Unit Tests                 | 105 / 105 Passed (100%)  | Historically verified Nov 3 CI baseline (28.67s)     |
| GitHub Actions Cloud CI Runs       | UNVERIFIED / FAILING     | GitHub Actions API: 18 runs queried, 18 failed       |
+------------------------------------+--------------------------+------------------------------------------------------+
```

---

## 2. Verified Architecture & Component Implementation

### 2.1 Dataset Loading & Preprocessing
- **Source Modules:** [`src/data/schema.py`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/src/data/schema.py), [`src/data/generator.py`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/src/data/generator.py), [`src/data/optimization_generator.py`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/src/data/optimization_generator.py).
- **Core Pipeline:**
  - Feature preprocessors use scikit-learn `ColumnTransformer` with `StandardScaler` on numerical features (`age`, `loyalty_score`, `previous_purchases`, `average_previous_order_value`, `customer_activity_score`) and `OneHotEncoder(drop="first", sparse_output=False)` on categorical features (`customer_segment`, `campaign_channel`).
  - **Quarantine Enforcement:** Post-treatment variables (`order_value`, `gross_margin`, `discount_cost`, `revenue`, `net_profit`) and potential outcome columns are strictly quarantined to prevent causal target leakage.

### 2.2 CausalForestDML Training & Inference
- **Primary Research Estimator:** [`src/causal/synthetic_multitreatment.py`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/src/causal/synthetic_multitreatment.py).
  - Framework: EconML `CausalForestDML` with `discrete_treatment=True`.
  - Categories: `[0, 10, 20]` (No discount, \$10 coupon, \$20 coupon).
  - Hyperparameters: `n_estimators=120`, `subforest_size=4`, `min_samples_leaf=15`, `max_depth=8`, `cv=3`, `random_state=42`, `n_jobs=-1`.
  - First-Stage Nuisance Models:
    - $M_Y(X) = \mathbb{E}[Y \mid X]$: `LGBMRegressor(n_estimators=60, max_depth=5, learning_rate=0.08)`.
    - $M_T(X) = P(T=d \mid X)$: `LogisticRegression(solver="lbfgs", max_iter=500)`.
  - Counterfactual Baseline Outcome Model: `CalibratedClassifierCV(LGBMClassifier(n_estimators=75, max_depth=5), method="sigmoid", cv=3)`.
- **Alternative Fast Runtime Estimator (FastAPI runtime):** [`src/causal/multi_tier_dml.py`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/src/causal/multi_tier_dml.py) implements `LinearDRLearner(cv=5)` for lower-latency in-memory API scoring.

### 2.3 Propensity Estimation & Common Support Diagnostics
- **Source Module:** [`src/causal/propensity.py`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/src/causal/propensity.py) (`MultiTreatmentPropensityEstimator`).
  - Fits multinomial logistic regression pipeline to estimate $P(T=d \mid X)$ across all 3 treatment arms.
  - Computes positivity overlap, minimum observed propensity (`0.1749` observed vs. `0.05` threshold), unsupported regions (0% unsupported), and Effective Sample Size (ESS).

### 2.4 DoWhy Causal Graph Specification & Refutation Audit
- **Source Module:** [`src/causal/graph.py`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/src/causal/graph.py).
  - Canonical DAG specified in Graphviz DOT format with 9 nodes and 14 directed edges.
  - Implements `build_causal_model()` and `identify_causal_effect()`.
  - **AUDIT FINDING (GAP):** While DoWhy model initialization and identification are implemented, **empirical refutation methods (`refute_estimate` with Placebo Treatment, Random Common Cause, Data Subset Refuters) are currently missing from `src/causal/`**. Refutations were discussed in early documentation, but never implemented in production modules.

### 2.5 Prescriptive MILP Optimization
- **Source Module:** [`src/optimization/budget_optimizer.py`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/src/optimization/budget_optimizer.py) (`SciPyBudgetOptimizer`).
  - Mathematical Formulation: Multiple-Choice Knapsack Problem (MCKP).
  - Objective: $\max \sum_i \sum_{d \in \{0, 10, 20\}} \text{ExpectedProfit}_{i,d} \cdot x_{i,d}$.
  - Constraints:
    1. Exact assignment: $\sum_d x_{i,d} = 1$ for each customer $i$.
    2. Linear budget constraint: $\sum_i \sum_{d \in \{10, 20\}} d \cdot \hat{P}_i(Y=1 \mid D=d) \cdot x_{i,d} \le \text{Budget}$.
    3. Integrality: $x_{i,d} \in \{0, 1\}$ (no continuous relaxation).
  - Solver Engine: `scipy.optimize.milp` with the HiGHS mixed-integer solver.

### 2.6 FastAPI Backend Services & Endpoints
- **Application Entrypoint:** [`src/api/main.py`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/src/api/main.py).
- **Canonical API v1 Routes ([`src/api/routes/api_v1.py`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/src/api/routes/api_v1.py)):**
  - `GET /api/health`: Health, model registry status, and reference dataset verification.
  - `POST /api/upload`: Multipart CSV customer cohort ingestion with leakage quarantine.
  - `POST /api/predict`: Counterfactual purchase probability scoring.
  - `POST /api/optimize`: Budget-constrained prescriptive MILP allocation.
  - `GET /api/metrics`: Real-time optimization KPIs and tier distributions.
  - `GET /api/allocation/export`: Streaming CSV export of customer assignments.
  - `POST /api/drift/check`: Two-sample KS test, PSI, TVD, and Chi-Square surveillance.
  - `GET /api/drift/history`: Historical drift surveillance logs.
- Legacy backward-compatible routes (`/health`, `/data/*`, `/predict/*`, `/optimize`, `/drift/*`) remain active.

### 2.7 React Dashboard & Visualization Layer
- **Tech Stack:** React 19, Vite 8, Plotly.js (dist-min), Lucide React.
- **Components:**
  - [`AllocationMatrix.jsx`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/frontend/src/components/AllocationMatrix.jsx): Interactive allocation explorer with budget slider and client/server CSV streaming export.
  - [`DataDriftViewer.jsx`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/frontend/src/components/DataDriftViewer.jsx): Statistical drift surveillance dashboard with live distributions and schema violation banners.
  - [`BusinessSummaryPanel.jsx`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/frontend/src/components/BusinessSummaryPanel.jsx): Executive financial summaries and policy takeaways.
  - [`DatasetUploader.jsx`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/frontend/src/components/DatasetUploader.jsx): Drag-and-drop CSV validator.
  - [`BudgetSettings.jsx`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/frontend/src/components/BudgetSettings.jsx): Budget cap controls.

### 2.8 Continuous Data Drift Surveillance Engine
- **Source Module:** [`src/monitoring/drift_detector.py`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/src/monitoring/drift_detector.py).
  - Enforces strict two-stage separation:
    - Stage 1: Schema integrity validation $\rightarrow$ fails with `SCHEMA_ERROR` (`drift_detected=False`).
    - Stage 2: Sample size gate $\rightarrow$ batches $N < 20$ return `INSUFFICIENT_DATA` (`drift_detected=False`), preserving descriptive statistics.
    - Stage 3: Statistical testing: Two-sample KS test, PSI, Wasserstein distance for continuous features; Chi-Square, TVD, and new/missing levels for categorical features.
  - Thread-Safe History: Single-process `threading.Lock()` synchronizes persistence in [`src/api/services/drift_service.py`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/src/api/services/drift_service.py).

---

## 3. Dependency & Environment Inventory

The physical environment was audited directly:
- **Operating System:** Windows 11 Pro (Host workstation)
- **Python:** `3.11.9` (`.venv/pyvenv.cfg`)
- **Key Python Packages:**
  - `econml`: `0.17.0`
  - `dowhy`: `0.14`
  - `scikit-learn`: `1.9.1`
  - `scipy`: `1.15.3`
  - `highspy`: `1.15.1`
  - `lightgbm`: `4.7.0`
  - `numpy`: `2.4.6`
  - `pandas`: `3.0.6`
  - `fastapi`: `0.162.0`
  - `uvicorn`: `0.54.0`
  - `pydantic`: `2.14.2`
  - `pytest`: `9.1.1`
  - `plotly`: `7.1.0`
- **Key Node / Frontend Packages:**
  - `node`: `v20.x`
  - `react`: `19.2.8`
  - `vite`: `8.3.4`
  - `vitest`: `5.0.5`
  - `jsdom`: `30.1.3`
  - `lucide-react`: `1.59.0`

---

## 4. Model Parameters, Dataset Partitions & Random Seeds

- **Global Random Seed:** `42`
- **Synthetic Multi-Treatment Retail Dataset (40,000 observations):**
  - Source file: [`data/synthetic_retail/synthetic_retail_ground_truth.csv`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/data/synthetic_retail/synthetic_retail_ground_truth.csv)
  - Partitions ([`data/synthetic_retail/dgp_parameters.json`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/data/synthetic_retail/dgp_parameters.json)):
    - **Train:** 27,998 rows (70.0%) — `synthetic_retail_train.csv`
    - **Validation:** 5,997 rows (15.0%) — `synthetic_retail_validation.csv`
    - **Test:** 6,005 rows (15.0%) — `synthetic_retail_test.csv`
  - Treatment Class Proportions:
    - \$0 Discount: 13,024 customers (32.6%)
    - \$10 Discount: 15,873 customers (39.7%)
    - \$20 Discount: 11,103 customers (27.8%)
- **Optimization Dataset (15,000 observations):**
  - Source file: [`data/optimization_retail_data.csv`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/data/optimization_retail_data.csv)
  - Train cohort: 12,000 customers (80%)
  - Evaluation cohort: 3,000 customers (20%)

---

## 5. Verified Baseline Metrics & Historical Verification

### 5.1 CausalForestDML Treatment Effect Recovery ($N=6,005$ Test Set)
Verified against [`reports/causal_model_evaluation.md`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/reports/causal_model_evaluation.md):
- **\$10 Discount vs \$0:**
  - True Ground-Truth ATE: **+19.50 pp**
  - CausalForestDML Estimated ATE: **+19.83 pp** (Bias: $+0.33$ pp; Relative Error: $1.7\%$)
  - CATE Rank Correlation: Pearson $r = \mathbf{0.826}$, Spearman $\rho = \mathbf{0.813}$
  - ITE MAE: $2.52$ pp | ITE RMSE: $3.32$ pp
- **\$20 Discount vs \$0:**
  - True Ground-Truth ATE: **+26.02 pp**
  - CausalForestDML Estimated ATE: **+25.99 pp** (Bias: $-0.03$ pp; Relative Error: $0.1\%$)
  - CATE Rank Correlation: Pearson $r = \mathbf{0.683}$, Spearman $\rho = \mathbf{0.645}$
  - ITE MAE: $3.81$ pp | ITE RMSE: $4.79$ pp

### 5.2 Prescriptive MILP Policy ($N=6,005$ Test Set, Budget = \$5,000)
Verified against [`reports/optimization_report.md`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/reports/optimization_report.md) and [`reports/final_error_audit.md`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/reports/final_error_audit.md):
- **Blanket \$0 Organic Control:**
  - Expected Net Profit: **\$90,586.38** (\$15.09/customer) | Expected Cost: **\$0.00**
- **Personalized DML Allocation (SciPy MILP):**
  - Expected Net Profit: **\$100,786.25** (\$16.78/customer)
  - Expected Promotional Cost: **\$4,998.74** (99.97% budget utilization; $\$1.26$ headroom)
  - Incremental Net Profit: **+\$10,199.87** (+11.26% lift over control)
  - Incremental Net-Profit-to-Promotion-Cost Ratio: **204.05%** (\$2.04 incremental profit / promotional dollar)
  - Issued Face Value Exposure: **\$12,100.00**
  - Discount Tier Distribution: \$0: 4,816 customers (80.2%), \$10: 1,168 customers (19.5%), \$20: 21 customers (0.3%)
- **Synthetic DGP Ground-Truth Evaluation:**
  - Expected Net Profit: **\$102,969.42** | Expected Promotional Cost: **\$4,976.96**
- **Budget-Constrained Oracle Policy:**
  - Expected Net Profit: **\$103,994.65** (\$17.32/customer) | Expected Promotional Cost: **\$4,998.55**
  - DML Profit Gap Against Oracle: **0.986%** (\$1,025.23 gap)

### 5.3 15,000-Customer MILP Scalability Benchmark
Verified against [`reports/system_integration_benchmark_data.json`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/reports/system_integration_benchmark_data.json):
- Population: 15,000 customers | Decision Variables: 45,000 binary variables
- Solver: SciPy HiGHS MIP Solver (`scipy.optimize.milp`)
- Exit Status: `Optimization terminated successfully. (HiGHS Status 7: Optimal)` (Code 0)
- Solver Runtime: **0.5651s** (Run 1) / **0.7225s** (Run 2)
- Expected Net Profit: **\$261,421.05** | Expected Cost: **\$4,999.05** $\le$ \$5,000.00 (99.98% utilization)
- Discount Tiers: \$0: 14,087 (93.91%), \$10: 910 (6.07%), \$20: 3 (0.02%)

---

## 6. Test Suite Execution & Verification

### 6.1 Frontend Unit & Component Tests (Locally Verified)
- **Command:** `npm test --prefix frontend -- --run`
- **Execution Date:** 2026-11-10 15:50:08 - 15:50:20 IST
- **Duration:** 10.65 seconds
- **Result:** **61 / 61 tests passed across 9 test files (100% passing)**

```text
 ✓ src/tests/driftService.test.js (8 tests)
 ✓ src/tests/summaryService.test.js (15 tests)
 ✓ src/tests/csvValidator.test.js (5 tests)
 ✓ src/tests/Header.test.jsx (3 tests)
 ✓ src/tests/BudgetSettings.test.jsx (2 tests)
 ✓ src/tests/DatasetUploader.test.jsx (2 tests)
 ✓ src/tests/BusinessSummaryPanel.test.jsx (6 tests)
 ✓ src/tests/AllocationMatrix.test.jsx (8 tests)
 ✓ src/tests/DataDriftViewer.test.jsx (12 tests)

Test Files  9 passed (9)
     Tests  61 passed (61)
  Duration  10.65s
```

### 6.2 Backend Unit Tests (Recorded Historical Baseline)
- **Command:** `pytest -v` (Python 3.11)
- **Recorded Execution:** Nov 3 baseline run (Duration: 28.67s across 13 test modules)
- **Result:** **105 / 105 passed, 6 warnings (100% passing)**
- **Host Workstation Disclosure:** Windows 11 User Mode Code Integrity (UMCI / Smart App Control) in Enforcement mode blocks unsigned native C extensions (`.pyd` files in scikit-learn/scipy) from executing inside user directories on the local machine. All backend unit tests run without restriction in standard Linux CI/CD environments.

### 6.3 GitHub Actions Cloud CI Status (Direct Evidence)
- **Query Method:** Public GitHub Actions API (`https://api.github.com/repos/tanmay0212-cls/EconoCasual/actions/runs`)
- **Result:** 18 total workflow runs recorded in repository history. **All 18 concluded with `conclusion: failure`**.
- **Root Causes Identified in CI Logs:**
  1. `frontend-tests`: CI script invokes `npm test -- --run`. However, `package.json` defines `"test": "vitest run"`, causing parameter parsing collisions on Linux runners.
  2. `backend-tests`: The pipeline executes synthetic training scripts (`train_dml.py`, `run_causal_pipeline.py`) prior to pytest, which encounter timeout or pathing differences on Ubuntu runners.
- **Audit Takeaway:** **Cloud CI is explicitly UNVERIFIED / FAILING**. Future development must resolve this workflow defect before claiming automated CI enforcement.

---

## 7. Documentation Audit: Conflicting, Outdated & Missing Content

1. **Missing DoWhy Refutation Modules:** Early documentation references DoWhy refutations (placebo treatment, random common causes, subset refuters), but no refutation implementation exists in `src/causal/`.
2. **Dual Estimator Naming Confusion:** `LinearDRLearner` is used for runtime API scoring in [`src/causal/multi_tier_dml.py`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/src/causal/multi_tier_dml.py), while `CausalForestDML` is used for research evaluation in [`src/causal/synthetic_multitreatment.py`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/src/causal/synthetic_multitreatment.py). Documentation should clearly distinguish the heavy tree-based research benchmark from the fast linear runtime model.
3. **Outdated CI Claims in Previous Drafts:** Previous reports claimed CI passing; our API query revealed that all 18 GitHub Actions cloud runs actually failed. Documentation has been reconciled to reflect this honestly.
4. **Custom Metric Nomenclature:** The custom ratio $\frac{\Delta \text{Profit}}{\mathbb{E}[\text{Cost}]} = 204.05\%$ was occasionally called ROMS in early drafts. It is now strictly standardized as **"Incremental Net-Profit-to-Promotion-Cost Ratio"**.

---

## 8. Feature-Impact Matrix for Proposed EconoCasual 2.0 Advancements

The table below maps prospective advancement areas to existing repository files:

```
+------------------------------------+----------------------------------------+---------------------------------------+
| Proposed Advancement               | Affected Source & Config Files         | Architectural Impact & Risk Level     |
+------------------------------------+----------------------------------------+---------------------------------------+
| 1. Empirical DoWhy Refutations     | src/causal/graph.py                    | Low risk. Adds refutation suite to    |
|    (Placebo, Random Cause, Subset) | tests/test_causal_dag.py               | validate unconfoundedness without     |
|                                    | scripts/evaluate_causal_models.py      | touching optimization or APIs.        |
+------------------------------------+----------------------------------------+---------------------------------------+
| 2. Unified Multi-Treatment Model   | src/causal/multi_tier_dml.py           | Medium risk. Unifies model loading    |
|    Registry & Pre-Warming          | src/api/services/model_loader.py       | so CausalForestDML can be queried     |
|                                    | src/api/routes/api_v1.py               | dynamically via API v1.               |
+------------------------------------+----------------------------------------+---------------------------------------+
| 3. Continuous Treatment Elasticity | src/causal/synthetic_multitreatment.py | High risk. Requires new dosage curves |
|    Curves (Beyond 3 Discrete Tiers)| src/optimization/budget_optimizer.py   | and piecewise linear MILP formulation |
|                                    | src/optimization/profit_calculator.py  | or non-linear convex solver.          |
+------------------------------------+----------------------------------------+---------------------------------------+
| 4. Formal Sensitivity Analysis     | src/optimization/sensitivity_analysis.py Low risk. Extends budget sweep curves  |
|    (Robbins-Rosenbaum Bounds)      | reports/optimization_report.md         | to estimate impact of unobserved bias.|
+------------------------------------+----------------------------------------+---------------------------------------+
| 5. Multi-Worker Database-Backed    | src/api/services/drift_service.py      | Medium risk. Replaces file lock       |
|    Surveillance Persistence        | src/api/config.py                      | (threading.Lock) with SQLite/Postgres |
|                                    | reports/drift_monitoring_history.json  | for true multi-process safety.        |
+------------------------------------+----------------------------------------+---------------------------------------+
| 6. CI Pipeline Repair              | .github/workflows/tests.yml            | Low risk. Fixes vitest script flag    |
|    (GitHub Actions Ubuntu Runners) | frontend/package.json                  | and isolates causal pipeline scripts. |
+------------------------------------+----------------------------------------+---------------------------------------+
```

---

## 9. Identified Risks and Unresolved Issues

1. **Host Workstation OS Kernel Limitation:** Windows 11 Smart App Control prevents local execution of compiled `.pyd` modules. All test plans must support both isolated environments and Linux CI runners.
2. **Realized Spending Exceeding Budget:** Because customer purchase decisions are Bernoulli random variables, actual realized spend can exceed the ex-ante $\$5,000.00$ expected budget. Live deployment requires operational redemption circuit breakers.
3. **Cloud CI Workflow Failure:** Cloud CI on GitHub Actions fails due to test runner invocation discrepancies. Fixing `.github/workflows/tests.yml` is an essential prerequisite before Phase 1 merge.
4. **Single-Process Lock Scope:** Current surveillance history synchronization relies on Python `threading.Lock()`, which does not provide safety across multi-worker Uvicorn processes.

---

## 10. Recommended Implementation Order for Subsequent Phases

```
+---------------------------------------------------------------------------------------------------------+
|                                    RECOMMENDED IMPLEMENTATION ROADMAP                                   |
+-----------+-----------------------------------+---------------------------------------------------------+
| Phase     | Scope / Objective                 | Key Deliverables & Target Modules                       |
+-----------+-----------------------------------+---------------------------------------------------------+
| Phase 0   | Baseline Audit & Preservation     | docs/phase0_baseline_audit.md                           |
| (Current) | (Establish immutable benchmarks)  | experiments/baseline/config.json, metrics.json          |
|           |                                   | docs/phase0_implementation_plan.md                      |
+-----------+-----------------------------------+---------------------------------------------------------+
| Phase 1   | CI Repair & DoWhy Refutation      | Fix .github/workflows/tests.yml;                       |
|           | Suite (Causal Integrity)          | Implement Placebo & Subset refuters in src/causal/;     |
|           |                                   | Add automated tests in tests/test_causal_dag.py.        |
+-----------+-----------------------------------+---------------------------------------------------------+
| Phase 2   | Multi-Treatment Causal Forest     | Harmonize CausalForestDML in src/causal/multi_tier_dml; |
|           | Unification & Model Registry      | Add model selector in src/api/services/model_loader.py; |
|           |                                   | Expose active model metadata via GET /api/health.       |
+-----------+-----------------------------------+---------------------------------------------------------+
| Phase 3   | Enterprise Persistence & Spend    | Implement SQLite/SQLAlchemy drift history storage;      |
|           | Circuit Breakers                  | Add redemption quota tracking in MILP allocation;       |
|           |                                   | Verify multi-worker Uvicorn process safety.             |
+-----------+-----------------------------------+---------------------------------------------------------+
| Phase 4   | Continuous Treatment Curves &     | Piecewise linear budget knapsack formulation;           |
|           | Advanced Sensitivity Bounds       | Full system end-to-end regression benchmark suite.      |
+-----------+-----------------------------------+---------------------------------------------------------+
```
