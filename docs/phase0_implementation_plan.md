# EconoCasual 2.0 — Phase 0: Implementation Roadmap & Execution Plan

**Project:** EconoCasual 2.0 – Dynamic Pricing via Double Machine Learning  
**Author:** Senior Causal Machine Learning Engineer & Python Architect  
**Date:** 2026-11-10 15:52:00 IST / 10:22:00 UTC  
**Baseline Commit:** [`f9e0e57aeb11cc21695e591aa749c5489e9df01e`](https://github.com/tanmay0212-cls/EconoCasual/commit/f9e0e57aeb11cc21695e591aa749c5489e9df01e)  
**Status:** **APPROVED FOR SUBSEQUENT EXECUTION (WAITING FOR USER APPROVAL)**

---

## 1. Architectural Principles & Non-Negotiable Invariants

Before embarking on EconoCasual 2.0 enhancements, all architectural modifications must strictly obey the following invariants:

1. **Strict Preservation of Canonical Metrics:**
   - Baseline Expected Profit ($0 allocation): **$90,586.38**
   - Optimized Expected Profit (SciPy MILP): **$100,786.25**
   - Incremental Expected Profit: **$10,199.87**
   - Ex-Ante Expected Promotional Cost: **$4,998.74**
   - Marketing Budget Ceiling: **$5,000.00**
   - Incremental Net-Profit-to-Promotion-Cost Ratio: **204.05%**
   - Budget-Constrained Oracle Ground Truth: **$103,994.65** (0.986% gap)
2. **Backward-Compatible API Contracts:**
   - Existing FastAPI endpoints under `/api` and legacy aliases (`/health`, `/optimize`, `/predict/effects`, `/drift/check`) must maintain 100% schema backward-compatibility.
3. **No Unjustified Retraining:**
   - The verified model artifacts ([`models/multitreatment_model.joblib`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/models/multitreatment_model.joblib) and [`models/dml_model.joblib`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/models/dml_model.joblib)) must remain deployable without forced retraining.
4. **Verifiable Quality & Evidence-First Reporting:**
   - No test, CI result, or performance metric may be claimed without executable command evidence.

---

## 2. Identified Deficiencies from Phase 0 Baseline Audit

The baseline audit identified five concrete structural and methodological gaps in the current repository:

```
+----+-----------------------------------------+------------------------------------------------------------------------+
| #  | Gap Identified in Baseline Audit        | Description & Root Cause                                               |
+----+-----------------------------------------+------------------------------------------------------------------------+
| 1  | Cloud CI Failures in GitHub Actions     | All 18 GitHub Actions runs failed due to parameter collisions          |
|    |                                         | ('npm test -- --run' vs 'vitest run') and training script timeouts.    |
| 2  | Missing Empirical DoWhy Refutations     | DoWhy DAG initialization exists in src/causal/graph.py, but empirical  |
|    |                                         | refutations (Placebo, Random Cause, Subset) were never coded.          |
| 3  | Dual Estimator Dichotomy                | CausalForestDML is used in research evaluation, but LinearDRLearner is |
|    |                                         | hardcoded in runtime API services without runtime model switching.     |
| 4  | Single-Process Lock Scope               | drift_service.py uses threading.Lock(), which fails to protect against |
|    |                                         | concurrent write collisions across multi-worker Uvicorn processes.    |
| 5  | Ex-Ante Expectation vs Realized Budget  | Prescriptive MILP guarantees E[Cost] <= $5k; stochastic customer       |
|    |                                         | redemption draws can exceed cash budget without operational cutoff.   |
+----+-----------------------------------------+------------------------------------------------------------------------+
```

---

## 3. Phased Implementation Roadmap

### Phase 1: CI Pipeline Remediation & DoWhy Empirical Refutation Suite

- **Objective:** Fix the broken GitHub Actions CI pipeline and complete the formal causal identification refutation suite.
- **Key Tasks:**
  1. Fix `.github/workflows/tests.yml`:
     - Update frontend step to `npm run test` or `npx vitest run`.
     - Decouple model training from unit test execution; ensure `pytest -v` runs directly against existing serialized artifacts.
  2. Implement `src/causal/refutation.py`:
     - Add `refute_placebo_treatment()`: Replaces discount offer with random placebo; asserts estimated effect drops to near zero ($p > 0.05$).
     - Add `refute_random_common_cause()`: Injects synthetic confounder; asserts original CATE estimate does not deviate significantly.
     - Add `refute_data_subset()`: Evaluates robustness across random 80% subsamples.
  3. Add unit test coverage in `tests/test_causal_dag.py`.
- **Target Files:**
  - `.github/workflows/tests.yml`
  - `src/causal/refutation.py` (New module)
  - `tests/test_causal_dag.py`
  - `docs/causal_refutation_report.md`

---

### Phase 2: CausalForestDML & Model Registry Harmonization

- **Objective:** Enable runtime selection between fast linear estimators (`LinearDRLearner`) and high-fidelity causal forests (`CausalForestDML`).
- **Key Tasks:**
  1. Refactor [`src/api/services/model_loader.py`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/src/api/services/model_loader.py):
     - Support loading both `LinearDRLearner` and `SyntheticMultiTreatmentDML` (CausalForestDML).
     - Provide runtime model selection via configuration setting (`settings.active_causal_model`).
  2. Update [`src/api/routes/api_v1.py`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/src/api/routes/api_v1.py):
     - Expose active estimator class and hyperparameters in `GET /api/health`.
     - Allow optional `model_name` query parameter in `POST /api/predict`.
  3. Verify latency and throughput benchmarks across both estimators.
- **Target Files:**
  - `src/api/config.py`
  - `src/api/services/model_loader.py`
  - `src/api/routes/api_v1.py`
  - `tests/test_api_v1.py`

---

### Phase 3: Enterprise Persistence & Operational Spend Circuit Breakers

- **Objective:** Upgrade drift surveillance persistence to support multi-process workers and introduce cash-outflow protection.
- **Key Tasks:**
  1. Upgrade [`src/api/services/drift_service.py`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/src/api/services/drift_service.py):
     - Implement database-backed history logging using SQLite (via SQLAlchemy) with connection pooling and atomic transactions.
     - Maintain JSON export as an offline fallback.
  2. Implement Operational Spend Safety in [`src/optimization/budget_optimizer.py`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/src/optimization/budget_optimizer.py):
     - Add `max_realized_risk_budget`: Calculate binomial confidence upper bound on realized spend:
       $$\text{UpperSpend}_{99\%} = \mathbb{E}[\text{Cost}] + 2.33 \cdot \sqrt{\sum_i x_{i,d}^2 \cdot d^2 \cdot P_i (1 - P_i)}$$
     - Provide conservative allocation mode ensuring $\text{UpperSpend}_{99\%} \le B_{\text{cap}}$.
  3. Update React Dashboard [`BudgetSettings.jsx`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/frontend/src/components/BudgetSettings.jsx) and [`AllocationMatrix.jsx`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/frontend/src/components/AllocationMatrix.jsx) to display risk-adjusted expected spend.
- **Target Files:**
  - `src/api/services/drift_service.py`
  - `src/optimization/budget_optimizer.py`
  - `src/api/schemas/optimize.py`
  - `frontend/src/components/AllocationMatrix.jsx`
  - `frontend/src/components/BudgetSettings.jsx`
  - `tests/test_budget_optimizer.py`

---

### Phase 4: Continuous Treatment Curves & Sensitivity Analysis

- **Objective:** Extend prescriptive optimization to continuous discount percentages and formal sensitivity bounds.
- **Key Tasks:**
  1. Implement continuous dosage-response curves: $P(\text{purchase} \mid X, d)$ for $d \in [0, 30]$.
  2. Formulate piecewise linear MILP approximations for continuous knapsack optimization.
  3. Implement Rosenbaum sensitivity bounds for unobserved confounding.
  4. Comprehensive system integration regression verification.

---

## 4. Verification Gates & Success Criteria

Every phase transition requires satisfying strict verification gates:

```
+--------------------------+-----------------------------------------------------------------------------------+
| Verification Gate        | Acceptance Criteria                                                               |
+--------------------------+-----------------------------------------------------------------------------------+
| Gate 1: Unit Test Suite  | 100% pass on all 61 frontend Vitest tests and 105 backend Pytest tests.           |
| Gate 2: Non-Regression   | Baseline profit ($90,586.38) and optimized profit ($100,786.25) remain identical.|
| Gate 3: Cloud CI Pass    | GitHub Actions workflow runs to completion with 'conclusion: success'.            |
| Gate 4: API Backward-    | All existing v1 and legacy routes respond with 200 OK and backward-compatible     |
|         Compatibility    | payload schemas.                                                                  |
| Gate 5: Documentation    | Every changed module accompanied by updated documentation and reproducible specs. |
+--------------------------+-----------------------------------------------------------------------------------+
```

---

## 5. Summary & Next Step

Phase 0 has completed the baseline audit and preserved all benchmark artifacts:
- [`docs/phase0_baseline_audit.md`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/docs/phase0_baseline_audit.md)
- [`experiments/baseline/config.json`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/experiments/baseline/config.json)
- [`experiments/baseline/metrics.json`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/experiments/baseline/metrics.json)
- [`docs/phase0_implementation_plan.md`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/docs/phase0_implementation_plan.md)

**Awaiting user approval before initiating Phase 1 implementation.**

