# EconoCausal: Dynamic Pricing via Double Machine Learning

[![CI Tests](https://github.com/tanmay0212-cls/EconoCasual/actions/workflows/tests.yml/badge.svg)](https://github.com/tanmay0212-cls/EconoCasual/actions/workflows/tests.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11.9-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![EconML](https://img.shields.io/badge/Causal%20ML-EconML%200.17-blue)](https://econml.azurewebsites.net/)
[![DoWhy](https://img.shields.io/badge/Causal%20DAG-DoWhy%200.14-orange)](https://py-why.github.io/dowhy/)
[![React 19](https://img.shields.io/badge/Frontend-React%2019%20%2B%20Vite%208-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Tests Passing](https://img.shields.io/badge/Tests-51%20Passed%20(39%20py%20%2B%2012%20js)-brightgreen)](https://github.com/tanmay0212-cls/EconoCasual)

> **Causal Machine Learning for Promotional Pricing Optimization**  
> Formulate causal DAGs, identify unconfounded adjustment sets, estimate heterogeneous customer price elasticity using Double Machine Learning (DML), and evaluate targeting effectiveness via Qini and uplift curves.

---

## Table of Contents

1. [Project Overview & Business Challenge](#1-project-overview--the-business-challenge)
2. [Causal Identification & Methodology](#2-causal-identification--methodology)
3. [Architecture & Implemented Features](#3-architecture--implemented-features)
4. [Repository Structure](#4-repository-structure)
5. [Installation & Quickstart](#5-installation--quickstart)
6. [Pipeline Execution Guide](#6-pipeline-execution-guide)
7. [Empirical Results & Causal Validation](#7-empirical-results--causal-validation)
8. [Automated Testing Suite](#8-automated-testing-suite)
9. [Causal Assumptions & Limitations](#9-causal-assumptions--limitations)
10. [Roadmap & Future Scope](#10-roadmap--future-scope)

---

## 1. Project Overview & The Business Challenge

In retail e-commerce, offering promotional discount codes is a high-stakes lever to drive customer acquisition and conversion. However, standard observational transaction logs suffer from severe **marketing selection bias (confounding)**:

* **The Problem:** Marketing campaigns disproportionately target loyal, high-lifetime-spend, and digitally engaged customers with promotional coupons.
* **The Illusion:** Because these customers have an inherently higher organic propensity to purchase regardless of discounts, standard unadjusted regressions (e.g. OLS or naive supervised classifiers) mistakenly attribute customer loyalty to discount effectiveness.
* **The Business Consequence:** Confounded models produce **artificially inflated elasticity estimates**, leading to wasted promotional margin on customers who would have purchased at full price.

### The EconoCausal Solution
EconoCausal resolves this problem using modern causal machine learning:
1. **Structural Causal DAGs (DoWhy):** Formulates domain assumptions, inspects causal pathways, and mathematically identifies the unconfounded back-door adjustment set.
2. **Double Machine Learning (EconML):** Employs Chernozhukov et al. (2018) orthogonalized cross-fitting to isolate the true causal effect of discounts from confounding covariates.
3. **Customer-Level CATE Estimation:** Estimates Conditional Average Treatment Effects ($\tau(X_i)$) with asymptotic 95% Wald confidence intervals to identify true price-sensitive cohorts.
4. **Uplift & Qini Evaluation:** Benchmarks causal targeting against random baseline policies using Radcliffe (2007) cumulative incremental gains.
5. **Interactive Analytics Dashboard:** React 19 + Vite interface with interactive DAG visualizers, drag-and-drop CSV validation, and marketing telemetry.

---

## 2. Causal Identification & Methodology

### 2.1 The Causal Directed Acyclic Graph (DAG)

```text
       [prior_purchases]       [prior_spend]       [engagement_score]
              \                     |                     /
               \                    |                    /
                +---> [discount_offered] (Treatment) <--+
                |             |             |
[customer_tenure]             |             +---> [discount_amount] (Mediator)
                |             v                          \
                +-----> [purchased] (Outcome) <-----------+
              /               |                            \
    [campaign_channel]        +-----------------------------> [revenue] (Downstream)
```

### 2.2 Approved Confounder Adjustment Set ($W$)
Under Pearl's Back-Door Criterion, the average treatment effect of `discount_offered` on `purchased` is non-parametrically identified by conditioning on the minimal adjustment set:
$$W = \{\text{prior\_purchases}, \text{prior\_spend}, \text{engagement\_score}, \text{customer\_tenure}, \text{campaign\_channel}\}$$

### 2.3 Strict Anti-Leakage Quarantine
* **`discount_amount`:** Post-treatment mediator ($T \to M \to Y$). Conditioning on discount amount mutes the direct mechanism and induces collider stratification bias.
* **`revenue`:** Post-treatment outcome ($Y \to \text{revenue}$). Conditioning on revenue conditions on the future and corrupts conversion probability.
* **`customer_id`:** Identifier. Strictly quarantined from all model feature matrices to prevent high-cardinality memorization.

### 2.4 Double Machine Learning (DML) Formulation
Under the partially linear CATE model:
$$Y = \tau(X) \cdot T + g(W) + \varepsilon, \quad \mathbb{E}[\varepsilon \mid X, W, T] = 0$$
$$T = m(W) + \eta, \quad \mathbb{E}[\eta \mid W] = 0$$

Using 5-fold cross-fitting, nuisance estimators $\hat{g}(W) = \mathbb{E}[Y \mid W]$ and $\hat{m}(W) = P(T=1 \mid W)$ are fit on out-of-fold partitions using Random Forest models. The second-stage regression orthogonalizes the residuals:
$$(Y - \hat{g}(W)) = \tau(X) \cdot (T - \hat{m}(W)) + \nu$$
This satisfies Neyman orthogonality, ensuring first-stage regularization bias converges at rate $o_P(N^{-1/4})$ and yields $\sqrt{N}$-consistent, asymptotically normal CATE estimates $\hat{\tau}(X)$.

---

## 3. Architecture & Implemented Features

### Milestone Breakdown

#### Week 1 (Completed): Causal Foundation & React Dashboard
* **Synthetic Data Generating Process (DGP):** Parameterized generator producing reproducible retail campaigns with documented non-linear confounding (`src/data/generator.py`).
* **DoWhy Causal DAG Engine:** DOT graph definition, graph acyclicity verification, and backdoor identification (`src/causal/graph.py`).
* **React 19 Analytics Dashboard:** Interactive DAG visualizer, client-side CSV validator (PapaParse), and marketing telemetry (`frontend/`).

#### Week 2 (Completed): Double Machine Learning & Causal Validation
* **EconML DML Estimator:** Production-grade `EconoCausalDML` wrapping `LinearDML` with `RandomForestRegressor` and `RandomForestClassifier` nuisance estimators, 5-fold cross-fitting, and serialization (`src/causal/dml.py`).
* **Customer CATE Interpreter:** CATE point estimates, 95% Wald confidence intervals, domain-justified categorization ($\pm 2.0$ pp threshold), and identifier isolation (`src/causal/interpreter.py`).
* **Uplift & Qini Visualizations:** Radcliffe Qini curve, cumulative uplift curve, AUUC, decile table, and interactive Plotly HTML figures (`src/causal/evaluation.py`).
* **Comprehensive Test Suite:** 39 backend pytest tests and 12 frontend Vitest tests covering edge cases, non-finite features, empty inputs, and seed reproducibility.

#### Week 3 (Planned / Future Scope):
* **Budget-Constrained Dynamic Pricing Optimizer:** Integer Linear Programming / Knapsack allocation to assign optimal discount depths under a marketing budget constraint.
* **REST API Layer:** FastAPI backend exposing CATE inference and optimization endpoints.

---

## 4. Repository Structure

```text
EconoCasual/
├── .github/
│   └── workflows/
│       └── tests.yml               # Automated GitHub Actions CI workflow
├── .gitignore                      # Git exclusion rules (models/*.joblib, caches, .env)
├── .env.example                    # Environment configuration template
├── pytest.ini                      # Pytest test discovery & execution settings
├── requirements.txt                # Pinned Python causal dependencies
├── README.md                       # Master project documentation
├── data/
│   └── README.md                   # Data guide: schemas, custom placement, & generator instructions
├── sample_data/
│   ├── synthetic_retail_campaign.csv # 2,500-customer reproducible benchmark dataset
│   ├── DATASET_METADATA.md         # Technical DGP specification & logit parameters
│   └── README.md                   # Sample data directory overview
├── docs/
│   └── CAUSAL_MODELING_SPEC.md     # Methodological DAG specification & backdoor identification
├── models/
│   ├── README.md                   # Model training and artifact documentation
│   ├── dml_metadata.json           # Tracked DML configuration, hyperparams, & benchmarks
│   └── dml_model.joblib            # (Ignored binary weights; generated via train_dml.py)
├── reports/
│   ├── week2_final_report.md       # Week 2 completion review & validation report
│   ├── week2_data_audit.md         # Pre-training dataset audit & feature approval
│   ├── dml_training_report.md      # DML training & cross-fitting execution diagnostics
│   ├── customer_treatment_effects.csv # 500-customer evaluation CATE estimates & 95% CIs
│   ├── customer_treatment_effects_summary.md # CATE distribution & taxonomy summary
│   ├── uplift_qini_report.md       # Qini score, AUUC, & decile analysis report
│   └── figures/                    # Standalone interactive Plotly HTML figures
│       ├── uplift_curve.html
│       ├── qini_curve.html
│       └── effect_distribution.html
├── scripts/
│   ├── run_causal_pipeline.py      # Week 1: Data validation & DoWhy DAG backdoor identification
│   ├── train_dml.py                # Week 2: Stratified split, DML training, & artifact export
│   ├── estimate_customer_effects.py# Week 2: Model reload, customer CATE inference, & CSV export
│   └── evaluate_uplift_qini.py     # Week 2: Uplift/Qini evaluation & Plotly HTML export
├── src/
│   ├── __init__.py
│   ├── causal/
│   │   ├── __init__.py             # Exposes EconoCausalDML, interpreter, & evaluation utilities
│   │   ├── dml.py                  # EconML LinearDML estimator & cross-fitting pipeline
│   │   ├── evaluation.py           # Ranking engine, Qini curves, AUUC, & Plotly charts
│   │   ├── graph.py                # Graphviz DOT DAG & DoWhy identification
│   │   └── interpreter.py          # Customer CATE interpreter, 95% CIs, & categorization
│   └── data/
│       ├── __init__.py
│       ├── generator.py            # Parameterized non-linear synthetic DGP
│       └── schema.py               # Schema definitions, binary domains, & data validation
├── tests/
│   ├── test_causal_dag.py          # Tests for DAG acyclicity & backdoor identification
│   ├── test_data_generation.py     # Tests for DGP domains, null checks, & schema rules
│   ├── test_dml.py                 # Tests for DML fitting, anti-leakage, seed reproducibility, & reload
│   ├── test_evaluation.py          # Tests for ranking, Qini curves, AUUC, ties, & edge cases
│   └── test_interpreter.py         # Tests for customer CATE alignment, 95% CIs, & non-finite inputs
└── frontend/                       # React 19 + Vite Analytics Dashboard
    ├── package.json                # Frontend package configuration & npm scripts
    ├── vite.config.js              # Vite bundler, code-splitting & Vitest configuration
    ├── index.html                  # HTML5 application shell
    ├── public/
    │   └── sample_retail_campaign.csv # Bundled static asset for client-side demo
    ├── src/
    │   ├── index.css               # Design system & dark-mode styling
    │   ├── App.jsx                 # Dashboard root coordinating DAG viewer & telemetry
    │   ├── components/             # Reusable React components
    │   │   ├── Header.jsx          # Header with dataset status badge
    │   │   ├── MetricsCards.jsx    # KPI telemetry summary cards
    │   │   ├── DatasetUploader.jsx # Drag-and-drop CSV upload & preview table
    │   │   ├── BudgetSettings.jsx  # Budget constraints & discount tier simulator
    │   │   ├── CausalDAGViewer.jsx # Interactive SVG Causal DAG visualizer & node inspector
    │   │   ├── MarketingTelemetry.jsx # Plotly charts for channel distribution & confounders
    │   │   └── FutureDeliverables.jsx # Wireframe placeholders for Week 2 and Week 3
    │   └── tests/                  # Frontend Vitest Suite (12 unit tests)
    │       ├── BudgetSettings.test.jsx
    │       ├── DatasetUploader.test.jsx
    │       ├── Header.test.jsx
    │       └── csvValidator.test.js
```

---

## 5. Installation & Quickstart

### Prerequisites
* **Python:** 3.11+ (tested on Python 3.11.9)
* **Node.js:** 20+ (tested on Node v20.x / npm 10.x)
* **Git:** 2.30+

### Clone the Repository
```bash
git clone https://github.com/tanmay0212-cls/EconoCasual.git
cd EconoCasual
```

### Setup Python Backend
```bash
# Create and activate virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate

# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Setup React Frontend
```bash
cd frontend
npm install
cd ..
```

---

## 6. Pipeline Execution Guide

Run the pipeline sequentially using the documented scripts:

### Step 1: Validate Dataset & Causal DAG Backdoor Identification
```bash
python scripts/run_causal_pipeline.py
```
*Validates the 2,500-record dataset schema, builds the Causal DAG, initializes DoWhy `CausalModel`, and identifies the minimal backdoor adjustment set.*

### Step 2: Train the EconML Double Machine Learning Estimator
```bash
python scripts/train_dml.py
```
*Performs stratified 80/20 train/eval split, fits `LinearDML` with 5-fold cross-fitting, serializes `models/dml_model.joblib` and `models/dml_metadata.json`, and outputs the ATE evaluation report.*

### Step 3: Estimate Customer-Level Treatment Effects (CATE)
```bash
python scripts/estimate_customer_effects.py
```
*Reloads the trained model, estimates CATE and 95% Wald confidence intervals for the 500 held-out customers, and exports `reports/customer_treatment_effects.csv`.*

### Step 4: Evaluate Uplift & Qini Curves
```bash
python scripts/evaluate_uplift_qini.py
```
*Computes the Radcliffe Qini score, AUUC, and decile targeting breakdown, exporting interactive Plotly visualizations to `reports/figures/`.*

### Step 5: Launch the React Analytics Dashboard
```bash
cd frontend
npm run dev
```
*Opens the interactive dashboard at `http://localhost:5173`.*

---

## 7. Empirical Results & Causal Validation

All metrics were evaluated on the **500 held-out evaluation customers** (20% test partition):

| Metric / Estimator | Value | Benchmark / Ground Truth | Validation Status |
| :--- | :---: | :---: | :---: |
| **Naive Observational Diff** | **+25.21 pp** (`0.2521`) | Confounded baseline | Severe upward bias (+10.75 pp) |
| **EconML LinearDML ATE** | **+17.25 pp** (`0.1725`) | Synthetic True Sample ATE: **+14.46 pp** | **COVERS TRUTH** (Inside 95% CI) |
| **95% Wald Confidence Interval** | **[+12.96 pp, +21.53 pp]** | Theoretical spec ATE: **+13.69 pp** | Statistically significant lift ($p < 0.001$) |
| **Confounding Bias Removed** | **+10.75 pp** | — | Bias purged via Neyman orthogonalization |
| **Latent ITE Rank Correlation** | **Spearman $\rho = +0.2104$** | $p = 2.0 \times 10^{-6}$ | Statistically significant elasticity ranking |
| **Decile 1 (Top 10%) Targeting Gain** | **8.50 incremental buyers** | Random baseline: 5.92 buyers | **+43.5% targeting advantage** |
| **Area Under Uplift Curve (AUUC)**| **0.2131** | Baseline overall uplift: +25.21 pp | Monotonically positive cumulative lift |

### Generated Visualization Artifacts
* **Uplift Curve:** [`reports/figures/uplift_curve.html`](reports/figures/uplift_curve.html)
* **Qini Curve:** [`reports/figures/qini_curve.html`](reports/figures/qini_curve.html)
* **CATE Dispersion:** [`reports/figures/effect_distribution.html`](reports/figures/effect_distribution.html)

---

## 8. Automated Testing Suite

The repository includes a comprehensive, multi-framework automated test suite:

### Backend Pytest Suite (39 Tests)
```bash
pytest -v
```
```text
tests/test_causal_dag.py        6 passed  (DAG acyclicity, DoWhy backdoor identification, input corruption)
tests/test_data_generation.py   7 passed  (Reproducibility, sample size, schema domains, business logic)
tests/test_dml.py              10 passed  (DML fit/shapes, anti-leakage, binary domains, serialization, reproducibility, non-finite features)
tests/test_evaluation.py        8 passed  (Ranking monotonicity, Qini/AUUC, zero-treated/control edge cases, tied effects, empty inputs)
tests/test_interpreter.py       8 passed  (Row alignment, customer ID quarantine, 95% CIs, non-finite checks, configurable threshold)
============================= 39 passed in 12.09s =============================
```

### Frontend Vitest Suite (12 Tests)
```bash
cd frontend
npm test -- --run
```
```text
 ✓ src/tests/csvValidator.test.js (5 tests)
 ✓ src/tests/Header.test.jsx (3 tests)
 ✓ src/tests/BudgetSettings.test.jsx (2 tests)
 ✓ src/tests/DatasetUploader.test.jsx (2 tests)
 Test Files  4 passed (4)
      Tests  12 passed (12)
```

---

## 9. Causal Assumptions & Limitations

> ### Critical Methodological Disclaimer
> Causal inference models do not provide deterministic mathematical certainty or guaranteed business revenue expansion. All treatment effect estimates are conditional on domain assumptions:

1. **Conditional Unconfoundedness:** We assume that discount assignment is conditionally independent of potential conversion outcomes given observed covariates $W$: $(Y(1), Y(0)) \perp T \mid W$. If unmeasured factors (e.g. competitor pricing, offline exposure, uncaptured seasonal intent) influenced both discount eligibility and purchase decisions, estimates retain omitted variable bias.
2. **Positivity (Overlap):** We assume every customer segment has non-zero probability of receiving and not receiving discounts: $0 < P(T=1 \mid W) < 1$. In our benchmark dataset, propensity scores are bounded in $[0.18, 0.82]$.
3. **No Interference (SUTVA):** We assume offering a promotional discount to customer $i$ does not alter customer $j$'s conversion likelihood (no social coupon sharing or inventory stockouts).
4. **Parametric Linearity Assumption:** `LinearDML` parameterizes CATE as a linear function $\tau(X) = \langle \theta, X \rangle + \alpha$. Non-linear feature interactions are not captured unless explicitly specified or estimated via non-parametric causal forests.
5. **Observational Qini Curves:** Qini and uplift curves in observational data evaluate raw sample conversion contrasts, which can reflect residual covariate imbalances in intermediate deciles.

---

## 10. Roadmap & Future Scope

* [x] **Week 1:** Causal DAG specification, DoWhy backdoor identification, synthetic DGP, and React dashboard.
* [x] **Week 2:** EconML Double Machine Learning, cross-fitting, customer-level CATE estimation, Wald CIs, Qini/uplift curves, and automated test suite.
* [ ] **Week 3 (Planned):** Budget-constrained dynamic pricing optimization using Integer Linear Programming / Knapsack allocation to assign optimal discount face values ($\$10$, $\$20$) maximizing net incremental revenue under campaign budget caps.
* [ ] **Week 4 (Planned):** FastAPI backend integration connecting the React frontend to the DML estimator and knapsack policy allocator.

---

## Authors & Acknowledgments

* **Lead Engineer:** Tanmay Yamana ([tanmayyamana999@gmail.com](mailto:tanmayyamana999@gmail.com))
* **Repository:** [https://github.com/tanmay0212-cls/EconoCasual](https://github.com/tanmay0212-cls/EconoCasual)
* **Frameworks:** EconML (Microsoft Research), DoWhy (PyWhy), Scikit-Learn, React, Vite, Plotly.
