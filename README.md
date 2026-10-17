# EconoCausal: Dynamic Pricing via Double Machine Learning

**Organization:** Infotact Solutions  
**Milestone:** Week 1 — Causal ML Foundation & React Analytics Dashboard  
**Author:** Tanmay Yamana (`tanmayyamana999@gmail.com`)  
**Date:** October 17, 2026

---

## 1. Project Objective & Problem Statement

### The Business Challenge
In modern retail e-commerce, pricing and promotional discount strategies are among the highest-leverage revenue drivers. However, standard observational transaction logs suffer from severe **marketing selection bias (confounding)**:
* Marketing campaigns disproportionately target **high-engagement**, **high-lifetime-spend**, and **loyal** customers with promotional discount codes.
* Because these loyalty segments possess an inherently higher baseline propensity to purchase regardless of discounts, standard unadjusted regressions (e.g., OLS or naive supervised classifiers) mistakenly attribute customer loyalty to discount effectiveness.
* This confounding leads to **artificially inflated elasticity estimates**, suboptimal price promotions, and wasted marketing expenditure on customers who would have purchased at full price.

### EconoCausal's Solution
EconoCausal is an end-to-end causal machine learning platform combining **Causal Directed Acyclic Graphs (DAGs)**, the **DoWhy** framework, and **Double Machine Learning (DML)** to:
1. Formulate and validate domain causal assumptions using structural graph theory.
2. Formally identify the unconfounded back-door adjustment set.
3. Prepare for heterogeneous Conditional Average Treatment Effect (CATE) estimation to isolate true consumer price elasticity without confounding bias.

---

## 2. Week 1 Scope & Implemented Features

### Backend Causal Foundation
* **Reproducible Synthetic DGP:** Parameterized Data Generating Process generating $N \ge 1,000$ (default: 2,500) customer records with documented non-linear confounding.
* **Causal Variable Classification:**
  * **Treatment ($T$):** `discount_offered` (Binary: 0 or 1).
  * **Outcome ($Y$):** `purchased` (Binary: 0 or 1).
  * **Observed Confounders ($W$):** Pre-treatment customer behavior variables (`prior_purchases`, `prior_spend`, `engagement_score`, `customer_tenure`, `campaign_channel`).
  * **Post-Treatment Variables:** `discount_amount` and `revenue` are strictly excluded as confounders to prevent mediator suppression and collider stratification bias.
* **DoWhy Causal DAG & Model Initialization:** Directed Acyclic Graph specified in Graphviz DOT format. Step 2 **Effect Identification via Back-door Criterion** executes cleanly and resolves the minimal adjustment set.
* **Schema & Domain Validator:** Automated checks for required columns, null values, binary domain constraints, and post-treatment consistency.

### Frontend Analytics Dashboard (React 19 + Vite)
* **Header & Live Status Indicator:** Real-time badge indicating whether active data is the bundled `Synthetic Sample Campaign (N=2,500)` or a `Custom Uploaded File`. Quick actions to reload sample data or download the sample CSV template.
* **Client-Side CSV Ingestion:** Drag-and-drop CSV parser powered by `PapaParse` with 100% in-browser processing (zero external server transmission). Provides instant schema error reporting and a missing-value inspector.
* **First 10 Rows Preview Table:** Color-coded table mapping each column to its causal taxonomy role (*Treatment*, *Outcome*, *Confounder*, *Post-Treatment*).
* **Marketing Budget & Pricing Bounds:** Configurable campaign budget cap (initially `$5,000`) with strict non-negative validation, multi-currency switcher (`$`, `€`, `£`, `₹`, `C$`), configurable discount options (`$10`, `$20`, custom tiers), and real-time recipient capacity simulations.
* **Interactive Causal DAG Visualizer:** Interactive SVG diagram illustrating causal and confounding pathways, complete with an interactive node inspector explaining why each variable is included or excluded.
* **Exploratory Marketing Telemetry (Plotly):** Interactive charts showing customer volume across outreach channels and pre-treatment spend versus engagement distributions.
* **Future Deliverables Wireframes:** Dedicated placeholders for Week 2 (Heterogeneous Uplift via DML) and Week 3 (Constrained Dynamic Pricing Optimizer) with clear disclaimers that causal estimates and optimizations are pending future milestones.

---

## 3. Technology Stack

### Backend & Causal Inference
* **Python:** 3.11.9
* **Causal Inference Engine:** `dowhy (0.14)`
* **Data Processing & ML:** `pandas (3.0.6)`, `numpy (2.4.6)`, `scikit-learn (1.9.1)`, `scipy (1.15.3)`
* **Graph Theory & Visualization:** `networkx (3.6.1)`, `pydot (4.0.1)`
* **Test Runner:** `pytest (9.1.1)`

### Frontend & Analytics Dashboard
* **Framework:** React 19 (`react`, `react-dom`)
* **Bundler & Tooling:** Vite 8 (`vite`, Rolldown bundler)
* **Icons:** `lucide-react (1.59.0)`
* **CSV Parsing:** `papaparse (5.7.0)`
* **Visualizations:** `plotly.js-dist-min (4.1.2)`, `react-plotly.js (4.1.0)`
* **Test Runner:** `vitest (5.0.5)`, `@testing-library/react (16.3.3)`, `jsdom (30.1.3)`

---

## 4. Project Folder Structure

```text
EconoCasual/
├── .gitignore                      # Git exclusion rules for .venv, node_modules, credentials, and caches
├── pytest.ini                      # Pytest runner configuration
├── requirements.txt                # Pinned Python causal dependencies
├── README.md                       # Comprehensive internship documentation
├── sample_data/
│   ├── synthetic_retail_campaign.csv # 2,500-record reproducible synthetic dataset
│   └── DATASET_METADATA.md         # Variable dictionary and DGP mechanics
├── docs/
│   └── CAUSAL_MODELING_SPEC.md     # Methodological specification & confounder justifications
├── src/                            # Python Causal ML Foundation
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generator.py            # Parameterized non-linear synthetic DGP
│   │   └── schema.py               # Schema definitions, domain rules, and dataset validation
│   └── causal/
│       ├── __init__.py
│       └── graph.py                # Graphviz DOT DAG, DoWhy CausalModel builder, & identification
├── scripts/
│   └── run_causal_pipeline.py      # Automated pipeline verification script
├── tests/                          # Backend Pytest Suite
│   ├── test_causal_dag.py          # Tests for DAG acyclicity & DoWhy backdoor identification
│   └── test_data_generation.py     # Tests for DGP domains, null checks, & logic consistency
└── frontend/                       # React 19 + Vite Analytics Dashboard
    ├── package.json                # Frontend package configuration & npm scripts
    ├── vite.config.js              # Vite bundler, code-splitting & Vitest configuration
    ├── index.html                  # HTML5 application shell
    ├── public/
    │   └── sample_retail_campaign.csv # Static asset for browser loading and template download
    └── src/
        ├── index.css               # Modern dark-mode design system & layout styling
        ├── App.jsx                 # Dashboard root coordinating data, budget, and visualizers
        ├── components/
        │   ├── Header.jsx          # Header with active dataset status indicator
        │   ├── MetricsCards.jsx    # Top telemetry KPI cards
        │   ├── DatasetUploader.jsx # Drag-and-drop CSV upload, missing values, & preview table
        │   ├── BudgetSettings.jsx  # Budget constraints, currency selector, & discount options
        │   ├── CausalDAGViewer.jsx # Interactive SVG Causal DAG visualizer & node inspector
        │   ├── MarketingTelemetry.jsx # Plotly charts for channel distribution & confounders
        │   └── FutureDeliverables.jsx # Wireframe placeholders for Week 2 and Week 3
        ├── utils/
        │   └── csvValidator.js     # Client-side causal schema & domain validation rules
        └── tests/                  # Frontend Vitest Suite
            ├── csvValidator.test.js # Unit tests for CSV validation & missing value calculations
            ├── BudgetSettings.test.jsx # Unit tests for budget inputs & non-negative constraints
            ├── DatasetUploader.test.jsx # Unit tests for file upload, error states, and preview
            └── Header.test.jsx     # Unit tests for dataset status badges
```

---

## 5. Installation & Run Instructions

### A. Python Causal ML Foundation (Backend)

#### 1. Setup Environment
```powershell
# In project root:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### 2. Run Pipeline Verification
```powershell
python scripts/run_causal_pipeline.py
```
*Executes the complete Week 1 pipeline: generates 2,500 synthetic records, validates schema integrity, defines the Causal DAG, initializes DoWhy `CausalModel`, and outputs the minimal back-door adjustment set.*

#### 3. Run Backend Pytest Suite
```powershell
pytest -v
```

---

### B. React Analytics Dashboard (Frontend)

#### 1. Setup & Installation
```powershell
cd frontend
npm install
```

#### 2. Start Development Server
```powershell
npm run dev
```
Open **`http://localhost:5173`** in your browser to view the interactive dashboard.

#### 3. Run Frontend Unit Tests
```powershell
npm test
```

#### 4. Run Production Build
```powershell
npm run build
```
Generates minified production assets in `frontend/dist/`.

---

## 6. Dataset Schema & Causal Assumptions

### Canonical Variable Taxonomy

| Column Name | Data Type | Causal Role | Domain / Format | Description & Confounder Justification |
| :--- | :--- | :--- | :--- | :--- |
| `customer_id` | `string` | Identifier | `CUST_00001`+ | Unique customer tracking identifier. |
| `prior_purchases` | `integer` | Observed Confounder ($W_1$) | $\ge 0$ | Pre-treatment order count. Repeat buyers are targeted with loyalty coupons and have higher baseline conversion. |
| `prior_spend` | `float` | Observed Confounder ($W_2$) | $\ge \$5.00$ | Cumulative monetary spend. High spenders qualify for premium discounts and possess higher purchasing power. |
| `engagement_score` | `float` | Observed Confounder ($W_3$) | $0.0 - 100.0$ | Digital app/email activity index. Engaged users see push notifications and have organic purchase intent. |
| `customer_tenure` | `integer` | Observed Confounder ($W_4$) | $1 - 72$ mo | Account age in months. Retention campaigns target specific tenure cohorts with habitual brand loyalty. |
| `campaign_channel` | `category` | Observed Confounder ($W_5$) | `email`, `push`, `sms`, `social` | Delivery medium. Delivery probability of discounts and baseline channel response rates differ. |
| `discount_offered` | `binary` | **Treatment ($T$)** | $0$ or $1$ | Promotional discount assignment ($1 =$ discount offered, $0 =$ standard price). |
| `discount_amount` | `float` | Post-treatment Mediator | $\ge \$0.00$ | Realized dollar face value. Strictly excluded as confounder to avoid mediator suppression bias. |
| `purchased` | `binary` | **Primary Outcome ($Y$)** | $0$ or $1$ | Campaign purchase conversion ($1 =$ converted, $0 =$ did not purchase). |
| `revenue` | `float` | Downstream Outcome | $\ge \$0.00$ | Realized basket spend post-discount. Strictly excluded as confounder to avoid collider stratification. |

### Causal Identification Assumption
Under Pearl's Back-Door Criterion, the average treatment effect of `discount_offered` on `purchased` is identified by conditioning on the minimal adjustment set:
$$Z = \{\text{prior\_purchases}, \text{prior\_spend}, \text{engagement\_score}, \text{customer\_tenure}, \text{campaign\_channel}\}$$

---

## 7. Tests Performed & Actual Results

### Backend Pytest Suite (`13 Passed` in `4.43s`)
* `test_reproducible_generation`: Identical seed yields exact bitwise identical data.
* `test_minimum_sample_size_requirement`: Validates sample size $\ge 1,000$ and generator constraints.
* `test_required_columns_present`: Validates presence of all 10 canonical columns.
* `test_no_null_values`: Verifies 0 null or NaN occurrences.
* `test_binary_variable_domains`: Verifies treatment and outcome are strictly $\{0, 1\}$.
* `test_post_treatment_logic_consistency`: Validates post-treatment business rules.
* `test_schema_validator_catches_errors`: Validates detection of missing columns and non-binary values.
* `test_dag_metadata_structure`: Verifies node taxonomy and roles.
* `test_no_post_treatment_confounders`: Verifies post-treatment variables are excluded from $W$.
* `test_dag_is_strictly_acyclic`: Verifies DAG acyclicity via `networkx`.
* `test_dowhy_causal_model_initialization`: Verifies DoWhy `CausalModel` instantiation.
* `test_dowhy_backdoor_identification`: Verifies DoWhy identifies the exact backdoor adjustment set.
* `test_causal_model_rejects_corrupted_data`: Verifies input error handling.

### Frontend Vitest Suite (`12 Passed` in `5.08s`)
* `csvValidator.test.js` (5 tests): Clean CSV validation, empty dataset rejection, missing column error detection, non-binary treatment flagging, accurate missing value counting.
* `Header.test.jsx` (3 tests): Title rendering, synthetic sample status badge, custom uploaded status badge.
* `BudgetSettings.test.jsx` (2 tests): Budget input rendering ($5,000 default), non-negative budget validation.
* `DatasetUploader.test.jsx` (2 tests): Ingestion error rendering, first 10 rows preview table rendering.

### Production Build
* `npm run build`: Succeeded in `1.98s` producing minified assets with manual code-splitting.

---

## 8. Known Limitations & Planned Week 2 Work

### Methodological Guardrails
1. **The Graph Is a Structural Hypothesis:** The causal DAG formalizes domain assumptions; it does not empirically prove causality without identification assumptions.
2. **Identification $\neq$ Estimation:** Week 1 establishes the structural identification (Step 2 of DoWhy). Per requirements, **no treatment effect estimation or Double Machine Learning model has been trained yet**.

### Planned Week 2 Work
* **Double Machine Learning (DML) Implementation:** Implement Chernozhukov et al. DML engine.
* **Nuisance Model Estimation:** 5-fold cross-fitting with LightGBM / Gradient Boosting to estimate treatment propensity $\hat{m}(W)$ and outcome response $\hat{g}(W)$.
* **Residual Orthogonal Regression:** Orthogonalized regression of $\tilde{Y} = Y - \hat{g}(W)$ on $\tilde{T} = T - \hat{m}(W)$ to recover unconfounded Conditional Average Treatment Effects (CATE).
* **Heterogeneous Uplift Curves:** Segment customers by price sensitivity for personalized promotional targeting.

### Planned Week 3 Work
* **Constrained Policy Optimization:** Implement integer linear programming / knapsack optimization to allocate discrete discount tiers ($10, $20) maximizing incremental revenue subject to the configured campaign budget cap.
