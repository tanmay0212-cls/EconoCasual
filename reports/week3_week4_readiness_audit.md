# Week 3 & Week 4 Readiness Audit: Technical Feasibility, Data Gaps & Architectural Roadmap

**Project:** EconoCausal — Dynamic Pricing via Double Machine Learning  
**Milestone:** Pre-Implementation Architectural & Data Readiness Audit (Transition from Week 2 to Weeks 3 & 4)  
**Author:** Senior Machine Learning & Causal AI Engineer  
**Date:** November 3, 2026  
**Status:** **AUDIT COMPLETE — READY FOR STAKEHOLDER APPROVAL**  

---

## Executive Summary & Audit Verdict

This audit rigorously evaluates the existing **Week 1 (Causal DAG & Data Foundation)** and **Week 2 (Double Machine Learning & CATE Inference)** codebase against the upcoming requirements for **Week 3 (Budget-Constrained SciPy Optimization & Personalized Allocation Matrix)** and **Week 4 (FastAPI REST Service, Automated Drift Detection, Dashboard Integration & Natural-Language Summaries)** as specified in the original INFOTACT project specification.

### Audit Verdict: **PARTIAL READINESS (Architecture Sound, Critical Gaps in Data & Models)**
* **The Good:** The Week 1 and Week 2 foundations are rock solid. Pearl's backdoor identification, the 5-fold orthogonalized cross-fitting DML pipeline, anti-leakage quarantining, and the testing framework (39 backend tests + 12 frontend tests) are robust, modular, and fully functional.
* **The Blocker:** The current dataset (`sample_data/synthetic_retail_campaign.csv`) and trained estimator (`models/dml_model.joblib`) were strictly formulated around a **single binary treatment** ($T \in \{0, 1\}$ representing generic "discount offered" vs "control") on a **single binary outcome** ($Y \in \{0, 1\}$ purchase conversion).
* **The Reality Check:** The existing dataset **does not contain dose-response elasticity for $0, $10, and $20 discounts**, lacks counterfactual baseline order values, contains **no temporal tracking or timestamps for data drift detection**, and the current CATE output cannot directly power a multi-tier budget optimizer without extension.
* **The Solution:** We must **NOT** alter the existing Week 1 or Week 2 files or models. Instead, we specify an additive, non-breaking extension:
  1. A multi-tier synthetic dataset generator producing realistic dose-response elasticity ($0, $10, $20) and baseline basket values.
  2. Multi-batch synthetic datasets with time-indexed distributions for drift detection.
  3. A multi-arm causal DML extension and counterfactual revenue predictor.
  4. A SciPy-based mixed-integer linear programming (MILP) budget optimizer.
  5. A modular FastAPI service with automated drift statistical testing and natural-language reporting.

---

## Summary Scorecard

| Requirement | Target Milestone | Current Codebase Status | Gap Severity | Primary Action Required |
| :--- | :---: | :---: | :---: | :--- |
| **Personalized $0, $10, $20 Discounts** | Week 3 | Binary $T \in \{0, 1\}$ only | **CRITICAL** | Generate multi-tier dataset with dose-response elasticity; extend DML to multi-treatment. |
| **Budget-Constrained SciPy Optimization** | Week 3 | Placeholder UI only | **CRITICAL** | Implement SciPy MILP optimizer (`scipy.optimize.milp`) maximizing incremental revenue. |
| **Customer Allocation Matrix** | Week 3 | Wireframe placeholder | **HIGH** | Build allocation matrix generator mapping customers to optimal tiers ($0, $10, $20). |
| **Expected Revenue & Profit Formulation** | Week 3 | Conversion rate lift only | **CRITICAL** | Formulate baseline conversion $Y(0)$, basket size $\mathbb{E}[S \mid X]$, and net profit margin. |
| **Automated Data Drift Detection** | Week 4 | No timestamps / single batch | **CRITICAL** | Generate time-stamped reference and drifted batches; implement KS-test & PSI monitoring. |
| **Causal Engine REST API** | Week 4 | Standalone CLI scripts only | **MEDIUM** | Build FastAPI application exposing inference, optimization, and drift endpoints. |
| **Dashboard Integration** | Week 4 | Partial React UI (Static CSV) | **MEDIUM** | Wire React frontend to FastAPI endpoints for real-time optimization and telemetry. |
| **Natural-Language Summaries** | Week 4 | Static markdown reports | **LOW** | Implement automated narrative synthesis engine generating plain-language executive summaries. |

---

## Detailed Answers to the 9 Audit Questions

### 1. Does the current dataset support estimating different causal effects for $0, $10, and $20 discounts?
**Answer: NO.**
* **Code Evidence:** In `src/data/generator.py` (lines 141–153), purchase conversion is generated as:
  $$\text{logit}(P(\text{purchased} = 1 \mid T, W)) = -1.30 + (0.70 \times \text{discount\_offered}) + 0.60 \times \text{eng} + 0.45 \times \text{purchases} + \dots$$
  The causal boost is driven **exclusively** by the binary indicator `discount_offered` ($+0.70$ logit).
* While `discount_amount` is generated in lines 132–137 as a random post-treatment draw from $\{10, 15, 20, 25, 35\}$, **`discount_amount` does not enter the conversion probability function**.
* Consequently, in the current data, a customer who received $\$10$ has the exact same underlying conversion probability lift as a customer who received $\$20$ or $\$35$. There is **zero dose-response elasticity** in the synthetic DGP. Any model fit to this data would find that conversion differences between $\$10$ and $\$20$ are pure sampling noise.

### 2. Is the existing treatment binary or multi-valued?
**Answer: The existing treatment is strictly BINARY ($T \in \{0, 1\}$).**
* In `src/data/schema.py`:
  * `TREATMENT_COL: str = "discount_offered"`
  * Validation rules explicitly enforce binary domains: `set(df[binary_col].dropna().unique()).issubset({0, 1})`.
  * `discount_amount` is explicitly categorized under `POST_TREATMENT_VARS` as a mediator, not a treatment.
* In `src/causal/dml.py`:
  * `validate_inputs()` enforces: `if not t_vals.issubset({0, 1}): raise ValueError(...)`.
  * The EconML estimator is trained with `discrete_treatment=True` on a binary 1D array ($0$ or $1$).

### 3. Does the current DML model support multiple discount levels?
**Answer: NO.**
* The current `EconoCausalDML` wrapper in `src/causal/dml.py` is configured for a single binary treatment.
* Calling `self.estimator.effect(X)` returns a 1D vector $\hat{\tau}(X) \in \mathbb{R}^N$, representing the scalar conversion lift $\mathbb{E}[Y(1) - Y(0) \mid X]$.
* The validation methods and preprocessing pipelines reject non-binary treatment inputs.
* To support multiple discount tiers ($0, $10, $20), the DML model must be extended to **multi-arm treatment estimation**, where EconML estimates a vector of treatment effects $\boldsymbol{\tau}(X) = [\tau_{10}(X), \tau_{20}(X)]$ relative to the control baseline $T=0$.

### 4. Do we have sufficient information to calculate expected revenue, discount cost, and profit?
**Answer: NO (Insufficient information for multi-tier net profit optimization).**
* To compute expected profit for customer $i$ under discount tier $d \in \{0, 10, 20\}$, an optimizer requires:
  $$\text{Expected Profit}(i, d) = P(\text{purchase} \mid X_i, d) \times \left( \mathbb{E}[\text{Basket Value} \mid X_i, d] - \text{COGS} \right) - \text{Cost}(i, d)$$
* **Current Gaps:**
  1. **Baseline Conversion $P(Y=0 \mid X_i)$:** The current DML model only estimates the *lift* $\tau(X_i) = P(Y=1 \mid X_i, 1) - P(Y=1 \mid X_i, 0)$. It does not expose the baseline counterfactual conversion probability $P(Y(0) = 1 \mid X_i)$.
  2. **Expected Basket Value $\mathbb{E}[\text{Basket} \mid X_i]$:** In the current dataset, `revenue` is only observed post-purchase and has $\$0.50 \times \text{discount\_amount}$ mechanically subtracted. There is no predictive model for counterfactual gross basket size.
  3. **Cost Accounting:** The current dataset has no explicit cost structure (e.g. redemption cost vs send cost, margin percentage / COGS).
* Without these baseline models, an optimizer cannot compute absolute dollar revenue or net profit across tiers.

### 5. Can the existing customer-level effect estimates be used directly by a budget optimizer?
**Answer: NO, not for the multi-tier personalized allocation required by Week 3.**
* The existing outputs in `reports/customer_treatment_effects.csv` provide a single scalar $\hat{\tau}_i$ representing the conversion rate lift of an unspecified binary discount.
* If passed directly into an optimizer:
  * The optimizer could only make a binary yes/no decision at a single fixed cost (e.g. "give coupon or not").
  * It **cannot decide whether customer $i$ should receive $\$0$, $\$10$, or $\$20$**, because it has no information about the marginal benefit of moving from $\$10$ to $\$20$.
  * Furthermore, $\hat{\tau}_i$ is an incremental *conversion probability* ($\Delta P$), not incremental *revenue* ($\Delta \$$). A high conversion lift on a customer who spends $\$15$ is worth less than a modest lift on a customer who spends $\$300$.

### 6. Does the current dataset contain timestamps or historical batches suitable for data drift detection?
**Answer: NO.**
* `sample_data/synthetic_retail_campaign.csv` contains exactly 10 columns: `customer_id`, `prior_purchases`, `prior_spend`, `engagement_score`, `customer_tenure`, `discount_offered`, `discount_amount`, `purchased`, `revenue`, and `campaign_channel`.
* There are **no timestamps, no transaction dates, and no batch identifiers**.
* The dataset represents a single static snapshot ($N=2,500$) generated at one point in time. Automated drift detection requires comparing a reference baseline batch against subsequent operational batches over time.

### 7. Which new synthetic data fields and datasets are required?
**Answer: Two distinct synthetic datasets must be created (leaving the Week 1/2 dataset untouched):**
1. **Multi-Tier Promotional Campaign Dataset (`data/synthetic_multi_tier_campaign.csv`):**
   * Multi-valued treatment: `discount_tier` $\in \{0, 10, 20\}$.
   * Dose-response causal DGP: Customer conversion probability incorporates price elasticity where $\$20$ discount yields diminishing marginal lift over $\$10$, differentiated across customer segments (e.g. bargain hunters vs VIPs).
   * Counterfactual potential outcomes: $Y(0), Y(10), Y(20)$ and baseline basket values for evaluation.
   * Realized metrics: `gross_basket_value`, `discount_cost`, and `net_revenue`.
2. **Time-Indexed Production Drift Batches (`data/drift/`):**
   * `campaign_batch_reference.csv`: Baseline serving distribution (Batch 0, Q1).
   * `campaign_batch_covariate_drift.csv`: Covariate shift (Batch 1, Q2) simulating changing customer acquisition (e.g. sudden drop in `customer_tenure` or surge in mobile/social traffic).
   * `campaign_batch_concept_drift.csv`: Concept drift (Batch 2, Q3) simulating macroeconomic shifts or seasonal changes altering baseline conversion and price elasticity.
   * Required field: `event_timestamp` (ISO 8601).

### 8. Which Week 2 model components can be reused without modification?
**Answer:**
1. `src/causal/graph.py`: The DAG definition, DOT graph generation, acyclicity verification, and backdoor identification.
2. `src/causal/evaluation.py`: Qini curves, cumulative uplift curves, AUUC integration, decile table generation, and Plotly visualization builders (reusable for pairwise tier comparisons and policy evaluation).
3. `src/data/schema.py`: Base validation utilities (`validate_dataset`), null checks, non-negative range validators, and schema checking patterns.
4. Preprocessor pipeline architecture in `src/causal/dml.py`: `ColumnTransformer` with `StandardScaler` for numeric confounders and `OneHotEncoder` for categorical channels.
5. Anti-leakage quarantine principles and verification suites.

### 9. Which components must be retrained or extended?
**Answer:**
1. **DGP Generator (`src/data/generator.py`):** Add `generate_multi_tier_campaign_data()` and `generate_drift_batches()`, preserving the original `generate_synthetic_campaign_data()` for backward compatibility.
2. **DML Estimator (`src/causal/dml.py`):** Create `MultiTierCausalDML` supporting discrete multi-arm treatments $T \in \{0, 10, 20\}$, outputting multidimensional CATE vectors $[\tau_{10}(X), \tau_{20}(X)]$.
3. **Interpreter Layer (`src/causal/interpreter.py`):** Extend to translate multi-tier CATE vectors into tier-specific conversion probabilities and incremental expected values.
4. **New Optimization Module (`src/optimization/budget_optimizer.py`):** Implement SciPy-based budget optimizer solving the multi-choice knapsack problem and generating the Customer Allocation Matrix.
5. **New Drift Detection Module (`src/monitoring/drift_detector.py`):** Implement Kolmogorov-Smirnov (KS) test, Population Stability Index (PSI), and Wasserstein distance for continuous and categorical covariates.
6. **New REST API (`src/api/`):** FastAPI application wrapping the causal engine, optimizer, drift detector, and summary generator.
7. **Frontend Dashboard (`frontend/`):** Replace `FutureDeliverables.jsx` wireframes with live Allocation Matrix tables, budget sliders, and drift monitoring visualizers.

---

## Detailed Gap Analysis

```
+----------------------------------------------------------------------------------------------------+
|                                    CURRENT CAPABILITIES (WEEKS 1-2)                                |
+-----------------------------------+----------------------------------------------------------------+
| Binary Treatment (T in {0, 1})    | Single discount offered vs standard pricing                    |
| LinearDML (Single Output)         | Estimates scalar tau(X) = E[Y(1) - Y(0) | X]                   |
| Static Cross-Sectional Data       | Single CSV (N=2,500), no timestamps, no drift batches           |
| Offline Evaluation Scripts        | Standalone Python CLI scripts + Static React UI                |
+-----------------------------------+----------------------------------------------------------------+
                                    |
                                    v (TRANSITION & EXTENSION)
+----------------------------------------------------------------------------------------------------+
|                                   REQUIRED CAPABILITIES (WEEKS 3-4)                                |
+-----------------------------------+----------------------------------------------------------------+
| Multi-Tier Treatment (0, 10, 20)  | Dose-response elasticity with diminishing returns              |
| Multi-Arm CATE Estimator          | Estimates vector [tau_10(X), tau_20(X)] relative to T=0        |
| Revenue & Cost Counterfactuals    | Models baseline conversion Y(0) and basket size E[Basket | X]  |
| SciPy Mixed-Integer Optimizer     | Solves budget-constrained knapsack allocation (BILP / MILP)    |
| Customer Allocation Matrix        | Optimal tier assignment (0, 10, 20) maximizing expected profit |
| Automated Drift Engine            | KS-test, PSI, Wasserstein distance across temporal batches     |
| FastAPI REST Engine               | Real-time endpoints for CATE, Optimization, Drift & Narratives |
| Live Dashboard Integration        | React 19 UI dynamically invoking API endpoints                 |
+----------------------------------------------------------------------------------------------------+
```

### Gap 1: Dose-Response Elasticity & Multi-Tier Causal Data
* In the current dataset, discount face value is a non-causal random draw.
* Real-world retail economics dictates that price elasticity is non-linear:
  * For low-intent, price-sensitive shoppers: $\$10$ yields moderate lift, $\$20$ yields substantial lift.
  * For high-intent brand loyalists: $\$10$ captures the purchase; $\$20$ wastes $\$10$ of margin without increasing conversion probability (deadweight loss).
  * For completely disengaged customers: neither $\$10$ nor $\$20$ induces conversion.
* A new synthetic dataset must explicitly encode segment-specific dose-response curves.

### Gap 2: Counterfactual Expected Revenue & Margin Modeling
* To maximize *revenue* (or *profit*) rather than raw conversion probability, the optimization objective must be defined in currency units ($\$$).
* For each customer $i$ and discount tier $d \in \{0, 10, 20\}$:
  $$\mathbb{E}[\text{Revenue} \mid X_i, d] = P(\text{purchase} \mid X_i, d) \times \mathbb{E}[\text{Gross Basket} \mid X_i] - \text{Discount Cost}(i, d)$$
  $$\text{where } P(\text{purchase} \mid X_i, d) = P(Y(0) = 1 \mid X_i) + \tau_d(X_i)$$
* This requires:
  1. A baseline conversion estimator $\hat{g}_0(X) = \hat{P}(Y=1 \mid X, T=0)$.
  2. A basket size estimator $\hat{S}(X) = \hat{\mathbb{E}}[\text{basket\_value} \mid X]$.
  3. A multi-arm treatment lift estimator $\hat{\tau}_d(X)$ for $d \in \{10, 20\}$.

### Gap 3: Optimization Formulation
* Week 3 requires allocating discounts under a fixed budget $B$ (e.g. $\$5,000$).
* Decision variable for customer $i$ and discount tier $d \in \{0, 10, 20\}$:
  $$z_{i, d} \in \{0, 1\}, \quad \text{where } z_{i, d} = 1 \text{ if customer } i \text{ receives tier } d$$
* Constraints:
  1. **Mutually Exclusive Assignment:** $\sum_{d \in \{0, 10, 20\}} z_{i, d} = 1 \quad \forall i \in \{1, \dots, N\}$.
  2. **Budget Constraint:**
     $$\sum_{i=1}^N \sum_{d \in \{10, 20\}} \text{Expected Cost}(i, d) \cdot z_{i, d} \le B$$
     Where $\text{Expected Cost}(i, d) = d \times P(\text{purchase} \mid X_i, d)$ (coupons redeemed only upon conversion).
  3. **Objective:**
     $$\max_{\{z_{i, d}\}} \sum_{i=1}^N \sum_{d \in \{0, 10, 20\}} \mathbb{E}[\text{Net Value}(i, d)] \cdot z_{i, d}$$
* This is a **Multiple-Choice Knapsack Problem (MCKP)**, which can be solved exactly via `scipy.optimize.milp` (Mixed-Integer Linear Programming) using HiGHS solver, or via linear relaxation (`scipy.optimize.linprog`) with greedy rounding for large populations.

### Gap 4: Temporal & Batch Indexing for Drift Detection
* Data drift occurs when the serving distribution $P_{\text{production}}(X, Y)$ diverges from the training distribution $P_{\text{reference}}(X, Y)$.
* Types of drift required for Week 4:
  1. **Covariate Drift ($P(X)$ shifts, $P(Y \mid X)$ constant):** e.g., shift in marketing traffic channel, customer tenure drops due to new user acquisition campaign, or engagement scores drop.
  2. **Concept Drift ($P(Y \mid X)$ shifts):** e.g., macroeconomic inflation dampens overall conversion rate, or discount elasticity changes.
* Current project has zero temporal tracking. Time-indexed batch generation is mandatory.

---

## Technical Specifications for New Synthetic Datasets

> [!IMPORTANT]
> **Preservation Rule:** The existing files `sample_data/synthetic_retail_campaign.csv` and `data/README.md` will **remain strictly untouched** to preserve Week 1 and Week 2 reproducibility and test integrity.

### Dataset 1: Multi-Tier Retail Campaign Dataset (`data/synthetic_multi_tier_campaign.csv`)
* **Sample Size ($N$):** 3,000 customer records.
* **Random Seed:** 42 (parameterized and fully reproducible).
* **Treatment Variable:** `discount_tier` $\in \{0, 10, 20\}$:
  * Control ($0$): Standard pricing ($d = \$0$).
  * Tier 10 ($10$): $\$10$ promotional discount.
  * Tier 20 ($20$): $\$20$ promotional discount.
* **Confounders ($W$):** Identical to Week 1 DAG to preserve causal consistency:
  `prior_purchases`, `prior_spend`, `engagement_score`, `customer_tenure`, `campaign_channel`.
* **Dose-Response Elasticity DGP:**
  $$\text{logit}(P(\text{purchased} = 1 \mid d, W)) = \alpha_0 + \tau(d, W) + \boldsymbol{\beta}^T W$$
  $$\tau(d, W) = \begin{cases} 
  0 & \text{if } d = 0 \\
  \tau_{10}^{\text{base}} + \gamma_1 \cdot \text{elasticity}(W) & \text{if } d = 10 \\
  \tau_{10}^{\text{base}} + \Delta\tau_{20}^{\text{base}} + (\gamma_1 + \gamma_2) \cdot \text{elasticity}(W) & \text{if } d = 20 
  \end{cases}$$
  *Where $\text{elasticity}(W)$ is higher for mid-engagement / moderate-spend users (price sensitive) and lower for VIPs.*
* **Basket Value & Realized Revenue DGP:**
  $$\text{Gross Basket Value} = \$45.00 + 0.08 \times \text{prior\_spend} + 12.0 \times \log(1 + \text{prior\_purchases}) + \mathcal{N}(0, 10)$$
  $$\text{Net Revenue} = \mathbb{I}(\text{purchased} = 1) \times \max(\$5.00, \text{Gross Basket Value} - d)$$
  $$\text{Discount Cost} = \mathbb{I}(\text{purchased} = 1) \times d$$

### Dataset 2: Drift Monitoring Batches (`data/drift/`)
* **Batch 0 (Reference Baseline):** `campaign_batch_reference.csv` ($N = 1,000$, Q1).
  * Generated with standard baseline distributions matching the training population.
* **Batch 1 (Covariate Drift):** `campaign_batch_covariate_drift.csv` ($N = 1,000$, Q2).
  * `customer_tenure`: Shifted from Gamma(2.5, 8.0) to Exponential(mean=6) (influx of new accounts).
  * `campaign_channel`: Shifted from 40% email to 65% social / push.
  * `engagement_score`: Mean shifted from 52.0 to 38.0.
* **Batch 2 (Concept Drift):** `campaign_batch_concept_drift.csv` ($N = 1,000$, Q3).
  * Baseline conversion intercept $\alpha_0$ shifted down by $-0.60$ (macro conversion drop).
  * Treatment elasticity $\tau(d)$ attenuated by 35% (discount fatigue).

---

## Recommended Architecture for Week 3 & Week 4

```
                                  +--------------------------------------------------+
                                  |                 React 19 Frontend                |
                                  |   (Budget Simulator, Allocation Matrix, Drift)   |
                                  +-------------------------+------------------------+
                                                            | HTTP / JSON
                                                            v
+----------------------------------------------------------------------------------------------------+
|                                         FastAPI REST Service                                       |
|                                                                                                    |
|  POST /predict/cate          POST /optimize/allocation      POST /monitor/drift   GET /summary     |
+---------------+-----------------------------+------------------------+------------------+----------+
                |                             |                        |                  |
                v                             v                        v                  v
+-------------------------------+  +--------------------+  +--------------------+  +-----------------+
|      MultiTierCausalDML       |  |   SciPy Optimizer  |  |    Drift Engine    |  | Narrative Agent |
| (EconML Multi-Arm LinearDML)  |  | (scipy.optimize    |  | (KS-Test, PSI,     |  | (Executive      |
|                               |  |   .milp / HiGHS)   |  |  Wasserstein)      |  |  Summary Gen)   |
+---------------+---------------+  +----------+---------+  +----------+---------+  +--------+--------+
                |                             |                       |                     |
                v                             v                       v                     v
      Multi-Arm CATE Vector           Allocation Matrix         Drift Alerts        Plain-Language
     [tau_10(X), tau_20(X)]           (Cust -> Tier $)         & P-Value Report        KPI Report
```

### Week 3 Architecture: Budget-Constrained Dynamic Pricing Engine

#### 1. Estimator: `MultiTierCausalDML` (`src/causal/multi_tier_dml.py`)
* EconML's `LinearDML` natively supports discrete multi-arm treatments when treatment $T$ is passed as an integer array $\{0, 1, 2\}$ representing $\{0, \$10, \$20\}$.
* EconML automatically fits:
  * A multinomial propensity model: $\hat{m}_d(W) = P(T = d \mid W)$ for $d \in \{0, 1, 2\}$.
  * An outcome regression model: $\hat{g}(W) = \mathbb{E}[Y \mid W]$.
  * Orthogonalized second-stage regressions yielding:
    $$\boldsymbol{\tau}(X_i) = \begin{bmatrix} \tau_{10}(X_i) \\ \tau_{20}(X_i) \end{bmatrix}$$
* A companion `BaselineOutcomeModel` estimates $\hat{Y}_0(X) = P(Y=1 \mid X, T=0)$ and expected gross basket size $\hat{S}(X)$ to convert conversion lift into expected monetary revenue.

#### 2. Optimizer: `SciPyBudgetOptimizer` (`src/optimization/budget_optimizer.py`)
* **Formulation:** Mixed-Integer Linear Program (MILP) using `scipy.optimize.milp` with the HiGHS backend.
* **Inputs:**
  * Customer feature matrix $X$ ($N$ customers).
  * Predicted conversion vectors: $[P_{i, 0}, P_{i, 10}, P_{i, 20}]$.
  * Predicted gross basket values: $[S_{i, 0}, S_{i, 10}, S_{i, 20}]$.
  * Campaign budget cap: $B$ (e.g. $\$5,000$).
  * Optimization objective: Maximize expected net incremental revenue subject to budget.
* **Outputs:**
  * **Customer Allocation Matrix:**
    A pandas DataFrame / JSON table containing:
    `[customer_id, optimal_discount, baseline_conversion, optimal_conversion, expected_cost, expected_revenue, expected_profit, allocation_reason]`.
  * **Aggregate Campaign KPI Summary:**
    Total budget spent, total expected conversions, incremental conversions over zero-discount baseline, expected ROI, and tier breakdown percentages.

---

### Week 4 Architecture: REST API, Monitoring & Dashboard Integration

#### 1. REST API Layer (`src/api/main.py`)
Built with **FastAPI** and **Pydantic v2**:
* `GET /health`: Healthcheck, version, and loaded model metadata.
* `POST /predict/cate`: Accepts a batch of customer feature records, returns multi-tier CATE vectors $[\tau_{10}, \tau_{20}]$ and confidence intervals.
* `POST /optimize/allocate`: Accepts customer records and budget parameters, executes SciPy MILP solver, and returns the Allocation Matrix.
* `POST /monitor/drift`: Accepts a candidate batch against the reference baseline, computes KS and PSI drift metrics, and returns alert flags.
* `POST /insights/summary`: Generates automated natural-language executive summaries of optimization and drift findings.

#### 2. Automated Drift Detection Engine (`src/monitoring/drift_detector.py`)
* **Numerical Covariates (`prior_purchases`, `prior_spend`, `engagement_score`, `customer_tenure`):**
  * Two-sample **Kolmogorov-Smirnov (KS) test** (`scipy.stats.ks_2samp`): Tests null hypothesis that sample distributions are identical. Flag drift if $p < 0.05$.
  * **Wasserstein Distance** (`scipy.stats.wasserstein_distance`): Measures absolute distributional shift.
* **Categorical Covariates (`campaign_channel`):**
  * **Population Stability Index (PSI):**
    $$\text{PSI} = \sum_{k} (P_k - Q_k) \times \ln\left(\frac{P_k}{Q_k}\right)$$
    Thresholds: $\text{PSI} < 0.10$ (No Drift), $0.10 \le \text{PSI} < 0.25$ (Moderate Shift), $\text{PSI} \ge 0.25$ (Significant Drift).
* **Concept Drift (CATE & Target Distribution):**
  * Tracks mean predicted CATE and observed conversion rates across batches.

#### 3. Natural-Language Summary Generator (`src/monitoring/narrative_generator.py`)
* Converts quantitative optimization results and drift diagnostics into structured, executive-ready English text:
  * "Under a $\$5,000$ campaign budget, the causal optimizer allocated $\$20$ discounts to 142 highly price-sensitive customers (28.4%), $\$10$ discounts to 215 moderate-elasticity customers (43.0%), and $\$0$ to 143 baseline converters (28.6%), driving an estimated $+84.2$ incremental conversions and $\$5,840$ net incremental profit."
  * "Drift Alert: Significant covariate drift detected in `customer_tenure` ($p < 0.001$, $\text{PSI} = 0.28$) and `campaign_channel` due to a 25% increase in paid social acquisitions. Model retraining is recommended."

#### 4. Frontend Dashboard Integration (`frontend/`)
* Extend the React 19 application:
  * Connect `BudgetSettings.jsx` budget slider directly to `POST /optimize/allocate`.
  * Replace `FutureDeliverables.jsx` placeholders with:
    1. **Live Allocation Matrix Component:** Searchable, sortable table displaying customer tier assignments, expected ROI, and allocation rationale.
    2. **Drift Monitoring Dashboard:** Interactive charts showing PSI scores, KS test p-values, and distribution comparisons between baseline and production batches.
    3. **Executive Summary Card:** Real-time natural-language insights generated by the backend.

---

## Dependencies and Risk Analysis

### Technical Dependencies to Add
```text
# Additions to requirements.txt for Weeks 3 & 4:
fastapi>=0.110.0          # High-performance REST API framework
uvicorn[standard]>=0.28.0 # ASGI web server
pydantic>=2.6.0           # Request/response validation & serialization
scipy>=1.12.0             # Optimization (HiGHS MILP solver) & Statistical tests (KS, Wasserstein)
```
*(All are mature, standard packages with zero licensing conflicts.)*

### Methodological & Operational Risks

| Risk Category | Risk Description | Severity | Mitigation Strategy |
| :--- | :--- | :---: | :--- |
| **Causal Inference** | Positivity violation in multi-arm DML (insufficient support for tier 20 in certain subgroups). | **HIGH** | Parameterize multi-tier synthetic DGP to guarantee overlap across all covariate strata ($0.15 \le P(T=d \mid W) \le 0.65$). |
| **Causal Inference** | SUTVA violation (cannibalization or coupon sharing). | **MEDIUM** | State causal assumptions explicitly in metadata and audit documentation. |
| **Optimization** | SciPy MILP computation time exploding for large customer cohorts ($N > 50,000$). | **MEDIUM** | Implement a dual-mode optimizer: Exact MILP for $N \le 10,000$; Greedy Linear Programming Relaxation with knapsack density sorting for $N > 10,000$. |
| **Drift Monitoring** | False positive alert fatigue from sensitive KS tests on large sample sizes. | **MEDIUM** | Pair statistical p-values ($p < 0.01$) with effect size thresholds (Wasserstein distance $> 0.15$ or $\text{PSI} > 0.20$). |
| **Software Architecture** | Breaking changes to Week 1/2 tests when adding multi-tier functionality. | **CRITICAL** | Keep original `EconoCausalDML` and `synthetic_retail_campaign.csv` intact; build multi-tier and drift modules as additive new files. |

---

## Clear Implementation Sequence

We propose executing Weeks 3 and 4 in 4 structured, verifiable phases:

```
Phase 3.1: Multi-Tier Data & Dose-Response DGP
├── Generate data/synthetic_multi_tier_campaign.csv (N=3,000; tiers: 0, 10, 20)
└── Add unit tests verifying schema, overlap, and non-linear elasticity

Phase 3.2: Multi-Arm DML & Counterfactual Revenue Modeling
├── Implement src/causal/multi_tier_dml.py (Multi-arm EconML LinearDML)
├── Implement baseline conversion Y(0) and basket value predictors
└── Validate CATE recovery against synthetic dose-response ground truth

Phase 3.3: SciPy Budget Optimizer & Allocation Matrix
├── Implement src/optimization/budget_optimizer.py (SciPy MILP / HiGHS)
├── Generate Allocation Matrix export (customer_id -> tier -> profit)
└── Add unit tests for budget caps, mutual exclusivity, and empty/edge cases

Phase 4.1: Drift Simulation Batches & Monitoring Engine
├── Generate data/drift/ (Reference, Covariate Drift, Concept Drift batches)
├── Implement src/monitoring/drift_detector.py (KS-test, PSI, Wasserstein)
└── Implement src/monitoring/narrative_generator.py (Executive summary engine)

Phase 4.2: FastAPI REST Service & End-to-End Dashboard Integration
├── Implement src/api/ (endpoints: /predict/cate, /optimize/allocate, /monitor/drift, /insights/summary)
├── Update React frontend components to consume REST endpoints
└── Execute full end-to-end integration test suite
```

---

## Conclusion & Next Step

The EconoCausal codebase is thoroughly documented, rigorously tested, and causally sound. However, the step up to Week 3 (SciPy optimization) and Week 4 (Drift & REST API) requires the additive synthetic datasets and model extensions detailed above.

**Audit Status:** Completed. All constraints respected (no Kaggle datasets, no modification of original Week 1/2 datasets or models, no premature implementation).  
**Awaiting Stakeholder Approval:** We await your confirmation of these audit findings and architectural recommendations before proceeding with Phase 3.1 implementation.

