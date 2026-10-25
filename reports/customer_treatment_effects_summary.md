# Customer Treatment Effect Estimation & Interpretation Report
**Project:** EconoCausal — Dynamic Pricing via Double Machine Learning  
**Milestone:** Week 2 — Customer-Level CATE Inference & Interpretation Layer  
**Model:** `LinearDML` with `RandomForest` Nuisance Estimators  
**Near-Zero Threshold:** $\pm 2.0\%$ (configurable default: $\pm 0.02$)  
**Status:** **INFERENCE COMPLETE & VALIDATED**

---

## 1. Overview & Methodological Framing

This report provides the customer-level causal treatment effect estimation for the **500 evaluation customers** in the held-out test partition. Using the trained `LinearDML` model, we estimate the conditional uplift on purchase conversion probability if a promotional discount is offered versus standard pricing.

### Crucial Causal Interpretation Principles:
1. **CATE vs. ITE:**
   These estimates are **Conditional Average Treatment Effects (CATE)**: $\tau(X_i) = \mathbb{E}[Y(1) - Y(0) \mid X = x_i]$. They represent the expected treatment effect for customer cohorts sharing the same modeled pre-treatment characteristics $X=x_i$. They are **not** deterministic Individual Treatment Effects (ITE), as true individual counterfactuals are unobservable.
2. **Transparent Effect Categorization (Not "Proven Persuadables"):**
   Customers with $\hat{\tau}(X_i) > 0$ are classified as having a **positive estimated effect** (the discount may increase purchase probability). They must **not** be treated as definitively proven "persuadables", because individual real-world responses retain unmodeled residual variation.
3. **Quarantine of Customer Identifiers:**
   `customer_id` was retained strictly for row alignment and audit reporting. It was strictly excluded from feature preprocessors and model estimators.

---

## 2. Category Distribution & Proportions

| Effect Category | Definition / Threshold | Customer Count | Proportion (%) | Plain-Language Interpretation |
| :--- | :--- | :---: | :---: | :--- |
| **Positive Estimated Effect** | $\hat{\tau}(X_i) > +2.0\%$ | **500** | **100.0%** | Discount is estimated to meaningfully increase purchase conversion. |
| **Near-Zero Estimated Effect** | $|\hat{\tau}(X_i)| \le 2.0\%$ | **0** | **0.0%** | Little to no estimated change in conversion probability. |
| **Negative Estimated Effect** | $\hat{\tau}(X_i) < -2.0\%$ | **0** | **0.0%** | Discount is estimated to decrease conversion (e.g. brand cheapening). |
| **Total Evaluation Population** | — | **500** | **100.0%** | Full held-out evaluation sample. |

---

## 3. CATE Distributional Summary Statistics

| Statistic | Percentage Points Lift (pp) | Raw Scale ($[0, 1]$) | Interpretation |
| :--- | :---: | :---: | :--- |
| **Minimum CATE** | +9.22% | 0.092234 | Lowest estimated conversion uplift across customer profiles. |
| **25th Percentile (Q1)** | +13.85% | 0.138503 | Lower quartile response boundary. |
| **Median CATE** | +15.48% | 0.15479 | Median customer sensitivity to discount promotion. |
| **75th Percentile (Q3)** | +22.98% | 0.229833 | Highly responsive cohort boundary. |
| **Maximum CATE** | +26.24% | 0.262406 | Peak conversion lift observed in high-elasticity segments. |
| **Mean CATE (ATE)** | +17.25% | 0.172452 | Population average treatment effect over evaluation set. |
| **Standard Deviation** | 4.45% | 0.04453 | Heterogeneity spread across customer attribute space. |

---

## 4. Uncertainty & Confidence Intervals

`LinearDML` provides customer-level Wald confidence intervals $\hat{\tau}(X_i) \pm z_{1-\alpha/2} \cdot \text{SE}(\hat{\tau}(X_i))$ derived from the asymptotic variance of the second-stage linear regression parameters.

### Limitations of Confidence Intervals:
* **Parametric Linearity Assumption:** These confidence intervals reflect sampling variance in the estimated parameter vectors $\hat{\theta}$ and $\hat{\alpha}$, assuming the linear CATE formulation $\tau(X) = X'\theta + \alpha$ is structurally correct.
* **Does Not Capture Model Misspecification:** Confidence intervals do not account for potential bias from omitted non-linear interactions or residual confounding.
* **Does Not Reflect Idiosyncratic Noise:** They measure uncertainty about the *conditional expected mean*, not the prediction interval for a single stochastic purchase event.

---

## 5. Sample Output Preview (First 10 Customers)

The complete dataset is saved to `reports/customer_treatment_effects.csv`.

| customer_id | estimated_effect | estimated_effect_percentage_points | ci_lower | ci_upper | ci_lower_percentage_points | ci_upper_percentage_points | effect_category |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| CUST_00124 | 0.115633 | 11.5633 | -0.06825 | 0.299515 | -6.825 | 29.9515 | positive |
| CUST_00770 | 0.242727 | 24.2727 | 0.153439 | 0.332015 | 15.3439 | 33.2015 | positive |
| CUST_01317 | 0.172858 | 17.2858 | 0.014458 | 0.331258 | 1.4458 | 33.1258 | positive |
| CUST_01222 | 0.241233 | 24.1233 | 0.14056 | 0.341906 | 14.056 | 34.1906 | positive |
| CUST_01809 | 0.155918 | 15.5918 | 0.056821 | 0.255014 | 5.6821 | 25.5014 | positive |
| CUST_01514 | 0.147913 | 14.7913 | 0.001235 | 0.294591 | 0.1235 | 29.4591 | positive |
| CUST_02021 | 0.150381 | 15.0381 | 0.064849 | 0.235913 | 6.4849 | 23.5913 | positive |
| CUST_00657 | 0.232171 | 23.2171 | 0.130381 | 0.333962 | 13.0381 | 33.3962 | positive |
| CUST_00609 | 0.161652 | 16.1652 | 0.070832 | 0.252472 | 7.0832 | 25.2472 | positive |
| CUST_01057 | 0.236614 | 23.6614 | 0.124599 | 0.348629 | 12.4599 | 34.8629 | positive |

---

## 6. Methodological Safeguards & Next Steps

> **Causal Disclaimer:**  
> These figures represent Conditional Average Treatment Effects (CATE) for customer subgroups conditioned on modeled pre-treatment covariates. They do not constitute deterministic individualized proofs ('persuadables'). Causal interpretation strictly relies on conditional unconfoundedness, positivity, DAG validity, and model specification quality.

### Next Steps for Week 3 (Policy & Budget Optimization):
1. **Cost-Benefit Ranking:** Compare each customer's estimated conversion lift against the cost of the promotional discount face value.
2. **Constrained Knapsack Allocation:** Solve an optimization program to allocate promotional codes to customers maximizing incremental profit subject to the marketing budget constraint.
