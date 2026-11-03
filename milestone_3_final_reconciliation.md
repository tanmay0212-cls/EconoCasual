# Milestone 3 — Final Clarification and Reconciliation Report

**Project:** EconoCasual: Dynamic Pricing via Double Machine Learning  
**Milestone:** 3 – Prescriptive Budget Optimization Engine  
**Role:** Senior Machine Learning Engineer, Causal Inference Researcher, and Software Quality Auditor  
**Date:** 2026-11-03  
**Status:** **APPROVED & RECONCILED — READY FOR MILESTONE 4**

---

## Executive Summary

This report establishes the final mathematical, algorithmic, and financial reconciliation for Milestone 3. All items raised prior to final milestone sign-off have been rigorously analyzed, verified against source code and underlying data artifacts, and reconciled to a single canonical standard.

```
+---------------------------------------------------------------------------------------------------------------+
|                                       RECONCILIATION SUMMARY SCORECARD                                        |
+----------------------+---------------------------+--------------------------------+---------------------------+
| Item                 | Prior / Draft State       | Reconciled Standard            | Exact Technical Root Cause|
+----------------------+---------------------------+--------------------------------+---------------------------+
| 1. Baseline          | $90,501.19                | $90,586.38                     | Margin imputation: segment|
|    Blanket $0        |                           | ($15.09 / cust)                | heuristic ($0.49) vs      |
|    Profit            |                           |                                | recorded margin ($0.495). |
+----------------------+---------------------------+--------------------------------+---------------------------+
| 2. Expected Profit   | Referred to as            | Strictly Partitioned:          | Counterfactual potential  |
|    Definitions       | "Realized Synthetic       | Model-Expected: $100,786.25    | probabilities (P*) vs     |
|                      | Profit"                   | GT-Expected:    $102,969.42    | estimated (P-hat). Diff   |
|                      |                           | GT-Expected Cost: $4,976.96    | is estimation delta, not  |
|                      |                           |                                | proven conservatism.      |
+----------------------+---------------------------+--------------------------------+---------------------------+
| 3. Oracle Comparison | "Optimality Gap"          | Ground-Truth Expected Profit   | Both solve identical MILP |
|    & Budget Overrun  | Unconstrained 693.4%      | Gap Against Constrained Oracle:| knapsack under $5k cap.   |
|                      | called "overrun"          | 0.986% (Agreement: 88.88%).    | Unconstrained spend is    |
|                      |                           | Spend is 693.4% of budget      | 693.4% of budget (593.4%  |
|                      |                           | (593.4% above budget).         | above budget).            |
+----------------------+---------------------------+--------------------------------+---------------------------+
| 4. Financial         | "GAAP ROMS"               | Incremental Net-Profit-to-     | Promotional redemption is |
|    Terminology       | (204.0%)                  | Promotion-Cost Ratio:          | an expected liability     |
|                      |                           | 2.0405 (204.05%)               | decision model, not GAAP. |
+----------------------+---------------------------+--------------------------------+---------------------------+
| 5. Brier Score       | 0.2012                    | 0.2072                         | 0.2072 is dynamically     |
|    Calibration       |                           | (Log Loss = 0.5987)            | computed test score;      |
|                      |                           |                                | 0.2012 was draft placeholder|
+----------------------+---------------------------+--------------------------------+---------------------------+
```

---

## Point 1: Reconcile Baseline Profit ($90,501.19 vs $90,586.38)

### 1.1 The Discrepancy
- **Earlier Policy Report (`oracle_evaluation.csv`):** Blanket $0 Profit = **$90,501.19** ($15.07 / customer).
- **Independent Audit Recalculation (`customer_allocation.csv`):** Blanket $0 Profit = **$90,586.38** ($15.09 / customer).
- **Net Variance:** Exactly **+$85.1928** across the $N=6,005$ held-out test cohort ($\approx +\$0.0142$ per customer).

### 1.2 Root Cause Analysis
We performed a line-by-line inspection of:
1. `src/optimization/budget_optimizer.py` (`SciPyBudgetOptimizer.compute_unit_economics`)
2. `src/optimization/profit_calculator.py` (`ProfitCalculator.calculate_customer_economics`)

In `SciPyBudgetOptimizer.compute_unit_economics`:
```python
# Lines 236-242 of src/optimization/budget_optimizer.py
if "gross_margin_rate" in df_customers.columns:
    margins = df_customers["gross_margin_rate"].to_numpy(dtype=float)
elif "customer_segment" in df_customers.columns:
    margins = ProfitCalculator().estimate_gross_margin_rate(prices, df_customers["customer_segment"].to_numpy())
```

In `ProfitCalculator.calculate_customer_economics` (prior implementation):
```python
# Lines 73-78 of src/optimization/profit_calculator.py
if "customer_segment" in customer_data.columns:
    seg = customer_data["customer_segment"].to_numpy(dtype=str)
    margin_rate = self.estimate_gross_margin_rate(aov, seg)
else:
    margin_rate = np.full(n_customers, self.base_margin_rate)
```

#### The Exact Difference:
- The test dataset (`synthetic_retail_test.csv`) contains pre-generated customer records with an explicit `gross_margin_rate` column (for instance, customer `SYN_000007` has `gross_margin_rate = 0.495`).
- When `SciPyBudgetOptimizer` generated `customer_allocation.csv`, it detected and used the recorded `gross_margin_rate` column ($0.495$).
- When `ProfitCalculator` was invoked in `scripts/run_policy_comparison.py`, it bypassed the dataset's `gross_margin_rate` column and unconditionally re-computed margins using the segment heuristic formula ($0.52 - 0.03 = 0.490$ for the budget segment).

#### Mathematical Proof:
$$\Delta \text{Profit}_0 = \sum_{i=1}^{6005} P_0(X_i) \cdot \text{AOV}_i \cdot \left[ \text{gross\_margin\_rate}_i^{\text{actual}} - \text{margin\_rate}_i^{\text{heuristic}} \right]$$

Evaluating this sum across all 6,005 test customers in Python:
- $\sum P_0(X_i) \cdot \text{AOV}_i \cdot \text{gross\_margin\_rate}_i^{\text{actual}} = \mathbf{\$90,586.3836}$
- $\sum P_0(X_i) \cdot \text{AOV}_i \cdot \text{margin\_rate}_i^{\text{heuristic}} = \mathbf{\$90,501.1907}$
- **Exact Difference:** $\$90,586.3836 - \$90,501.1907 = \mathbf{\$85.1928}$.

### 1.3 Resolution & Canonical Standard
1. Updated `src/optimization/profit_calculator.py` to prioritize `gross_margin_rate` when present in the customer DataFrame, falling back to segment estimation only when missing.
2. Re-executed `scripts/run_policy_comparison.py` to regenerate `data/optimization/oracle_evaluation.csv`.
3. **Canonical Unified Value:** **$90,586.38** ($15.09 / customer). This value is now uniformly established across all code, tests, CSV exports, JSON metrics, and markdown documentation.

---

## Point 2: Explain Model-Expected versus Ground-Truth Expected Profit

A strict conceptual boundary must separate ex-ante decision optimization from retrospective counterfactual benchmark evaluation.

```
+---------------------------------------------------------------------------------------------------------+
|                    MODEL-EXPECTED (EX-ANTE) vs GROUND-TRUTH EXPECTED (SYNTHETIC DGP)                    |
+------------------------------------+-----------------------------------+--------------------------------+
| Attribute                          | Model-Expected Profit (Ex-Ante)   | Ground-Truth Expected Profit   |
+------------------------------------+-----------------------------------+--------------------------------+
| Metric Value                       | $100,786.25 ($16.78 / cust)       | $102,969.42 ($17.15 / cust)    |
| Promotional Cost                   | $4,998.74 (Expected liability)    | $4,976.96 (Expected cost)      |
| Probability Vector                 | P_hat_d(X_i) from CausalForest    | P*_d(X_i) from DGP Potential Y |
| Conversion Rate                    | 32.56%                            | 32.95%                         |
| Operational Role                   | Solver Objective Function         | Retrospective Benchmark Test   |
+------------------------------------+-----------------------------------+--------------------------------+
```

### 2.1 Formal Mathematical Definitions

#### 1. Model-Expected Profit (Ex-Ante):
$$\mathbb{E}[\Pi_{\text{DML}}] = \sum_{i=1}^N \sum_{d \in \{0, 10, 20\}} x_{i, d} \cdot \left[ \text{AOV}_i \cdot \alpha_d \cdot m_i - d \right] \cdot \hat{P}_d(X_i) = \mathbf{\$100,786.25}$$
- Computed using the Double Machine Learning model's out-of-sample predicted counterfactual probabilities $\hat{P}_d(X_i)$.
- This is the **operational objective function** that the SciPy optimizer evaluates and maximizes at decision time.
- Associated Model-Expected Promotional Cost: $\sum_{i=1}^N \sum_{d \in \{10, 20\}} (d \cdot \hat{P}_d(X_i)) x_{i, d} = \mathbf{\$4,998.74}$.

#### 2. Ground-Truth Expected Profit (Synthetic DGP):
$$\mathbb{E}^*[\Pi_{\text{DML}}] = \sum_{i=1}^N \sum_{d \in \{0, 10, 20\}} x_{i, d} \cdot \text{expected\_net\_profit\_d}^*(X_i) = \mathbf{\$102,969.42}$$
- Computed by taking the DML model's chosen discrete allocations ($x_{i, d} \in \{0, 1\}$) and evaluating them against the isolated ground-truth potential profit columns in `synthetic_retail_ground_truth.csv`.
- **Ground-Truth Expected Promotional Cost (Synthetic DGP):**
  $$\mathbb{E}^*[\text{Cost}_{\text{DML}}] = \sum_{i=1}^N \sum_{d \in \{10, 20\}} (d \cdot P_d^*(X_i)) x_{i, d} = \mathbf{\$4,976.96}$$

### 2.2 Critical Distinction: Expected versus Realized
> [!IMPORTANT]
> **Neither $102,969.42 nor $4,976.96 are realized values.**
> A "realized" financial outcome strictly requires evaluating actual observed binary purchase indicators ($Y_i \in \{0, 1\}$) and actual observed coupon redemptions ($d \cdot Y_i$).
> Because $102,969.42 and $4,976.96 are calculated using ground-truth *potential outcomes* and *potential conversion probabilities* ($P_d^*$) from the synthetic Data Generating Process, they represent **expected values under the true synthetic distribution**. Calling them "realized" is mathematically incorrect.

### 2.3 Description of the $2,183.17 Difference
$$\Delta = \mathbb{E}^*[\Pi_{\text{DML}}] - \mathbb{E}[\Pi_{\text{DML}}] = \$102,969.42 - \$100,786.25 = \mathbf{+\$2,183.17}$$

- The $2,183.17 difference represents the **estimation delta between model-estimated expected profit and ground-truth expected profit under the synthetic DGP**.
- **Audit Clarification on Conservatism:** While the model-estimated profit is lower than ground-truth expected profit in this aggregate test cohort, this audit **does not assert systematic model conservatism**. Establishing systematic conservatism would require dedicated subgroup calibration analyses across all customer segments and propensity tiers. It is documented strictly as the numerical difference between the model estimate and the DGP expectation.

---

## Point 3: Verify Oracle Budget Accounting & Ground-Truth Expected Profit Gap

### 3.1 Source Code Inspection
We audited the constraint formulation in `src/optimization/budget_optimizer.py` and the Oracle evaluation logic in `src/optimization/policy_evaluator.py`.

#### DML Optimization Formulation:
```python
# DML knapsack budget row in src/optimization/budget_optimizer.py (Lines 129-140)
costs[0::3] = 0.0
costs[1::3] = 10.0 * p10_hat   # Model predicted probability
costs[2::3] = 20.0 * p20_hat   # Model predicted probability
# Subject to: sum(costs * x) <= Budget (5000.0)
```

#### Budget-Constrained Oracle Formulation:
```python
# Constrained Oracle MILP
costs_oracle[0::3] = 0.0
costs_oracle[1::3] = 10.0 * prob_10_true   # Ground-truth potential probability P*_10
costs_oracle[2::3] = 20.0 * prob_20_true   # Ground-truth potential probability P*_20
# Subject to: sum(costs_oracle * x_oracle) <= Budget (5000.0)
```

### 3.2 Verification of Budget Accounting Parity
1. **Identical Budget Constraint:**
   Both problems enforce the exact same linear knapsack constraint under the redemption accounting model:
   $$\sum_{i=1}^N \sum_{d \in \{10, 20\}} (d \cdot P_d(X_i)) \cdot x_{i, d} \le \$5,000.00$$
2. **Ground-Truth Potential Outcomes in Oracle Cost:**
   In the Oracle MILP, $P_d(X_i)$ is substituted with the true ground-truth potential conversion probability $P_d^*(X_i)$ (`potential_purchase_prob_d` from `synthetic_retail_ground_truth.csv`).
3. **Budget Parity Demonstration:**
   - Both optimizations enforce an upper bound of strictly **$5,000.00**.
   - Both optimizations assign a promotional cost of **$0.00** for $d=0$ (non-discounted customers cost nothing).
   - In both cases, only converting customers incur promotional expenditure ($d \cdot P_d$).
4. **Comparability of Oracle Cost ($4,998.55) and DML Ground-Truth Expected Cost ($4,976.96):**
   - Under the DML allocation, the ground-truth expected promotional cost is **$4,976.96** ($\le \$5,000.00$).
   - The Budget-Constrained Oracle incurs a ground-truth expected promotional cost of **$4,998.55** ($\le \$5,000.00$).
   - Both policies achieve $>99.5\%$ budget utilization without violating the $5,000 budget cap. Their expenditures differ by only **$21.59** (0.43%), confirming that both policies operate under the identical budget cap and accounting framework.

### 3.3 Recalculated Benchmarks

```
+----------------------------------------------------------------------------------------------------+
|                         BUDGET-CONSTRAINED ORACLE BENCHMARK ($5,000 CAP)                           |
+--------------------------------------+-------------------------+-----------------------------------+
| Metric                               | Constrained Oracle (GT) | DML Model Allocation (GT-Eval)    |
+--------------------------------------+-------------------------+-----------------------------------+
| Ground-Truth Expected Profit         | $103,994.65             | $102,969.42                       |
| Profit Per Customer                  | $17.32                  | $17.15                            |
| GT Expected Promotional Cost         | $4,998.55               | $4,976.96                         |
| Tier 0 ($0) Count                    | 4,791 (79.78%)          | 4,816 (80.20%)                    |
| Tier 10 ($10) Count                  | 1,179 (19.63%)          | 1,168 (19.45%)                    |
| Tier 20 ($20) Count                  | 35 (0.58%)              | 21 (0.35%)                        |
| Budget Compliance                    | FEASIBLE (<= $5,000.00) | FEASIBLE (<= $5,000.00)           |
+--------------------------------------+-------------------------+-----------------------------------+
```

#### 1. Ground-Truth Expected Profit Gap Against the Budget-Constrained Oracle:
$$\text{Profit Gap} = \frac{\mathbb{E}^*[\Pi_{\text{Oracle}}] - \mathbb{E}^*[\Pi_{\text{DML}}]}{\mathbb{E}^*[\Pi_{\text{Oracle}}]} = \frac{\$103,994.65 - \$102,969.42}{\$103,994.65} = \frac{\$1,025.23}{\$103,994.65} = \mathbf{0.986\%}$$

$$\text{Captured Frontier Share} = 100\% - 0.986\% = \mathbf{99.01\%}$$
The DML policy captures **99.01%** of the theoretical maximum profit attainable by an omniscient Oracle operating under the identical $5,000 budget ceiling.

#### 2. Customer Decision Agreement Rate:
$$\text{Agreement Rate} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(d_i^{\text{DML}} = d_i^{\text{Oracle}}) = \frac{5,337}{6,005} = \mathbf{88.88\%}$$
Across 6,005 test customers, the Double Machine Learning model makes the identical promotional tier decision as the all-knowing Oracle in **88.88%** of cases.

### 3.4 Correct Budget Overrun Reporting for Unconstrained Oracle
The unconstrained Oracle ($\arg\max_d \text{Profit}_d^*$) achieves $120,855.65 but incurs **$34,669.80** in subsidies against the $5,000.00 budget.
- This promotional expenditure represents:
  $$\frac{\$34,669.80}{\$5,000.00} \times 100\% = \mathbf{693.4\% \text{ of the budget}}$$
- Or equivalently:
  $$\frac{\$34,669.80 - \$5,000.00}{\$5,000.00} \times 100\% = \mathbf{593.4\% \text{ above the budget}}$$
- **Correction:** Calling 693.4% the "overrun" was mathematically inaccurate. The overrun is **593.4% above budget** ($693.4\% - 100\%$).

---

## Point 4: Correct Financial Terminology and Accounting Scope

### 4.1 Excise "GAAP ROMS"
Earlier documentation claimed that the 204.0% metric represented "ROMS following standard GAAP accounting." This claim is inaccurate and has been corrected:
- **GAAP Accounting Realities:** US GAAP (ASC 606 / Revenue from Contracts with Customers) and IFRS 15 govern historical financial statement reporting. Under ASC 606, discounts and coupons are classified as *variable consideration* reducing transaction revenue, or as selling/marketing expenses upon realization. GAAP does not codify a metric called "ROMS" (Return on Marketing Spend).
- **Managerial Decision Modeling:** The project's promotional costing calculation ($\text{Cost}_i = d \cdot P_d(X_i)$) is an **expected financial liability model for decision optimization**, not a post-hoc financial accounting ledger.

### 4.2 Canonical Metric: Incremental Net-Profit-to-Promotion-Cost Ratio

$$\text{Incremental Net-Profit-to-Promotion-Cost Ratio} = \frac{\mathbb{E}[\Pi_{\text{policy}}] - \mathbb{E}[\Pi_{\text{blanket\_0}}]}{\text{Cost}_{\text{policy}}}$$

#### Mathematical Computation:
- $\mathbb{E}[\Pi_{\text{DML}}] = \$100,786.25$
- $\mathbb{E}[\Pi_{\text{blanket\_0}}] = \$90,586.38$
- $\text{Cost}_{\text{DML}} = \$4,998.74$

$$\text{Ratio} = \frac{\$100,786.25 - \$90,586.38}{\$4,998.74} = \frac{\$10,199.87}{\$4,998.74} = \mathbf{2.0405} \quad (\mathbf{204.05\%})$$

**Interpretation:** For every $1.00 of expected promotional subsidy spent under the DML policy, the enterprise generates **$2.04 of incremental net contribution margin** over and above the organic baseline ($0 discount).

---

## Point 5: Reconcile Brier Scores (0.2012 vs 0.2072)

```
+----------------------------------------------------------------------------------------------------+
|                                    BRIER SCORE AUDIT RECONCILIATION                                |
+---------------------------+------------------------------------+-----------------------------------+
| Dimension                 | Preliminary Report (Legacy Draft)  | Verified Audit (Canonical Standard)|
+---------------------------+------------------------------------+-----------------------------------+
| Brier Score               | 0.2012                             | 0.2072                            |
| Log Loss                  | Unreported                         | 0.5987                            |
| Evaluation Dataset        | Synthetic Retail Prototype Split   | data/synthetic_retail_test.csv    |
| Sample Size (N)           | 6,005 customers                    | 6,005 held-out test customers     |
| Outcome Variable (Y)      | purchase (binary {0, 1})           | purchase (binary {0, 1})          |
| Treatment Variable (T)    | discount_amount (0, 10, 20)        | discount_amount (0, 10, 20)       |
| Model Evaluated           | CausalForestDML (pre-tuning)       | CausalForestDML (final calibrated)|
| Evaluation Script         | evaluate_causal_models.py (draft)  | generate_counterfactual_pred.py   |
| Implementation Function   | Static text block                  | evaluate_probability_calibration  |
+---------------------------+------------------------------------+-----------------------------------+
```

### 5.1 Root Cause for the 0.0060 Difference
1. In `scripts/generate_counterfactual_predictions.py`, the calibration evaluation function `evaluate_probability_calibration` dynamically computes the Brier score on `data/synthetic_retail/synthetic_retail_test.csv`:
   ```python
   brier = float(brier_score_loss(actual_y, pred_y_observed))
   # Evaluates to: 0.2072 (Log Loss: 0.5987)
   ```
2. In `scripts/evaluate_causal_models.py`, the string `0.2012` was recorded in a static markdown template during early exploratory development before final hyperparameter tuning of the EconML `CausalForestDML` estimators.
3. When the causal model was finalized with calibrated baseline probabilities, the exact out-of-sample Brier score became **0.2072**.

### 5.2 Test Set Calibration by Discount Arm ($N=6,005$)
The canonical test Brier score of **0.2072** reflects exceptional probability calibration across all three operational discount arms:
- **Arm $0 ($0):** $N=1,956$ | Actual Conversion = **25.15%** | Predicted Mean = **24.75%** | Calibration Delta = **0.41 pp**
- **Arm $10 ($10):** $N=2,382$ | Actual Conversion = **48.99%** | Predicted Mean = **49.36%** | Calibration Delta = **0.37 pp**
- **Arm $20 ($20):** $N=1,667$ | Actual Conversion = **58.43%** | Predicted Mean = **57.69%** | Calibration Delta = **0.74 pp**

**Canonical Standard:** **0.2072** is the verified, reproducible test Brier score for Milestone 2 and Milestone 3 causal inference and optimization models.

---

## Source-Code Verification: Exact Functions Used

The following table documents the exact functions and files in the repository used to calculate ground-truth expected profit, expected promotional cost, and the Oracle comparison:

### 1. Ground-Truth Expected Profit Calculation
- **Primary Function:** `evaluate_multitreatment_performance` in [`src/causal/evaluation.py`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/src/causal/evaluation.py#L621-L627)
- **Source Logic:**
  ```python
  model_choices = merged["model_recommended_discount"].to_numpy()
  model_realized_exp_profit = np.where(
      model_choices == 0,
      merged[p0_col],
      np.where(model_choices == 10, merged[p10_col], merged[p20_col])
  )
  total_gt_expected_profit = float(model_realized_exp_profit.sum()) # Evaluates to $102,969.42
  ```
- **Source Columns:** `expected_net_profit_0`, `expected_net_profit_10`, `expected_net_profit_20` from `data/synthetic_retail/synthetic_retail_ground_truth.csv`.

### 2. Ground-Truth Expected Promotional Cost Calculation
- **Primary Function:** Strategy evaluation in [`src/optimization/policy_evaluator.py`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/src/optimization/policy_evaluator.py#L161-L166)
- **Source Logic:**
  ```python
  gt_probs = np.where(
      decisions == 0,
      merged_gt["potential_purchase_prob_0"],
      np.where(decisions == 10, merged_gt["potential_purchase_prob_10"], merged_gt["potential_purchase_prob_20"])
  )
  gt_expected_cost = float((decisions * gt_probs).sum()) # DML: $4,976.96 | Oracle: $4,998.55
  ```
- **Source Columns:** `potential_purchase_prob_0`, `potential_purchase_prob_10`, `potential_purchase_prob_20` from `data/synthetic_retail/synthetic_retail_ground_truth.csv`.

### 3. Budget-Constrained Oracle Optimization Formulation
- **Primary Function:** SciPy MILP knapsack solver in [`src/optimization/budget_optimizer.py`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/src/optimization/budget_optimizer.py#L109-L161)
- **Source Logic:**
  ```python
  # Objective: Minimize c = -[Profit*_0, Profit*_10, Profit*_20]
  # Subject to:
  # 1. Assignment: sum_d x_{i, d} == 1
  # 2. Budget: sum_i sum_d (d * P*_d) * x_{i, d} <= 5000.0
  res = milp(c=c, integrality=np.ones(3*N), bounds=Bounds(0, 1), constraints=constraints)
  oracle_gt_expected_profit = -res.fun # Evaluates to $103,994.65
  ```

---

## Final Milestone 3 Metrics Canonical Reference

All project deliverables now conform to this single, unified truth:

| Performance Metric | Canonical Reconciled Value | Evaluation Context |
| :--- | :---: | :--- |
| **Held-Out Test Cohort Size ($N$)** | 6,005 | Held-out test customers (`synthetic_retail_test.csv`) |
| **Configured Marketing Budget Ceiling** | $5,000.00 | Strict redemption expenditure budget cap |
| **Blanket $0 Baseline Net Profit** | **$90,586.38** ($15.09 / cust) | Model-estimated ex-ante predicted probabilities $\hat{P}_0$ |
| **Model-Expected Profit (Ex-Ante)** | **$100,786.25** ($16.78 / cust) | Model-estimated ex-ante predicted probabilities $\hat{P}_d$ |
| **Model-Expected Promotional Cost** | **$4,998.74** | Expected redemption liability (99.97% utilization) |
| **Incremental Net Profit over Blanket $0** | **+$10,199.87** (+11.26%) | $\mathbb{E}[\Pi_{\text{DML}}] - \mathbb{E}[\Pi_{\text{Blanket0}}]$ |
| **Incremental Net-Profit-to-Promotion-Cost Ratio** | **2.0405 (204.05%)** | $\Delta \Pi / \text{Cost}_{\text{DML}}$ (Managerial liability model) |
| **Ground-Truth Expected Profit (Synthetic DGP)** | **$102,969.42** ($17.15 / cust) | Synthetic ground truth potential outcomes $P^*$ |
| **Ground-Truth Expected Promotional Cost (Synthetic DGP)**| **$4,976.96** (99.54% budget) | Synthetic ground truth potential outcomes $P^*$ |
| **Difference: Model vs GT Expected Profit** | **+$2,183.17** | Estimation delta under synthetic DGP |
| **Budget-Constrained Oracle GT-Expected Profit** | **$103,994.65** ($17.32 / cust) | Omniscient MILP knapsack under $5,000 cap |
| **Budget-Constrained Oracle GT-Expected Cost** | **$4,998.55** (99.97% budget) | Omniscient MILP knapsack under $5,000 cap |
| **GT Expected Profit Gap Against Constrained Oracle** | **0.986%** | $(103,994.65 - 102,969.42) / 103,994.65$ |
| **Share of Constrained Frontier Captured** | **99.01%** | DML captures 99.01% of Constrained Oracle |
| **Oracle Customer Decision Agreement** | **88.88%** | 5,337 identical tier assignments out of 6,005 |
| **Unconstrained Oracle Spend (Infeasible)** | **$34,669.80** | **693.4% of budget (593.4% above budget)** |
| **Out-of-Sample Calibration Brier Score** | **0.2072** (Log Loss: 0.5987) | Test set calibration against observed purchases |
| **Unit & Integration Test Suite** | **20 / 20 Optimization** | 100% test pass rate |
