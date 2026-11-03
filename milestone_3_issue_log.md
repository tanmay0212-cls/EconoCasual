# EconoCasual Milestone 3 Issue & Audit Log

**Project:** EconoCasual: Dynamic Pricing via Double Machine Learning  
**Milestone:** 3 – Prescriptive Budget Optimization Engine  
**Auditor:** Senior Causal ML Engineer & Software Quality Auditor  
**Date:** 2026-11-03  

---

## 1. Issue Summary

| Issue ID | Severity | Category | Component | Status |
| :---: | :---: | :--- | :--- | :---: |
| **ISSUE-01** | **MEDIUM** | Evaluation Methodology | `policy_evaluator.py`, `oracle_evaluation.csv` | Resolved in Review |
| **ISSUE-02** | **LOW** | Economic Assumptions | `profit_calculator.py` | Resolved (Code Unified) |
| **ISSUE-03** | **LOW** | Scalability / Solver | `budget_optimizer.py` | Architectural Note |
| **ISSUE-04** | **LOW** | Causal Modeling | `profit_calculator.py` | Future Enhancement |
| **ISSUE-05** | **LOW** | Data Consistency | `profit_calculator.py`, `budget_optimizer.py` | Resolved ($90,586.38 Unified) |
| **ISSUE-06** | **MEDIUM** | Financial Terminology | Documentation & Metrics | Resolved (Terminology Corrected) |
| **ISSUE-07** | **LOW** | Metrics Verification | `causal_model_evaluation.md` | Resolved (0.2072 Reconciled) |
| **ISSUE-08** | **LOW** | Terminology Precision | Audit & Reconciliation Reports | Resolved (Ground-Truth Expected Terminology) |

---

## 2. Detailed Issue Records

### ISSUE-01: Unconstrained Oracle vs Budget-Constrained Oracle Benchmark
- **Severity:** Medium
- **Location:** `src/optimization/policy_evaluator.py`, `scripts/run_policy_comparison.py`, `data/optimization/oracle_evaluation.csv`
- **Description:** In the initial strategy comparison table, the Oracle policy was evaluated using the *unconstrained* optimal decision rule ($\arg\max_d \text{Profit}_d$), which assumes infinite marketing budget. On the 6,005 test customers, this unconstrained Oracle spends **$34,669.80** in discount subsidies.
- **Budget Overrun Characterization:** A spend of $34,669.80 against a $5,000 budget represents **693.4% of the budget**, or **593.4% above the budget** (not a 693.4% overrun).
- **Finding:** While labeling an unconstrained policy as INFEASIBLE under a $5,000 cap is technically correct, it does not answer the executive question: *"What is the theoretical maximum profit achievable by an all-knowing Oracle subject to the exact same $5,000 budget constraint?"*
- **Resolution:** We implemented an independent budget-constrained Oracle solver using SciPy MILP directly on the true potential outcomes. Under the $5,000 cap:
  - **Ground-Truth Expected Profit (Budget-Constrained Oracle):** **$103,994.65** ($17.32 / customer) at a ground-truth expected promotional cost of **$4,998.55**.
  - **Ground-Truth Expected Profit (DML Policy Allocation):** **$102,969.42** ($17.15 / customer) at a ground-truth expected promotional cost of **$4,976.96**.
  - **Ground-Truth Expected Profit Gap Against the Budget-Constrained Oracle:** **0.986%** (The DML model captures **99.01%** of the true constrained Oracle frontier).
  - **Customer Decision Agreement Rate:** **88.88%**.
- **Action Taken:** Both the Unconstrained Theoretical Frontier ($120.8k) and the Budget-Constrained Oracle ($104.0k) are explicitly reported in `milestone_3_recalculated_metrics.json` and `milestone_3_final_reconciliation.md`.

---

### ISSUE-02 & ISSUE-05: Baseline Blanket $0 Profit Discrepancy ($90,501.19 vs $90,586.38)
- **Severity:** Low
- **Location:** `src/optimization/profit_calculator.py` vs `src/optimization/budget_optimizer.py`
- **Description:** An $85.19 variance existed between the initial policy comparison report ($90,501.19) and the allocation audit ($90,586.38).
- **Root Cause Identified:** 
  - `SciPyBudgetOptimizer.compute_unit_economics` preferred the explicit pre-treatment `gross_margin_rate` column present in `synthetic_retail_test.csv` (which incorporates customer-specific margin attributes).
  - In contrast, `ProfitCalculator.calculate_customer_economics` unconditionally re-estimated margins from customer segment alone ($0.52 - 0.03 = 0.49$ for budget segment), discarding customer-specific recorded margin rates.
  - Evaluating $\sum P_0 \cdot \text{AOV} \cdot \text{margin}_{\text{actual}} = \$90,586.38$, whereas $\sum P_0 \cdot \text{AOV} \cdot \text{margin}_{\text{estimated}} = \$90,501.19$. Difference across 6,005 customers is exactly $\mathbf{\$85.1928}$.
- **Resolution:** Updated `ProfitCalculator.calculate_customer_economics` to check for `gross_margin_rate` before falling back to segment heuristics. Re-ran `scripts/run_policy_comparison.py`. All reports, CSV files, and JSON metric artifacts now uniformly reflect the canonical value: **$90,586.38** ($15.09/customer).

---

### ISSUE-06: Financial Terminology and GAAP Accounting Characterization
- **Severity:** Medium
- **Location:** All milestone reports and summaries
- **Description:** The 204.0% performance metric was previously described as "ROMS following standard GAAP accounting."
- **Audit Clarification:**
  1. Standard GAAP accounting does not define "ROMS" (Return on Marketing Spend). In financial accounting under ASC 606 / IFRS 15, coupon discounts are recognized as variable consideration reducing transaction price, or as marketing expenses upon coupon redemption.
  2. The project's promotional costing metric ($\text{Cost} = \sum d \cdot P_d$) is an **expected-cost managerial decision model**, representing the expected financial liability incurred upon redemption, not standard historical GAAP reporting.
  3. The 204.0% figure represents incremental net profit over organic baseline divided by expected promotional subsidy expenditure.
- **Resolution:**
  - Renamed metric to: **"Incremental Net-Profit-to-Promotion-Cost Ratio"**.
  - Formal formula documented:
    $$\text{Incremental Ratio} = \frac{\text{Profit}_{\text{policy}} - \text{Profit}_{\text{blanket\_0}}}{\text{Cost}_{\text{policy}}} = \frac{\$100,786.25 - \$90,586.38}{\$4,998.74} = \frac{\$10,199.87}{\$4,998.74} = \mathbf{2.0405} \; (204.05\%)$$
  - Any reference to GAAP accounting has been excised and replaced with managerial decision modeling notation.

---

### ISSUE-07: Brier Score Reconciliation (0.2012 vs 0.2072)
- **Severity:** Low
- **Location:** `reports/causal_model_evaluation.md` vs `scripts/generate_counterfactual_predictions.py`
- **Description:** Earlier narrative documentation reported an overall Brier score of `0.2012`, while the independent audit computed `0.2072` ($\Delta = 0.0060$).
- **Audit Findings:**
  - `0.2072` is the true, dynamically computed Brier score on `data/synthetic_retail/synthetic_retail_test.csv` ($N=6,005$) against the observed binary outcome (`purchase`) under observed treatments (`discount_amount`).
  - `0.2012` was a static placeholder string recorded in an early markdown draft during preliminary model prototyping prior to hyperparameter finalization.
- **Resolution:** Reconciled both figures, confirmed `0.2072` as the verified canonical test-set calibration score, and fully documented calibration breakdowns by discount arm.

---

### ISSUE-08: Terminology Precision for Expected vs Realized Values and Model Difference
- **Severity:** Low
- **Location:** `milestone_3_final_reconciliation.md`, `milestone_3_independent_review.md`
- **Description:** Previous reports referred to $102,969.42 as "Realized Synthetic Profit" and $4,976.96 as "Realized Cost", and attributed the $2,183.17 difference to "model conservatism."
- **Audit Corrections:**
  1. $102,969.42 is calculated using ground-truth potential profit columns (`expected_net_profit_d`), and $4,976.96 is calculated using ground-truth potential probabilities ($d \cdot P_d^*$). Because neither is calculated using actual observed binary purchase/redemption outcomes, they must be designated:
     - **Ground-Truth Expected Profit (Synthetic DGP): $102,969.42**
     - **Ground-Truth Expected Promotional Cost (Synthetic DGP): $4,976.96**
     - The ex-ante model prediction is designated: **Model-Expected Profit (Ex-Ante): $100,786.25**.
  2. The $2,183.17 discrepancy is strictly described as the **difference between model-estimated expected profit and ground-truth expected profit under the synthetic DGP**. It is not claimed as systematic conservatism without additional subgroup or calibration evidence.
  3. The Oracle comparison is formally designated: **Ground-Truth Expected Profit Gap Against the Budget-Constrained Oracle (0.986%)**.
- **Status:** Fully resolved across all project reports and JSON schemas.

---

### ISSUE-03: MILP Branch-and-Bound Computational Scaling for Enterprise Cohorts
- **Severity:** Low
- **Location:** `src/optimization/budget_optimizer.py`
- **Description:** For the 6,005 test customers ($18,015$ binary variables), the SciPy HiGHS MILP solver executes in ~0.50 seconds. However, if deployed on 1,000,000 customers ($3,000,000$ binary variables), solving a binary integer knapsack problem to provable integrality tolerance could experience latency or memory pressure.
- **Recommendation:** For enterprise batches exceeding 100,000 records:
  1. Batch customers into strategic micro-segments (e.g. 20,000 per sub-problem).
  2. Or utilize Dantzig's Greedy Knapsack heuristic (ranking customers by marginal return-on-spend $\frac{\Delta \text{Profit}}{\Delta \text{Cost}}$) as an $O(N \log N)$ near-optimal approximation.

---

### ISSUE-04: Fixed Basket Expansion Multipliers
- **Severity:** Low
- **Location:** `src/optimization/profit_calculator.py`
- **Description:** The calculator assumes fixed basket expansion factors: $1.00$ ($0), $1.03$ ($10 coupon), and $1.06$ ($20 coupon).
- **Implication:** While coupons empirically increase basket size in retail, price elasticity of basket size is heterogeneous across customers.
- **Recommendation:** In future iterations, fit a secondary causal estimator for continuous spend conditional on purchase (similar to the Hillstrom continuous spend model) to personalize basket expansion predictions.
