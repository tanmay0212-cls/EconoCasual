# Prescriptive Optimization Summary: Budget-Constrained Discount Allocation

**Project:** EconoCausal — Dynamic Pricing via Double Machine Learning  
**Milestone:** Week 3 — SciPy Prescriptive Optimization Engine & Allocation Matrix  
**Optimization Method:** Mixed-Integer Linear Programming (`scipy.optimize.milp` / HiGHS Solver)  
**Campaign Budget Cap:** **$5,000.00**  
**Evaluation Population:** **3,000 customers** (held-out test cohort)  
**Status:** **OPTIMIZATION COMPLETE & CONSTRAINTS VERIFIED**

---

## 1. Executive Summary & Core Results

The Week 3 prescriptive optimization engine translates the multi-treatment causal model's
counterfactual predictions into an actionable, profit-maximizing promotional discount policy.
Under a strict **$5,000.00 marketing budget**, the solver allocated personalized
discounts across the 3,000 evaluation customers.

### Key Optimization Highlights:
* **Budget Feasibility:** Total expected discount cost is **$4,604.20**, achieving **92.1% budget utilization** without violating the budget ceiling.
* **Superiority Over Blanket Policies:**
  * **Blanket $10** costs **$16,246.60** (324.9% of budget) — **severely violates the budget by $11,246.60**.
  * **Blanket $20** costs **$35,705.84** (714.1% of budget) — **grossly over budget**.
  * **Personalized Allocation** stays 100% compliant while generating **$53,082.35** in expected net profit ($+2,066.60 net contribution lift over baseline).
* **Targeting Efficiency:**
  * **$0 (Standard Pricing):** 2,162 customers (72.1%) — customers with high organic conversion (VIPs) or low margin elasticity.
  * **$10 Coupon:** 836 customers (27.9%) — customers with strong incremental conversion lift relative to coupon cost.
  * **$20 Coupon:** 2 customers (0.1%) — high-basket, high-elasticity shoppers where large lift justifies the $20 redemption.

---

## 2. Business Objectives & Economic Accounting

### 2.1 Accounting Assumptions (Redemption Model)
In accordance with retail e-commerce accounting standards, promotional discounts are incurred
**only upon coupon redemption** (successful conversion):

$$\mathbb{E}[\text{Discount Cost}_{i, d}] = d \times \hat{P}(\text{purchase} = 1 \mid X_i, d)$$

| Economic Metric | Formula | Business Meaning |
| :--- | :--- | :--- |
| **Predicted Purchase Probability** | $\hat{P}_{i, d} \in [0, 1]$ | Counterfactual probability of customer $i$ purchasing under discount $d$. |
| **Expected Gross Revenue** | $\text{price}_i \times \hat{P}_{i, d}$ | Gross sales volume before discount deductions. |
| **Expected Discount Cost** | $d \times \hat{P}_{i, d}$ | Coupon redemption cost ($0 if customer does not buy). |
| **Expected Net Revenue** | $(\text{price}_i - d) \times \hat{P}_{i, d}$ | Net cash collected after deducting redeemed promotional face value. |
| **Expected Gross Profit** | $\text{price}_i \times \text{margin}_i \times \hat{P}_{i, d}$ | Gross product margin before marketing promotional expense. |
| **Expected Net Profit (Contribution)** | $(\text{price}_i \times \text{margin}_i - d) \times \hat{P}_{i, d}$ | **Primary Objective:** True bottom-line cash contribution to operating profit. |

### 2.2 Distinguishing the PDF Objective vs Profit Maximization
* **PDF Predicted-Revenue Objective:** The original INFOTACT prompt specifies *"maximize predicted revenue under a fixed marketing budget"*.
  Under this formulation, the objective optimizes net revenue $(\text{price} - d) \times \hat{P}$.
* **Profit-Based Business Objective (Default):** Maximizes true contribution margin $(\text{price} \times \text{margin} - d) \times \hat{P}$, preventing the optimizer from allocating high discounts to low-margin items.
* Both objectives are supported via `objective_type='net_profit'` and `objective_type='net_revenue'`. In all reports, both expected revenue and expected profit are reported simultaneously.

---

## 3. Mathematical Optimization Formulation

The allocation problem is modeled as a **Multiple-Choice Knapsack Problem (MCKP)** solved via
**Binary Integer Linear Programming (BILP)** using `scipy.optimize.milp` with the **HiGHS** backend:

$$\max_{z_{i, d}} \sum_{i=1}^N \sum_{d \in \{0, 10, 20\}} V_{i, d} \cdot z_{i, d}$$

**Subject to:**
1. **Mutually Exclusive Tier Allocation (Exactly one action per customer):**
   $$\sum_{d \in \{0, 10, 20\}} z_{i, d} = 1 \quad \forall i \in \{1, \dots, N\}$$
2. **Strict Marketing Budget Ceiling:**
   $$\sum_{i=1}^N \sum_{d \in \{10, 20\}} \left( d \times \hat{P}_{i, d} \right) \cdot z_{i, d} \le \text{Budget}$$
3. **Binary Integrality Domain:**
   $$z_{i, d} \in \{0, 1\} \quad \forall i, d$$

* **Solver Status:** `0` (Optimization terminated successfully. (HiGHS Status 7: Optimal))
* **Optimality Certificate:** HiGHS Status 7 (Global Optimal Solution Verified).
* **Runtime:** Solved in under **0.15 seconds** for 3,000 customers using sparse CSC constraint matrices.

---

## 4. Strategy Comparison: Fair Benchmark

All 4 strategies were evaluated on the **exact same 3,000 customer population** using
identical economic pricing and margin profiles:

| Strategy | Budget Cap | Exp Discount Cost | Budget Util (%) | Compliant? | Exp Conversions | Exp Gross Rev ($) | Exp Net Rev ($) | Exp Net Profit ($) | Profit Lift ($) | Profit Lift (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| No Discount (All $0) | $5,000.00 | $0.00 | 0.0% | YES | 1,152.6 | $106,473.01 | $106,473.01 | $51,015.75 | $0.00 | 0.0% |
| Blanket $10 Discount | $5,000.00 | $16,246.60 | 324.9% | NO (Over Budget) | 1,624.7 | $137,756.90 | $121,510.30 | $47,828.63 | $-3,187.12 | -6.2% |
| Blanket $20 Discount | $5,000.00 | $35,705.84 | 714.1% | NO (Over Budget) | 1,785.3 | $146,789.04 | $111,083.20 | $31,841.21 | $-19,174.54 | -37.6% |
| Optimized Personalized (net_profit) | $5,000.00 | $4,604.20 | 92.1% | YES | 1,308.1 | $121,447.78 | $116,843.59 | $53,082.35 | $+2,066.60 | +4.1% |

### Strategic Insights:
1. **The Blanket Policy Fallacy:** Blanket policies ($10 or $20 for everyone) are completely infeasible in production. Offering $10 to all 3,000 customers would require a budget of **$16,246.60** (>3x the allowed budget cap).
2. **Margin Protection:** Even if unlimited budget were available, blanket $20 generates less net profit than blanket $10 because the $20 coupon wipes out product margins on low-priced items.
3. **Personalization Advantage:** The SciPy optimizer selectively deploys discounts where elasticity is high and basket margin is sufficiently thick to absorb coupon redemption.

---

## 5. Allocation Matrix (Sample Preview)

The full customer-level policy is saved to [`reports/allocation_matrix.csv`](allocation_matrix.csv) (3,000 records).

| customer_id | predicted_purchase_probability_0 | predicted_purchase_probability_10 | predicted_purchase_probability_20 | predicted_effect_10_vs_0 | predicted_effect_20_vs_0 | recommended_discount | expected_discount_cost | expected_revenue | expected_net_profit |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| OPT_012001 | 0.369754 | 0.520709 | 0.575259 | 0.150955 | 0.205505 | 0 | 0.0 | 31.4919 | 10.0459 |
| OPT_012002 | 0.70823 | 0.83962 | 0.887738 | 0.13139 | 0.179508 | 0 | 0.0 | 31.2684 | 12.6668 |
| OPT_012003 | 0.147162 | 0.279962 | 0.369383 | 0.1328 | 0.222221 | 0 | 0.0 | 5.1345 | 1.5855 |
| OPT_012004 | 0.707007 | 0.775342 | 0.756712 | 0.068335 | 0.049705 | 0 | 0.0 | 81.6876 | 43.9398 |
| OPT_012005 | 0.117624 | 0.206689 | 0.306662 | 0.089065 | 0.189038 | 0 | 0.0 | 4.5344 | 1.3785 |
| OPT_012006 | 0.606399 | 0.793445 | 0.835957 | 0.187046 | 0.229558 | 0 | 0.0 | 37.0752 | 16.0721 |
| OPT_012007 | 0.328354 | 0.446816 | 0.489546 | 0.118462 | 0.161192 | 0 | 0.0 | 6.9644 | 2.0294 |
| OPT_012008 | 0.76958 | 0.840851 | 0.820861 | 0.071271 | 0.051281 | 0 | 0.0 | 159.0722 | 88.2851 |
| OPT_012009 | 0.398235 | 0.564628 | 0.604475 | 0.166393 | 0.20624 | 10 | 5.6463 | 58.3656 | 23.9451 |
| OPT_012010 | 0.332851 | 0.448948 | 0.504912 | 0.116097 | 0.172061 | 0 | 0.0 | 21.4522 | 10.8977 |

---

## 6. Optimization Validation & Guardrail Verification

| Validation Check | Target Requirement | Empirical Result | Status |
| :--- | :--- | :---: | :---: |
| **Budget Feasibility** | Total Cost $\le$ $5,000.00$ | **$4,604.20** | ✅ **PASS** |
| **Exact Single Action** | $\sum_d z_{i, d} = 1$ for all $i$ | **100% adherence** (3,000 / 3,000) | ✅ **PASS** |
| **Valid Discount Domain** | Recommended $\in \{0, 10, 20\}$ | **100% valid** | ✅ **PASS** |
| **Solver Termination** | Status 0 (Optimal solution found) | **Optimal (HiGHS Status 7)** | ✅ **PASS** |
| **Edge Case: Zero Budget** | $B = 0 \implies$ All $0 | **Tested & Verified** | ✅ **PASS** |
| **Reproducibility** | Deterministic outputs across runs | **100% Bit-Exact** | ✅ **PASS** |

---

## 7. Artifacts Generated

* **Engine Module:** `src/optimization/budget_optimizer.py`
* **Allocation Matrix:** `reports/allocation_matrix.csv`
* **Comprehensive Summary:** `reports/optimization_summary.md`
* **Test Suite:** `tests/test_optimization.py`
