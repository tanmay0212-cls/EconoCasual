# EconoCasual 2.0 — Phase 3: Uncertainty-Aware & Fairness-Constrained MILP

**Document Status:** Complete & Verified  
**Optimization Solver:** SciPy MILP / HiGHS Branch-and-Cut Mixed-Integer Linear Programming Engine  
**Evaluation Cohort:** Synthetic Retail Test Split (`data/synthetic_retail/synthetic_retail_test.csv`, $N = 6,005$)  
**Promotional Budget Ceiling:** $\$5,000.00$ (Ex-Ante Expected Redemption Expenditure)  
**Allocation Artifact:** `data/optimization/advanced_allocations.csv` ($6,005$ customer decisions across 4 regimes)  
**Benchmark Artifact:** `experiments/results/advanced_optimization_benchmarks.json`  

---

## Executive Summary

Phase 3 introduces an advanced operations research optimization layer to EconoCasual, extending the baseline HiGHS mixed-integer linear programming (MILP) solver to account for **causal estimation uncertainty** and **business-defined demographic fairness constraints**.

The original baseline optimizer (`SciPyBudgetOptimizer`) maximized unconstrained nominal expected profit, resulting in significant exposure to high-variance customer estimates and severe demographic under-representation of budget customers ($8.9\%$ allocation rate vs $36.0\%$ for VIPs). The new advanced optimizer (`AdvancedMILPOptimizer`) provides configurable risk-aversion modes and fairness constraints to balance financial return, statistical certainty, and equitable brand exposure.

### Key Empirical Findings

1. **Uncertainty De-Risking:** In Risk-Adjusted Mode ($\lambda = 1.0$), the optimizer sacrifices only **$\$133.96$** ($0.13\%$) in nominal expected profit while increasing the conservative lower-confidence-bound profit by **$+\$386.39$** ($+0.40\%$, from $\$95,741.92$ to $\$96,128.31$) and reducing selected customer estimation uncertainty by **$3.5\%$** (mean SE drops from $0.0346$ to $0.0334$).
2. **The "Price of Fairness":** Enforcing a $15.0\%$ minimum allocation floor on the budget segment increases budget customer allocation by **$+68.8\%$** ($157 \to 265$ customers) and raises their demographic parity ratio from **$0.448$ to $0.739$**, with an economic trade-off of only **$\$47.81$** ($0.047\%$ of expected profit).
3. **Combined Resilience:** Combining risk-aversion ($\lambda = 1.0$) and fairness ($15\%$ floor) achieves an expected profit of **$\$100,481.63$**, conservative LCB profit of **$\$96,137.18$**, and demographic parity within budget ($99.94\%$ utilization), executing in **$0.315$ seconds**.

---

## 1. Mathematical Formulation

### 1.1 Decision Variables and Domain
For each customer $i \in \{1, \dots, N\}$ and discount tier $d \in \mathcal{D} = \{0, 10, 20\}$:
$$z_{i, d} \in \{0, 1\}$$
where $z_{i, d} = 1$ if customer $i$ is assigned promotional discount $d$, and $0$ otherwise.

### 1.2 Structural Assignment and Budget Constraints

1. **Exact Assignment (Mutually Exclusive Tiers):**
   $$\sum_{d \in \{0, 10, 20\}} z_{i, d} = 1 \quad \forall i \in \{1, \dots, N\}$$

2. **Ex-Ante Expected Promotional Budget Ceiling:**
   $$\sum_{i=1}^N \sum_{d \in \{10, 20\}} \text{Cost}_{i, d} \cdot z_{i, d} \le B$$
   where $\text{Cost}_{i, d} = d \cdot P(Y = 1 \mid X_i, d)$ is the expected redemption expenditure under the redemption accounting model.

> [!WARNING]
> **Cash Budget Disclaimer:** Expected promotional cost is an *ex-ante mathematical expectation* derived from predicted redemption probabilities. Realized redemption expenditure depends on actual customer purchase events and may exhibit finite-sample variance around this expectation. It does not guarantee cash outlays will stay strictly below the budget in a single promotional campaign.

---

## 2. Uncertainty-Aware Optimization Formulations

### 2.1 Mode 1: Baseline Nominal Expected Profit
Maximizes total nominal expected net monetary contribution:
$$\max_{z} \sum_{i=1}^N \sum_{d \in \mathcal{D}} \Pi_{i, d} \cdot z_{i, d}$$
where $\Pi_{i, d} = \hat{P}_{i, d} \cdot U_{i, d}$, and $U_{i, d} = \text{OrderValue}_{i, d} \cdot \text{MarginRate}_i - d$.

### 2.2 Mode 2: Conservative Lower-Confidence-Bound (LCB) Optimization
Substitutes the 95% conservative lower confidence bound for treatment effect lifts:
$$\tau_{\text{LCB}, i, d} = \hat{\tau}_{i, d} - z_{1-\alpha/2} \cdot \text{SE}(\hat{\tau}_{i, d})$$
$$P_{\text{LCB}, i, d} = \text{clip}\left(\hat{P}_{i, 0} + \tau_{\text{LCB}, i, d}, \ 0, \ 1\right)$$
$$\max_{z} \sum_{i=1}^N \left( \Pi_{i, 0} \cdot z_{i, 0} + \sum_{d \in \{10, 20\}} P_{\text{LCB}, i, d} \cdot U_{i, d} \cdot z_{i, d} \right)$$

### 2.3 Mode 3: Risk-Adjusted Mean-Variance Optimization
Rather than subtracting ungrounded heuristic penalty scores, we translate CATE estimation variance directly into monetary profit variance:
$$\Delta \text{Profit}_{i, d} = \tau_{i, d} \cdot U_{i, d} \quad \implies \quad \text{SE}(\Delta \text{Profit}_{i, d}) = \text{SE}(\tau_{i, d}) \cdot U_{i, d}$$

The objective maximizes expected profit penalized by incremental monetary profit standard error:
$$\max_{z} \sum_{i=1}^N \left( \Pi_{i, 0} \cdot z_{i, 0} + \sum_{d \in \{10, 20\}} \left( \Pi_{i, d} - \lambda \cdot \text{SE}(\tau_{i, d}) \cdot U_{i, d} \right) \cdot z_{i, d} \right)$$
where $\lambda \ge 0$ is the risk-aversion hyperparameter.

---

## 3. Business-Defined Fairness Constraints

### 3.1 Ethical Scope and Attribute Quarantine
Fairness constraints are restricted strictly to business-defined customer segments (`budget`, `regular`, `premium`, `vip`) derived from historical shopping behavior. **No protected personal characteristics** (age, gender, ethnicity, location) are utilized.

### 3.2 Formulation: Minimum Segment Allocation Floor
To prevent strategic neglect of budget or regular tiers, marketing leadership configures minimum allocation floors $\rho_{\min, s}$:
$$\sum_{i \in s} \sum_{d \in \{10, 20\}} z_{i, d} \ge \lceil N_s \cdot \rho_{\min, s} \rceil \quad \forall s \in \mathcal{S}$$

### 3.3 Formulation: Relative Demographic Parity Floor
Guarantees that each business segment achieves at least a fraction $\gamma_{\min}$ of the overall campaign allocation rate:
$$\frac{1}{N_s} \sum_{i \in s} (z_{i, 10} + z_{i, 20}) \ge \gamma_{\min} \cdot \frac{1}{N} \sum_{i=1}^N (z_{i, 10} + z_{i, 20})$$
Linearized in SciPy MILP constraint matrix $A$:
$$\sum_{i \in s} (z_{i, 10} + z_{i, 20}) - \frac{\gamma_{\min} N_s}{N} \sum_{i=1}^N (z_{i, 10} + z_{i, 20}) \ge 0$$

---

## 4. Empirical Benchmark Scorecard

Evaluated on held-out test data ($N = 6,005$ customers, $\$5,000.00$ budget cap):

| Optimization Regime | Expected Profit | Conservative LCB Profit | Expected Cost | Tier Allocation ($0 / $10 / $20) | Mean SE of Selected | Solver Status | Runtime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Original MILP (Baseline)** | $\$100,636.58$ | $\$95,741.92$ | $\$4,998.79$ | $4,815 \ / \ 1,170 \ / \ 20$ | $0.0346$ | `Optimal (0)` | $0.726$s |
| **2. Uncertainty-Aware MILP ($\lambda=1.0$)** | $\$100,502.62$ | $\$96,128.31$ | $\$4,999.93$ | $4,783 \ / \ 1,190 \ / \ 32$ | $0.0334$ | `Optimal (0)` | $0.442$s |
| **3. Fairness-Constrained MILP ($15\%$ floor)** | $\$100,588.77$ | $\$95,854.60$ | $\$4,997.37$ | $4,783 \ / \ 1,205 \ / \ 17$ | $0.0341$ | `Optimal (0)` | $0.412$s |
| **4. Combined Uncertainty & Fairness** | $\$100,481.63$ | $\$96,137.18$ | $\$4,997.19$ | $4,776 \ / \ 1,198 \ / \ 31$ | $0.0333$ | `Optimal (0)` | $0.315$s |

---

## 5. Sensitivity Analysis: Risk Aversion & Price of Fairness

### 5.1 Risk-Aversion Efficient Frontier ($\lambda \in [0.0, 2.0]$)

| Risk Aversion ($\lambda$) | Expected Profit | Conservative LCB Profit | Expected Cost | Mean Selected SE | Allocation ($0 / $10 / $20) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $\lambda = 0.0$ (Neutral) | $\$100,636.58$ | $\$95,741.92$ | $\$4,998.79$ | $0.0346$ | $4,815 \ / \ 1,170 \ / \ 20$ |
| $\lambda = 0.5$ | $\$100,595.68$ | $\$95,971.82$ | $\$4,998.12$ | $0.0340$ | $4,797 \ / \ 1,183 \ / \ 25$ |
| $\lambda = 1.0$ (Balanced) | $\$100,502.62$ | $\$96,128.31$ | $\$4,999.93$ | $0.0334$ | $4,783 \ / \ 1,190 \ / \ 32$ |
| $\lambda = 1.5$ | $\$100,248.92$ | $\$96,269.30$ | $\$4,997.13$ | $0.0326$ | $4,760 \ / \ 1,209 \ / \ 36$ |
| $\lambda = 2.0$ (Conservative) | $\$99,978.97$ | $\$96,306.26$ | $\$4,998.28$ | $0.0319$ | $4,729 \ / \ 1,242 \ / \ 34$ |

**Monotone Efficient Frontier:** As risk aversion $\lambda$ increases from $0.0 \to 2.0$:
- Mean standard error drops monotonically from $0.0346 \to 0.0319$ (a **$7.8\%$ variance reduction**).
- Guaranteed lower-bound profit rises monotonically from $\$95,741.92 \to \$96,306.26$ (a **$+\$564.34$ downside gain**).
- The marginal cost of de-risking remains modest: only $\$657.61$ ($0.65\%$) in nominal profit is forfeited at $\lambda = 2.0$.

### 5.2 The "Price of Fairness" (Budget Segment Allocation Floor Sweep)

| Budget Segment Floor | Actual Budget Rate | Budget Parity Ratio | Expected Profit | Profit Loss vs Unconstrained | Allocation ($0 / $10 / $20) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $8.9\%$ (Unconstrained) | $8.9\%$ | $0.448$ | $\$100,636.58$ | $\$0.00$ | $4,815 \ / \ 1,170 \ / \ 20$ |
| $12.0\%$ Floor | $12.0\%$ | $0.598$ | $\$100,625.30$ | $\$11.28$ | $4,801 \ / \ 1,186 \ / \ 18$ |
| $15.0\%$ Floor | $15.0\%$ | $0.739$ | $\$100,588.77$ | $\$47.81$ | $4,783 \ / \ 1,205 \ / \ 17$ |
| $18.0\%$ Floor | $18.0\%$ | $0.875$ | $\$100,509.84$ | $\$126.74$ | $4,766 \ / \ 1,224 \ / \ 15$ |
| $20.0\%$ Floor | $20.0\%$ | $0.963$ | $\$100,432.19$ | $\$204.39$ | $4,755 \ / \ 1,237 \ / \ 13$ |

**Fairness Insight:** Achieving near-perfect demographic parity ($0.963$ parity ratio) for budget shoppers costs only **$\$204.39$** ($0.20\%$) in expected profit, demonstrating that fair promotional targeting is highly economically viable in this retail portfolio.

---

## 6. Segment-Level Diagnostic Comparison

Comparison of segment outcomes between Original MILP and Fairness-Constrained MILP ($15\%$ floor):

| Segment | Customers | Original Allocation % | Original Parity | Fairness Allocation % | Fairness Parity | Change in Customers Targeted |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Budget** | $1,769$ | $8.9\%$ | $0.448$ | **$15.0\%$** | **$0.739$** | $+108$ customers ($+68.8\%$) |
| **Regular** | $2,378$ | $20.1\%$ | $1.012$ | $18.3\%$ | $0.901$ | $-42$ customers |
| **Premium** | $1,253$ | $27.0\%$ | $1.361$ | $25.1\%$ | $1.231$ | $-24$ customers |
| **VIP** | $605$ | $36.0\%$ | $1.818$ | $34.0\%$ | $1.673$ | $-12$ customers |

---

## 7. Automated Test Suite Verification

Comprehensive unit test suite executed (`tests/test_advanced_optimizer.py`):

```bash
pytest tests/test_advanced_optimizer.py -v
```

```text
tests/test_advanced_optimizer.py::test_optimizer_parameter_validation PASSED [ 11%]
tests/test_advanced_optimizer.py::test_milp_exact_assignment_and_budget_cap PASSED [ 22%]
tests/test_advanced_optimizer.py::test_optimization_modes_execution PASSED [ 33%]
tests/test_advanced_optimizer.py::test_risk_aversion_uncertainty_reduction PASSED [ 44%]
tests/test_advanced_optimizer.py::test_fairness_constraint_increases_segment_allocation PASSED [ 55%]
tests/test_advanced_optimizer.py::test_zero_budget_edge_case PASSED      [ 66%]
tests/test_advanced_optimizer.py::test_infeasible_budget_and_fairness_handling PASSED [ 77%]
tests/test_advanced_optimizer.py::test_optimization_reproducibility PASSED [ 88%]
tests/test_advanced_optimizer.py::test_to_dict_serialization PASSED      [100%]

============================== 9 passed in 2.85s ==============================
```

Combined multi-phase test suite (`tests/test_doubly_robust.py`, `tests/test_uncertainty.py`, `tests/test_advanced_optimizer.py`):
```text
============================= 29 passed in 9.25s ==============================
```

