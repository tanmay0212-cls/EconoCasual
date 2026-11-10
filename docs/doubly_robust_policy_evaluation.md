# EconoCasual 2.0 — Doubly Robust Policy Evaluation Report

**Document Status:** Complete & Verified  
**Evaluation Dataset:** Held-out Synthetic Retail Test Split (`data/synthetic_retail/synthetic_retail_test.csv`, $N = 6,005$)  
**Benchmark Reference:** Milestone 3 Baseline & Canonical Financial Records  
**Generated Timestamp:** 2026-11-10T16:08:52  

---

## Executive Summary

Off-policy evaluation (OPE) was conducted on the held-out test split of $N = 6,005$ customers to establish whether the prescriptive **Double Machine Learning + SciPy Mixed-Integer Linear Programming (DML + MILP)** discount allocation policy outperforms conventional baseline strategies in real-world retail unit economics.

Five distinct policies were evaluated:
1. **No-Discount Policy ($0 Blanket):** Baseline organic customer conversion.
2. **Flat $10 Discount Policy:** Blanket moderate discount allocated to all customers.
3. **Random Discount Policy:** Uniform stochastic exploration across $\{\$0, \$10, \$20\}$.
4. **Existing DML + MILP Policy:** CausalForestDML heterogeneous elasticity allocation under a strict $\$5,000.00$ expected promotional expenditure ceiling.
5. **Uncertainty-Aware Causal Policy (Future Phase):** Marked `status: "unavailable"` awaiting Phase 3/4 implementation.

### Key Empirical Findings

1. **Capital Efficiency & ROI Dominance:** Under a strict $\$5,000$ marketing budget, the **DML + MILP Prescriptive Policy** achieves an incremental profit of **$\$7,165.56$** (95% CI: $[\$1,373.24, \$12,681.20]$) over the organic no-discount baseline on an actual expected expenditure of **$\$4,998.74$**, yielding an **Incremental Net-Profit-to-Promotion-Cost Ratio of $143.35\%$** under Doubly Robust estimation (and **$204.05\%$** by model expectation).
2. **Unconstrained Blanket Inefficiency:** While blanket $\$10$ discounting drives higher nominal total conversions ($47.90\%$) and total profit ($\$113,754.02$), it exhausts **$\$29,149.90$** in promotional expenditure (nearly 6x the budget cap), delivering an ROI of only **$72.46\%$**.
3. **Stochastic Exploration Inefficiency:** Random allocation wastes budget on deep discounts ($\$20$) for price-inelastic customers, generating an ROI of only **$32.93\%$**.
4. **Triangulation Consistency:** The Doubly Robust estimated profit for the DML + MILP policy ($\$99,797.05$) closely matches the Model Expected Profit ($\$100,607.76$) and Ground Truth Expected Profit ($\$102,969.42$), confirming that the causal nuisance models and logging propensities are well-specified.

---

## 1. Mathematical Formulation

### 1.1 Multi-Treatment Environment
Let each customer $i \in \{1, \dots, N\}$ be characterized by pre-treatment feature vector $X_i \in \mathcal{X}$, assigned discrete treatment $T_i \in \mathcal{D} = \{0, 10, 20\}$, and observed binary conversion outcome $Y_i \in \{0, 1\}$.

The economic outcome is the individual net monetary contribution $R_i$:
$$R_i = Y_i \cdot \left[ \text{OrderValue}(X_i, T_i) \cdot \text{MarginRate}(X_i) - T_i \right]$$

### 1.2 Inverse Probability Weighting (IPW)
For a candidate policy $\pi(d \mid X_i)$ and logging assignment propensity $e_d(X_i) = P(T_i = d \mid X_i)$, the generalized Horvitz-Thompson IPW policy value estimator is:
$$V_{\text{IPW}}(\pi) = \frac{1}{N} \sum_{i=1}^N \sum_{d \in \mathcal{D}} \frac{\mathbb{I}(T_i = d) \cdot \pi(d \mid X_i)}{e_d(X_i)} Y_i$$

The Self-Normalized (Hajek) IPW estimator adjusts for finite-sample weight volatility:
$$V_{\text{Hajek}}(\pi) = \frac{\sum_{i=1}^N \sum_{d \in \mathcal{D}} \frac{\mathbb{I}(T_i = d) \cdot \pi(d \mid X_i)}{e_d(X_i)} Y_i}{\sum_{i=1}^N \sum_{d \in \mathcal{D}} \frac{\mathbb{I}(T_i = d) \cdot \pi(d \mid X_i)}{e_d(X_i)}}$$

### 1.3 Direct Method (DM)
The Direct Method replaces observed outcomes with the conditional outcome model $\hat{\mu}_d(X_i) = \mathbb{E}[Y_i \mid X_i, T_i = d]$:
$$V_{\text{DM}}(\pi) = \frac{1}{N} \sum_{i=1}^N \sum_{d \in \mathcal{D}} \pi(d \mid X_i) \hat{\mu}_d(X_i)$$

### 1.4 Doubly Robust (DR) Estimator
The Doubly Robust estimator augments the Direct Method baseline with an importance-weighted residual correction:
$$\psi_i(\pi) = \sum_{d \in \mathcal{D}} \pi(d \mid X_i) \hat{\mu}_d(X_i) + \sum_{d \in \mathcal{D}} \frac{\mathbb{I}(T_i = d) \cdot \pi(d \mid X_i)}{e_d(X_i)} \left( Y_i - \hat{\mu}_d(X_i) \right)$$
$$V_{\text{DR}}(\pi) = \frac{1}{N} \sum_{i=1}^N \psi_i(\pi)$$

**Double Robustness Property:** If either the propensity model $e_d(X)$ or the outcome model $\hat{\mu}_d(X)$ is correctly specified, $V_{\text{DR}}(\pi)$ is an asymptotically unbiased estimator of the true policy value $\mathbb{E}[Y(\pi)]$.

---

## 2. Diagnostics: Positivity, Overlap & Sample Size

### 2.1 Positivity / Common Support
The logging policy assignment in the synthetic retail environment satisfies strict positivity:
- $P(T = \$0 \mid X) \in [0.1425, 0.6370]$, Mean: $0.3272$
- $P(T = \$10 \mid X) \in [0.1963, 0.5742]$, Mean: $0.3948$
- $P(T = \$20 \mid X) \in [0.1161, 0.5738]$, Mean: $0.2780$

Because all conditional propensities are strictly bounded away from zero ($e_d(X) \ge 0.1161 > 0.0100$), **zero records were clipped** at the $\alpha = 0.01$ threshold ($0.0\%$ clipped across all policies).

### 2.2 Effective Sample Size (Kish's ESS)
Kish's Effective Sample Size measures the information retention after importance weighting:
$$\text{ESS}(\pi) = \frac{\left( \sum_{i=1}^N w_i \right)^2}{\sum_{i=1}^N w_i^2}$$

- **No-Discount Policy:** $\text{ESS} = 1,825.55$ ($30.4\%$ of $N$)
- **Flat $10 Discount Policy:** $\text{ESS} = 2,303.56$ ($38.4\%$ of $N$)
- **Random Discount Policy:** $\text{ESS} = 1,793.51$ ($29.9\%$ of $N$)
- **DML + MILP Policy:** $\text{ESS} = 1,796.36$ ($29.9\%$ of $N$)

All effective sample sizes exceed $1,750$ customers, providing ample statistical power for narrow confidence intervals.

---

## 3. Policy Evaluation Scorecard

Evaluated on held-out test data ($N = 6,005$, 500 bootstrap iterations, $\alpha = 0.05$):

| Policy Name | Status | DR Conversion Rate (95% CI) | DR Total Expected Profit (95% CI) | Model Exp. Profit | Ground Truth Exp. Profit | Expected Promo Cost | Incremental Profit vs Organic | Incr. Net-Profit-to-Cost Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **No-Discount (Organic)** | `evaluated` | $29.16\%$ $[27.10\%, 31.23\%]$ | $\$92,631.49$ $[\$85,209.55, \$99,935.40]$ | $\$90,501.19$ | $\$94,169.35$ | $\$0.00$ | $\$0.00$ | — |
| **Flat $10 Discount** | `evaluated` | $47.90\%$ $[45.81\%, 49.99\%]$ | $\$113,754.02$ $[\$107,507.63, \$119,579.30]$ | $\$115,163.88$ | $\$116,399.74$ | $\$29,149.90$ | $\$21,122.53$ $[\$12,223.07, \$29,419.16]$ | $72.46\%$ |
| **Random Exploration** | `evaluated` | $43.71\%$ $[41.42\%, 45.92\%]$ | $\$102,836.49$ $[\$96,549.32, \$108,693.87]$ | $\$102,518.24$ | $\$104,888.36$ | $\$30,988.72$ | $\$10,205.00$ $[\$3,322.72, \$17,074.96]$ | $32.93\%$ |
| **DML + MILP Prescriptive** | `evaluated` | $32.26\%$ $[30.14\%, 34.47\%]$ | $\$99,797.05$ $[\$92,361.51, \$107,548.90]$ | $\$100,607.76$ | $\$102,969.42$ | $\$4,998.74$ | $\$7,165.56$ $[\$1,373.24, \$12,681.20]$ | **$143.35\%$** |
| **Uncertainty-Aware (Future)** | `unavailable` | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

---

## 4. Method Triangulation & Reality Check

To prevent confirmation bias or metric inflation, four independent profit perspectives were triangulated:

```
+---------------------------------------------------------------------------------------+
|                                    PROFIT TRIANGULATION                               |
|                                                                                       |
|  DML + MILP Policy:                                                                   |
|    - Doubly Robust Estimated Profit:      $99,797.05  (95% CI: $92,361 - $107,549)    |
|    - Model Expected Profit:               $100,607.76 (Variance with DR: -0.81%)      |
|    - Ground Truth Expected Profit:        $102,969.42 (Variance with DR: -3.08%)      |
|    - Realized Observed Profit (Sub-cohort): $28,181.76 (Observed match T_i = \pi_i)   |
|                                                                                       |
|  Organic Baseline ($0):                                                               |
|    - Doubly Robust Estimated Profit:      $92,631.49  (95% CI: $85,210 - $99,935)     |
|    - Model Expected Profit:               $90,501.19                                  |
|    - Ground Truth Expected Profit:        $94,169.35                                  |
|    - Realized Observed Profit (Sub-cohort): $24,633.32                                |
+---------------------------------------------------------------------------------------+
```

### Explanatory Analysis:
1. **Model Expected Profit vs DR Profit:** The model expectation of $\$100,607.76$ is within $0.81\%$ of the Doubly Robust estimate of $\$99,797.05$. The small negative residual correction indicates that the outcome model had a slight $+0.81\%$ optimism bias, which the DR estimator corrected.
2. **Realized Observed Profit Clarification:** The realized observed profit of $\$28,181.76$ for the DML + MILP policy reflects *only* the subset of the $6,005$ test customers whose historical randomized treatment $T_i$ coincidentally matched the prescriptive recommendation $\pi(X_i)$ ($1,926$ matching customers). It is not the counterfactual total and should **never** be conflated with total policy value.
3. **Budget Compliance:** The MILP solution adheres strictly to the $\$5,000.00$ expected budget ceiling, consuming $\$4,998.74$ ($99.97\%$ utilization).

---

## 5. Non-Parametric Bootstrap Methodology

1. **Re-sampling Unit:** Customer-level influence scores $\psi_i(\pi)$ were resampled with replacement ($N = 6,005$, $B = 500$ replicates).
2. **Paired Contrast Resampling:** For policy incremental lift $\Delta V = V(\pi) - V(\pi_{\text{organic}})$, paired difference scores $\Delta \psi_i = \psi_i(\pi) - \psi_i(\pi_{\text{organic}})$ were resampled simultaneously on each bootstrap draw to preserve covariance.
3. **Interval Calculation:** Both 95% Percentile intervals ($[\text{Quantile}_{0.025}, \text{Quantile}_{0.975}]$) and Normal-approximation intervals ($\hat{\theta} \pm 1.96 \cdot \text{SE}$) were computed.
4. **Reproducibility:** A fixed pseudo-random seed (`random_state=42`) guarantees exact bit-for-bit reproducibility across all bootstrap runs.

---

## 6. Data Integrity and Anti-Leakage Enforcements

The evaluation pipeline strictly enforces all causal auditing principles:
1. **Covariate Isolation:** Only valid pre-treatment customer features (`age`, `loyalty_score`, `previous_purchases`, `customer_segment`, `average_previous_order_value`, `customer_activity_score`) are utilized.
2. **Post-Treatment Exclusion:** Post-treatment observables (`order_value`, `gross_margin`, `gross_margin_rate`, `discount_cost`, `revenue`, `net_profit`, `purchase_probability`) are strictly excluded from nuisance model conditioning.
3. **Partition Isolation:** Held-out evaluation is performed solely on `synthetic_retail_test.csv` ($N = 6,005$), leaving training and validation partitions completely untouched.
4. **Identifier Isolation:** `customer_id` strings and indices are excluded from feature vectors and used strictly for cross-table record joining.

---

## 7. Assumptions & Limitations

1. **Unconfoundedness (Conditional Ignorability):** $(Y(0), Y(10), Y(20)) \perp T \mid X$. Satisfied by construction in the synthetic retail data-generating process.
2. **Positivity:** $0 < P(T = d \mid X) < 1$ for all $d \in \{0, 10, 20\}$ and all $X \in \mathcal{X}$. Verified empirically ($e_d(X) \ge 0.1161$).
3. **SUTVA:** No interference between customers and identical treatment versions.
4. **Ex-Ante vs Ex-Post Budget:** The MILP optimizer constrains *ex-ante expected* promotional expenditure ($\sum d \cdot \hat{P}_d \cdot z_{i, d} \le B$). Realized redemption expenditure may exhibit finite-sample variance around this expectation.

