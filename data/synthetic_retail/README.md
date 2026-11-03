# Multi-Treatment Synthetic Retail Dataset

**Project:** EconoCasual: Dynamic Pricing via Double Machine Learning  
**Sample Size:** 40,000 unique customer observations  
**Treatment Options:** T0: $0 (No discount), T1: $10 discount, T2: $20 discount  
**Seed:** 42 (Fully reproducible)

---

## 1. Purpose & Analytical Scope

This dataset provides an observational retail marketing benchmark with **heterogeneous treatment effects**, realistic marketing selection confounding, and known **ground-truth counterfactual potential outcomes**.

Unlike purely observational data where counterfactual outcomes $Y(d)$ are unobservable (the fundamental problem of causal inference), synthetic retail data allows direct mathematical verification of:
1. Doubly robust Double Machine Learning (DML) estimator convergence.
2. Individual Treatment Effect (ITE) ranking and uplift accuracy.
3. Multiple-Choice Knapsack budget optimization versus true oracle decisions.

---

## 2. Customer Features (Pre-Treatment Covariates)

All customer characteristics reflect pre-treatment status prior to promotional discount assignment:

| Column Name | Type | Description | Domain / Bounds |
| :--- | :---: | :--- | :--- |
| `customer_id` | String | Unique synthetic customer identifier | `SYN_000001` — `SYN_040000` |
| `age` | Integer | Customer age in years | $18 - 80$ years |
| `loyalty_score` | Float | Customer loyalty and engagement score | $[0.0, 100.0]$ |
| `previous_purchases` | Integer | Count of lifetime completed orders | $0 - 35$ orders |
| `customer_segment` | Categorical | Customer demographic tier | `budget`, `regular`, `premium`, `vip` |
| `average_previous_order_value` | Float | Historical average order value in USD | $\$25.00 - \$350.00$ |
| `customer_activity_score` | Float | Digital engagement & site recency index | $[0.0, 100.0]$ |
| `campaign_date` | Date | Date of promotional offer distribution | `2026-01-15` — `2026-09-30` |

---

## 3. Treatment Assignment & Overlap Mechanism

The promotional coupon assignment is governed by a multinomial logit propensity model reflecting marketing team selection bias (targeting active and high-spend customers), while preserving strong overlap ($0.10 \le P(T=d \mid X) \le 0.65$):

- **Treatment Column:** `discount_amount` with values `0`, `10`, `20`.
- **Distribution:**
  - **`$0` Discount (Control):** 13,024 customers (32.6%)
  - **`$10` Discount (Moderate):** 15,873 customers (39.7%)
  - **`$20` Discount (Deep):** 11,103 customers (27.8%)

---

## 4. Financial & Accounting Definitions

Accounting rules follow promotional retail standards:

| Metric | Formula / Definition | Rationale |
| :--- | :--- | :--- |
| **`purchase`** | Binary $0$ or $1$ | Whether the customer converted in the campaign period |
| **`purchase_probability`** | $P(Y=1 \mid X, T)$ | Latent conversion probability under assigned discount |
| **`order_value`** | Baseline order value with coupon basket expansion | Zero if customer did not purchase |
| **`discount_cost`** | `discount_amount * purchase` | Incurred **only upon customer redemption** at checkout |
| **`revenue`** | $\max(0, \text{order\_value} - \text{discount\_cost})$ | Net cash receipts received from the customer |
| **`gross_margin_rate`**| Product profit margin rate ($[0.35, 0.70]$) | Pre-discount merchandise profitability |
| **`gross_margin`** | $\text{order\_value} \times \text{gross\_margin\_rate}$ | Gross contribution before promotional expense |
| **`net_profit`** | $\text{gross\_margin} - \text{discount\_cost}$ | Net commercial contribution margin |

---

## 5. Ground-Truth Counterfactuals (`synthetic_retail_ground_truth.csv`)

Ground-truth potential outcomes and true treatment effects are preserved in a separate evaluation file:
- **`potential_purchase_prob_0` / `_10` / `_20`:** True counterfactual probabilities $P(Y(d) = 1 \mid X)$.
- **`true_ite_10_vs_0`:** True individual uplift for $10 coupon ($\tau_{10}(X_i)$). True ATE = **+19.58 pp**.
- **`true_ite_20_vs_0`:** True individual uplift for $20 coupon ($\tau_{20}(X_i)$). True ATE = **+26.06 pp**.
- **`expected_net_profit_0` / `_10` / `_20`:** Expected net profit under each tier.
- **`optimal_unconstrained_discount`:** Oracle profit-maximizing discount depth:
  - `$0` optimal: 2,992 customers (7.5%) — Organic loyal buyers.
  - `$10` optimal: 29,920 customers (74.8%) — Elastic sweet spot.
  - `$20` optimal: 7,088 customers (17.7%) — High-hurdle or high-basket buyers.

> **CRITICAL ANTI-LEAKAGE RULE:** Potential outcomes and ground-truth columns are strictly quarantined in `synthetic_retail_ground_truth.csv`. They are never present in model training or validation splits.

---

## 6. Dataset Partitions

```text
data/synthetic_retail/
├── synthetic_retail_train.csv       # 27,998 rows (70% stratified training partition)
├── synthetic_retail_validation.csv  # 5,997 rows (15% stratified validation partition)
├── synthetic_retail_test.csv        # 6,005 rows (15% held-out test partition)
├── synthetic_retail_ground_truth.csv# 40,000 rows (Evaluation-only potential outcomes)
├── dgp_parameters.json              # Exact DGP coefficients, distribution parameters, & seed
└── README.md                        # Documentation
```

