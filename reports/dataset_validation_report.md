# EconoCasual Dataset Validation & Quality Assurance Report

**Project:** EconoCasual: Dynamic Pricing via Double Machine Learning  
**Milestone:** Multi-Treatment Synthetic Retail Generation & Hillstrom Benchmark Integration  
**Date:** 2026-11-03 16:36:58  
**Validation Status:** **ALL CHECKS PASSED (100% COMPLIANT)**

---

## 1. Executive Summary

This report establishes the data foundation for multi-treatment causal estimation and benchmarking across two distinct paradigms:
1. **Multi-Treatment Synthetic Retail Dataset (40,000 observations):** Observational retail data supporting monetary coupon deductions ($0, $10, $20) with known heterogeneous ground-truth potential outcomes and realistic selection confounding.
2. **Hillstrom MineThatData Benchmark Dataset (64,000 observations):** Authentic randomized controlled trial (RCT) measuring behavioral email communication uplift (No E-Mail vs. Men's E-Mail vs. Women's E-Mail).

Both datasets were subjected to a rigorous 10-point audit covering data integrity, domain constraints, treatment overlap, potential outcome consistency, and anti-leakage compliance.

---

## 2. Multi-Treatment Synthetic Retail Dataset Audit

### 2.1 File Artifacts & Dimensions
| Artifact Name | Path | Rows | Columns | Share | Role |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Training Partition** | `data/synthetic_retail/synthetic_retail_train.csv` | 27,998 | 17 | 70.0% | Model training & cross-fitting |
| **Validation Partition** | `data/synthetic_retail/synthetic_retail_validation.csv` | 5,997 | 17 | 15.0% | Hyperparameter tuning & overlap audit |
| **Test Partition** | `data/synthetic_retail/synthetic_retail_test.csv` | 6,005 | 17 | 15.0% | Out-of-sample policy evaluation |
| **Ground Truth (Isolated)**| `data/synthetic_retail/synthetic_retail_ground_truth.csv` | 40,000 | 15 | 100.0% | Oracle counterfactual benchmark (No Leakage) |

### 2.2 Data Integrity & Hygiene
- **Missing Values:** Exactly **0** null cells across all 40,000 records.
- **Duplicate Records:** Exactly **0** duplicate `customer_id` rows.
- **Partition Disjointness:** **Zero customer overlap** between Train, Validation, and Test splits ($	ext{Train} \cap 	ext{Val} = \emptyset$, $	ext{Train} \cap 	ext{Test} = \emptyset$, $	ext{Val} \cap 	ext{Test} = \emptyset$).
- **Numerical Domains:**
  - `age`: $[18, 80]$ years (Mean: 43.1, Std: 12.2)
  - `loyalty_score`: $[0.0, 100.0]$ (Mean: 51.8, Std: 19.2)
  - `previous_purchases`: $[0, 35]$ orders (Mean: 8.2)
  - `average_previous_order_value`: $[\$25.00, \$350.00]$ (Mean: \$85.74)
  - `customer_activity_score`: $[0.0, 100.0]$ (Mean: 57.9)

### 2.3 Treatment Overlap & Representation
The assignment mechanism provides balanced coverage across all three treatment tiers:
| Discount Tier | Customers | Share | Mean Baseline Prob ($p_0$) | Observed Conversion | Realized Revenue | Net Profit |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$0 (Control)** | 13,024 | 32.6% | 29.20% | 25.13% | \$23.77 | \$12.66 |
| **$10 (Moderate)**| 15,873 | 39.7% | 29.34% | 49.01% | \$38.83 | \$18.15 |
| **$20 (Deep)** | 11,103 | 27.8% | 28.83% | 58.43% | \$51.43 | \$22.12 |

### 2.4 Ground-Truth Causal Effects & Heterogeneity
- **True Average Treatment Effect ($10 vs $0):** **+19.58 percentage points** (Standard Deviation: 5.70 pp, Range: [1.0 pp, 32.8 pp])
- **True Average Treatment Effect ($20 vs $0):** **+26.07 percentage points** (Standard Deviation: 6.52 pp, Range: [4.2 pp, 45.0 pp])
- **Potential Outcome Consistency:** Observed binary purchase matches counterfactual potential outcomes exactly ($Y = Y_d$ with zero discrepancies).
- **Oracle Optimal Discount Distribution (Unconstrained):**
  - **$0 Discount Best:** 2,992 customers (7.5%) — Organic buyers whose high baseline conversion does not justify coupon subsidization.
  - **$10 Discount Best:** 29,920 customers (74.8%) — Elastic customers where a $10 coupon provides the sweet spot of incremental margin.
  - **$20 Discount Best:** 7,088 customers (17.7%) — High-hurdle or high-basket customers where the steeper coupon unlocks significant incremental spend.

> **Validation Note:** $20 is **not** automatically the best discount for all customers. For 82.3% of the customer base, offering $20 erodes net profit compared to $0 or $10.

---

## 3. Hillstrom MineThatData Email Marketing Audit

### 3.1 Experimental Design & Dimensions
- **Total Records:** 64,000 customers.
- **Experimental Protocol:** Randomized Controlled Trial (RCT) with equal 1/3 random assignment.
- **Missing Values:** Exactly **0** null cells.
- **Partition Disjointness:** **Zero customer overlap** across train (44,796 rows), validation (9,598 rows), and test (9,606 rows).

### 3.2 Treatment Group Breakdown
| Group ID | Treatment Name | Raw Segment String | Customers | Share | Site Visit Rate | Conversion Rate | Mean Spend |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`0`** | `no_email` | `No E-Mail` | 21,306 | 33.29% | 10.62% | 0.57% | \$0.65 |
| **`1`** | `mens_email` | `Mens E-Mail` | 21,307 | 33.29% | 18.28% | 1.25% | \$1.42 |
| **`2`** | `womens_email` | `Womens E-Mail` | 21,387 | 33.42% | 15.14% | 0.88% | \$1.08 |

### 3.3 Experimental Empirical Uplift Over Control
- **Men's E-Mail Uplift:**
  - Visit Uplift: **+7.66 percentage points** (Relative Lift: +72.1%)
  - Conversion Uplift: **+0.68 percentage points** (Relative Lift: +119.3%, $p < 0.0001$)
  - Revenue Lift: **+$0.77 per customer** (Relative Lift: +118.5%)
- **Women's E-Mail Uplift:**
  - Visit Uplift: **+4.52 percentage points** (Relative Lift: +42.6%)
  - Conversion Uplift: **+0.31 percentage points** (Relative Lift: +54.4%, $p < 0.001$)
  - Revenue Lift: **+$0.43 per customer** (Relative Lift: +66.2%)

---

## 4. Methodological Comparison Between Datasets

| Dimension | Multi-Treatment Synthetic Retail | Hillstrom MineThatData |
| :--- | :--- | :--- |
| **Data Nature** | Synthetic Confounded Observational Log | Real-World Randomized Controlled Trial (RCT) |
| **Treatment Mechanism** | Confounded (Marketing team targets active users) | Unconfounded ($P(T \mid X) = 1/3$ by randomized design) |
| **Treatment Definition** | Monetary Coupon Deductions ($0, $10, $20) | Marketing Communication (No Email, Men's, Women's) |
| **Financial Cost** | High variable cost ($10, $20 coupon per purchase) | Zero/negligible marginal cost per email sent |
| **Primary Evaluation Goal**| Constrained Knapsack Margin Optimization | Uplift Modeling & Targeting Preference |
| **Ground-Truth Availability**| Exact counterfactual potential outcomes known | Realized outcomes observable only for assigned treatment |

---

## 5. Anti-Leakage Compliance Matrix

| Dataset | Split File | Pre-Treatment Features | Quarantined Post-Treatment & Ground-Truth Variables | Audit Result |
| :--- | :--- | :--- | :--- | :---: |
| **Synthetic Retail** | `synthetic_retail_train.csv` | `age`, `loyalty_score`, `previous_purchases`, `customer_segment`, `average_previous_order_value`, `customer_activity_score`, `campaign_date` | `potential_purchase_*`, `true_ite_*`, `optimal_unconstrained_discount`, `order_value`, `revenue`, `gross_margin`, `net_profit` | **PASS (100% Isolated)** |
| **Hillstrom** | `hillstrom_train.csv` | `recency`, `history_segment`, `history`, `mens`, `womens`, `zip_code`, `newbie`, `channel` | `visit`, `conversion`, `spend` (reserved exclusively as target outcomes) | **PASS (100% Isolated)** |

---

## 6. Audit Conclusion

Both datasets strictly satisfy all milestone criteria. Data structures, documentation, ground-truth separations, and test partitions are fully verified and ready for causal estimator benchmarking.
