# Optimization Dataset Report: Multi-Tier Promotional Discount Data

**Project:** EconoCausal — Dynamic Pricing via Double Machine Learning  
**Milestone:** Week 3 — Multi-Arm Causal Dataset for Budget-Constrained Optimization  
**Report Date:** 2026-11-03  
**Status:** ✅ GENERATION COMPLETE — 24/24 VALIDATION CHECKS PASSED

---

## 1. Executive Summary

This report documents the synthetic `optimization_retail_data.csv` dataset designed to support
three-arm causal treatment estimation, budget-constrained revenue optimization, and
counterfactual profit modelling for Week 3 of EconoCausal.

The dataset is a **clean additive extension** of the existing project. It does **not** modify,
overwrite, or depend on the Week 1 binary-treatment dataset
(`sample_data/synthetic_retail_campaign.csv`) or the trained Week 2 `LinearDML` model.

### Key Results

| Property | Value |
|:---|---:|
| Records | **15,000** |
| Columns | **22** |
| Treatment arms | **$0 / $10 / $20** |
| Validation result | **✅ 24 / 24 checks passed, 0 critical failures** |
| Overall purchase rate | **50.37%** |
| Ground-truth ATE (T=10 vs T=0) | **+15.45 pp** |
| Ground-truth ATE (T=20 vs T=0) | **+21.32 pp** |
| Total gross revenue | **$653,189.26** |
| Total discount cost | **$81,660.00** |
| Total net profit after discount | **$223,648.58** |

---

## 2. How This Dataset Differs from the Original Week 1 Dataset

| Dimension | Week 1 Dataset (`synthetic_retail_campaign.csv`) | Week 3 Dataset (`optimization_retail_data.csv`) |
|:---|:---|:---|
| **Purpose** | Binary DML causal identification & CATE estimation | Multi-arm causal estimation & budget-constrained optimization |
| **Treatment variable** | `discount_offered` ∈ {0, 1} — binary | `discount_level` ∈ {0, 10, 20} — three discrete arms |
| **Treatment effect** | Single scalar τ(X) = E[Y(1)−Y(0)\|X] | Two-vector [τ₁₀(X), τ₂₀(X)] capturing dose-response elasticity |
| **Discount cost in DGP** | `discount_amount` is random post-treatment draw ($10–$35); **does not affect conversion probability** | `discount_amount` IS the causal treatment; conversion lift is explicitly parameterized per tier |
| **Economics** | `revenue` only; no margin, no profit columns | `gross_revenue`, `discount_cost`, `net_revenue`, `gross_profit_before_discount`, `net_profit_after_discount` |
| **Customer features** | `prior_purchases`, `prior_spend`, `engagement_score`, `customer_tenure`, `campaign_channel` | `age`, `loyalty_score`, `previous_purchases`, `customer_segment`, `campaign_channel`, `product_price`, `gross_margin_rate`, `customer_timestamp` |
| **Temporal structure** | None (single static snapshot) | `customer_timestamp` ∈ {2026-Q1, Q2, Q3, Q4} for drift detection batches |
| **Ground-truth potential outcomes** | None | `y0_potential`, `y10_potential`, `y20_potential`, `tau_10_true`, `tau_20_true` |
| **Sample size** | 2,500 | 15,000 |

---

## 3. Schema & Variable Dictionary

| Column | Type | Role | Domain / Range | Notes |
|:---|:---|:---|:---|:---|
| `customer_id` | `str` | Identifier | `OPT_000001`–`OPT_015000` | Unique; never used as a model feature |
| `age` | `int` | Pre-treatment confounder | [18, 75] | Normal(42, 12) clipped; younger = more price-sensitive |
| `loyalty_score` | `float` | Pre-treatment confounder | [1.0, 99.0] | Beta(2,3)×100; right-skewed; VIPs boosted +20 pts |
| `previous_purchases` | `int` | Pre-treatment confounder | [0, 29] | Poisson(λ by segment); proxy for brand loyalty |
| `customer_segment` | `str` | Pre-treatment confounder | 4 levels | `budget_seeker`, `casual_shopper`, `regular_buyer`, `vip_loyalist` |
| `campaign_channel` | `str` | Pre-treatment confounder | 4 levels | `email`, `push`, `sms`, `social` |
| `customer_timestamp` | `str` | Temporal batch ID | 4 levels | 2026-Q1 through Q4; for drift monitoring |
| `product_price` | `float` | Context variable | [$8, $461.55] | LogNormal per segment; determines absolute revenue impact |
| `gross_margin_rate` | `float` | Context variable | [0.16, 0.69] | Normal per segment; determines profit sensitivity |
| `discount_level` | `int` | **Treatment (T)** | {0, 10, 20} | Dollar face-value of discount tier |
| `discount_amount` | `float` | **Treatment alias** | {0.0, 10.0, 20.0} | Equals `discount_level`; dollar amount |
| `purchase` | `int` | **Primary outcome (Y)** | {0, 1} | Binary conversion; sampled from tier-specific logit |
| `gross_revenue` | `float` | Post-treatment outcome | ≥ 0 | `product_price × purchase` |
| `discount_cost` | `float` | Post-treatment cost | {0, 10, 20} | `discount_amount × purchase` (redemption model) |
| `net_revenue` | `float` | Post-treatment outcome | Can be negative | `gross_revenue − discount_cost` |
| `gross_profit_before_discount` | `float` | Post-treatment outcome | ≥ 0 | `gross_revenue × gross_margin_rate` |
| `net_profit_after_discount` | `float` | **Optimization target** | Can be negative | `gross_profit_before_discount − discount_cost` |
| `y0_potential` | `float` | Ground truth | (0, 1) | P(purchase=1 \| X, T=0); never used as model input |
| `y10_potential` | `float` | Ground truth | (0, 1) | P(purchase=1 \| X, T=10) |
| `y20_potential` | `float` | Ground truth | (0, 1) | P(purchase=1 \| X, T=20) |
| `tau_10_true` | `float` | Ground truth CATE | (0, 1) | `y10_potential − y0_potential`; true $10 lift |
| `tau_20_true` | `float` | Ground truth CATE | (0, 1) | `y20_potential − y0_potential`; true $20 lift |

---

## 4. Data-Generating Process

### 4.1 Customer Segment Assignment

Segments are assigned from a Categorical distribution:

| Segment | Probability | Typical Product Price | Typical Margin | Baseline Conversion |
|:---|---:|---:|---:|---:|
| `budget_seeker` | 25% | ~$35 | ~30% | ~14% |
| `casual_shopper` | 30% | ~$55 | ~38% | ~31% |
| `regular_buyer` | 30% | ~$80 | ~45% | ~39% |
| `vip_loyalist` | 15% | ~$150 | ~55% | ~62% |

### 4.2 Treatment Assignment (Confounded Multinomial Propensity)

Each customer's probability of receiving each tier is governed by segment-specific
logit biases plus $\mathcal{N}(0, 1)$ noise through a softmax:

$$P(T = d \mid \text{segment}) \propto \exp(\alpha_d^{[\text{segment}]} + \eta_d),\quad \eta_d \sim \mathcal{N}(0, 1)$$

| Segment | P(T=0) | P(T=10) | P(T=20) |
|:---|---:|---:|---:|
| `budget_seeker` | 23.6% | 32.0% | 44.4% |
| `casual_shopper` | 31.4% | 38.4% | 30.1% |
| `regular_buyer` | 36.3% | 36.9% | 26.7% |
| `vip_loyalist` | 46.2% | 29.2% | 24.6% |

This introduces **realistic marketing selection bias** (more engaged / price-sensitive customers
are more likely to receive larger discounts) while maintaining **full overlap** (positivity) —
every tier has ≥ 23% probability in its most-biased segment.

### 4.3 Outcome Generation (Heterogeneous Treatment Effects)

$$\text{logit}\bigl(P(\text{purchase}=1 \mid d, X)\bigr) = \underbrace{\alpha_0^{[\text{seg}]}}_{\text{baseline}} + \underbrace{\beta^\top z(X)}_{\text{covariate effects}} + \underbrace{\gamma^{[\text{channel}]}}_{\text{channel}} + \underbrace{\tau_d^{[\text{seg}]} + \delta \cdot s(X)}_{\text{treatment lift}}$$

Where:
- $z(X)$ = normalised age, loyalty, previous purchases
- $s(X)$ = individual price-sensitivity modifier: $(1 - \text{loyalty\_norm}) \times \max(0, 1 - \text{age\_norm})$
- $\delta = 0.30$ (heterogeneity coefficient)

**Segment-specific treatment lifts on logit scale:**

| Segment | $\tau_{10}$ (logit) | $\tau_{20}$ (logit) |
|:---|---:|---:|
| `budget_seeker` | +0.65 | +1.10 |
| `casual_shopper` | +0.55 | +0.85 |
| `regular_buyer` | +0.45 | +0.62 |
| `vip_loyalist` | +0.12 | +0.18 |

### 4.4 Ground-Truth CATE by Segment (Actual from Generated Data)

| Segment | Mean τ₁₀ | Mean τ₂₀ | τ₂₀ − τ₁₀ (marginal $10→$20) |
|:---|---:|---:|---:|
| `budget_seeker` | +15.45 pp | +25.32 pp | +9.87 pp |
| `casual_shopper` | +19.16 pp | +26.18 pp | +7.02 pp |
| `regular_buyer` | +16.98 pp | +20.79 pp | +3.81 pp |
| `vip_loyalist` | +4.76 pp | +5.70 pp | +0.94 pp |
| **Population** | **+15.45 pp** | **+21.32 pp** | **+5.87 pp** |

**Key finding:** The marginal conversion gain from upgrading from $10 → $20 is
substantially smaller than from $0 → $10 in every segment (diminishing returns).

---

## 5. Economics & Accounting Model

### 5.1 Accounting Conventions

> **Discount Incurrence Model: Redemption-Only**  
> Discount cost is recorded **only when a customer receives an offer AND makes a purchase**.
> A customer who receives a coupon but does not convert incurs zero discount cost.

| Economic Metric | Formula | Accounting Justification |
|:---|:---|:---|
| `gross_revenue` | `product_price × purchase` | Full pre-discount revenue |
| `discount_cost` | `discount_amount × purchase` | Redeemed coupons only |
| `net_revenue` | `gross_revenue − discount_cost` | Cash received after coupon redemption |
| `gross_profit_before_discount` | `gross_revenue × gross_margin_rate` | Gross margin (COGS already netted via rate) |
| `net_profit_after_discount` | `gross_profit_before_discount − discount_cost` | **Optimization target** |

### 5.2 Diminishing Returns: $20 Discount Does NOT Always Maximize Profit

This is a core design requirement. Validation Check 13 confirmed:

| Segment | Avg Net Profit T=10 | Avg Net Profit T=20 | Conclusion |
|:---|---:|---:|:---|
| `budget_seeker` | **+$0.39** | **−$3.63** | ✅ $20 destroys profit (thin margin $35 × 30%) |
| `casual_shopper` | **+$6.73** | **+$1.40** | ✅ $20 earns less despite higher conversion |
| `regular_buyer` | ~$15 | ~$10 | $10 tier more efficient |
| `vip_loyalist` | ~$32 | ~$28 | $20 barely worth it (tiny incremental conversion lift) |

**Business implication:** A naive policy of "give everyone the biggest discount" would
destroy $4 of margin per budget_seeker customer. The optimizer must balance incremental
conversion probability against per-unit cost.

### 5.3 Population Economics Summary

| Metric | Value |
|:---|---:|
| Total gross revenue | $653,189.26 |
| Total discount cost | $81,660.00 |
| Total net revenue | $571,529.26 |
| Total net profit after discount | $223,648.58 |
| Avg net profit — T=0 | $21.22 per customer |
| Avg net profit — T=10 | $15.20 per customer |
| Avg net profit — T=20 | $8.05 per customer |

> [!NOTE]
> The population-level averages across tiers do NOT indicate that $0 is always optimal —
> they reflect assignment bias (VIPs who convert easily are disproportionately in T=0).
> The optimizer must condition on customer features, not raw tier averages.

---

## 6. Treatment Distribution and Overlap

| Tier | Count | Proportion |
|:---|---:|---:|
| T=0 ($0) | 4,964 | 33.1% |
| T=10 ($10) | 5,250 | 35.0% |
| T=20 ($20) | 4,786 | 31.9% |

Every segment×tier cell contains ≥ 23% representation, satisfying the positivity
(overlap) assumption required for non-parametric causal identification.

---

## 7. Conversion Rates

### By Discount Level
| Tier | Conversion Rate |
|:---|---:|
| T=0 | 41.98% |
| T=10 | 52.88% (+10.90 pp over T=0) |
| T=20 | 56.31% (+14.33 pp over T=0) |

### By Segment (across all tiers)
| Segment | Conversion Rate |
|:---|---:|
| `budget_seeker` | 30.37% |
| `casual_shopper` | 49.57% |
| `regular_buyer` | 55.04% |
| `vip_loyalist` | 76.81% |

---

## 8. Validation Results — 24/24 Checks Passed

| Check | Description | Result |
|:---|:---|:---:|
| C1 | All 22 required columns present | ✅ PASS |
| C2 | Zero missing values | ✅ PASS |
| C3 | No duplicate customer_id | ✅ PASS |
| C4 | discount_level ∈ {0, 10, 20} | ✅ PASS |
| C5 | discount_amount == discount_level | ✅ PASS |
| C6 | purchase ∈ {0, 1} | ✅ PASS |
| C7a | No discount_cost when purchase == 0 | ✅ PASS |
| C7b | discount_cost == discount_amount when purchase == 1 | ✅ PASS |
| C8a | gross_revenue = product_price × purchase | ✅ PASS |
| C8b | net_revenue = gross_revenue − discount_cost | ✅ PASS |
| C8c | net_profit_after_discount accounting correct | ✅ PASS |
| C9-age | age ∈ [18, 75] | ✅ PASS |
| C9-loyalty_score | loyalty_score ∈ [1, 99] | ✅ PASS |
| C9-previous_purchases | previous_purchases ≥ 0 | ✅ PASS |
| C9-product_price | product_price ∈ [$8, $600] | ✅ PASS |
| C9-gross_margin_rate | gross_margin_rate ∈ [0.10, 0.80] | ✅ PASS |
| C10 | Positivity — every segment ≥ 5% in each tier | ✅ PASS |
| C11a | y10_potential ≥ y0_potential (100.00% of rows) | ✅ PASS |
| C11b | y20_potential ≥ y10_potential (100.00% of rows) | ✅ PASS |
| C12a | tau_10_true all non-negative | ✅ PASS |
| C12b | tau_20_true all non-negative | ✅ PASS |
| C12c | No effects > 60 pp | ✅ PASS |
| C13-budget_seeker | T=20 net profit < T=10 net profit for budget_seeker | ✅ PASS |
| C13-casual_shopper | T=20 net profit < T=10 net profit for casual_shopper | ✅ PASS |

---

## 9. Artifacts

| File | Size | Description |
|:---|:---|:---|
| `data/optimization_retail_data.csv` | ~2.1 MB | 15,000-row multi-arm synthetic dataset |
| `data/optimization_dgp_parameters.json` | 5.6 KB | Full documented DGP parameter manifest |
| `src/data/optimization_generator.py` | — | Parameterized DGP generator module |
| `scripts/generate_optimization_dataset.py` | — | Runner script (does not modify Week 1 files) |
| `scripts/validate_optimization_dataset.py` | — | Validation script (24 checks, exit code 0) |

### Files NOT modified (Week 1/2 Preservation Confirmed)
- `sample_data/synthetic_retail_campaign.csv` — untouched ✅
- `src/data/generator.py` — untouched ✅
- `src/data/schema.py` — untouched ✅
- `models/dml_model.joblib` — untouched ✅
- All 39 pytest tests — untouched ✅

---

## 10. Ready for Week 3 — Pending Review

This dataset provides everything the Week 3 multi-arm causal estimator and SciPy optimizer need:

| Optimizer Input | Provided By |
|:---|:---|
| Customer feature matrix (X) | `age`, `loyalty_score`, `previous_purchases`, `customer_segment`, `campaign_channel` |
| Multi-arm treatment labels (T ∈ {0, 10, 20}) | `discount_level` |
| Binary purchase outcome (Y) | `purchase` |
| Counterfactual profit target per tier | `net_profit_after_discount` (computed from `product_price`, `gross_margin_rate`, `discount_amount`) |
| Ground-truth benchmarks for model evaluation | `tau_10_true`, `tau_20_true`, `y0_potential`, `y10_potential`, `y20_potential` |
| Temporal batch IDs for drift detection (Week 4) | `customer_timestamp` ∈ {2026-Q1, Q2, Q3, Q4} |

**Awaiting approval to proceed with Phase 3.2: Multi-Arm DML estimator implementation.**

