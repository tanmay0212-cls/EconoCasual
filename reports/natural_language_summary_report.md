# EconoCasual Natural-Language Business Summary Report

**Project:** EconoCasual: Dynamic Pricing via Double Machine Learning  
**Milestone:** Week 4 — Natural-Language Business Summary & Executive Synthesis  
**Authors:** AI/ML Engineering & Product Architecture Team  
**Date:** November 3, 2026  
**Status:** Approved & Validated (Backend: 82/82 Tests Passing, Frontend: 47/47 Tests Passing)

---

## 1. Executive Summary & Business Objective

While Double Machine Learning (DML) and Mixed-Integer Linear Programming (MILP) provide mathematical rigor for price elasticity estimation and constrained budget optimization, executive stakeholders (such as a Marketing Director or Chief Commercial Officer) require **clear, transparent, and actionable business translations** rather than raw regression coefficients or decision matrices.

The **EconoCasual Natural-Language Executive Summary Engine** bridges this gap:
- Translates mathematical optimization results, knapsack discount allocations, expected financial lifts, and statistical data drift telemetry into **clear business language**.
- Operates **100% deterministically** using rule-based synthesis without external paid LLM APIs, ensuring zero runtime cost, zero latency, and zero hallucination risk.
- Strictly adheres to causal accounting principles: **clearly distinguishes model-predicted expectations from historical realized business outcomes**, without making misleading claims of guaranteed earnings or causal certainty.
- Explicitly flags when required metrics are missing or inconsistent, avoiding fabricated numbers or percentages.
- Provides interactive metric inspection so marketing leaders can audit every single claim down to its underlying raw data point.

---

## 2. Architecture & Summary Generation Methodology

The summary architecture is implemented in [`frontend/src/services/summaryService.js`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/frontend/src/services/summaryService.js) and presented via [`frontend/src/components/BusinessSummaryPanel.jsx`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/frontend/src/components/BusinessSummaryPanel.jsx).

```
[ Optimization Results (MILP) ]         [ Drift Detection Engine (KS/PSI) ]
       │                                                 │
       ├── Allocation Matrix (3,000 rows)                ├── Continuous Tests (p-vals, PSI)
       ├── Expected Revenue & Profit                     ├── Categorical Frequencies (TVD)
       └── Budget Utilization ($4,604 / $5,000)          └── Covariate Shift Status
       │                                                 │
       └────────────────────────┬────────────────────────┘
                                │
                                ▼
         [ Deterministic Summary Generation Service ]
                   (summaryService.js)
        ├── Rule-based synthesis templates
        ├── Anti-hallucination metric validators
        ├── Expected vs. Realized qualifier injector
        └── Missing metric gracefully handled
                                │
                                ▼
         [ Executive Decision Briefing Panel ]
              (BusinessSummaryPanel.jsx)
        ├── Run ID & Timestamp metadata
        ├── Key Executive Takeaway Banner
        ├── 6 Structured Business Sections
        ├── Expandable Underlying Metrics Drawers
        └── Causal Accounting Governance Notice
```

---

## 3. The Six Core Dashboard Summary Sections

### Section 1: Optimization Recommendation
- **Business Purpose:** Provide the core marketing decision regarding who receives coupons and who pays full catalog price.
- **Synthesized Example Output:**
  > **Headline:** Target promotional discounts to 838 customers (27.9% of cohort)  
  > **Narrative:**  
  > "The causal optimizer recommends targeting personalized discounts to 838 customers: 836 allocated $10 coupons and 2 allocated $20 coupons. For the remaining 2,162 customers, the model recommends maintaining full price ($0) to avoid subsidizing organic purchases where baseline conversion propensity is already high."
- **Underlying Metrics Supporting Claim:** Total Customer Cohort (3,000), Allocated $0 (2,162), Allocated $10 (836), Allocated $20 (2).

### Section 2: Budget Utilization Summary
- **Business Purpose:** Transparently account for authorized campaign expenditures, remaining reserve, and redemption timing.
- **Synthesized Example Output:**
  > **Headline:** Allocated $4,604.20 of the available $5,000.00 budget (92.1%)  
  > **Narrative:**  
  > "The optimized strategy allocates $4,604.20 of the available $5,000.00 expected discount budget (92.1% utilization), leaving $395.80 in reserve.  
  > *Accounting note:* In accordance with retail promotional standards, expected discount cost is accrued only when a customer redeems the coupon upon purchasing ($\text{Expected Cost} = \text{Discount} \times \hat{P}_{\text{purchase}}$)."
- **Underlying Metrics Supporting Claim:** Authorized Budget Cap ($5,000.00), Projected Discount Cost ($4,604.20), Remaining Reserve ($395.80), Budget Utilization (92.1%).

### Section 3: Customer Allocation Summary
- **Business Purpose:** Explain the tier breakdown and commercial rationale behind discount depths.
- **Synthesized Example Output:**
  > **Headline:** 72.1% full price, 27.9% discounted  
  > **Narrative:**  
  > "The optimizer recommended no discount ($0) for 2,162 customers (72.1% of cohort), preserving product margins where customer elasticity does not justify discounting. Moderate discounts ($10) were assigned to 836 customers (27.9%), capturing price-responsive buyers with optimal incremental return. Deep discounts ($20) were reserved for 2 highly elastic customers (0.1%) where expected margin gain outweighs the steeper $20 coupon expense."
- **Underlying Metrics Supporting Claim:** Total Evaluated (3,000 customers), $0 Tier (2,162 / 72.1%), $10 Tier (836 / 27.9%), $20 Tier (2 / 0.1%).

### Section 4: Expected Revenue & Profit Comparison
- **Business Purpose:** Contrast the personalized policy against alternative marketing benchmarks (No Discount vs. Blanket Promotions).
- **Synthesized Example Output:**
  > **Headline:** Model projects +$2,066.60 (+4.1%) profit lift over no-discount baseline  
  > **Narrative:**  
  > "The model estimates that the personalized strategy produces a higher expected contribution ($53,082.35) than blanket baseline pricing ($51,015.75), representing an expected lift of $2,066.60. In comparison, unsegmented blanket discounting ($10 to all customers) would incur $16,246.60 in discount expenses, severely breaching campaign budget limits and eroding net margins.  
  > *Projections notice:* Expected revenue and profit figures are model-predicted counterfactual expectations based on estimated purchase probabilities. They represent statistical projections, not guaranteed actual receipts."
- **Underlying Metrics Supporting Claim:** Projected Gross Revenue ($121,447.78), Projected Net Profit ($53,082.35), Baseline Profit ($51,015.75), Blanket $10 Promotion Cost ($16,246.60).

### Section 5: Data Drift & Covariate Shift Surveillance
- **Business Purpose:** Warn the marketing director if customer demographics or acquisition funnels have shifted significantly.
- **Synthesized Example Output:**
  > **Headline:** Severe covariate shift detected across 4 of 7 customer features  
  > **Narrative:**  
  > "The reference and current customer distributions differ substantially. Statistically significant drift was detected in age, loyalty_score, customer_segment, campaign_channel. Model performance and conversion elasticity should be monitored closely. Covariate shift ($P(X)$) does not automatically invalidate causal treatment effect estimates, but common support overlap must be audited."
- **Underlying Metrics Supporting Claim:** Overall Status (Severe), Features Evaluated (7), Drifted Features Count (4 / 7), High Severity Features (4).

### Section 6: Recommended Operational Next Actions
- **Business Purpose:** Concrete, actionable next steps for marketing campaign operations.
- **Synthesized Example Output:**
  > **Headline:** Action protocol for promotional campaign launch and governance  
  > **Narrative:**  
  > "1. Deployment: Proceed with targeted discount distribution for the 838 identified elastic customers, maintaining promotional expenditure within the authorized budget.  
  > 2. Telemetry Audit: Because severe demographic drift was detected in incoming cohorts, verify common support overlap and recalibrate propensity scores $e(X)$. Do not trigger automated model retraining without offline counterfactual holdout testing.  
  > 3. Holdout Governance: Reserve a 5% randomized holdout group receiving $0 discount to measure realized conversion lift and validate counterfactual DML projections against actual financial results."

---

## 4. User Experience & Dashboard Panel Integration

The summary panel is integrated into [`frontend/src/App.jsx`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/frontend/src/App.jsx) as a primary navigation view:

1. **Prominent Navigation Tab:** Positioned as the default landing view (`activeTab === 'summary'`) with a "Week 4" highlight badge.
2. **Metadata & Timestamp Header:** Displays the associated optimization Run ID (`run_opt_20261103_073602`), ISO generation timestamp, and the synthesis engine mode.
3. **One-Click Metric Inspection ("Inspect Supporting Metrics"):** Every summary card contains an expandable drawer listing exact underlying numbers, formulas, and metric definitions.
4. **Seamless Deep-Dive Navigation:** Provides one-click action buttons to navigate directly to the **3,000-Customer Allocation Matrix** or the **Data Drift Monitoring Console**.
5. **Causal Accounting Governance Notice:** Prominently placed footer callout reminding users that all projections are counterfactual expectations and reinforcing the mandate for randomized holdout verification.

---

## 5. Handling Missing, Inconsistent, and Edge-Case Data

The service adheres to strict data integrity safeguards:
- **Missing Optimization Data:** Generates an informative card: *"No active optimization results were provided for evaluation. Run a budget optimization to generate customer-level discount allocations."*
- **Missing Drift Telemetry:** Generates: *"No recent customer drift inspection was provided. Run a drift check against the baseline reference dataset."*
- **Inconsistent Populations (NaN / Null):** Explains: *"The optimization result record is missing valid customer population totals. Allocation recommendations cannot be summarized without verifiable customer counts."*
- **Zero Budget / All-$0 Policy:** Accurately states: *"Under the active campaign constraints, the optimizer recommends no discounts ($0) for all evaluated customers. The model projects that promotional coupon costs would exceed expected incremental contribution."*
- **No Invented Numbers:** If a metric is null or non-finite, helper functions return `"Unavailable"` rather than fabricating `0.0` or placeholder estimates.

---

## 6. Automated Testing Suite & Verification

The summary engine was verified through automated Vitest unit and integration suites:

- **Formatting & Edge Cases:** [`frontend/src/tests/summaryService.test.js`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/frontend/src/tests/summaryService.test.js) (15 tests passing)
  - Currency, count, and percentage formatting
  - Standard complete optimization + drift inputs
  - Empty, null, and undefined parameter handling
  - Zero budget and all-$0 allocations
  - Inconsistent and corrupted customer counts
  - Verification of counterfactual wording and disclaimer presence
- **Component & UI Interaction:** [`frontend/src/tests/BusinessSummaryPanel.test.jsx`](file:///c:/Users/tanma/OneDrive/Desktop/EconoCasual/frontend/src/tests/BusinessSummaryPanel.test.jsx) (6 tests passing)
  - Renders title, badge, run ID, and executive takeaway
  - Renders all 6 dashboard summary sections
  - Verifies synthesized business narratives
  - Toggles and displays underlying supporting metrics
  - Handles missing optimization data gracefully
  - Displays methodological notice box

### Full Suite Status
- **Backend Pytest Suite:** **82 passed / 82 total** (100% green).
- **Frontend Vitest Suite:** **47 passed / 47 total** across 8 test suites (100% green).
- **Production Vite Build:** Built cleanly in 2.93s with 0 warnings.

