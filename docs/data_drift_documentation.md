# EconoCasual Data Drift & Covariate Shift Documentation

**Project:** EconoCasual: Dynamic Pricing via Double Machine Learning  
**Module:** Automated Continuous Drift Surveillance & Model Monitoring  
**Source Code:** `src/monitoring/drift_detector.py`  
**API Route:** `POST /drift/check`  
**Dashboard Component:** `frontend/src/components/DataDriftViewer.jsx`

---

## 1. Overview & Monitoring Strategy

In production algorithmic pricing, incoming customer cohorts inevitably experience distribution shifts due to advertising campaign changes, seasonal shopping behaviors, and macroeconomic shifts.

The EconoCasual drift surveillance engine continuously tests incoming customer observations against the approved baseline reference distribution (`data/drift_reference.csv`, 5,000 customers). It enforces a strict **two-stage architecture**:
1. **Schema & Model-Input Validation:** Verifies required columns, expected types, null rates, and domain bounds, isolating pipeline corruptions.
2. **Statistical Two-Sample Hypothesis Testing:** Applies continuous and categorical divergence metrics to identify true demographic shifts.

---

## 2. Statistical Formulation & Metrics

### 2.1 Continuous Features (`age`, `loyalty_score`, `previous_purchases`, `product_price`, `gross_margin_rate`)

1. **Two-Sample Kolmogorov-Smirnov Test:**
   - Evaluates whether the empirical cumulative distribution function (ECDF) of the current batch $F_{\text{curr}}(x)$ diverges significantly from the reference baseline $F_{\text{ref}}(x)$:
     $$D = \sup_x |F_{\text{ref}}(x) - F_{\text{curr}}(x)|$$
   - Flagged if asymptotic p-value $p < \alpha = 0.05$.

2. **Population Stability Index (PSI):**
   - Bins continuous features into $K = 10$ quantiles derived from the reference population.
   - For reference bin proportion $P_k$ and current bin proportion $Q_k$:
     $$\text{PSI} = \sum_{k=1}^K (P_k - Q_k) \cdot \ln\left( \frac{P_k + \epsilon}{Q_k + \epsilon} \right)$$
   - $\epsilon = 10^{-4}$ provides Laplace smoothing.
   - **Thresholds:**
     - $\text{PSI} < 0.10$: Stable / No significant drift.
     - $0.10 \le \text{PSI} < 0.25$: Moderate drift (warning).
     - $\text{PSI} \ge 0.25$: Severe drift (action required).

3. **Wasserstein-1 Distance (Earth Mover's Distance):**
   - Quantifies the minimum transportation work required to transform $F_{\text{curr}}$ into $F_{\text{ref}}$:
     $$W_1(P, Q) = \int_{-\infty}^{\infty} |F_{\text{ref}}(x) - F_{\text{curr}}(x)| \, dx$$

---

### 2.2 Categorical Features (`customer_segment`, `campaign_channel`)

1. **Total Variation Distance (TVD):**
   - For discrete category proportions $P(c)$ and $Q(c)$:
     $$\delta(P, Q) = \frac{1}{2} \sum_{c \in \mathcal{C}} |P(c) - Q(c)|$$
   - Flagged if $\text{TVD} \ge 0.15$.

2. **Pearson's Chi-Square Contingency Test:**
   - Tests goodness-of-fit against expected proportions:
     $$\chi^2 = \sum_{c \in \mathcal{C}} \frac{(O_c - E_c)^2}{E_c}$$
   - Evaluated at degrees of freedom $df = |\mathcal{C}| - 1$.

3. **New / Missing Category Detection:**
   - Any unseen category level or complete disappearance of a known baseline segment immediately triggers a high severity flag.

---

## 3. Schema Integrity vs. Statistical Drift Separation

```
[ Incoming Inference Batch ]
            │
            ▼
[ Stage 1: Schema Integrity Validation ]
   ├── Missing Required Features Check (e.g. 'age', 'customer_segment')
   ├── Unexpected Column Quarantine
   ├── Null Rate Check (nulls must be 0)
   └── Boundary Checks:
         - age in [18, 100]
         - loyalty_score in [0, 100]
         - product_price > 0
         - gross_margin_rate in (0, 1]
            │
       [Is Valid?] ──► NO ──► Return Schema Error (Pipeline corrupt; Halt)
            │
           YES
            ▼
[ Stage 2: Statistical Drift Testing ]
   ├── KS Test, PSI, Wasserstein Distance (Numeric)
   └── Chi-Square, TVD, New Categories (Categorical)
            │
            ▼
[ Stage 3: Severity Synthesis ]
   ├── "No significant drift detected"
   ├── "Potential drift detected"
   └── "Severe or widespread drift detected"
```

---

## 4. Causal Methodological Governance: Covariate Shift vs. Concept Drift

A foundational tenet of the EconoCasual platform is that **covariate drift ($P(X)$) does NOT automatically invalidate causal treatment effect estimates ($\tau(X)$)**:

1. **Covariate Shift ($P_{\text{curr}}(X) \ne P_{\text{ref}}(X)$):**
   - Customer demographics or acquisition channels have shifted (e.g. an older demographic arriving via social ads).
   - Under Pearl's unconfoundedness assumption $(Y(d) \perp T \mid X)$, the Conditional Average Treatment Effect $\tau(X)$ remains mathematically unbiased on the overlapping support.
2. **Concept Drift ($P_{\text{curr}}(Y \mid X, T) \ne P_{\text{ref}}(Y \mid X, T)$):**
   - The underlying behavioral price elasticity has changed (e.g. macro inflation or brand crisis). This requires randomized holdout verification to detect.
3. **Prohibition of Automated Retraining:**
   - Automated model retraining without human review and empirical validation is strictly prohibited. Uncontrolled retraining on drifted, non-randomly targeted customer data risks feedback loops and severe policy destabilization.

