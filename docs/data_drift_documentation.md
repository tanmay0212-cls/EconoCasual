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
   ├── Type Integrity Check (numeric vs string coercion)
   ├── Null Rate Check (nulls must be 0)
   └── Boundary Checks:
         - age in [18, 100]
         - loyalty_score in [0, 100]
         - product_price > 0
         - gross_margin_rate in (0, 1]
            │
       [Is Valid?] ──► NO ──► Return Schema Error (OverallDriftStatus.SCHEMA_ERROR, drift_detected=False)
            │
           YES
            ▼
[ Stage 2: Sample Size Policy Gate ]
   ├── N < min_sample_size (default: 20) ──► Return Inconclusive (OverallDriftStatus.INSUFFICIENT_DATA, drift_detected=False)
   └── N >= min_sample_size ──► Proceed to Two-Sample Hypothesis Tests
            │
            ▼
[ Stage 3: Statistical Hypothesis Testing ]
   ├── KS Test, PSI, Wasserstein Distance (Numeric Features)
   ├── Chi-Square, TVD, Emerging/Missing Categories (Categorical Features)
   └── Counterfactual Prediction Probability Drift (P(d=0), P(d=10), P(d=20))
            │
            ▼
[ Stage 4: Severity & Status Synthesis ]
   ├── "No significant drift detected" (All features stable)
   ├── "Potential drift detected" (1-2 moderate features, PSI 0.10-0.25)
   └── "Severe or widespread drift detected" (>= 3 moderate or >= 1 high severity feature)
```

### 3.1 Strict Separation of Ingestion Failures from Statistical Drift
Historically, schema validation defects (e.g., missing columns or corrupted types) were occasionally conflated with severe demographic drift. In EconoCasual, schema violations are strictly decoupled:
- Schema violations represent **pipeline / data contract defects**, returning `overall_status: "Schema validation failed"` and `drift_detected: false`.
- The dashboard displays schema violations in a high-priority red alert banner listing the exact violated constraints.
- Statistical two-sample tests are skipped to prevent arithmetic errors or skewed statistics on malformed inputs.

### 3.2 Configurable Minimum Sample Size Policy
Statistical two-sample tests (KS test, Chi-square, PSI) suffer from high variance and severe under-power when evaluated on tiny sample sizes ($N < 20$).
- The system enforces a configurable `min_sample_size` parameter (default: 20 records).
- Batches with $N < \text{min\_sample\_size}$ return `overall_status: "Insufficient data for statistical tests"` with `drift_detected: false`.
- Crucially, descriptive statistics (mean, median, standard deviation, category frequencies) are preserved and returned to allow operators to inspect the limited sample without asserting false stability or false alarm drift.

---

## 4. Multi-Tenant Detector Isolation & Threshold Integrity

In production API environments, client requests can specify custom sensitivity thresholds (such as `ks_p_value_threshold` or `psi_high_threshold`).
- **Immutable Singleton Instance:** The cached detector singleton (`get_detector()`) maintains immutable default baseline configurations.
- **Dynamic Parameter Injection:** Per-request threshold overrides are passed explicitly into `detect_drift(data, ks_alpha=..., psi_high_threshold=..., min_sample_size=...)` and downstream calculation methods without mutating singleton instance attributes.
- **Parameter Validation:** Thresholds are strictly validated at the API boundary ($0 < \alpha < 1$, $\text{PSI} > 0$, $\text{min\_sample\_size} \ge 1$) to ensure numerical stability.

---

## 5. Causal Methodological Governance: Covariate Shift vs. Prediction Drift vs. Concept Drift

A foundational tenet of the EconoCasual platform is that **covariate drift ($P(X)$) does NOT automatically invalidate causal treatment effect estimates ($\tau(X)$)**:

1. **Covariate Shift ($P_{\text{curr}}(X) \ne P_{\text{ref}}(X)$):**
   - Customer demographics or acquisition channels have shifted (e.g. an older demographic arriving via social ads).
   - Under Pearl's unconfoundedness assumption $(Y(d) \perp T \mid X)$, the Conditional Average Treatment Effect $\tau(X)$ remains mathematically unbiased on the overlapping support.
   - Recommended response: Recalibrate propensity scores $e(X)$ and budget constraints; do not retrain causal trees.

2. **Prediction Drift ($P_{\text{curr}}(P_d) \ne P_{\text{ref}}(P_d)$):**
   - Evaluates whether the distribution of counterfactual conversion probabilities ($P_0, P_{10}, P_{20}$) produced by the causal policy differs across populations.
   - If counterfactual probabilities are not present or computable, the API returns `prediction_drift_status: "unavailable"` without fabricating placeholder numbers.

3. **Concept Drift ($P_{\text{curr}}(Y \mid X, T) \ne P_{\text{ref}}(Y \mid X, T)$):**
   - The underlying behavioral price elasticity has changed (e.g. macroeconomic shifts or competitor pricing changes).
   - Concept drift cannot be confirmed from observational feature logs alone; it strictly requires randomized holdout verification (A/B testing).

4. **Prohibition of Automated Retraining:**
   - Automated model retraining without human review and empirical validation is strictly prohibited. Uncontrolled retraining on drifted, non-randomly targeted customer data risks feedback loops and severe policy destabilization.

---

## 6. Surveillance History & Storage Concurrency

1. **Thread-Safe History Persistence:**
   - Surveillance run metrics are persisted to `reports/drift_monitoring_history.json`.
   - Write operations are protected by a process-level `threading.Lock()` to prevent race conditions during concurrent API invocations.

2. **Corruption & Failure Handling:**
   - If the persistence file is corrupted or unparseable, the service raises an explicit `HTTPException(500)` rather than masking the fault by returning an empty history `[]`.
   - The frontend dashboard cleanly distinguishes between a genuinely empty run history (`[]`) and a server/storage outage.

