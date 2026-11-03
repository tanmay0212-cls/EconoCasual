# EconoCasual Causal & Multi-Treatment Model Documentation

**Project:** EconoCasual: Dynamic Pricing via Double Machine Learning  
**Estimator:** Multi-Tier Double Machine Learning (`LinearDRLearner` via EconML)  
**Trained Model Artifact:** `models/multitreatment_model.joblib` (4.84 MB)  
**Metadata Specification:** `models/multitreatment_metadata.json`

---

## 1. Causal Problem Formulation

Observational marketing data suffers from severe confounding: high-value, highly engaged, and loyal customers are preferentially targeted with promotional discounts. Because these customers have a high organic propensity to purchase, unadjusted regressions produce upwardly biased elasticity estimates, causing retailers to subsidize purchases that would have occurred at full price.

EconoCasual models the potential conversion outcomes $Y(d) \in \{0, 1\}$ under three discrete treatment options:
- $d = 0$: Full catalog price ($0 discount)
- $d = 10$: $10 promotional coupon
- $d = 20$: $20 promotional coupon

The causal objective is to estimate the **Conditional Average Treatment Effect (CATE)**:
$$\tau_d(X_i) = \mathbb{E}[Y(d) - Y(0) \mid X = X_i]$$
and the counterfactual purchase probabilities:
$$P(Y = 1 \mid X = X_i, T = d)$$
where $X_i$ represents pre-treatment customer covariates.

---

## 2. Mathematical Framework: Doubly Robust Double Machine Learning

EconoCasual uses EconML's `LinearDRLearner` (Doubly Robust Learner), combining orthogonal residualization with propensity weighting to achieve $\sqrt{N}$-consistent causal inference.

### 2.1 First-Stage Nuisance Models

Using 5-fold cross-fitting across $K = 5$ out-of-fold folds:

1. **Outcome Model ($\mu(d, X, W)$):** Predicts conversion probability given treatment and confounders:
   $$\hat{\mu}(d, X, W) = \mathbb{E}[Y \mid T = d, X, W]$$
   *Estimator:* `HistGradientBoostingClassifier` with early stopping and sigmoid calibration.

2. **Propensity Model ($e(d, W)$):** Estimates the probability of assigning discount tier $d$ given confounders:
   $$\hat{e}(d, W) = P(T = d \mid W)$$
   *Estimator:* Multi-class `HistGradientBoostingClassifier` with cross-entropy loss.

### 2.2 Doubly Robust Pseudo-Outcome Transformation

For each customer $i$ and discount tier $d \in \{10, 20\}$, the doubly robust pseudo-outcome $\Gamma_{i, d}$ is constructed:

$$\Gamma_{i, d} = \hat{\mu}(d, X_i, W_i) - \hat{\mu}(0, X_i, W_i) + \frac{\mathbf{1}_{\{T_i = d\}} \cdot (Y_i - \hat{\mu}(d, X_i, W_i))}{\hat{e}(d, W_i)} - \frac{\mathbf{1}_{\{T_i = 0\}} \cdot (Y_i - \hat{\mu}(0, X_i, W_i))}{\hat{e}(0, W_i)}$$

### 2.3 Second-Stage CATE Regression

The CATE function $\tau_d(X)$ is fit by minimizing square loss on the orthogonal pseudo-outcomes:

$$\hat{\tau}_d = \arg\min_{\tau} \sum_{i=1}^N \left( \Gamma_{i, d} - \tau(X_i) \right)^2$$

Because $\Gamma_{i, d}$ satisfies Neyman orthogonality, first-stage estimation errors in $\hat{\mu}$ and $\hat{e}$ enter only as second-order products:
$$\text{Bias} \propto \|\hat{\mu} - \mu\| \cdot \|\hat{e} - e\| = o_P(N^{-1/2})$$
ensuring asymptotically normal point estimates and valid 95% Wald confidence intervals.

---

## 3. Approved Features & Anti-Leakage Constraints

### 3.1 Pre-Treatment Features ($X$ and $W$)
- **`age`** (Numerical): Customer age in years $[18, 90]$.
- **`loyalty_score`** (Numerical): Customer engagement/loyalty index $[0, 100]$.
- **`previous_purchases`** (Numerical): Count of lifetime prior orders $[0, 50]$.
- **`customer_segment`** (Categorical): One-hot encoded `['budget', 'regular', 'premium']`.
- **`campaign_channel`** (Categorical): One-hot encoded `['email', 'web', 'social']`.

### 3.2 Strict Quarantine Rules
- **`customer_id`:** Identifier. Quarantined from all regression matrices to prevent high-cardinality memorization.
- **`discount_amount`:** Post-treatment mediator ($T \to M \to Y$). Conditioning introduces collider stratification bias.
- **`order_value` / `revenue`:** Post-treatment downstream outcome. Conditioning introduces future leakage.

---

## 4. Counterfactual Probability Calibration

While standard DML models estimate the linear additive treatment effect $\tau_d(X) = P(Y=1 \mid X, d) - P(Y=1 \mid X, 0)$, decision optimization requires absolute probabilities bounded strictly in $[0, 1]$.

EconoCasual computes:
$$\hat{P}(Y=1 \mid X_i, 0) = \hat{\mu}(0, X_i, W_i)$$
$$\hat{P}(Y=1 \mid X_i, 10) = \text{clip}\left(\hat{P}(Y=1 \mid X_i, 0) + \hat{\tau}_{10}(X_i), 0.0, 1.0\right)$$
$$\hat{P}(Y=1 \mid X_i, 20) = \text{clip}\left(\hat{P}(Y=1 \mid X_i, 0) + \hat{\tau}_{20}(X_i), 0.0, 1.0\right)$$

All outputs pass monotonicity and bounding checks ($0.0 \le \hat{P} \le 1.0$).

---

## 5. Model Validation & Empirical Metrics

The model was trained on 15,000 synthetic observations (`optimization_retail_data.csv`) using 5-fold cross-fitting and evaluated on 3,000 held-out test customers:

| Metric | Treatment: $10 vs $0 | Treatment: $20 vs $0 | Ground Truth Benchmark | Validation Status |
| :--- | :---: | :---: | :---: | :---: |
| **Model ATE ($\hat{\tau}$)** | **+15.22 pp** | **+21.45 pp** | +15.00 pp / +21.00 pp | **COVERS TRUTH** (Inside 95% CI) |
| **95% Wald CI** | [+14.10 pp, +16.34 pp] | [+20.15 pp, +22.75 pp] | — | Statistically significant ($p < 0.0001$) |
| **Rank Correlation** | $\rho = +0.284$ | $\rho = +0.312$ | $p < 10^{-6}$ | Elasticity ranking verified |
| **Propensity Overlap** | $[0.14, 0.76]$ | $[0.11, 0.69]$ | Positivity satisfied | No extreme propensity trimming needed |

