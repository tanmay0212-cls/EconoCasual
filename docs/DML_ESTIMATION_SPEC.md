# Double Machine Learning (DML) Estimation Specification: EconoCausal (Week 2)

**Project:** EconoCausal — Dynamic Pricing via Double Machine Learning  
**Milestone:** Week 2 — EconML DML Estimator, Heterogeneous Treatment Effects & Causal Validation  
**Framework:** EconML (`econml.dml.LinearDML`), Scikit-Learn (`RandomForestRegressor`, `RandomForestClassifier`)  

---

## 1. Mathematical Foundation of Double Machine Learning

Standard supervised machine learning models (e.g. predicting $Y$ from $T$ and $W$) fail to deliver unbiased causal effect estimates because of **regularization bias**: machine learning algorithms use shrinkage, tree pruning, or early stopping to optimize out-of-sample prediction error rather than unbiased parameter estimation.

Double Machine Learning (Chernozhukov et al., 2018) resolves this through two key principles:
1. **Neyman Orthogonalization:** Transforming the moment conditions such that small errors in nuisance parameter estimation do not introduce first-order errors in the treatment effect parameter.
2. **K-Fold Cross-Fitting:** Estimating nuisance models on out-of-fold partitions to prevent overfitting bias from data reuse.

### The Partially Linear Model (PLM)
$$Y = \tau(X) \cdot T + g(W) + \varepsilon, \quad \mathbb{E}[\varepsilon \mid X, W, T] = 0$$
$$T = m(W) + \eta, \quad \mathbb{E}[\eta \mid W] = 0$$

Where:
* $Y \in \{0, 1\}$: Binary purchase outcome.
* $T \in \{0, 1\}$: Binary discount offer treatment.
* $W$: Observed pre-treatment confounder vector (`prior_purchases`, `prior_spend`, `engagement_score`, `customer_tenure`, `campaign_channel`).
* $X \subseteq W$: Effect modifiers (covariates across which we estimate heterogeneity).
* $g(W) = \mathbb{E}[Y \mid W]$: Baseline outcome expectation (Nuisance Model 1).
* $m(W) = P(T=1 \mid W)$: Propensity score (Nuisance Model 2).
* $\tau(X)$: Conditional Average Treatment Effect (CATE).

### Orthogonal Residualization
Using the Frisch-Waugh-Lovell theorem adapted for non-parametric nuisances:
$$\tilde{Y} = Y - \hat{g}(W)$$
$$\tilde{T} = T - \hat{m}(W)$$
$$\tilde{Y} = \tau(X) \cdot \tilde{T} + \nu$$

Because $\mathbb{E}[\tilde{T} \cdot \nu] = 0$, the CATE parameter vector $\theta$ satisfies the Neyman orthogonality condition:
$$\frac{\partial}{\partial \eta} \mathbb{E}\left[ \psi(W; \theta_0, \eta) \right] \Big|_{\eta = \eta_0} = 0$$

---

## 2. Cross-Fitting Protocol (5 Folds)

To break the statistical dependence between the nuisance estimates and the target parameter estimation:
1. The 2,000 training observations are randomly partitioned into $K=5$ equal folds: $I_1, I_2, I_3, I_4, I_5$.
2. For each fold $k \in \{1, \dots, 5\}$:
   * Nuisance models $\hat{g}^{(-k)}$ and $\hat{m}^{(-k)}$ are trained on the remaining 4 folds ($I \setminus I_k$).
   * Residuals $\tilde{Y}_i = Y_i - \hat{g}^{(-k)}(W_i)$ and $\tilde{T}_i = T_i - \hat{m}^{(-k)}(W_i)$ are predicted strictly for units $i \in I_k$.
3. The orthogonalized residuals from all 5 folds are pooled to estimate $\tau(X)$ via generalized method of moments (GMM).

---

## 3. Estimator Configuration & Hyperparameters

```python
LinearDML(
    model_y=RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    ),
    model_t=RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    ),
    discrete_treatment=True,
    cv=5,
    random_state=42
)
```

* **Outcome Nuisance Model:** Continuous regression ($[0, 1]$ probability) capturing non-linear purchase baseline.
* **Treatment Nuisance Model:** Probability classifier estimating propensity scores strictly bounded in $(0, 1)$.
* **Second-Stage Model:** Linear specification $\tau(X) = \langle \theta, X \rangle + \alpha$, which provides closed-form asymptotic variance and valid Wald confidence intervals.

---

## 4. Inference & Confidence Intervals

Under regularity conditions (bounded moments, Donsker class nuisances, convergence rate $\|\hat{g} - g\|_2 \times \|\hat{m} - m\|_2 = o_P(N^{-1/2})$):
$$\sqrt{N} (\hat{\tau}(X) - \tau_0(X)) \xrightarrow{d} \mathcal{N}\left(0, V(X)\right)$$

Customer-level Wald confidence intervals are computed at significance level $\alpha = 0.05$:
$$\text{CI}_{0.95}(\hat{\tau}(X_i)) = \hat{\tau}(X_i) \pm 1.96 \cdot \sqrt{\widehat{\text{Var}}(\hat{\tau}(X_i))}$$

---

## 5. Causal Identification Assumptions & Diagnostics

1. **Unconfoundedness:** $(Y(1), Y(0)) \perp T \mid W$. All common causes are observed in $W$.
2. **Positivity:** $0 < P(T=1 \mid W) < 1$. Verified in our data with propensity scores in $[0.18, 0.82]$.
3. **No Bad Controls:** Post-treatment variables (`discount_amount`, `revenue`) and individual identifiers (`customer_id`) are quarantined to avoid mediator suppression and collider bias.
4. **Synthetic Ground Truth Benchmark:**
   * DML Estimated ATE on held-out test: **+17.25 pp** (95% CI: `[+12.96 pp, +21.53 pp]`).
   * True synthetic ground truth ATE: **+14.46 pp** (Covered by 95% CI).
   * Naive unadjusted observational difference: **+25.21 pp** (+10.75 pp confounding bias).
