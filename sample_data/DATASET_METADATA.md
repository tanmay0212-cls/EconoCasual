# Dataset Metadata: Synthetic Retail Marketing Campaign

> **LABEL: SYNTHETIC DATASET**  
> *This dataset was synthetically generated using a parameterized Data Generating Process (DGP) for causal inference research, testing, and benchmarking. It does NOT contain real consumer data or personally identifiable information (PII).*

---

## 1. Overview
* **File:** `synthetic_retail_campaign.csv`
* **Sample Size ($N$):** 2,500 records
* **Random Seed:** 42 (fully reproducible via `src/data/generator.py`)
* **Use Case:** Dynamic pricing and promotional discount evaluation in retail marketing.

---

## 2. Variable Dictionary

| Column Name | Data Type | Role in Causal DAG | Description & Domain |
| :--- | :--- | :--- | :--- |
| `customer_id` | `string` | Identifier | Unique customer code (`CUST_00001` - `CUST_02500`). |
| `prior_purchases` | `integer` | Observed Confounder ($W_1$) | Historical number of orders placed by customer ($\ge 0$). Skewed Poisson distribution. |
| `prior_spend` | `float` | Observed Confounder ($W_2$) | Cumulative historical dollar spend ($\ge \$5.00$). Lognormal distribution correlated with purchase count. |
| `engagement_score` | `float` | Observed Confounder ($W_3$) | App and email activity index scaled between $0.0$ and $100.0$. |
| `customer_tenure` | `integer` | Observed Confounder ($W_4$) | Customer account age in months ($1 \le \text{tenure} \le 72$). |
| `campaign_channel` | `category` | Observed Confounder ($W_5$) | Marketing delivery medium: `email`, `push`, `sms`, `social`. |
| `discount_offered` | `binary (0/1)` | **Treatment ($T$)** | Whether the customer was assigned a promotional discount code ($1$) or standard pricing ($0$). |
| `discount_amount` | `float` | Post-treatment Mediator | Realized dollar face-value of discount (\$10-\$35 if offered, strictly \$0 if not offered). |
| `purchased` | `binary (0/1)` | **Primary Outcome ($Y$)** | Conversion: whether customer completed a purchase during campaign ($1$) or not ($0$). |
| `revenue` | `float` | Downstream Outcome | Realized basket dollar value post-discount (strictly \$0 if `purchased == 0`). |

---

## 3. Data Generating Process (DGP) Mechanics

### Treatment Assignment (Marketing Targeting Bias):
Marketers do not distribute promotional discounts randomly. Highly engaged users, long-tenured loyalty members, and high-spend customers are disproportionately targeted:
$$\text{logit}(P(\text{discount\_offered} = 1 \mid W)) = \beta_0 + \beta_1 \cdot \text{engagement} + \beta_2 \cdot \text{prior\_purchases} + \beta_3 \cdot \log(\text{prior\_spend}) + \beta_4 \cdot \text{tenure} + \gamma_{\text{channel}}$$

### Outcome Generation (Conversion):
The purchase decision is driven simultaneously by customer baseline propensity and the true causal incentive of the discount:
$$\text{logit}(P(\text{purchased} = 1 \mid T, W)) = \alpha_0 + \tau \cdot \text{discount\_offered} + \alpha_1 \cdot \text{engagement} + \alpha_2 \cdot \text{prior\_purchases} + \alpha_3 \cdot \log(\text{prior\_spend}) + \alpha_4 \cdot \text{tenure} + \delta_{\text{channel}}$$
* **True Causal Treatment Parameter ($\tau$):** $+0.70$ on logit scale ($\approx 10-15\%$ percentage-point lift).
* **Confounding Consequence:** Because $W$ drives both discount receipt and organic purchase conversion, naive unadjusted differences between treated and control groups exhibit upward selection bias.

---

## 4. Post-Treatment Guardrails
* `discount_amount` and `revenue` are downstream consequences of the treatment decision.
* **Strict Causal Rule:** Post-treatment variables are never included as common causes. Controlling for them would induce mediator suppression or collider stratification bias.
