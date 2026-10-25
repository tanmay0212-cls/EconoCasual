# EconoCausal Data Guide

This document describes how to obtain, generate, and structure datasets for the EconoCausal causal inference pipeline.

## 1. Primary Dataset Location

* **Sample/Bundled Dataset:** [`sample_data/synthetic_retail_campaign.csv`](../sample_data/synthetic_retail_campaign.csv)  
  A reproducible 2,500-customer synthetic campaign dataset generated using a parameterized Data Generating Process (DGP) with documented non-linear confounding.
* **Frontend Static Asset:** [`frontend/public/sample_retail_campaign.csv`](../frontend/public/sample_retail_campaign.csv)  
  Identical dataset bundled for in-browser client-side drag-and-drop testing.
* **Custom Dataset Placement:**  
  When running with custom operational retail logs, place your CSV file in `data/` (or provide its path via CLI arguments).

## 2. How to Generate or Regenerate the Synthetic Dataset

To generate a fresh synthetic campaign dataset:

```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# or: .\.venv\Scripts\Activate.ps1  # Windows PowerShell

# Run the data generator
python -c "
from src.data.generator import generate_synthetic_campaign_data
df = generate_synthetic_campaign_data(n_samples=2500, random_seed=42)
df.to_csv('sample_data/synthetic_retail_campaign.csv', index=False)
print(f'Generated {len(df)} records.')
"
```

Or execute the complete causal foundation pipeline:
```bash
python scripts/run_causal_pipeline.py
```

## 3. Required Data Schema

Any incoming retail campaign CSV must provide the following 10 canonical columns:

| Column Name | Type | Causal Taxonomy | Domain / Format | Description |
| :--- | :--- | :--- | :--- | :--- |
| `customer_id` | `string` | Identifier | `CUST_00001`+ | Unique customer code. Quarantined from feature matrices to prevent memorization. |
| `prior_purchases` | `integer` | Observed Confounder ($W_1$) | $\ge 0$ | Pre-treatment order count. Repeat buyers are disproportionately targeted. |
| `prior_spend` | `float` | Observed Confounder ($W_2$) | $\ge \$0.00$ | Cumulative pre-campaign monetary spend. Reflects customer purchasing power. |
| `engagement_score`| `float` | Observed Confounder ($W_3$) | $0.0 - 100.0$ | Digital engagement index (app opens, email click-through rate). |
| `customer_tenure` | `integer` | Observed Confounder ($W_4$) | $1 - 72$ mo | Account age in months. Accounts for loyalty and tenure cohort differences. |
| `campaign_channel`| `category` | Observed Confounder ($W_5$) | `email`, `push`, `sms`, `social` | Delivery medium utilized for the promotional campaign. |
| `discount_offered`| `binary` | **Treatment ($T$)** | $0$ or $1$ | Whether the customer received a discount offer ($1$) or standard pricing ($0$). Canonical alias: `discount_received`. |
| `discount_amount` | `float` | Post-treatment Mediator | $\ge \$0.00$ | Realized dollar face-value of discount. Strictly excluded from confounder adjustment set. |
| `purchased` | `binary` | **Primary Outcome ($Y$)** | $0$ or $1$ | Campaign conversion ($1 =$ purchased, $0 =$ did not purchase). Canonical alias: `purchase`. |
| `revenue` | `float` | Downstream Outcome | $\ge \$0.00$ | Realized post-discount basket value. Strictly excluded from confounder adjustment set. |

## 4. Synthetic Data Notice & Redistribution Terms

The synthetic retail campaign dataset contained in `sample_data/` is entirely computer-generated using mathematical distributions. It contains **no real consumer data, no financial records, and no Personally Identifiable Information (PII)**. It is safe for open distribution, research, and automated benchmarking.
