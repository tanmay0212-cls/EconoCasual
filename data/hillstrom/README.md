# Hillstrom MineThatData Email Marketing Dataset Benchmark

**Source:** Kevin Hillstrom (MineThatData E-Mail Analytics Data Mining Challenge 2008)  
**Catalog Reference:** [scikit-uplift Hillstrom Documentation](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_hillstrom.html) | [TensorFlow Datasets](https://tensorflow.google.cn/datasets/catalog/hillstrom)  
**Total Records:** 64,000 customers (Randomized Controlled Trial / A/B/C Test)

---

## 1. Overview & Project Role

In **EconoCasual**, the Hillstrom dataset serves as an authentic, real-world randomized controlled benchmark for multi-treatment marketing uplift modeling.

Unlike observational retail datasets that require Double Machine Learning to purge marketing selection confounding, the Hillstrom dataset represents a **pure randomized experiment (RCT)**. Treatment assignment was completely unconfounded by design ($T \perp (Y(0), Y(1), Y(2))$), providing an empirical gold standard for causal uplift algorithms.

> **Methodological Contrast:**
> - **Hillstrom MineThatData:** Evaluates non-monetary marketing communication uplift (receiving Men's vs. Women's vs. No promotional email). There is no coupon code or monetary discount deduction.
> - **Synthetic Retail Dataset:** Evaluates price elasticity under monetary coupon deductions ($0 vs. $10 vs. $20 discounts), where discounts directly impact product margin and retail net profit.

---

## 2. Experimental Treatment Groups

The experiment randomized 64,000 past customers into three equal-sized cohorts:

| Group ID | Treatment Name | Raw Segment String | Customers | Share | Description |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **`0`** | `no_email` | `No E-Mail` | 21,306 | 33.29% | **Control Group:** Received no promotional communication. Measures organic baseline conversion. |
| **`1`** | `mens_email` | `Mens E-Mail` | 21,307 | 33.29% | **Treatment 1:** Received promotional email featuring men's apparel and accessories. |
| **`2`** | `womens_email` | `Womens E-Mail` | 21,387 | 33.42% | **Treatment 2:** Received promotional email featuring women's apparel and accessories. |

---

## 3. Pre-Treatment Covariates

All pre-treatment customer features were recorded prior to email distribution:

| Feature | Type | Description | Values / Range |
| :--- | :---: | :--- | :--- |
| `customer_id` | String | Unique synthetic identifier assigned during ingestion | `HILL_000001` — `HILL_064000` |
| `recency` | Integer | Months since last purchase prior to campaign | $1 - 12$ months |
| `history_segment`| Categorical | Binned historical dollar spend category | e.g. `$100 - $200`, `$200 - $350`, `$1,000+` |
| `history` | Float | Exact historical dollar spend in the past year | $\$29.99 - \$3,345.93$ (Mean: $\$242.09$) |
| `mens` | Binary | Customer purchased men's merchandise in past year | $0$ or $1$ ($55.1\%$ positive) |
| `womens` | Binary | Customer purchased women's merchandise in past year | $0$ or $1$ ($54.9\%$ positive) |
| `zip_code` | Categorical | Geographic classification of customer zip code | `Urban` ($40.1\%$), `Suburban` ($44.9\%$), `Rural` ($15.0\%$) |
| `newbie` | Binary | New customer acquired within past 12 months | $0$ or $1$ ($50.2\%$ new) |
| `channel` | Categorical | Channel customer used to purchase in past year | `Web` ($44.1\%$), `Phone` ($43.8\%$), `Multichannel` ($12.0\%$) |

---

## 4. Post-Treatment Outcomes

The campaign tracked customer behavioral responses over the 2-week observation window:

| Outcome | Type | Description | Control (0) | Men's Email (1) | Women's Email (2) |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **`visit`** | Binary | Visited e-commerce site within 2 weeks | 10.62% | 18.28% (+7.66 pp) | 15.14% (+4.52 pp) |
| **`conversion`** | Binary | Completed a purchase transaction within 2 weeks | 0.57% | 1.25% (+0.68 pp) | 0.88% (+0.31 pp) |
| **`spend`** | Float | Total dollar sales generated within 2 weeks | $0.65 | $1.42 (+$0.77) | $1.08 (+$0.43) |

> **Anti-Leakage Rule:** `visit`, `conversion`, and `spend` represent post-treatment outcomes. They must **never** be included as baseline confounder features in causal modeling.

---

## 5. Dataset File Structure

```text
data/hillstrom/
├── README.md                      # This documentation file
├── raw/
│   └── hillstrom_raw.csv          # Untouched downloaded dataset (64,000 rows, 3.96 MB)
└── processed/
    ├── hillstrom_processed.csv    # Master cleaned dataset with standardized treatment encoding
    ├── hillstrom_train.csv        # Stratified training partition (44,796 rows, 70%)
    ├── hillstrom_validation.csv   # Stratified validation partition (9,598 rows, 15%)
    └── hillstrom_test.csv         # Stratified held-out test partition (9,606 rows, 15%)
```

---

## 6. How to Reproduce

```bash
# 1. Download the raw dataset from official source
python scripts/download_hillstrom.py

# 2. Preprocess and generate stratified analytical splits
python scripts/preprocess_hillstrom.py
```

