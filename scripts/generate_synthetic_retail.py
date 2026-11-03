"""
Generate Multi-Treatment Synthetic Retail Dataset for EconoCasual.
Supports T0 ($0), T1 ($10), and T2 ($20) discount levels with heterogeneous price elasticity,
observed confounding, and rigorous accounting standards.

Outputs:
  data/synthetic_retail/synthetic_retail_train.csv
  data/synthetic_retail/synthetic_retail_validation.csv
  data/synthetic_retail/synthetic_retail_test.csv
  data/synthetic_retail/synthetic_retail_ground_truth.csv
  data/synthetic_retail/dgp_parameters.json
"""
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))

def softmax(logits):
    exp_z = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def generate_multi_treatment_synthetic_data(n_samples: int = 40000, random_seed: int = 42):
    np.random.seed(random_seed)
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "data" / "synthetic_retail"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("GENERATING MULTI-TREATMENT SYNTHETIC RETAIL DATASET")
    print("=" * 75)
    print(f"Target sample size: {n_samples:,} customer records")
    print(f"Random seed: {random_seed}")

    # -------------------------------------------------------------------------
    # 1. Customer Pre-Treatment Characteristics (Covariates X)
    # -------------------------------------------------------------------------
    customer_ids = [f"SYN_{i+1:06d}" for i in range(n_samples)]

    # Age: Normal centered at 43 years, bounded [18, 80]
    age = np.clip(np.random.normal(loc=43.5, scale=12.5, size=n_samples), 18, 80).astype(int)

    # Customer Segment: Budget (30%), Regular (40%), Premium (20%), VIP (10%)
    segment_choices = ["budget", "regular", "premium", "vip"]
    segment_probs = [0.30, 0.40, 0.20, 0.10]
    customer_segment = np.random.choice(segment_choices, size=n_samples, p=segment_probs)

    # Segment multipliers for economic traits
    segment_wealth_map = {"budget": 0.70, "regular": 1.00, "premium": 1.45, "vip": 2.10}
    segment_loyalty_bias = {"budget": -12.0, "regular": 0.0, "premium": 14.0, "vip": 25.0}

    # Loyalty score: [0, 100] influenced by segment
    loyalty_base = np.random.beta(a=2.5, b=2.5, size=n_samples) * 70.0 + 15.0
    loyalty_bias = np.array([segment_loyalty_bias[s] for s in customer_segment])
    loyalty_score = np.clip(loyalty_base + loyalty_bias + np.random.normal(0, 5, n_samples), 0.0, 100.0).round(1)

    # Previous Purchases: Discrete count [0, 30] correlated with loyalty and segment
    prev_mean = 2.0 + (loyalty_score / 15.0) + np.array([segment_wealth_map[s] * 2.5 for s in customer_segment])
    previous_purchases = np.clip(np.random.poisson(lam=prev_mean), 0, 35).astype(int)

    # Average Previous Order Value: Log-normal distribution scaled by segment wealth
    base_aov = np.array([segment_wealth_map[s] * 75.0 for s in customer_segment])
    average_previous_order_value = np.clip(
        base_aov * np.random.lognormal(mean=0.0, sigma=0.25, size=n_samples),
        25.0,
        350.0
    ).round(2)

    # Customer Activity Score: [0, 100] reflecting recent site engagement & recency
    activity_base = (
        0.35 * loyalty_score
        + 1.2 * np.clip(previous_purchases, 0, 15)
        + np.random.normal(loc=30.0, scale=12.0, size=n_samples)
    )
    customer_activity_score = np.clip(activity_base, 0.0, 100.0).round(1)

    # Campaign Date: Spread over a 9-month retail calendar
    start_date = datetime(2026, 1, 15)
    random_days = np.random.randint(0, 250, size=n_samples)
    campaign_date = [(start_date + timedelta(days=int(d))).strftime("%Y-%m-%d") for d in random_days]

    # Baseline product price & margin for the active campaign
    # Product price aligns with customer's historical order value with slight variance
    product_price = np.clip(
        average_previous_order_value * np.random.normal(loc=1.0, scale=0.08, size=n_samples),
        25.0,
        380.0
    ).round(2)

    # Gross margin rate: Retail margin between 40% and 65% (mean ~52%)
    gross_margin_rate = np.clip(
        0.52 + 0.04 * (average_previous_order_value > 120.0) - 0.03 * (customer_segment == "budget") + np.random.normal(0, 0.02, size=n_samples),
        0.35,
        0.70
    ).round(3)

    # -------------------------------------------------------------------------
    # 2. Confounded Treatment Assignment Mechanism
    # -------------------------------------------------------------------------
    # Marketing teams preferentially target active/loyal users, creating selection bias
    # Softmax ensures full overlap / positivity across all 3 tiers: P(T=d | X) in [0.10, 0.65]
    logit_0 = np.zeros(n_samples)
    
    # $10 discount propensity: targeted at regular and budget customers with moderate activity
    logit_10 = (
        -0.10
        + 0.015 * (customer_activity_score - 50.0)
        + 0.010 * (loyalty_score - 50.0)
        + 0.35 * (customer_segment == "regular")
        + 0.20 * (customer_segment == "budget")
        - 0.30 * (customer_segment == "vip")
    )

    # $20 discount propensity: targeted at higher spenders and premium/VIP segments
    logit_20 = (
        -0.35
        + 0.020 * (customer_activity_score - 50.0)
        + 0.015 * (average_previous_order_value - 90.0) / 30.0
        + 0.40 * (customer_segment == "premium")
        + 0.55 * (customer_segment == "vip")
        - 0.45 * (customer_segment == "budget")
    )

    logits_matrix = np.column_stack([logit_0, logit_10, logit_20])
    treatment_propensities = softmax(logits_matrix)

    # Sample treatment assignment per customer
    treatment_levels = np.array([0, 10, 20])
    assigned_treatment_idx = np.array([
        np.random.choice([0, 1, 2], p=treatment_propensities[i])
        for i in range(n_samples)
    ])
    discount_amount = treatment_levels[assigned_treatment_idx]

    # -------------------------------------------------------------------------
    # 3. Ground-Truth Potential Purchase Probabilities & Heterogeneous CATE
    # -------------------------------------------------------------------------
    # 3.1 Baseline organic purchase probability (T0 = $0 discount)
    z_baseline = (
        -1.40
        + 0.025 * (loyalty_score - 50.0)
        + 0.028 * (customer_activity_score - 50.0)
        + 0.060 * np.log1p(previous_purchases)
        - 0.010 * (age - 40.0)
        + 0.30 * (customer_segment == "vip")
        + 0.15 * (customer_segment == "premium")
        - 0.25 * (customer_segment == "budget")
    )
    p0 = np.clip(sigmoid(z_baseline), 0.02, 0.92)

    # 3.2 Heterogeneous Treatment Lift for $10 Discount (tau_10)
    # Sweet spot for budget/regular segments with moderate loyalty
    tau_10_raw = (
        0.16
        + 0.08 * (customer_segment == "budget")
        + 0.05 * (customer_segment == "regular")
        - 0.06 * (customer_segment == "vip")
        - 0.0015 * (loyalty_score - 40.0) # Organic loyal users need less discount
        + 0.0012 * (customer_activity_score - 45.0)
    )
    tau_10 = np.clip(tau_10_raw, 0.01, 0.35)
    p10 = np.clip(p0 + tau_10, 0.03, 0.97)

    # 3.3 Heterogeneous Treatment Lift for $20 Discount (tau_20)
    # Effective for high-hurdle budget shoppers or high-AOV premium buyers,
    # but provides diminishing returns for already-loyal buyers
    tau_20_raw = (
        0.23
        + 0.10 * (customer_segment == "budget")
        + 0.06 * (average_previous_order_value > 110.0)
        - 0.08 * (customer_segment == "vip") # VIP already buys organically
        - 0.0020 * (loyalty_score - 40.0)
        + 0.0015 * (customer_activity_score - 45.0)
    )
    tau_20 = np.clip(tau_20_raw, tau_10 + 0.01, 0.45)
    p20 = np.clip(p0 + tau_20, 0.05, 0.99)

    # Individual Treatment Effects (ITE) relative to $0
    ite_10_vs_0 = (p10 - p0).round(4)
    ite_20_vs_0 = (p20 - p0).round(4)
    ite_20_vs_10 = (p20 - p10).round(4)

    # -------------------------------------------------------------------------
    # 4. Realized & Potential Business Outcomes
    # -------------------------------------------------------------------------
    # Common random shock per individual coupling counterfactual potential outcomes
    # Ensures Rubin Causal Model consistency: Y = Y(D) = sum_d 1{D=d} * Y(d)
    u_shock = np.random.uniform(0, 1, size=n_samples)
    y0_pot = (u_shock < p0).astype(int)
    y10_pot = (u_shock < p10).astype(int)
    y20_pot = (u_shock < p20).astype(int)

    # Realized purchase probability under the assigned discount
    realized_prob = np.where(
        discount_amount == 0,
        p0,
        np.where(discount_amount == 10, p10, p20)
    )
    
    # Realized binary purchase satisfies consistency: Y = Y_d for assigned d
    purchase = np.where(
        discount_amount == 0,
        y0_pot,
        np.where(discount_amount == 10, y10_pot, y20_pot)
    )

    # Basket expansion elasticity: coupons increase order value slightly
    basket_boost = np.where(discount_amount == 10, 1.03, np.where(discount_amount == 20, 1.06, 1.00))
    realized_order_val = np.where(purchase == 1, (product_price * basket_boost).round(2), 0.0)

    # Accounting Definitions:
    # 1. Discount cost is incurred only when a customer redeems upon purchase
    discount_cost = np.where(purchase == 1, discount_amount.astype(float), 0.0).round(2)
    # 2. Realized revenue: net cash collected from customer
    revenue = np.maximum(0.0, realized_order_val - discount_cost).round(2)
    # 3. Gross margin: gross product profit before discount deduction
    gross_margin = np.where(purchase == 1, (realized_order_val * gross_margin_rate).round(2), 0.0)
    # 4. Net profit / contribution: gross margin minus promotional discount cost
    net_profit = np.where(purchase == 1, (gross_margin - discount_cost).round(2), 0.0)

    # -------------------------------------------------------------------------
    # 5. Potential Economic Value & Optimal Strategic Policy (Ground Truth)
    # -------------------------------------------------------------------------
    # For every customer, compute expected contribution under each option d in {0, 10, 20}:
    # E[Profit_d] = (price * margin - d) * P(Y_d = 1)
    exp_profit_0 = (product_price * gross_margin_rate * p0).round(2)
    exp_profit_10 = ((product_price * 1.03 * gross_margin_rate - 10.0) * p10).round(2)
    exp_profit_20 = ((product_price * 1.06 * gross_margin_rate - 20.0) * p20).round(2)

    # Optimal discount decision that maximizes expected net profit (unconstrained)
    profits_matrix = np.column_stack([exp_profit_0, exp_profit_10, exp_profit_20])
    best_discount_idx = np.argmax(profits_matrix, axis=1)
    optimal_discount_tier = treatment_levels[best_discount_idx]

    # -------------------------------------------------------------------------
    # 6. Assemble Full DataFrame & Segregate Ground Truth
    # -------------------------------------------------------------------------
    # Analytical Customer Records (Safe for model training - ZERO LEAKAGE)
    analytical_df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": age,
        "loyalty_score": loyalty_score,
        "previous_purchases": previous_purchases,
        "customer_segment": customer_segment,
        "average_previous_order_value": average_previous_order_value,
        "customer_activity_score": customer_activity_score,
        "campaign_date": campaign_date,
        "discount_amount": discount_amount,
        "purchase": purchase,
        "purchase_probability": realized_prob.round(4),
        "order_value": realized_order_val,
        "gross_margin_rate": gross_margin_rate,
        "gross_margin": gross_margin,
        "discount_cost": discount_cost,
        "revenue": revenue,
        "net_profit": net_profit,
    })

    # Ground-Truth Potential Outcomes & CATE (Stored SEPARATELY for evaluation only)
    ground_truth_df = pd.DataFrame({
        "customer_id": customer_ids,
        "assigned_discount": discount_amount,
        "potential_purchase_prob_0": p0.round(4),
        "potential_purchase_prob_10": p10.round(4),
        "potential_purchase_prob_20": p20.round(4),
        "true_ite_10_vs_0": ite_10_vs_0,
        "true_ite_20_vs_0": ite_20_vs_0,
        "true_ite_20_vs_10": ite_20_vs_10,
        "potential_purchase_0": y0_pot,
        "potential_purchase_10": y10_pot,
        "potential_purchase_20": y20_pot,
        "expected_net_profit_0": exp_profit_0,
        "expected_net_profit_10": exp_profit_10,
        "expected_net_profit_20": exp_profit_20,
        "optimal_unconstrained_discount": optimal_discount_tier,
    })

    # -------------------------------------------------------------------------
    # 7. Stratified Train / Validation / Test Splitting (70% / 15% / 15%)
    # -------------------------------------------------------------------------
    # Stratify by discount_amount and purchase to prevent distribution mismatch
    strata = analytical_df["discount_amount"].astype(str) + "_" + analytical_df["purchase"].astype(str)
    
    train_indices = []
    val_indices = []
    test_indices = []

    for _, group in analytical_df.groupby(strata):
        shuffled = np.random.permutation(group.index.to_numpy())
        n = len(shuffled)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        
        train_indices.extend(shuffled[:n_train])
        val_indices.extend(shuffled[n_train:n_train + n_val])
        test_indices.extend(shuffled[n_train + n_val:])

    train_df = analytical_df.loc[train_indices].sort_values("customer_id").reset_index(drop=True)
    val_df = analytical_df.loc[val_indices].sort_values("customer_id").reset_index(drop=True)
    test_df = analytical_df.loc[test_indices].sort_values("customer_id").reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 8. Export Data Files
    # -------------------------------------------------------------------------
    train_path = out_dir / "synthetic_retail_train.csv"
    val_path = out_dir / "synthetic_retail_validation.csv"
    test_path = out_dir / "synthetic_retail_test.csv"
    gt_path = out_dir / "synthetic_retail_ground_truth.csv"
    param_path = out_dir / "dgp_parameters.json"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    ground_truth_df.to_csv(gt_path, index=False)

    # Parameters metadata file
    dgp_params = {
        "dataset_name": "EconoCasual Multi-Treatment Synthetic Retail Dataset",
        "generated_timestamp": datetime.now().isoformat(),
        "random_seed": random_seed,
        "n_samples": n_samples,
        "partitions": {
            "train": {"rows": len(train_df), "pct": 70.0, "file": train_path.name},
            "validation": {"rows": len(val_df), "pct": 15.0, "file": val_path.name},
            "test": {"rows": len(test_df), "pct": 15.0, "file": test_path.name},
            "ground_truth": {"rows": len(ground_truth_df), "file": gt_path.name},
        },
        "treatments": {
            "0": {"name": "No Discount ($0)", "count": int((discount_amount == 0).sum())},
            "10": {"name": "Moderate Discount ($10)", "count": int((discount_amount == 10).sum())},
            "20": {"name": "Deep Discount ($20)", "count": int((discount_amount == 20).sum())},
        },
        "true_average_treatment_effects": {
            "ate_10_vs_0": float(np.mean(p10 - p0)),
            "ate_20_vs_0": float(np.mean(p20 - p0)),
            "ate_20_vs_10": float(np.mean(p20 - p10)),
        },
        "optimal_discount_distribution_unconstrained": {
            "$0": int((optimal_discount_tier == 0).sum()),
            "$10": int((optimal_discount_tier == 10).sum()),
            "$20": int((optimal_discount_tier == 20).sum()),
        },
        "accounting_definitions": {
            "discount_cost": "Incurred upon customer conversion (discount_amount * purchase)",
            "revenue": "Net receipts collected (max(0, order_value - discount_cost))",
            "gross_margin": "order_value * gross_margin_rate * purchase",
            "net_profit": "gross_margin - discount_cost",
        }
    }

    with open(param_path, "w", encoding="utf-8") as f:
        json.dump(dgp_params, f, indent=2)

    print("\n[OK] Synthetic retail datasets successfully exported:")
    print(f"  - Train:        {train_path} ({len(train_df):,} rows)")
    print(f"  - Validation:   {val_path} ({len(val_df):,} rows)")
    print(f"  - Test:         {test_path} ({len(test_df):,} rows)")
    print(f"  - Ground Truth: {gt_path} ({len(ground_truth_df):,} rows)")
    print(f"  - Parameters:   {param_path}")

    print("\nTreatment Group Breakdown:")
    for d in [0, 10, 20]:
        sub = analytical_df[analytical_df["discount_amount"] == d]
        p_rate = sub["purchase"].mean() * 100
        avg_rev = sub["revenue"].mean()
        avg_prof = sub["net_profit"].mean()
        print(f"  - Discount ${d:2d}: N={len(sub):,} ({len(sub)/n_samples*100:.1f}%) | Purchase={p_rate:.2f}% | Mean Rev=${avg_rev:.2f} | Mean Profit=${avg_prof:.2f}")

    print("\nGround-Truth True Causal Effects:")
    print(f"  - True ATE ($10 vs $0): +{np.mean(p10 - p0)*100:.2f} percentage points")
    print(f"  - True ATE ($20 vs $0): +{np.mean(p20 - p0)*100:.2f} percentage points")
    print(f"  - Unconstrained Profit-Optimal Policy:")
    print(f"      $0 Discount best for:  {(optimal_discount_tier == 0).sum():,} customers ({(optimal_discount_tier == 0).mean()*100:.1f}%)")
    print(f"      $10 Discount best for: {(optimal_discount_tier == 10).sum():,} customers ({(optimal_discount_tier == 10).mean()*100:.1f}%)")
    print(f"      $20 Discount best for: {(optimal_discount_tier == 20).sum():,} customers ({(optimal_discount_tier == 20).mean()*100:.1f}%)")

if __name__ == "__main__":
    generate_multi_treatment_synthetic_data()
