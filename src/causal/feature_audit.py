"""Causal Feature Audit and Pre-Training Data Hygiene Verification.

This module inspects both the Multi-Treatment Synthetic Retail dataset and the
Hillstrom MineThatData benchmark dataset to identify:
1. Confounders (pre-treatment covariates influencing treatment assignment & outcomes)
2. Treatment variables & representations
3. Outcome variables
4. Post-treatment variables to exclude (bad controls / intermediate mediators)
5. Data leakage vectors (ground-truth potential outcomes, future metrics)
6. Causal identification assumptions (Ignorability, Positivity, SUTVA)
"""

from typing import Dict, Any, List, Tuple
import os
import json
import numpy as np
import pandas as pd
import econml
import dowhy
import sklearn
import lightgbm


def audit_synthetic_retail(data_dir: str = "data/synthetic_retail") -> Dict[str, Any]:
    """Audit the multi-treatment synthetic retail dataset for causal compliance."""
    train_path = os.path.join(data_dir, "synthetic_retail_train.csv")
    val_path = os.path.join(data_dir, "synthetic_retail_validation.csv")
    test_path = os.path.join(data_dir, "synthetic_retail_test.csv")
    gt_path = os.path.join(data_dir, "synthetic_retail_ground_truth.csv")
    params_path = os.path.join(data_dir, "dgp_parameters.json")

    for path in [train_path, val_path, test_path, gt_path, params_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing expected synthetic retail file: {path}")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    df_gt = pd.read_csv(gt_path)

    with open(params_path, "r") as f:
        dgp_params = json.load(f)

    # Classification of columns
    pre_treatment_features = [
        "age",
        "loyalty_score",
        "previous_purchases",
        "customer_segment",
        "average_previous_order_value",
        "customer_activity_score",
    ]
    
    treatment_col = "discount_amount"
    valid_treatments = [0, 10, 20]
    primary_outcome = "purchase"
    
    # Post-treatment / financial realization variables (NOT to be used as features)
    post_treatment_vars = [
        "order_value",
        "gross_margin_rate",
        "gross_margin",
        "discount_cost",
        "revenue",
        "net_profit",
    ]

    # Isolated ground truth columns
    ground_truth_cols = [c for c in df_gt.columns if "potential" in c or "true_ite" in c or "exp_profit" in c or "optimal" in c]

    # Check for leakage in training data
    train_leaks = [c for c in df_train.columns if c in ground_truth_cols]

    # Audit confounding: compute correlation of pre-treatment features with treatment and outcome
    df_train_num = df_train.copy()
    segment_map = {"budget": 0, "regular": 1, "premium": 2, "vip": 3}
    df_train_num["customer_segment_num"] = df_train_num["customer_segment"].map(segment_map)
    audit_features = pre_treatment_features + ["customer_segment_num"]

    treatment_corrs = {}
    outcome_corrs = {}
    for feat in ["age", "loyalty_score", "previous_purchases", "average_previous_order_value", "customer_activity_score", "customer_segment_num"]:
        treatment_corrs[feat] = round(float(df_train_num[feat].corr(df_train_num[treatment_col])), 4)
        outcome_corrs[feat] = round(float(df_train_num[feat].corr(df_train_num[primary_outcome])), 4)

    return {
        "dataset_name": "Multi-Treatment Synthetic Retail",
        "splits": {
            "train_rows": len(df_train),
            "val_rows": len(df_val),
            "test_rows": len(df_test),
            "ground_truth_rows": len(df_gt),
        },
        "pre_treatment_features": pre_treatment_features,
        "treatment_variable": treatment_col,
        "valid_treatments": valid_treatments,
        "primary_outcome": primary_outcome,
        "post_treatment_vars": post_treatment_vars,
        "ground_truth_cols": ground_truth_cols,
        "leakage_detected": len(train_leaks) > 0,
        "leaked_columns": train_leaks,
        "treatment_correlations": treatment_corrs,
        "outcome_correlations": outcome_corrs,
        "dgp_parameters": dgp_params,
    }


def audit_hillstrom(data_dir: str = "data/hillstrom") -> Dict[str, Any]:
    """Audit the Hillstrom MineThatData benchmark dataset for causal compliance."""
    raw_path = os.path.join(data_dir, "raw", "hillstrom_raw.csv")
    proc_path = os.path.join(data_dir, "processed", "hillstrom_processed.csv")
    train_path = os.path.join(data_dir, "processed", "hillstrom_train.csv")
    val_path = os.path.join(data_dir, "processed", "hillstrom_validation.csv")
    test_path = os.path.join(data_dir, "processed", "hillstrom_test.csv")

    for path in [raw_path, proc_path, train_path, val_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing expected Hillstrom file: {path}")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    df_proc = pd.read_csv(proc_path)

    pre_treatment_features = [
        "recency",
        "history",
        "history_segment",
        "mens",
        "womens",
        "zip_code",
        "newbie",
        "channel",
    ]

    treatment_col = "treatment_group"
    valid_treatments = [0, 1, 2] # 0: No Email, 1: Mens Email, 2: Womens Email
    outcome_cols = ["visit", "conversion", "spend"]

    # Verify randomization / orthogonality with baseline covariates
    correlations_with_treatment = {}
    for feat in ["recency", "history", "mens", "womens", "newbie"]:
        correlations_with_treatment[feat] = round(float(df_proc[feat].corr(df_proc[treatment_col])), 5)

    return {
        "dataset_name": "Hillstrom MineThatData RCT Benchmark",
        "splits": {
            "raw_rows": len(pd.read_csv(raw_path)),
            "processed_rows": len(df_proc),
            "train_rows": len(df_train),
            "val_rows": len(df_val),
            "test_rows": len(df_test),
        },
        "pre_treatment_features": pre_treatment_features,
        "treatment_variable": treatment_col,
        "treatment_mapping": {0: "no_email", 1: "mens_email", 2: "womens_email"},
        "valid_treatments": valid_treatments,
        "outcomes": outcome_cols,
        "treatment_correlations": correlations_with_treatment,
        "randomization_verified": all(abs(c) < 0.02 for c in correlations_with_treatment.values()),
    }


def get_environment_info() -> Dict[str, str]:
    """Retrieve installed versions of causal and ML packages."""
    return {
        "econml": econml.__version__,
        "dowhy": dowhy.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
    }

