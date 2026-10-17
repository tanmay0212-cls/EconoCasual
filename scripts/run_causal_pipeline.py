"""Causal Pipeline Verification Script for EconoCausal Week 1.

This script executes the complete Week 1 verification workflow:
1. Generates and saves a reproducible synthetic retail marketing dataset.
2. Performs schema and domain validation.
3. Constructs the Causal Directed Acyclic Graph (DAG).
4. Initializes DoWhy CausalModel.
5. Runs Step 2 Effect Identification (Back-door criterion).
6. Outputs structural summaries and causal disclaimers.
"""

import os
import sys
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.generator import generate_synthetic_campaign_data
from src.data.schema import (
    REQUIRED_COLUMNS,
    TREATMENT_COL,
    OUTCOME_COL,
    OBSERVED_CONFOUNDERS,
    POST_TREATMENT_VARS,
    validate_dataset,
)
from src.causal.graph import (
    CAUSAL_DAG_DOT,
    build_causal_model,
    identify_causal_effect,
    get_dag_metadata,
)


def run_pipeline() -> None:
    print("=" * 75)
    print("  ECONOCAUSAL - DYNAMIC PRICING VIA DOUBLE MACHINE LEARNING")
    print("  Week 1: Causal Foundation & Structural DAG Verification")
    print("=" * 75)

    sample_data_dir = os.path.join(PROJECT_ROOT, "sample_data")
    os.makedirs(sample_data_dir, exist_ok=True)
    csv_path = os.path.join(sample_data_dir, "synthetic_retail_campaign.csv")

    # Step 1: Generate or load reproducible synthetic data
    print(f"\n[Step 1/5] Generating reproducible synthetic dataset (N=2,500)...")
    df = generate_synthetic_campaign_data(
        n_samples=2500,
        random_seed=42,
        output_filepath=csv_path,
    )
    print(f"  -> Saved to: {csv_path}")
    print(f"  -> Total records: {len(df):,}")
    print(f"  -> Columns ({len(df.columns)}): {list(df.columns)}")

    # Step 2: Validate Schema and Causal Domains
    print(f"\n[Step 2/5] Validating schema, binary domains, and post-treatment logic...")
    validation = validate_dataset(df)
    if not validation["is_valid"]:
        print(f"  FAILED: Dataset validation failed with errors: {validation['errors']}")
        sys.exit(1)

    print("  -> Schema status: VALID")
    metrics = validation["metrics"]
    print(f"  -> Treatment ({TREATMENT_COL}) rate: {metrics['treatment_rate'] * 100:.2f}%")
    print(f"  -> Outcome ({OUTCOME_COL}) conversion rate: {metrics['conversion_rate'] * 100:.2f}%")
    print(f"  -> Total realized revenue: ${metrics['total_revenue']:,.2f}")
    print(f"  -> Marketing channels present: {metrics['channels']}")

    # Step 3: Inspect Causal DAG Specifications
    print(f"\n[Step 3/5] Defining Causal Directed Acyclic Graph (DAG)...")
    dag_meta = get_dag_metadata()
    print(f"  -> Treatment: {dag_meta['treatment']}")
    print(f"  -> Outcome: {dag_meta['outcome']}")
    print(f"  -> Observed Confounders (W): {dag_meta['observed_confounders']}")
    print(f"  -> Post-treatment downstream variables (excluded as confounders): {dag_meta['post_treatment_variables']}")
    print(f"  -> Total nodes: {len(dag_meta['nodes'])}, Total directed edges: {len(dag_meta['edges'])}")

    # Step 4: Initialize DoWhy CausalModel
    print(f"\n[Step 4/5] Initializing DoWhy CausalModel...")
    causal_model = build_causal_model(df)
    print("  -> DoWhy CausalModel successfully initialized!")

    # Step 5: Identification via Back-door Criterion
    print(f"\n[Step 5/5] Identifying causal effect via Back-door Criterion...")
    identified_estimand = identify_causal_effect(causal_model)
    backdoor_vars = identified_estimand.get_backdoor_variables()
    print("  -> Back-door identification SUCCEEDED.")
    print(f"  -> Minimal adjustment set to block non-causal paths: {backdoor_vars}")

    print("\n" + "-" * 75)
    print("  CRITICAL CAUSAL METHODOLOGY NOTES:")
    print("  1. The DAG specifies structural hypotheses based on marketing domain theory.")
    print("     The graph does NOT 'prove' causality itself; it makes assumptions testable.")
    print("  2. Post-treatment variables ('discount_amount', 'revenue') are strictly excluded")
    print("     from the adjustment set to prevent mediator/collider bias.")
    print("  3. Week 1 verifies model initialization and back-door identification.")
    print("     Estimation via Double Machine Learning (DML) is scheduled for Week 2.")
    print("-" * 75)
    print("\n[SUCCESS] Week 1 Causal Foundation verified successfully!\n")


if __name__ == "__main__":
    run_pipeline()
