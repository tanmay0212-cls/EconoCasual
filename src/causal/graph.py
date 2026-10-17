"""Causal Directed Acyclic Graph (DAG) specification and DoWhy modeling engine.

This module formalizes the structural causal assumptions for retail dynamic pricing
and marketing campaign intervention:
- Treatment: discount_offered (binary 0/1)
- Outcome: purchased (binary 0/1)
- Common Causes (Observed Confounders): pre-treatment customer history and channel
- Downstream Outcomes: discount_amount, revenue (strictly excluded as confounders)
"""

from typing import Dict, Any, Optional
import pandas as pd
from dowhy import CausalModel

from ..data.schema import (
    TREATMENT_COL,
    OUTCOME_COL,
    OBSERVED_CONFOUNDERS,
    POST_TREATMENT_VARS,
    validate_dataset,
)

# Canonical Causal DAG expressed in Graphviz DOT format
# Assumptions:
# 1. Prior purchases, prior spend, tenure, and engagement represent customer loyalty
#    and baseline purchase intent.
# 2. These pre-treatment confounders influence marketing discount assignment (Treatment)
#    due to promotional targeting policies.
# 3. They simultaneously influence purchase conversion (Outcome).
# 4. Treatment (discount_offered) causes purchase conversion.
# 5. discount_amount and revenue are strictly post-treatment mediators/outcomes and
#    must NEVER be adjusted for (conditioning on them introduces collider/mediator bias).
CAUSAL_DAG_DOT: str = """
digraph {
    prior_purchases -> discount_offered;
    prior_purchases -> purchased;
    prior_spend -> discount_offered;
    prior_spend -> purchased;
    engagement_score -> discount_offered;
    engagement_score -> purchased;
    customer_tenure -> discount_offered;
    customer_tenure -> purchased;
    campaign_channel -> discount_offered;
    campaign_channel -> purchased;
    discount_offered -> purchased;
    discount_offered -> discount_amount;
    purchased -> revenue;
    discount_amount -> revenue;
}
""".strip()


def build_causal_model(
    df: pd.DataFrame,
    graph_dot: Optional[str] = None,
) -> CausalModel:
    """Initialize a DoWhy CausalModel with validated data and causal DAG.

    Args:
        df: Input pandas DataFrame containing campaign observations.
        graph_dot: Optional custom DOT graph string. Defaults to CAUSAL_DAG_DOT.

    Returns:
        Initialized dowhy.CausalModel instance.

    Raises:
        ValueError: If dataset schema is invalid.
    """
    val_report = validate_dataset(df)
    if not val_report["is_valid"]:
        raise ValueError(
            f"Cannot initialize CausalModel: dataset failed validation. Errors: {val_report['errors']}"
        )

    dag_spec = graph_dot if graph_dot else CAUSAL_DAG_DOT

    # Initialize DoWhy CausalModel
    model = CausalModel(
        data=df,
        treatment=TREATMENT_COL,
        outcome=OUTCOME_COL,
        common_causes=OBSERVED_CONFOUNDERS,
        graph=dag_spec,
    )
    return model


def identify_causal_effect(model: CausalModel) -> Any:
    """Execute Step 2 of DoWhy framework: Causal Effect Identification.

    Uses the Back-door criterion to identify the adjustment set required
    to block all non-causal spurious paths between treatment and outcome.

    IMPORTANT NOTE:
    Identification formalizes the theoretical estimand under the specified DAG.
    It does NOT prove the DAG is correct or claim that causal effect has been
    empirically estimated (Week 2 Double Machine Learning will perform estimation).

    Args:
        model: An initialized dowhy.CausalModel.

    Returns:
        DoWhy IdentifiedEstimand object.
    """
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    return identified_estimand


def get_dag_metadata() -> Dict[str, Any]:
    """Retrieve structured metadata of the DAG for visualization, APIs, and inspection.

    Returns:
        Dictionary detailing nodes, edge connections, roles, and causal justifications.
    """
    nodes = [
        {
            "id": TREATMENT_COL,
            "label": "Discount Offered",
            "role": "treatment",
            "type": "binary",
            "description": "Marketing intervention: whether a promotional discount was offered to the customer.",
        },
        {
            "id": OUTCOME_COL,
            "label": "Purchased",
            "role": "outcome",
            "type": "binary",
            "description": "Primary conversion outcome: whether customer made a purchase.",
        },
        {
            "id": "prior_purchases",
            "label": "Prior Purchases",
            "role": "confounder",
            "type": "integer",
            "description": "Pre-treatment order frequency. Confounder: targeted for loyalty discounts and has higher baseline purchase rate.",
        },
        {
            "id": "prior_spend",
            "label": "Prior Spend ($)",
            "role": "confounder",
            "type": "continuous",
            "description": "Pre-treatment monetary spend. Confounder: high-value customers receive targeted promotions and have higher purchasing power.",
        },
        {
            "id": "engagement_score",
            "label": "Engagement Score",
            "role": "confounder",
            "type": "continuous",
            "description": "Pre-treatment digital activity score (0-100). Confounder: active users open promotional messages and convert more frequently.",
        },
        {
            "id": "customer_tenure",
            "label": "Customer Tenure (mo)",
            "role": "confounder",
            "type": "integer",
            "description": "Account age in months. Confounder: long-tenured customers qualify for loyalty perks and exhibit habitual purchasing.",
        },
        {
            "id": "campaign_channel",
            "label": "Campaign Channel",
            "role": "confounder",
            "type": "categorical",
            "description": "Outreach channel (email, push, sms, social). Confounder: delivery rates of discounts and baseline channel response rates vary.",
        },
        {
            "id": "discount_amount",
            "label": "Discount Amount ($)",
            "role": "post_treatment",
            "type": "continuous",
            "description": "Realized discount value. Post-treatment mediator: strictly excluded as confounder to prevent collider bias.",
        },
        {
            "id": "revenue",
            "label": "Revenue ($)",
            "role": "post_treatment",
            "type": "continuous",
            "description": "Realized basket spend post-discount. Downstream outcome: strictly excluded as confounder.",
        },
    ]

    edges = [
        {"source": "prior_purchases", "target": TREATMENT_COL, "type": "confounding"},
        {"source": "prior_purchases", "target": OUTCOME_COL, "type": "confounding"},
        {"source": "prior_spend", "target": TREATMENT_COL, "type": "confounding"},
        {"source": "prior_spend", "target": OUTCOME_COL, "type": "confounding"},
        {"source": "engagement_score", "target": TREATMENT_COL, "type": "confounding"},
        {"source": "engagement_score", "target": OUTCOME_COL, "type": "confounding"},
        {"source": "customer_tenure", "target": TREATMENT_COL, "type": "confounding"},
        {"source": "customer_tenure", "target": OUTCOME_COL, "type": "confounding"},
        {"source": "campaign_channel", "target": TREATMENT_COL, "type": "confounding"},
        {"source": "campaign_channel", "target": OUTCOME_COL, "type": "confounding"},
        {"source": TREATMENT_COL, "target": OUTCOME_COL, "type": "causal_treatment_effect"},
        {"source": TREATMENT_COL, "target": "discount_amount", "type": "post_treatment"},
        {"source": OUTCOME_COL, "target": "revenue", "type": "downstream_outcome"},
        {"source": "discount_amount", "target": "revenue", "type": "accounting_deduction"},
    ]

    return {
        "treatment": TREATMENT_COL,
        "outcome": OUTCOME_COL,
        "observed_confounders": OBSERVED_CONFOUNDERS,
        "post_treatment_variables": POST_TREATMENT_VARS,
        "nodes": nodes,
        "edges": edges,
        "dot_graph": CAUSAL_DAG_DOT,
        "causal_assumptions": [
            "Unconfoundedness conditional on observed pre-treatment history (prior purchases, spend, engagement, tenure, channel).",
            "Positivity: every customer segment has non-zero probability of receiving or not receiving discount.",
            "No unobserved confounders (critical identification assumption to be tested via refutations in Week 2).",
            "No interference (SUTVA): one customer's discount does not affect another customer's purchase decision.",
        ],
    }
