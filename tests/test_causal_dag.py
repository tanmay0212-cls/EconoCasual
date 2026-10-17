"""Unit tests for Causal DAG structure, DoWhy initialization, and effect identification."""

import pytest
import networkx as nx
from dowhy import CausalModel

from src.data.generator import generate_synthetic_campaign_data
from src.data.schema import (
    TREATMENT_COL,
    OUTCOME_COL,
    OBSERVED_CONFOUNDERS,
    POST_TREATMENT_VARS,
)
from src.causal.graph import (
    CAUSAL_DAG_DOT,
    build_causal_model,
    identify_causal_effect,
    get_dag_metadata,
)


def test_dag_metadata_structure():
    """Verify that DAG metadata conforms to required causal specifications."""
    meta = get_dag_metadata()
    assert meta["treatment"] == TREATMENT_COL
    assert meta["outcome"] == OUTCOME_COL
    assert set(meta["observed_confounders"]) == set(OBSERVED_CONFOUNDERS)
    assert set(meta["post_treatment_variables"]) == set(POST_TREATMENT_VARS)
    assert len(meta["nodes"]) == 9
    assert len(meta["edges"]) == 14


def test_no_post_treatment_confounders():
    """Verify strictly that post-treatment variables are never included as confounders."""
    meta = get_dag_metadata()
    confounders = set(meta["observed_confounders"])
    for post_var in POST_TREATMENT_VARS:
        assert post_var not in confounders, (
            f"Methodological violation: Post-treatment variable '{post_var}' "
            f"found in observed confounders!"
        )


def test_dag_is_strictly_acyclic():
    """Verify using networkx that the Causal DAG has no directed cycles (is a true DAG)."""
    # Parse DOT graph edges into a networkx DiGraph
    G = nx.DiGraph()
    for line in CAUSAL_DAG_DOT.split("\n"):
        line = line.strip()
        if "->" in line:
            src, tgt = line.replace(";", "").split("->")
            G.add_edge(src.strip(), tgt.strip())

    assert nx.is_directed_acyclic_graph(G), "Causal graph contains a cycle! Must be a strict DAG."
    assert TREATMENT_COL in G.nodes
    assert OUTCOME_COL in G.nodes

    # Verify path exists from treatment to outcome
    assert nx.has_path(G, TREATMENT_COL, OUTCOME_COL)


def test_dowhy_causal_model_initialization():
    """Verify DoWhy CausalModel successfully initializes with synthetic dataset."""
    df = generate_synthetic_campaign_data(n_samples=1000, random_seed=42)
    model = build_causal_model(df)

    assert isinstance(model, CausalModel)
    assert model._treatment == [TREATMENT_COL]
    assert model._outcome == [OUTCOME_COL]


def test_dowhy_backdoor_identification():
    """Verify DoWhy executes Step 2 Identification and identifies the backdoor adjustment set."""
    df = generate_synthetic_campaign_data(n_samples=1000, random_seed=42)
    model = build_causal_model(df)
    identified_estimand = identify_causal_effect(model)

    assert identified_estimand is not None
    backdoor_vars = identified_estimand.get_backdoor_variables()

    # The identified backdoor variables must match our observed confounders
    assert set(backdoor_vars) == set(OBSERVED_CONFOUNDERS)

    # Post-treatment variables must not be in the backdoor adjustment set
    for post_var in POST_TREATMENT_VARS:
        assert post_var not in backdoor_vars


def test_causal_model_rejects_corrupted_data():
    """Verify build_causal_model rejects corrupted datasets before DoWhy initialization."""
    df = generate_synthetic_campaign_data(n_samples=1000, random_seed=42)
    # Corrupt treatment column
    df[TREATMENT_COL] = 99

    with pytest.raises(ValueError, match="Cannot initialize CausalModel"):
        build_causal_model(df)
