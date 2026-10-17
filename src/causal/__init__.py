"""Causal graph specification, DAG metadata, and DoWhy modeling for EconoCausal."""

from .graph import (
    CAUSAL_DAG_DOT,
    build_causal_model,
    identify_causal_effect,
    get_dag_metadata,
)

__all__ = [
    "CAUSAL_DAG_DOT",
    "build_causal_model",
    "identify_causal_effect",
    "get_dag_metadata",
]
