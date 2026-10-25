"""Causal graph specification, DAG metadata, and DoWhy modeling for EconoCausal."""

from .graph import (
    CAUSAL_DAG_DOT,
    build_causal_model,
    identify_causal_effect,
    get_dag_metadata,
)
from .dml import (
    EconoCausalDML,
    DEFAULT_TREATMENT_NAMES,
    DEFAULT_OUTCOME_NAMES,
    DEFAULT_CONFOUNDERS,
    PROHIBITED_LEAKAGE_VARS,
    SYNTHETIC_SAMPLE_ATE,
    SYNTHETIC_SPEC_BENCHMARK_ATE,
)
from .interpreter import (
    CustomerEffectInterpreter,
    DEFAULT_NEAR_ZERO_THRESHOLD,
)
from .evaluation import (
    compute_ranking_table,
    compute_uplift_and_qini_metrics,
    plot_uplift_curve,
    plot_qini_curve,
    plot_effect_distribution,
)

__all__ = [
    "CAUSAL_DAG_DOT",
    "build_causal_model",
    "identify_causal_effect",
    "get_dag_metadata",
    "EconoCausalDML",
    "DEFAULT_TREATMENT_NAMES",
    "DEFAULT_OUTCOME_NAMES",
    "DEFAULT_CONFOUNDERS",
    "PROHIBITED_LEAKAGE_VARS",
    "SYNTHETIC_SAMPLE_ATE",
    "SYNTHETIC_SPEC_BENCHMARK_ATE",
    "CustomerEffectInterpreter",
    "DEFAULT_NEAR_ZERO_THRESHOLD",
    "compute_ranking_table",
    "compute_uplift_and_qini_metrics",
    "plot_uplift_curve",
    "plot_qini_curve",
    "plot_effect_distribution",
]
