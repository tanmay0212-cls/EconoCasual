"""Causal graph specification, DAG metadata, and DoWhy modeling for EconoCausal."""

try:
    from .graph import (
        CAUSAL_DAG_DOT,
        build_causal_model,
        identify_causal_effect,
        get_dag_metadata,
    )
except Exception:
    CAUSAL_DAG_DOT = ""
    build_causal_model = None
    identify_causal_effect = None
    get_dag_metadata = None
try:
    from .dml import (
        EconoCausalDML,
        DEFAULT_TREATMENT_NAMES,
        DEFAULT_OUTCOME_NAMES,
        DEFAULT_CONFOUNDERS,
        PROHIBITED_LEAKAGE_VARS,
        SYNTHETIC_SAMPLE_ATE,
        SYNTHETIC_SPEC_BENCHMARK_ATE,
    )
except Exception:
    EconoCausalDML = None
    DEFAULT_TREATMENT_NAMES = ["treatment"]
    DEFAULT_OUTCOME_NAMES = ["purchase"]
    DEFAULT_CONFOUNDERS = []
    PROHIBITED_LEAKAGE_VARS = []
    SYNTHETIC_SAMPLE_ATE = 0.10
    SYNTHETIC_SPEC_BENCHMARK_ATE = 0.10

try:
    from .interpreter import (
        CustomerEffectInterpreter,
        DEFAULT_NEAR_ZERO_THRESHOLD,
    )
except Exception:
    CustomerEffectInterpreter = None
    DEFAULT_NEAR_ZERO_THRESHOLD = 0.01

try:
    from .evaluation import (
        compute_ranking_table,
        compute_uplift_and_qini_metrics,
        plot_uplift_curve,
        plot_qini_curve,
        plot_effect_distribution,
    )
except Exception:
    compute_ranking_table = None
    compute_uplift_and_qini_metrics = None
    plot_uplift_curve = None
    plot_qini_curve = None
    plot_effect_distribution = None

# Week 3 multi-treatment causal engine
try:
    from .multi_tier_dml import (
        MultiTierCausalDML,
        MULTI_TREATMENT_LEVELS,
        DEFAULT_PRE_TREATMENT_CONFOUNDERS,
    )
except Exception:
    MultiTierCausalDML = None
    MULTI_TREATMENT_LEVELS = [0, 10, 20]
    DEFAULT_PRE_TREATMENT_CONFOUNDERS = []

# Phase 2 Uncertainty-Aware Estimation Engine
from .uncertainty import (
    CATEUncertaintyEstimator,
    TreatmentUncertaintyMetrics,
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
    "MultiTierCausalDML",
    "MULTI_TREATMENT_LEVELS",
    "DEFAULT_PRE_TREATMENT_CONFOUNDERS",
    "CATEUncertaintyEstimator",
    "TreatmentUncertaintyMetrics",
]
