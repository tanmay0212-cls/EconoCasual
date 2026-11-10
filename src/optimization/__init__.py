"""Optimization module for EconoCausal Week 3.

Provides SciPy-based mixed-integer linear programming (MILP) engines for
budget-constrained personalized discount allocation.
"""

from .budget_optimizer import (
    SciPyBudgetOptimizer,
    DISCOUNT_TIERS,
    DEFAULT_BUDGET,
    OptimizationResult,
)
from .advanced_optimizer import (
    AdvancedMILPOptimizer,
    AdvancedOptimizationResult,
    SegmentFairnessDiagnostic,
)

__all__ = [
    "SciPyBudgetOptimizer",
    "DISCOUNT_TIERS",
    "DEFAULT_BUDGET",
    "OptimizationResult",
    "AdvancedMILPOptimizer",
    "AdvancedOptimizationResult",
    "SegmentFairnessDiagnostic",
]

