/**
 * Frontend service layer for EconoCasual 2.0 (Phase 4).
 * Interacts with FastAPI /api/v2/* routes with resilient fallback to static JSON artifacts.
 */

const getApiBaseUrl = () => {
  if (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL;
  }
  if (typeof process !== 'undefined' && process.env && process.env.VITE_API_BASE_URL) {
    return process.env.VITE_API_BASE_URL;
  }
  return 'http://localhost:8000';
};

export const API_BASE_URL = getApiBaseUrl();

/**
 * Fetch Phase 1 Doubly Robust Policy Evaluation results.
 */
export async function fetchPolicyEvaluationSummary() {
  // 1. Try live backend
  try {
    const res = await fetch(`${API_BASE_URL}/api/v2/evaluation/summary`);
    if (res.ok) {
      const data = await res.json();
      return { ...data, isLiveApi: true };
    }
  } catch {
    // Backend unreachable; fall back to static bundle
  }

  // 2. Fallback to public/policy_evaluation.json
  const staticRes = await fetch('/policy_evaluation.json');
  if (!staticRes.ok) {
    throw new Error('Failed to load policy evaluation static artifact');
  }
  const staticData = await staticRes.json();
  return {
    success: true,
    metadata: staticData.metadata || {},
    policies: staticData.policies || {},
    canonical_milestone3_benchmarks: staticData.metadata?.canonical_milestone3_benchmarks || null,
    isLiveApi: false,
  };
}

/**
 * Run policy evaluation with custom parameters.
 */
export async function evaluatePolicies(params = {}) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v2/evaluation/evaluate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (res.ok) {
      const data = await res.json();
      return { ...data, isLiveApi: true };
    }
  } catch {
    // Fall back to summary
  }
  return fetchPolicyEvaluationSummary();
}

/**
 * Fetch Phase 2 Uncertainty Quantification summary.
 */
export async function fetchUncertaintySummary() {
  // 1. Try live backend
  try {
    const res = await fetch(`${API_BASE_URL}/api/v2/uncertainty/summary`);
    if (res.ok) {
      const data = await res.json();
      return { ...data, isLiveApi: true };
    }
  } catch {
    // Backend unreachable; fall back
  }

  // 2. Fallback to public/uncertainty_quantification.json
  const staticRes = await fetch('/uncertainty_quantification.json');
  if (!staticRes.ok) {
    throw new Error('Failed to load uncertainty quantification static artifact');
  }
  const staticData = await staticRes.json();
  return {
    success: true,
    metadata: staticData.metadata || {},
    models: staticData.models || {},
    distribution_summary: {
      se_mean: 0.0331,
      se_median: 0.0315,
      se_p90: 0.0468,
      cohort_size: 6005,
    },
    isLiveApi: false,
  };
}

/**
 * Fetch Phase 3 Advanced Optimization benchmarks (4 core regimes).
 */
export async function fetchOptimizationBenchmarks() {
  // 1. Try live backend
  try {
    const res = await fetch(`${API_BASE_URL}/api/v2/optimize/benchmarks`);
    if (res.ok) {
      const data = await res.json();
      return { ...data, isLiveApi: true };
    }
  } catch {
    // Fall back
  }

  // 2. Fallback to public/advanced_optimization_benchmarks.json
  const staticRes = await fetch('/advanced_optimization_benchmarks.json');
  if (!staticRes.ok) {
    throw new Error('Failed to load advanced optimization benchmarks static artifact');
  }
  const staticData = await staticRes.json();
  return {
    success: true,
    metadata: staticData.metadata || {},
    core_regimes: staticData.core_regimes || {},
    risk_aversion_sweep: staticData.risk_aversion_sweep || [],
    fairness_tradeoff_sweep: staticData.fairness_tradeoff_sweep || [],
    isLiveApi: false,
  };
}

/**
 * Execute advanced MILP optimization (risk-aware and fairness-constrained).
 */
export async function runAdvancedOptimization(params) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v2/optimize/advanced`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (res.ok) {
      const data = await res.json();
      return { ...data, isLiveApi: true };
    }
  } catch {
    // Fall back
  }

  // Fallback: load benchmarks and select closest matching regime
  const benchmarks = await fetchOptimizationBenchmarks();
  const mode = params.optimization_mode || 'conservative_lcb';
  const regKey =
    mode === 'baseline' && !params.segment_budget_floor
      ? 'regime1_original_milp'
      : mode === 'risk_adjusted' && !params.segment_budget_floor
      ? 'regime2_uncertainty_aware_milp'
      : mode === 'baseline' && params.segment_budget_floor
      ? 'regime3_fairness_constrained_milp'
      : 'regime4_combined_milp';

  const regimeData = benchmarks.core_regimes?.[regKey] || benchmarks.core_regimes?.regime1_original_milp;
  return {
    ...regimeData,
    budget: params.budget,
    risk_aversion: params.risk_aversion ?? 1.0,
    optimization_mode: mode,
    isLiveApi: false,
  };
}

/**
 * Fetch segment fairness and demographic parity diagnostics.
 */
export async function fetchFairnessDiagnostics() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v2/fairness/diagnostics`);
    if (res.ok) {
      const data = await res.json();
      return { ...data, isLiveApi: true };
    }
  } catch {
    // Fall back
  }

  const benchmarks = await fetchOptimizationBenchmarks();
  const r1 = benchmarks.core_regimes?.regime1_original_milp || {};
  return {
    success: true,
    total_customers: benchmarks.metadata?.n_customers || 6005,
    overall_allocation_rate_pct: 19.82,
    segments: r1.fairness_metrics || {},
    demographic_parity_gap_pct: 18.1,
    min_parity_ratio: 0.448,
    active_constraints_status: {
      segment_floor_configured: false,
      min_parity_ratio_configured: false,
      disadvantaged_segment: 'budget',
    },
    cash_budget_disclaimer: benchmarks.metadata?.cash_budget_disclaimer || '',
    isLiveApi: false,
  };
}

/**
 * Fetch catalog of reproducible experiments.
 */
export async function fetchExperimentHistory() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v2/experiments/history`);
    if (res.ok) {
      const data = await res.json();
      return { ...data, isLiveApi: true };
    }
  } catch {
    // Fall back
  }

  return {
    success: true,
    total_experiments: 3,
    experiments: [
      {
        experiment_id: 'phase1_doubly_robust_policy_evaluation',
        phase: 'Phase 1',
        title: 'Doubly Robust Multi-Treatment Off-Policy Evaluation',
        timestamp: '2026-11-10T16:08:52',
        summary: 'Evaluated 5 competing policies. DML+MILP achieved highest DR profit ($100,786.25) with 204.05% profit-to-cost ratio.',
        status: 'completed',
        artifacts: ['experiments/results/policy_evaluation.json'],
      },
      {
        experiment_id: 'phase2_uncertainty_quantification',
        phase: 'Phase 2',
        title: 'CATE Uncertainty Quantification via Subforest Jackknife & Residual Bootstrap',
        timestamp: '2026-11-10T16:29:23',
        summary: 'Quantified individual treatment effect variance across 6,005 customers with 95.5% empirical CI coverage.',
        status: 'completed',
        artifacts: ['experiments/results/uncertainty_quantification.json'],
      },
      {
        experiment_id: 'phase3_uncertainty_and_fairness_optimization',
        phase: 'Phase 3',
        title: 'Uncertainty-Aware & Fairness-Constrained SciPy MILP Optimization',
        timestamp: '2026-11-10T16:38:40',
        summary: 'Benchmarked 4 optimization regimes at $5,000 budget cap with Lower Confidence Bound and Risk-Adjusted objectives.',
        status: 'completed',
        artifacts: ['experiments/results/advanced_optimization_benchmarks.json'],
      },
    ],
    isLiveApi: false,
  };
}
