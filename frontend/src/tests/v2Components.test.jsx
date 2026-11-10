import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import PolicyEvaluationTab from '../components/PolicyEvaluationTab';
import UncertaintyAnalysisTab from '../components/UncertaintyAnalysisTab';
import FairnessAnalysisTab from '../components/FairnessAnalysisTab';
import AdvancedOptimizationTab from '../components/AdvancedOptimizationTab';
import * as v2Service from '../services/v2Service';

const MOCK_POLICY_EVAL = {
  success: true,
  isLiveApi: true,
  metadata: {
    evaluation_sample_size: 6005,
    min_propensity_clip: 0.01,
    canonical_milestone3_benchmarks: {
      dml_milp_expected_profit: 100786.25,
      dml_milp_incremental_profit: 10199.87,
      dml_milp_incremental_profit_to_cost_pct: 204.05,
    },
  },
  policies: {
    no_discount: {
      policy_name: 'No-Discount Policy',
      status: 'evaluated',
      is_available: true,
      description: 'Blanket $0 promotion to all customers (organic baseline).',
      economic_metrics: {
        dr_expected_profit: 92631.49,
        dr_profit_ci_95: [85209.55, 99935.40],
        incremental_profit_vs_no_discount: 0.0,
        expected_promotional_cost: 0.0,
        incremental_profit_to_cost_ratio: null,
      },
      conversion_metrics: {
        dr_conversion_rate: 0.2916,
      },
      diagnostics: {
        effective_sample_size: 1825.55,
      },
    },
    flat_10: {
      policy_name: 'Flat $10 Discount Policy',
      status: 'evaluated',
      is_available: true,
      description: 'Blanket $10 promotion to all customers regardless of individual elasticity.',
      economic_metrics: {
        dr_expected_profit: 113754.02,
        dr_profit_ci_95: [107507.63, 119579.30],
        incremental_profit_vs_no_discount: 21122.53,
        expected_promotional_cost: 29149.90,
        incremental_profit_to_cost_ratio: 0.7246,
      },
      conversion_metrics: {
        dr_conversion_rate: 0.4790,
      },
      diagnostics: {
        effective_sample_size: 1729.43,
      },
    },
    random_discount: {
      policy_name: 'Random Discount Policy',
      status: 'evaluated',
      is_available: true,
      description: 'Stochastic exploration assigning equal probability (1/3) across $0, $10, $20.',
      economic_metrics: {
        dr_expected_profit: 100412.30,
        dr_profit_ci_95: [94800.00, 106100.00],
        incremental_profit_vs_no_discount: 7780.81,
        expected_promotional_cost: 28500.00,
        incremental_profit_to_cost_ratio: 0.273,
      },
      conversion_metrics: {
        dr_conversion_rate: 0.4410,
      },
      diagnostics: {
        effective_sample_size: 1780.0,
      },
    },
    dml_milp: {
      policy_name: 'DML + MILP Prescriptive Policy',
      status: 'evaluated',
      is_available: true,
      description: 'Optimal budget-constrained allocation ($5,000 budget cap) via HiGHS MILP.',
      economic_metrics: {
        dr_expected_profit: 100786.25,
        dr_profit_ci_95: [98900.00, 102600.00],
        incremental_profit_vs_no_discount: 10199.87,
        expected_promotional_cost: 4998.74,
        incremental_profit_to_cost_ratio: 2.0405,
      },
      conversion_metrics: {
        dr_conversion_rate: 0.3842,
      },
      diagnostics: {
        effective_sample_size: 1980.5,
      },
    },
    uncertainty_aware_placeholder: {
      policy_name: 'Uncertainty-Aware Causal Policy (Phase 5 Placeholder)',
      status: 'unavailable',
      is_available: false,
      description: 'Reserved for future Lower-Confidence-Bound (LCB) policy deployment.',
      economic_metrics: null,
      conversion_metrics: null,
      diagnostics: null,
    },
  },
};

const MOCK_UNCERTAINTY = {
  success: true,
  isLiveApi: true,
  metadata: {
    n_samples: 6005,
    confidence_level: 0.95,
  },
  models: {
    asymptotic_subforest_model: {
      description: 'Asymptotic / Subforest Infinitesimal Jackknife heteroskedastic variance estimator',
      contrasts: {
        '10_vs_0': {
          cate_mae: 0.0252,
          cate_rmse: 0.0332,
          ate_bias: 0.0033,
          coverage_rate_pct: 95.52,
          mean_interval_width: 0.1299,
          reliability_counts: {
            reliable: 5104,
            moderate_uncertainty: 600,
            high_variance: 301,
          },
          segment_diagnostics: {
            budget: { coverage_rate_pct: 96.83, mean_interval_width: 0.1110 },
            regular: { coverage_rate_pct: 95.58, mean_interval_width: 0.1306 },
            premium: { coverage_rate_pct: 94.17, mean_interval_width: 0.1437 },
            vip: { coverage_rate_pct: 93.85, mean_interval_width: 0.1554 },
          },
        },
        '20_vs_0': {
          cate_mae: 0.0381,
          cate_rmse: 0.0479,
          ate_bias: 0.0003,
          coverage_rate_pct: 95.34,
          mean_interval_width: 0.1654,
          reliability_counts: {
            reliable: 4950,
            moderate_uncertainty: 720,
            high_variance: 335,
          },
          segment_diagnostics: {},
        },
      },
    },
  },
};

const MOCK_FAIRNESS = {
  success: true,
  isLiveApi: true,
  total_customers: 6005,
  overall_allocation_rate_pct: 19.82,
  demographic_parity_gap_pct: 18.1,
  min_parity_ratio: 0.448,
  active_constraints_status: {
    disadvantaged_segment: 'budget',
  },
  segments: {
    budget: {
      segment_name: 'budget',
      sample_size: 1769,
      allocation_count: 157,
      allocation_rate_pct: 8.88,
      average_discount_amount: 0.89,
      mean_incremental_profit: 0.51,
      parity_ratio_vs_overall: 0.448,
    },
    regular: {
      segment_name: 'regular',
      sample_size: 2378,
      allocation_count: 477,
      allocation_rate_pct: 20.06,
      average_discount_amount: 2.01,
      mean_incremental_profit: 1.87,
      parity_ratio_vs_overall: 1.012,
    },
    premium: {
      segment_name: 'premium',
      sample_size: 1253,
      allocation_count: 338,
      allocation_rate_pct: 26.98,
      average_discount_amount: 2.71,
      mean_incremental_profit: 2.53,
      parity_ratio_vs_overall: 1.361,
    },
    vip: {
      segment_name: 'vip',
      sample_size: 605,
      allocation_count: 218,
      allocation_rate_pct: 36.03,
      average_discount_amount: 3.65,
      mean_incremental_profit: 3.12,
      parity_ratio_vs_overall: 1.818,
    },
  },
};

const MOCK_BENCHMARKS = {
  success: true,
  metadata: {
    n_customers: 6005,
    budget: 5000.0,
    cash_budget_disclaimer: 'Expected promotional cost is an ex-ante mathematical expectation.',
  },
  core_regimes: {
    regime1_original_milp: {
      financial_kpis: {
        nominal_expected_profit: 100636.58,
        conservative_lcb_profit: 95741.92,
        total_expected_cost: 4998.79,
      },
      tier_counts: { 0: 4815, 10: 1170, 20: 20 },
      fairness_metrics: MOCK_FAIRNESS.segments,
    },
    regime2_uncertainty_aware_milp: {
      financial_kpis: {
        nominal_expected_profit: 100502.62,
        conservative_lcb_profit: 96128.31,
        total_expected_cost: 4999.93,
        budget_utilization_pct: 99.99,
      },
      tier_counts: { 0: 4783, 10: 1190, 20: 32 },
      uncertainty_kpis: { selected_customers_mean_uncertainty: 0.0334 },
      fairness_metrics: MOCK_FAIRNESS.segments,
      status_message: 'Optimization terminated successfully (HiGHS Optimal).',
      runtime_seconds: 0.442,
    },
    regime3_fairness_constrained_milp: {
      financial_kpis: {
        nominal_expected_profit: 100588.77,
        conservative_lcb_profit: 95854.60,
        total_expected_cost: 4997.37,
      },
      tier_counts: { 0: 4783, 10: 1205, 20: 17 },
      fairness_metrics: MOCK_FAIRNESS.segments,
    },
    regime4_combined_milp: {
      financial_kpis: {
        nominal_expected_profit: 100481.63,
        conservative_lcb_profit: 96137.18,
        total_expected_cost: 4997.19,
      },
      tier_counts: { 0: 4776, 10: 1198, 20: 31 },
      fairness_metrics: MOCK_FAIRNESS.segments,
    },
  },
};

describe('EconoCasual 2.0 (Phase 4) Frontend Components', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(v2Service, 'fetchPolicyEvaluationSummary').mockResolvedValue(MOCK_POLICY_EVAL);
    vi.spyOn(v2Service, 'fetchUncertaintySummary').mockResolvedValue(MOCK_UNCERTAINTY);
    vi.spyOn(v2Service, 'fetchFairnessDiagnostics').mockResolvedValue(MOCK_FAIRNESS);
    vi.spyOn(v2Service, 'fetchOptimizationBenchmarks').mockResolvedValue(MOCK_BENCHMARKS);
    vi.spyOn(v2Service, 'runAdvancedOptimization').mockResolvedValue(MOCK_BENCHMARKS.core_regimes.regime2_uncertainty_aware_milp);
  });

  it('renders PolicyEvaluationTab with all 5 policies and highlights DML', async () => {
    render(<PolicyEvaluationTab />);
    expect(screen.getByText(/Loading Doubly Robust Policy Evaluation results/i)).toBeDefined();

    await waitFor(() => {
      expect(screen.getByText(/Doubly Robust Off-Policy Evaluation/i)).toBeDefined();
    });

    expect(screen.getAllByText('No-Discount Policy').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Flat $10 Discount Policy').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Random Discount Policy').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('DML + MILP Prescriptive Policy').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/Uncertainty-Aware Causal Policy/i).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/Unavailable \(Future\)/i)).toBeDefined();
  });

  it('renders UncertaintyAnalysisTab with empirical coverage and reliability breakdown', async () => {
    render(<UncertaintyAnalysisTab />);
    expect(screen.getByText(/Loading CATE Uncertainty Quantification results/i)).toBeDefined();

    await waitFor(() => {
      expect(screen.getByText(/CATE Uncertainty Quantification & Reliability Analysis/i)).toBeDefined();
    });

    expect(screen.getByText('95.52%')).toBeDefined();
    expect(screen.getByText('0.1299')).toBeDefined();
    expect(screen.getByText(/Customer Estimate Reliability Breakdown/i)).toBeDefined();
    expect(screen.getByText(/Segment-Level CI Coverage & Width Breakdown/i)).toBeDefined();
  });

  it('renders FairnessAnalysisTab with customer tiers and parity ratios', async () => {
    render(<FairnessAnalysisTab />);
    expect(screen.getByText(/Loading Customer Segment Fairness Diagnostics/i)).toBeDefined();

    await waitFor(() => {
      expect(screen.getByText(/Customer Segment Fairness & Demographic Parity Diagnostics/i)).toBeDefined();
    });

    expect(screen.getAllByText('18.1%').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('0.448').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/Disadvantaged tier: Budget segment/i)).toBeDefined();
    expect(screen.getByText(/Segment-Level Allocation & Benefit Gap Matrix/i)).toBeDefined();
  });

  it('renders AdvancedOptimizationTab and runs advanced MILP solve', async () => {
    render(<AdvancedOptimizationTab />);

    await waitFor(() => {
      expect(screen.getByText(/Uncertainty-Aware & Fairness-Constrained SciPy MILP Optimizer/i)).toBeDefined();
    });

    expect(screen.getByText(/Budget & Objective Specification/i)).toBeDefined();
    expect(screen.getByText(/Business Fairness Constraints/i)).toBeDefined();

    const solveBtn = screen.getByRole('button', { name: /Run Advanced MILP Optimizer/i });
    expect(solveBtn).toBeDefined();

    fireEvent.click(solveBtn);

    await waitFor(() => {
      expect(v2Service.runAdvancedOptimization).toHaveBeenCalled();
    });
  });
});
