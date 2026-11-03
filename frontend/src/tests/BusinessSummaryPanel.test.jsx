import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import BusinessSummaryPanel from '../components/BusinessSummaryPanel';
import * as optimizationService from '../services/optimizationService';
import * as driftService from '../services/driftService';

const MOCK_OPT_DATA = {
  metadata: {
    run_id: 'run_opt_20261103_073602',
    timestamp: '2026-11-03T13:06:02Z',
    evaluation_population: 3000,
    default_budget: 5000.0,
  },
  kpis: {
    total_customers: 3000,
    count_discount_0: 2162,
    count_discount_10: 836,
    count_discount_20: 2,
    expected_revenue: 121447.78,
    expected_discount_cost: 4604.2,
    expected_net_profit: 53082.35,
    remaining_budget: 395.8,
    budget_utilization_pct: 92.1,
    budget_cap: 5000.0,
  },
  strategies: [
    {
      short_name: 'No Discount',
      discount: '$0',
      profit: 51015.75,
      cost: 0.0,
    },
    {
      short_name: 'Blanket $10',
      cost: 16246.6,
      profit: 47828.63,
    },
  ],
};

const MOCK_DRIFT_DATA = {
  report: {
    overall_status: 'Severe or widespread drift detected',
    total_features_evaluated: 7,
    drifted_features_count: 4,
    high_severity_count: 4,
    summary_message: '4 of 7 features exhibit statistically significant drift.',
    recommended_next_action: 'Acquisition cohort divergence detected. Re-calibrate propensity scores e(X).',
    features: [
      { feature_name: 'age', drift_detected: true },
      { feature_name: 'loyalty_score', drift_detected: true },
      { feature_name: 'customer_segment', drift_detected: true },
      { feature_name: 'campaign_channel', drift_detected: true },
    ],
  },
  metadata: {
    reference_file: 'data/drift_reference.csv',
    reference_size: 5000,
  },
};

describe('BusinessSummaryPanel Component', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('renders title, week badge, run ID, and executive takeaway banner', async () => {
    render(
      <BusinessSummaryPanel
        optimizationData={MOCK_OPT_DATA}
        driftData={MOCK_DRIFT_DATA}
      />
    );

    expect(screen.getByText(/Executive Decision Briefing & Natural-Language Synthesis/i)).toBeDefined();
    expect(screen.getAllByText(/Week 4 Deliverable/i).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('run_opt_20261103_073602')).toBeDefined();
    expect(screen.getByText(/Key Executive Takeaway:/i)).toBeDefined();
    expect(
      screen.getAllByText(/The causal optimizer recommends targeting personalized discounts/i).length
    ).toBeGreaterThanOrEqual(1);
  });

  it('renders all 6 required dashboard summary sections', () => {
    render(
      <BusinessSummaryPanel
        optimizationData={MOCK_OPT_DATA}
        driftData={MOCK_DRIFT_DATA}
      />
    );

    expect(screen.getByText('Optimization Recommendation')).toBeDefined();
    expect(screen.getByText('Budget Utilization Summary')).toBeDefined();
    expect(screen.getByText('Customer Allocation Summary')).toBeDefined();
    expect(screen.getByText('Expected Revenue & Profit Comparison')).toBeDefined();
    expect(screen.getByText('Data Drift & Covariate Shift Surveillance')).toBeDefined();
    expect(screen.getByText('Recommended Operational Next Actions')).toBeDefined();
  });

  it('displays accurate synthesized headlines and business narratives', () => {
    render(
      <BusinessSummaryPanel
        optimizationData={MOCK_OPT_DATA}
        driftData={MOCK_DRIFT_DATA}
      />
    );

    // Optimization Recommendation narrative
    expect(screen.getByText(/Target promotional discounts to 838 customers/i)).toBeDefined();
    expect(screen.getByText(/836 allocated \$10 coupons and 2 allocated \$20 coupons/i)).toBeDefined();

    // Budget Utilization narrative
    expect(screen.getByText(/Allocated \$4,604.20 of the available \$5,000.00 budget/i)).toBeDefined();

    // Customer Allocation narrative
    expect(screen.getByText(/72.1% full price, 27.9% discounted/i)).toBeDefined();

    // Profit lift narrative
    expect(screen.getByText(/Model projects \+\$2,066.60 \(\+4.1%\) profit lift/i)).toBeDefined();

    // Drift warning narrative
    expect(screen.getByText(/Severe covariate shift detected across 4 of 7 customer features/i)).toBeDefined();
  });

  it('allows expanding and inspecting supporting underlying metrics', () => {
    render(
      <BusinessSummaryPanel
        optimizationData={MOCK_OPT_DATA}
        driftData={MOCK_DRIFT_DATA}
      />
    );

    // Look for toggle buttons
    const toggleButtons = screen.getAllByRole('button', { name: /Inspect Supporting Metrics/i });
    expect(toggleButtons.length).toBe(6);

    // Click the 3rd section toggle (Customer Allocation)
    fireEvent.click(toggleButtons[2]);

    // Should reveal underlying customer metrics
    expect(screen.getByText('Total Evaluated')).toBeDefined();
    expect(screen.getByText('3,000 customers')).toBeDefined();
    expect(screen.getByText('$0 Tier (Full Price)')).toBeDefined();
    expect(screen.getByText('$10 Tier (Light Incentive)')).toBeDefined();
  });

  it('handles missing optimization data gracefully without crashing or fabricating numbers', () => {
    render(
      <BusinessSummaryPanel
        optimizationData={null}
        driftData={MOCK_DRIFT_DATA}
      />
    );

    expect(screen.getByText('Optimization metrics are unavailable')).toBeDefined();
    expect(screen.getByText('Budget figures are unavailable')).toBeDefined();
  });

  it('displays the causal accounting methodological notice', () => {
    render(
      <BusinessSummaryPanel
        optimizationData={MOCK_OPT_DATA}
        driftData={MOCK_DRIFT_DATA}
      />
    );

    expect(screen.getByText(/Causal Accounting & Counterfactual Projections Notice/i)).toBeDefined();
    expect(
      screen.getByText(/All revenue, cost, and profit figures are model-predicted counterfactual expectations/i)
    ).toBeDefined();
    expect(screen.getByText(/not guaranteed actual business earnings/i)).toBeDefined();
  });
});
