import { describe, it, expect } from 'vitest';
import {
  generateExecutiveSummary,
  formatCurrency,
  formatNumber,
  formatPct,
} from '../services/summaryService';

describe('summaryService formatting helpers', () => {
  it('formats currency values correctly and handles edge cases', () => {
    expect(formatCurrency(5000)).toBe('$5,000.00');
    expect(formatCurrency(4604.2)).toBe('$4,604.20');
    expect(formatCurrency(0)).toBe('$0.00');
    expect(formatCurrency(null)).toBe('Unavailable');
    expect(formatCurrency(undefined)).toBe('Unavailable');
    expect(formatCurrency('invalid')).toBe('Unavailable');
    expect(formatCurrency(NaN)).toBe('Unavailable');
  });

  it('formats customer numbers correctly and handles non-finite values', () => {
    expect(formatNumber(3000)).toBe('3,000');
    expect(formatNumber(0)).toBe('0');
    expect(formatNumber(null)).toBe('Unavailable');
    expect(formatNumber(undefined)).toBe('Unavailable');
    expect(formatNumber(NaN)).toBe('Unavailable');
  });

  it('formats percentages correctly and handles edge cases', () => {
    expect(formatPct(92.08)).toBe('92.1%');
    expect(formatPct(4.05, 2)).toBe('4.05%');
    expect(formatPct(0)).toBe('0.0%');
    expect(formatPct(null)).toBe('Unavailable');
    expect(formatPct(undefined)).toBe('Unavailable');
  });
});

describe('generateExecutiveSummary - Complete Realistic Inputs', () => {
  const MOCK_OPT_DATA = {
    metadata: {
      run_id: 'run_opt_20261103_073602',
      timestamp: '2026-11-03T13:06:02Z',
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
      features: [
        { feature_name: 'age', drift_detected: true },
        { feature_name: 'loyalty_score', drift_detected: true },
        { feature_name: 'customer_segment', drift_detected: true },
        { feature_name: 'campaign_channel', drift_detected: true },
      ],
    },
  };

  it('generates all 6 required dashboard sections with accurate values', () => {
    const summary = generateExecutiveSummary(MOCK_OPT_DATA, MOCK_DRIFT_DATA);

    expect(summary.runId).toBe('run_opt_20261103_073602');
    expect(summary.timestamp).toBe('2026-11-03T13:06:02Z');
    expect(summary.sections).toHaveLength(6);

    const sectionIds = summary.sections.map((s) => s.id);
    expect(sectionIds).toEqual([
      'optimization_recommendation',
      'budget_utilization',
      'customer_allocation',
      'revenue_profit_comparison',
      'drift_warning',
      'recommended_action',
    ]);
  });

  it('generates accurate optimization recommendation text without inventing numbers', () => {
    const summary = generateExecutiveSummary(MOCK_OPT_DATA, MOCK_DRIFT_DATA);
    const sec = summary.sections.find((s) => s.id === 'optimization_recommendation');

    expect(sec.status).toBe('available');
    expect(sec.headline).toContain('838 customers'); // 836 + 2
    expect(sec.body[0]).toContain('836 allocated $10 coupons');
    expect(sec.body[0]).toContain('2 allocated $20 coupons');
    expect(sec.body[1]).toContain('2,162 customers');
  });

  it('generates accurate budget utilization summary and distinguishes accounting basis', () => {
    const summary = generateExecutiveSummary(MOCK_OPT_DATA, MOCK_DRIFT_DATA);
    const sec = summary.sections.find((s) => s.id === 'budget_utilization');

    expect(sec.status).toBe('available');
    expect(sec.headline).toContain('$4,604.20');
    expect(sec.headline).toContain('$5,000.00');
    expect(sec.headline).toContain('92.1%');
    expect(sec.body[0]).toContain('leaving $395.80 in reserve');
    expect(sec.body[1]).toContain('accrued only when a customer redeems the coupon upon purchasing');
  });

  it('generates accurate profit comparison against baseline and blanket promotion', () => {
    const summary = generateExecutiveSummary(MOCK_OPT_DATA, MOCK_DRIFT_DATA);
    const sec = summary.sections.find((s) => s.id === 'revenue_profit_comparison');

    expect(sec.status).toBe('available');
    expect(sec.headline).toContain('$2,066.60'); // 53082.35 - 51015.75
    expect(sec.body[0]).toContain('higher expected contribution ($53,082.35) than blanket baseline pricing ($51,015.75)');
    expect(sec.body[1]).toContain('blanket discounting ($10 to all customers) would incur $16,246.60');
    expect(sec.body[2]).toContain('not guaranteed actual receipts');
  });

  it('generates data drift warning referencing drifted covariates', () => {
    const summary = generateExecutiveSummary(MOCK_OPT_DATA, MOCK_DRIFT_DATA);
    const sec = summary.sections.find((s) => s.id === 'drift_warning');

    expect(sec.status).toBe('available');
    expect(sec.headline).toContain('Severe covariate shift detected across 4 of 7');
    expect(sec.body[0]).toContain('age, loyalty_score, customer_segment, campaign_channel');
    expect(sec.body[1]).toContain('Covariate shift (P(X)) does not automatically invalidate causal treatment effect');
  });

  it('provides underlying metrics array for every available section', () => {
    const summary = generateExecutiveSummary(MOCK_OPT_DATA, MOCK_DRIFT_DATA);

    summary.sections.forEach((sec) => {
      expect(sec.underlyingMetrics).toBeDefined();
      expect(Array.isArray(sec.underlyingMetrics)).toBe(true);
      if (sec.status === 'available') {
        expect(sec.underlyingMetrics.length).toBeGreaterThanOrEqual(1);
      }
    });
  });
});

describe('generateExecutiveSummary - Edge Cases and Missing Metrics', () => {
  it('handles completely empty inputs gracefully without throwing exceptions', () => {
    const summary = generateExecutiveSummary({}, {});

    expect(summary).toBeDefined();
    expect(summary.hasErrors).toBe(true);
    expect(summary.sections).toHaveLength(6);

    const recSec = summary.sections.find((s) => s.id === 'optimization_recommendation');
    expect(recSec.status).toBe('unavailable');
    expect(recSec.headline).toContain('Optimization metrics are unavailable');

    const driftSec = summary.sections.find((s) => s.id === 'drift_warning');
    expect(driftSec.status).toBe('unavailable');
    expect(driftSec.headline).toContain('Data drift evaluation has not been executed');
  });

  it('handles null and undefined input parameters gracefully', () => {
    const summary = generateExecutiveSummary(null, null);

    expect(summary).toBeDefined();
    expect(summary.sections).toHaveLength(6);
    expect(summary.sections[0].status).toBe('unavailable');
  });

  it('handles zero budget allocation where all customers receive $0', () => {
    const zeroBudgetData = {
      kpis: {
        total_customers: 2500,
        count_discount_0: 2500,
        count_discount_10: 0,
        count_discount_20: 0,
        expected_revenue: 95000.0,
        expected_discount_cost: 0.0,
        expected_net_profit: 38000.0,
        remaining_budget: 0.0,
        budget_utilization_pct: 0.0,
        budget_cap: 0.0,
      },
    };

    const summary = generateExecutiveSummary(zeroBudgetData, {});
    const recSec = summary.sections.find((s) => s.id === 'optimization_recommendation');

    expect(recSec.status).toBe('available');
    expect(recSec.headline).toContain('Preserve full catalog pricing for all evaluated customers');
    expect(recSec.body[0]).toContain('no discounts ($0) for all 2,500 customers');
  });

  it('handles clean in-distribution drift report without false alarms', () => {
    const cleanDrift = {
      report: {
        overall_status: 'No significant drift detected',
        total_features_evaluated: 7,
        drifted_features_count: 0,
        high_severity_count: 0,
        features: [],
      },
    };

    const summary = generateExecutiveSummary({}, cleanDrift);
    const driftSec = summary.sections.find((s) => s.id === 'drift_warning');

    expect(driftSec.status).toBe('available');
    expect(driftSec.badge.type).toBe('success');
    expect(driftSec.headline).toContain('consistent with reference training baseline');
    expect(driftSec.body[0]).toContain('Kolmogorov-Smirnov p > 0.05');
  });

  it('handles inconsistent customer population (NaN / null) without crashing', () => {
    const badKpis = {
      kpis: {
        total_customers: null,
        count_discount_0: 'corrupted',
      },
    };

    const summary = generateExecutiveSummary(badKpis, {});
    const recSec = summary.sections.find((s) => s.id === 'optimization_recommendation');

    expect(recSec.status).toBe('partial');
    expect(recSec.headline).toContain('Customer counts are missing or inconsistent');
  });

  it('strictly includes the methodological notice disclaiming causal certainty and guaranteed revenue', () => {
    const summary = generateExecutiveSummary({}, {});

    expect(summary.methodologicalNotice).toBeDefined();
    expect(summary.methodologicalNotice).toContain('probabilistic projections');
    expect(summary.methodologicalNotice).toContain('not guaranteed actual business earnings');
  });
});

