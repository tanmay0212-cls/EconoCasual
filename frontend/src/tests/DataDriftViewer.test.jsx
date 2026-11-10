import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import DataDriftViewer from '../components/DataDriftViewer';
import * as driftService from '../services/driftService';

const MOCK_SHIFTED_PAYLOAD = {
  report: {
    overall_status: 'Severe or widespread drift detected',
    drifted_features_count: 4,
    total_features_evaluated: 7,
    high_severity_count: 3,
    reference_sample_size: 5000,
    current_sample_size: 3500,
    summary_message: '4 of 7 features exhibit statistically significant drift (3 high severity).',
    recommended_next_action: 'Acquisition cohort divergence detected. Re-calibrate propensity scores e(X). Do not automatically retrain causal forest.',
    methodological_notice: 'Methodological Notice: Covariate drift (P(X)) measures shift in customer features. It does NOT automatically imply concept drift (P(Y | X, T)) or invalidity of causal treatment effects.',
    schema_validation: {
      is_valid: true,
      errors: [],
      warnings: [],
    },
    features: [
      {
        feature_name: 'age',
        feature_type: 'numeric',
        drift_detected: true,
        severity: 'high',
        test_name: 'kolmogorov_smirnov',
        test_statistic: 0.185,
        p_value: 0.000001,
        psi: 0.284,
        wasserstein_distance: 7.2,
        reference_summary: { mean: 42.1, std: 11.2, median: 41.0, q25: 33.0, q75: 50.0 },
        current_summary: { mean: 49.3, std: 11.5, median: 49.0, q25: 41.0, q75: 58.0 },
        recommended_action: 'Significant age divergence; older customer demographic in current batch.',
      },
      {
        feature_name: 'customer_segment',
        feature_type: 'categorical',
        drift_detected: true,
        severity: 'high',
        test_name: 'chi_square',
        test_statistic: 142.5,
        p_value: 0.00001,
        tvd: 0.22,
        reference_summary: { proportions: { budget: 0.35, premium: 0.25, regular: 0.40 } },
        current_summary: { proportions: { budget: 0.20, premium: 0.45, regular: 0.35 } },
        new_categories: [],
        missing_categories: [],
        recommended_action: 'Segment distribution shifted significantly toward premium users.',
      },
      {
        feature_name: 'product_price',
        feature_type: 'numeric',
        drift_detected: false,
        severity: 'none',
        test_name: 'kolmogorov_smirnov',
        test_statistic: 0.012,
        p_value: 0.852,
        psi: 0.015,
        wasserstein_distance: 0.3,
        reference_summary: { mean: 85.0, std: 15.0, median: 84.5, q25: 75.0, q75: 95.0 },
        current_summary: { mean: 85.1, std: 14.9, median: 84.8, q25: 75.2, q75: 95.1 },
        recommended_action: 'Baseline product catalog prices remain stable.',
      },
    ],
  },
  distributions: {
    age: {
      labels: ['18-27', '27-36', '36-45', '45-54', '54-63', '63-72'],
      ref_pct: [12.5, 24.2, 31.0, 20.1, 9.2, 3.0],
      curr_pct: [4.1, 11.2, 23.5, 34.0, 20.2, 7.0],
    },
    customer_segment: {
      labels: ['budget', 'premium', 'regular'],
      ref_pct: [35.0, 25.0, 40.0],
      curr_pct: [20.0, 45.0, 35.0],
    },
  },
  metadata: {
    reference_file: 'data/drift_reference.csv',
    reference_size: 5000,
  },
  isLiveApi: false,
};

const MOCK_CLEAN_PAYLOAD = {
  report: {
    overall_status: 'No significant drift detected',
    drifted_features_count: 0,
    total_features_evaluated: 7,
    high_severity_count: 0,
    reference_sample_size: 5000,
    current_sample_size: 2500,
    summary_message: 'All 7 features are statistically consistent with the reference dataset.',
    recommended_next_action: 'Current customer distributions match training distribution. Proceed with production scoring.',
    methodological_notice: 'Methodological Notice: Covariate drift (P(X)) measures shift in customer features. It does NOT automatically imply concept drift (P(Y | X, T)) or invalidity of causal treatment effects.',
    schema_validation: {
      is_valid: true,
      errors: [],
      warnings: [],
    },
    features: [
      {
        feature_name: 'age',
        feature_type: 'numeric',
        drift_detected: false,
        severity: 'none',
        test_name: 'kolmogorov_smirnov',
        test_statistic: 0.015,
        p_value: 0.72,
        psi: 0.012,
        reference_summary: { mean: 42.1, std: 11.2 },
        current_summary: { mean: 42.0, std: 11.1 },
        recommended_action: 'Feature is stable.',
      },
    ],
  },
  distributions: {
    age: {
      labels: ['18-27', '27-36', '36-45', '45-54', '54-63', '63-72'],
      ref_pct: [12.5, 24.2, 31.0, 20.1, 9.2, 3.0],
      curr_pct: [12.6, 24.1, 30.9, 20.2, 9.1, 3.1],
    },
  },
  metadata: { reference_size: 5000 },
  isLiveApi: false,
};

describe('DataDriftViewer Component', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(driftService, 'fetchDriftHistory').mockResolvedValue([]);
  });

  it('renders title, week badge, and overall drift status banner', async () => {
    vi.spyOn(driftService, 'fetchDriftReport').mockResolvedValue(MOCK_SHIFTED_PAYLOAD);

    render(<DataDriftViewer />);

    expect(screen.getByText(/Automated Data Drift & Covariate Shift Monitoring/i)).toBeDefined();
    expect(screen.getByText(/Week 4 Deliverable/i)).toBeDefined();

    await waitFor(() => {
      expect(screen.getByText('Severe or widespread drift detected')).toBeDefined();
      expect(screen.getByText(/Severe Drift Detected/i)).toBeDefined();
    });

    expect(screen.getAllByText(/Re-calibrate propensity scores e\(X\)/i).length).toBeGreaterThanOrEqual(1);
  });

  it('displays summary KPI cards with correct metric values', async () => {
    vi.spyOn(driftService, 'fetchDriftReport').mockResolvedValue(MOCK_SHIFTED_PAYLOAD);

    render(<DataDriftViewer />);

    await waitFor(() => {
      expect(screen.getByText('Features Evaluated')).toBeDefined();
    });

    // Evaluated features
    expect(screen.getByText('7')).toBeDefined();
    // Drifted features: 4 / 7
    expect(screen.getByText('4')).toBeDefined();
    expect(screen.getByText('/ 7')).toBeDefined();
    // High severity features
    expect(screen.getByText('3')).toBeDefined();
    // Baseline sample size
    expect(screen.getByText('5,000')).toBeDefined();
    // Evaluation batch size
    expect(screen.getByText('3,500')).toBeDefined();
    // Schema validity
    expect(screen.getByText('100% Valid')).toBeDefined();
  });

  it('renders feature breakdown table with KS/Chi-Square metrics and severity tags', async () => {
    vi.spyOn(driftService, 'fetchDriftReport').mockResolvedValue(MOCK_SHIFTED_PAYLOAD);

    render(<DataDriftViewer />);

    await waitFor(() => {
      expect(screen.getByText('Feature-by-Feature Statistical Drift Breakdown')).toBeDefined();
    });

    // Check feature rows
    expect(screen.getAllByText('customer_segment').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('product_price').length).toBeGreaterThanOrEqual(1);

    // Check statistical tags
    const ksTags = screen.getAllByText('KS 2-Sample');
    expect(ksTags.length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('Chi-Square')).toBeDefined();

    // High severity badges
    const highBadges = screen.getAllByText('High Severity');
    expect(highBadges.length).toBeGreaterThanOrEqual(1);

    // Stable badge
    expect(screen.getByText('Stable')).toBeDefined();
  });

  it('switches evaluation cohort when clicking batch selector buttons', async () => {
    const fetchSpy = vi.spyOn(driftService, 'fetchDriftReport')
      .mockResolvedValueOnce(MOCK_SHIFTED_PAYLOAD)
      .mockResolvedValueOnce(MOCK_CLEAN_PAYLOAD);

    render(<DataDriftViewer />);

    await waitFor(() => {
      expect(screen.getByText('Severe or widespread drift detected')).toBeDefined();
    });

    // Click "In-Distribution (Clean)"
    const cleanBtn = screen.getByRole('button', { name: /In-Distribution \(Clean\)/i });
    fireEvent.click(cleanBtn);

    await waitFor(() => {
      expect(fetchSpy).toHaveBeenCalledWith('clean');
      expect(screen.getByText('No significant drift detected')).toBeDefined();
      expect(screen.getByText('All features stable')).toBeDefined();
    });
  });

  it('inspecting a feature updates the distribution comparison visualizer', async () => {
    vi.spyOn(driftService, 'fetchDriftReport').mockResolvedValue(MOCK_SHIFTED_PAYLOAD);

    render(<DataDriftViewer />);

    await waitFor(() => {
      expect(screen.getByText(/Empirical Distribution Comparison:/i)).toBeDefined();
    });

    // Click customer_segment quick select button
    const segmentBtns = screen.getAllByRole('button', { name: /customer_segment/i });
    fireEvent.click(segmentBtns[0]);

    await waitFor(() => {
      // Should show segment comparison bars/categories
      expect(screen.getByText('budget')).toBeDefined();
      expect(screen.getByText('premium')).toBeDefined();
    });
  });

  it('displays the causal methodological guidance explaining covariate shift vs concept drift', async () => {
    vi.spyOn(driftService, 'fetchDriftReport').mockResolvedValue(MOCK_SHIFTED_PAYLOAD);

    render(<DataDriftViewer />);

    await waitFor(() => {
      expect(screen.getByText(/Causal Methodological Guidance: Covariate Shift vs Concept Drift/i)).toBeDefined();
    });

    expect(screen.getByText(/Covariate Shift \(P\(X\) Drift\):/i)).toBeDefined();
    expect(screen.getByText(/Concept Drift \(P\(Y \| X, T\) Shift\):/i)).toBeDefined();
    expect(screen.getByText(/Do not automatically trigger unmonitored model retraining/i)).toBeDefined();
  });

  it('renders explicit unavailable notice when distributions are missing without fabricating charts', async () => {
    const payloadNoDist = {
      ...MOCK_SHIFTED_PAYLOAD,
      distributions: {},
    };
    vi.spyOn(driftService, 'fetchDriftReport').mockResolvedValue(payloadNoDist);

    render(<DataDriftViewer />);

    await waitFor(() => {
      expect(screen.getByText(/Distribution comparison histogram for/i)).toBeDefined();
      expect(screen.getByText(/is currently unavailable/i)).toBeDefined();
    });
  });

  it('renders explicit error banner when drift report API call fails', async () => {
    vi.spyOn(driftService, 'fetchDriftReport').mockRejectedValue(new Error('Network error: server unreachable'));

    render(<DataDriftViewer />);

    await waitFor(() => {
      expect(screen.getByText(/Error loading drift report:/i)).toBeDefined();
      expect(screen.getByText(/Network error: server unreachable/i)).toBeDefined();
    });
  });

  it('renders empty surveillance history notice when history is empty', async () => {
    vi.spyOn(driftService, 'fetchDriftReport').mockResolvedValue(MOCK_SHIFTED_PAYLOAD);
    vi.spyOn(driftService, 'fetchDriftHistory').mockResolvedValue([]);

    render(<DataDriftViewer />);

    await waitFor(() => {
      expect(screen.getByText(/No historical surveillance runs recorded yet/i)).toBeDefined();
    });
  });

  it('renders warning banner when surveillance history API request fails', async () => {
    vi.spyOn(driftService, 'fetchDriftReport').mockResolvedValue(MOCK_SHIFTED_PAYLOAD);
    vi.spyOn(driftService, 'fetchDriftHistory').mockRejectedValue(new Error('HTTP 500: Database unavailable'));

    render(<DataDriftViewer />);

    await waitFor(() => {
      expect(screen.getByText(/Surveillance History Unavailable:/i)).toBeDefined();
      expect(screen.getByText(/HTTP 500: Database unavailable/i)).toBeDefined();
    });
  });
});
