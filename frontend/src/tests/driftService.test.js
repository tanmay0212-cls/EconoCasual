import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { fetchDriftReport, fetchDriftHistory, API_BASE_URL } from '../services/driftService';

describe('driftService API and Fallback Hierarchy', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('connects to canonical live API endpoint (/api/drift/check) when available', async () => {
    const mockLiveReport = {
      overall_status: 'No significant drift detected',
      drift_detected: false,
      total_features_evaluated: 7,
      drifted_features_count: 0,
      severe_drift_count: 0,
      reference_sample_size: 5000,
      current_sample_size: 3000,
      features: [],
      distributions: {
        age: { labels: ['18-25', '25-35'], ref_pct: [20, 30], curr_pct: [22, 28] },
      },
    };

    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => mockLiveReport,
    });

    const result = await fetchDriftReport('clean');

    expect(global.fetch).toHaveBeenCalledWith(
      `${API_BASE_URL}/api/drift/check`,
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_reference: 'drift_reference' }),
      })
    );
    expect(result.isLiveApi).toBe(true);
    expect(result.report).toEqual(mockLiveReport);
    expect(result.distributions.age.labels).toEqual(['18-25', '25-35']);
  });

  it('falls back to legacy route (/drift/check) when /api/drift/check returns network error', async () => {
    const mockLegacyReport = {
      overall_status: 'Potential drift detected',
      drift_detected: true,
      features: [],
    };

    global.fetch = vi.fn()
      .mockRejectedValueOnce(new Error('Failed to fetch /api/drift/check'))
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => mockLegacyReport,
      });

    const result = await fetchDriftReport('shifted');

    expect(global.fetch).toHaveBeenNthCalledWith(1, `${API_BASE_URL}/api/drift/check`, expect.anything());
    expect(global.fetch).toHaveBeenNthCalledWith(2, `${API_BASE_URL}/drift/check`, expect.anything());
    expect(result.isLiveApi).toBe(true);
    expect(result.report.overall_status).toBe('Potential drift detected');
  });

  it('falls back to local static artifact (/drift_data.json) when both live and legacy routes fail', async () => {
    const mockStaticArtifact = {
      metadata: { reference_size: 5000 },
      batches: {
        shifted: {
          report: { overall_status: 'Severe or widespread drift detected', features: [] },
          distributions: { age: { labels: ['18-30'], ref_pct: [50], curr_pct: [70] } },
        },
      },
    };

    global.fetch = vi.fn()
      .mockRejectedValueOnce(new Error('Live API unreachable'))
      .mockRejectedValueOnce(new Error('Legacy API unreachable'))
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => mockStaticArtifact,
      });

    const result = await fetchDriftReport('shifted');

    expect(result.isLiveApi).toBe(false);
    expect(result.report.overall_status).toBe('Severe or widespread drift detected');
    expect(result.distributions.age.ref_pct).toEqual([50]);
  });

  it('throws an error when live API, legacy API, and static artifact all fail', async () => {
    global.fetch = vi.fn()
      .mockRejectedValueOnce(new Error('Live API 500'))
      .mockRejectedValueOnce(new Error('Legacy API 500'))
      .mockRejectedValueOnce(new Error('Static artifact 404'));

    await expect(fetchDriftReport('shifted')).rejects.toThrow(
      /Unable to retrieve drift detection data from either live API or local artifact/i
    );
  });

  it('fetchDriftHistory returns array of history items on 200 OK', async () => {
    const mockHistory = [
      { timestamp: '2026-11-10T12:00:00Z', run_id: 'run-1', overall_status: 'No significant drift detected' },
    ];

    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({ history: mockHistory }),
    });

    const result = await fetchDriftHistory();
    expect(result).toEqual(mockHistory);
  });

  it('fetchDriftHistory returns empty array when history is genuinely empty', async () => {
    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({ history: [] }),
    });

    const result = await fetchDriftHistory();
    expect(result).toEqual([]);
  });

  it('fetchDriftHistory throws when server returns HTTP 500 instead of swallowing error', async () => {
    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
    });

    await expect(fetchDriftHistory()).rejects.toThrow(/HTTP 500/);
  });

  it('handles live report with empty or missing distributions gracefully', async () => {
    const mockLiveReportWithoutDist = {
      overall_status: 'No significant drift detected',
      drift_detected: false,
      features: [],
    };

    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => mockLiveReportWithoutDist,
    });

    const result = await fetchDriftReport('clean');
    expect(result.distributions).toEqual({});
  });
});

