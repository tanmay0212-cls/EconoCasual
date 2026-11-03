/**
 * Service to load and manage Automated Data Drift Detection data and reports.
 * Connects directly to backend API (POST /drift/check) or precomputed static artifacts (/drift_data.json).
 */

const API_BASE_URL = 'http://localhost:8000';

/**
 * Fetch drift detection payload.
 * @param {string} batchType - 'shifted' | 'clean'
 * @returns {Promise<{ report: Object, distributions: Object, metadata: Object, isLiveApi: boolean }>}
 */
export async function fetchDriftReport(batchType = 'shifted') {
  // First attempt local precomputed static JSON for zero-latency instant UX
  try {
    const res = await fetch('/drift_data.json');
    if (res.ok) {
      const data = await res.json();
      const batchData = data.batches[batchType] || data.batches.shifted;
      return {
        report: batchData.report,
        distributions: batchData.distributions,
        metadata: data.metadata,
        isLiveApi: false,
      };
    }
  } catch (err) {
    console.warn('Could not load static drift_data.json, attempting live API fallback:', err);
  }

  // Live API fallback if static file not reachable
  try {
    const datasetRef = batchType === 'clean' ? 'drift_reference' : 'drift_current';
    const response = await fetch(`${API_BASE_URL}/drift/check`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset_reference: datasetRef }),
    });

    if (response.ok) {
      const liveReport = await response.json();
      return {
        report: liveReport,
        distributions: {},
        metadata: {
          reference_file: 'drift_reference',
          current_file: datasetRef,
        },
        isLiveApi: true,
      };
    }
  } catch (apiErr) {
    console.error('Failed to query live drift API:', apiErr);
  }

  throw new Error('Unable to retrieve drift detection data from either local artifact or live API.');
}

