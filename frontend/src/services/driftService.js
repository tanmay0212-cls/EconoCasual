/**
 * Service to load and manage Automated Data Drift Detection data and reports.
 * Connects directly to backend API (POST /api/drift/check) or precomputed static artifacts (/drift_data.json).
 */

const API_BASE_URL = 'http://localhost:8000';

/**
 * Fetch drift detection payload.
 * @param {string} batchType - 'shifted' | 'clean'
 * @returns {Promise<{ report: Object, distributions: Object, metadata: Object, isLiveApi: boolean }>}
 */
export async function fetchDriftReport(batchType = 'shifted') {
  const datasetRef = batchType === 'clean' ? 'drift_reference' : 'drift_current';

  // 1. Try live backend API first (/api/drift/check)
  try {
    const response = await fetch(`${API_BASE_URL}/api/drift/check`, {
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
    // Backend offline or unreachable; fall back to static or legacy endpoint
  }

  // 2. Try legacy route fallback (/drift/check)
  try {
    const legacyRes = await fetch(`${API_BASE_URL}/drift/check`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset_reference: datasetRef }),
    });
    if (legacyRes.ok) {
      const liveReport = await legacyRes.json();
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
  } catch {}

  // 3. Fall back to local precomputed static JSON for instant UX
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
    console.warn('Could not load static drift_data.json:', err);
  }

  throw new Error('Unable to retrieve drift detection data from either live API or local artifact.');
}

/**
 * Fetch historical drift surveillance check runs.
 * @returns {Promise<Array>}
 */
export async function fetchDriftHistory() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/drift/history`);
    if (response.ok) {
      const data = await response.json();
      return data.history || [];
    }
  } catch (err) {
    console.warn('Could not fetch drift history from live API:', err);
  }
  return [];
}


