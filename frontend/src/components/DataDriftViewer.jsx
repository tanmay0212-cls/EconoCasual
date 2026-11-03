import React, { useState, useEffect, useMemo } from 'react';
import {
  ShieldAlert,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  BarChart3,
  RefreshCw,
  Info,
  Layers,
  ArrowRight,
  TrendingDown,
  TrendingUp,
  Activity,
  FileCheck2,
  Sliders,
} from 'lucide-react';
import { fetchDriftReport, fetchDriftHistory } from '../services/driftService';

export default function DataDriftViewer() {
  const [batchType, setBatchType] = useState('shifted'); // 'shifted' | 'clean'
  const [dataPayload, setDataPayload] = useState(null);
  const [driftHistory, setDriftHistory] = useState([]);
  const [selectedFeature, setSelectedFeature] = useState('age');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Load drift report data
  const loadReport = async (type) => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await fetchDriftReport(type);
      setDataPayload(data);
      if (data.report?.features?.length > 0) {
        // Default to a drifted feature if available
        const firstDrifted = data.report.features.find((f) => f.drift_detected);
        if (firstDrifted) {
          setSelectedFeature(firstDrifted.feature_name);
        } else {
          setSelectedFeature(data.report.features[0].feature_name);
        }
      }
      // Load historical runs in parallel
      try {
        const history = await fetchDriftHistory();
        setDriftHistory(history);
      } catch (hErr) {
        console.warn('Could not load drift history:', hErr);
      }
    } catch (err) {
      setError(err.message || 'Failed to load drift report.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadReport(batchType);
  }, [batchType]);

  const report = dataPayload?.report;
  const distributions = dataPayload?.distributions || {};
  const metadata = dataPayload?.metadata;

  // Selected feature details
  const activeFeatureResult = useMemo(() => {
    if (!report?.features) return null;
    return report.features.find((f) => f.feature_name === selectedFeature) || report.features[0];
  }, [report, selectedFeature]);

  // Selected distribution data
  const activeDistData = useMemo(() => {
    if (!distributions || !selectedFeature) return null;
    return distributions[selectedFeature] || null;
  }, [distributions, selectedFeature]);

  // Overall status helper
  const getStatusBadge = (status) => {
    if (status === 'Severe or widespread drift detected') {
      return {
        label: 'Severe Drift Detected',
        className: 'badge-danger',
        icon: <ShieldAlert size={16} className="text-red" />,
        bannerClass: 'drift-banner-danger',
      };
    }
    if (status === 'Potential drift detected') {
      return {
        label: 'Potential Drift Detected',
        className: 'badge-warning',
        icon: <AlertTriangle size={16} className="text-amber" />,
        bannerClass: 'drift-banner-warning',
      };
    }
    return {
      label: 'No Significant Drift',
      className: 'badge-success',
      icon: <CheckCircle2 size={16} className="text-emerald" />,
      bannerClass: 'drift-banner-success',
    };
  };

  const getSeverityBadge = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'high':
        return <span className="badge badge-danger">High Severity</span>;
      case 'moderate':
        return <span className="badge badge-warning">Moderate</span>;
      default:
        return <span className="badge badge-success">Stable</span>;
    }
  };

  const statusInfo = getStatusBadge(report?.overall_status);

  return (
    <div className="drift-viewer-container">
      {/* 1. Header & Batch Selector Controls */}
      <div className="card drift-header-card">
        <div className="card-header-flex">
          <div className="title-with-icon">
            <div className="icon-badge icon-cyan">
              <Activity size={22} />
            </div>
            <div>
              <div className="drift-title-row">
                <h2 className="card-title">Automated Data Drift & Covariate Shift Monitoring</h2>
                <span className="badge badge-primary">Week 4 Deliverable</span>
              </div>
              <p className="card-subtitle">
                Continuous statistical distribution tracking comparing live customer cohorts against baseline reference data.
              </p>
            </div>
          </div>

          {/* Batch Selector Toggle */}
          <div className="batch-toggle-group" role="group" aria-label="Select evaluation batch">
            <span className="toggle-group-label">Evaluation Cohort:</span>
            <button
              type="button"
              className={`btn btn-toggle ${batchType === 'shifted' ? 'btn-toggle-active' : ''}`}
              onClick={() => setBatchType('shifted')}
              disabled={isLoading}
            >
              <ShieldAlert size={15} />
              <span>Future Batch (Shifted)</span>
            </button>
            <button
              type="button"
              className={`btn btn-toggle ${batchType === 'clean' ? 'btn-toggle-active' : ''}`}
              onClick={() => setBatchType('clean')}
              disabled={isLoading}
            >
              <CheckCircle2 size={15} />
              <span>In-Distribution (Clean)</span>
            </button>
            <button
              type="button"
              className="btn btn-secondary btn-icon-only"
              onClick={() => loadReport(batchType)}
              title="Refresh Drift Analysis"
              disabled={isLoading}
            >
              <RefreshCw size={15} className={isLoading ? 'spin-icon' : ''} />
            </button>
          </div>
        </div>

        {/* 2. Overall Status Banner */}
        {report && (
          <div className={`drift-status-banner ${statusInfo.bannerClass}`}>
            <div className="status-banner-main">
              <div className="status-icon-wrap">{statusInfo.icon}</div>
              <div className="status-banner-text">
                <div className="status-banner-header">
                  <h3 className="status-title">{report.overall_status}</h3>
                  <span className={`badge ${statusInfo.className}`}>{statusInfo.label}</span>
                </div>
                <p className="status-message">{report.summary_message}</p>
              </div>
            </div>
            <div className="status-banner-action">
              <span className="action-tag">Recommended Operational Protocol:</span>
              <p className="action-text">{report.recommended_next_action}</p>
            </div>
          </div>
        )}
      </div>

      {isLoading && (
        <div className="matrix-loading-state">
          <RefreshCw size={36} className="spin-icon text-cyan" />
          <p className="loading-text">Computing two-sample statistical tests & divergence metrics...</p>
        </div>
      )}

      {error && (
        <div className="alert alert-danger" style={{ margin: '1rem 0' }}>
          <XCircle size={18} />
          <div>
            <strong>Error loading drift report:</strong> {error}
          </div>
        </div>
      )}

      {!isLoading && report && (
        <>
          {/* 3. Drift Summary KPI Metrics */}
          <div className="metrics-grid">
            <div className="card metric-card">
              <div className="metric-header">
                <span className="metric-title">Features Evaluated</span>
                <Layers size={18} className="metric-icon text-cyan" />
              </div>
              <div className="metric-value">{report.total_features_evaluated}</div>
              <div className="metric-footer">
                <span className="metric-subtext">Covariates & business attributes</span>
              </div>
            </div>

            <div className="card metric-card">
              <div className="metric-header">
                <span className="metric-title">Drifted Features</span>
                <AlertTriangle size={18} className="metric-icon text-amber" />
              </div>
              <div className="metric-value">
                {report.drifted_features_count}
                <span className="metric-denominator"> / {report.total_features_evaluated}</span>
              </div>
              <div className="metric-footer">
                <span className={`metric-change ${report.drifted_features_count > 0 ? 'text-amber' : 'text-emerald'}`}>
                  {report.drifted_features_count > 0
                    ? `${Math.round((report.drifted_features_count / report.total_features_evaluated) * 100)}% divergent`
                    : 'All features stable'}
                </span>
              </div>
            </div>

            <div className="card metric-card">
              <div className="metric-header">
                <span className="metric-title">High Severity Features</span>
                <ShieldAlert size={18} className="metric-icon text-red" />
              </div>
              <div className="metric-value">{report.high_severity_count}</div>
              <div className="metric-footer">
                <span className={`metric-subtext ${report.high_severity_count > 0 ? 'text-red font-bold' : ''}`}>
                  {report.high_severity_count > 0 ? 'Requires acquisition review' : 'No critical divergence'}
                </span>
              </div>
            </div>

            <div className="card metric-card">
              <div className="metric-header">
                <span className="metric-title">Baseline Cohort Size</span>
                <FileCheck2 size={18} className="metric-icon text-emerald" />
              </div>
              <div className="metric-value">{report.reference_sample_size.toLocaleString()}</div>
              <div className="metric-footer">
                <span className="metric-subtext">Training reference customers</span>
              </div>
            </div>

            <div className="card metric-card">
              <div className="metric-header">
                <span className="metric-title">Evaluation Batch Size</span>
                <Activity size={18} className="metric-icon text-purple" />
              </div>
              <div className="metric-value">{report.current_sample_size.toLocaleString()}</div>
              <div className="metric-footer">
                <span className="metric-subtext">Incoming customer observations</span>
              </div>
            </div>

            <div className="card metric-card">
              <div className="metric-header">
                <span className="metric-title">Schema Integrity</span>
                <CheckCircle2 size={18} className="metric-icon text-emerald" />
              </div>
              <div className="metric-value text-emerald">
                {report.schema_validation?.is_valid ? '100% Valid' : 'Schema Error'}
              </div>
              <div className="metric-footer">
                <span className="metric-subtext">Zero model-input corruptions</span>
              </div>
            </div>
          </div>

          {/* 4. Feature Drift Breakdown Table */}
          <div className="card drift-table-card">
            <div className="card-header">
              <div className="title-with-icon">
                <div className="icon-badge icon-purple">
                  <Sliders size={18} />
                </div>
                <div>
                  <h3 className="card-title">Feature-by-Feature Statistical Drift Breakdown</h3>
                  <p className="card-subtitle">
                    Kolmogorov-Smirnov two-sample tests (continuous) and Chi-Square contingency tests (discrete) with PSI and TVD.
                  </p>
                </div>
              </div>
            </div>

            <div className="table-responsive">
              <table className="custom-table drift-table">
                <thead>
                  <tr>
                    <th>Feature Name</th>
                    <th>Type</th>
                    <th>Test Name</th>
                    <th>Statistic</th>
                    <th>P-Value</th>
                    <th>PSI / TVD</th>
                    <th>Drift Status</th>
                    <th>Recommended Operational Finding</th>
                    <th>Inspect</th>
                  </tr>
                </thead>
                <tbody>
                  {report.features.map((feat) => {
                    const isSelected = feat.feature_name === selectedFeature;
                    const metricVal = feat.psi != null ? `PSI: ${feat.psi.toFixed(3)}` : `TVD: ${feat.tvd?.toFixed(3)}`;
                    return (
                      <tr
                        key={feat.feature_name}
                        className={`drift-row ${isSelected ? 'drift-row-selected' : ''}`}
                        onClick={() => setSelectedFeature(feat.feature_name)}
                      >
                        <td>
                          <span className="font-bold feature-name-cell">{feat.feature_name}</span>
                        </td>
                        <td>
                          <span className="feature-type-tag">{feat.feature_type}</span>
                        </td>
                        <td>
                          <span className="test-name-tag">
                            {feat.test_name === 'kolmogorov_smirnov' ? 'KS 2-Sample' : 'Chi-Square'}
                          </span>
                        </td>
                        <td>
                          <code>{feat.test_statistic.toFixed(4)}</code>
                        </td>
                        <td>
                          <code className={feat.p_value < 0.05 ? 'text-red font-bold' : 'text-emerald'}>
                            {feat.p_value < 0.0001 ? '< 0.0001' : feat.p_value.toFixed(4)}
                          </code>
                        </td>
                        <td>
                          <code className={feat.psi >= 0.25 || feat.tvd >= 0.25 ? 'text-red font-bold' : ''}>
                            {metricVal}
                          </code>
                        </td>
                        <td>{getSeverityBadge(feat.severity)}</td>
                        <td className="action-cell">
                          <span className="action-cell-text">{feat.recommended_action}</span>
                        </td>
                        <td>
                          <button
                            type="button"
                            className={`btn btn-sm ${isSelected ? 'btn-primary' : 'btn-secondary'}`}
                            onClick={(e) => {
                              e.stopPropagation();
                              setSelectedFeature(feat.feature_name);
                            }}
                          >
                            <span>Inspect</span>
                            <ArrowRight size={13} />
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* 5. Distribution Comparison Visualizer */}
          {activeFeatureResult && (
            <div className="card distribution-visualizer-card">
              <div className="card-header">
                <div className="title-with-icon">
                  <div className="icon-badge icon-cyan">
                    <BarChart3 size={18} />
                  </div>
                  <div>
                    <div className="dist-title-row">
                      <h3 className="card-title">
                        Empirical Distribution Comparison: <code>{activeFeatureResult.feature_name}</code>
                      </h3>
                      {getSeverityBadge(activeFeatureResult.severity)}
                    </div>
                    <p className="card-subtitle">
                      Visualizing reference baseline vs current cohort distributions across bins/categories.
                    </p>
                  </div>
                </div>

                {/* Quick Feature Select Buttons */}
                <div className="feature-quick-select">
                  {report.features.map((f) => (
                    <button
                      key={f.feature_name}
                      type="button"
                      className={`btn btn-xs ${selectedFeature === f.feature_name ? 'btn-primary' : 'btn-secondary'}`}
                      onClick={() => setSelectedFeature(f.feature_name)}
                    >
                      {f.feature_name}
                      {f.drift_detected && <span className="drift-dot-indicator" />}
                    </button>
                  ))}
                </div>
              </div>

              <div className="distribution-body">
                {/* Visual Comparative Bars */}
                {activeDistData ? (
                  <div className="dist-chart-wrapper">
                    <div className="dist-chart-legend">
                      <div className="legend-item">
                        <span className="legend-swatch legend-ref" />
                        <span>Reference Baseline ({metadata?.reference_size || report.reference_sample_size} rows)</span>
                      </div>
                      <div className="legend-item">
                        <span className="legend-swatch legend-curr" />
                        <span>Current Batch ({report.current_sample_size} rows)</span>
                      </div>
                    </div>

                    <div className="dist-bars-container">
                      {activeDistData.labels.map((label, idx) => {
                        const refPct = activeDistData.ref_pct[idx] || 0;
                        const currPct = activeDistData.curr_pct[idx] || 0;
                        const diff = (currPct - refPct).toFixed(1);
                        return (
                          <div key={label} className="dist-bar-row">
                            <div className="dist-bar-label" title={label}>
                              {label}
                            </div>
                            <div className="dist-bars-dual">
                              {/* Reference Bar */}
                              <div className="dist-bar-track">
                                <div
                                  className="dist-bar-fill fill-ref"
                                  style={{ width: `${Math.min(refPct * 2, 100)}%` }}
                                >
                                  <span className="dist-bar-value">{refPct}%</span>
                                </div>
                              </div>
                              {/* Current Bar */}
                              <div className="dist-bar-track">
                                <div
                                  className={`dist-bar-fill ${activeFeatureResult.drift_detected ? 'fill-drifted' : 'fill-curr'}`}
                                  style={{ width: `${Math.min(currPct * 2, 100)}%` }}
                                >
                                  <span className="dist-bar-value">{currPct}%</span>
                                </div>
                              </div>
                            </div>
                            <div className="dist-diff-label">
                              <span
                                className={`diff-tag ${
                                  Math.abs(diff) >= 5
                                    ? diff > 0
                                    ? 'text-red'
                                    : 'text-amber'
                                    : 'text-muted'
                                }`}
                              >
                                {diff > 0 ? `+${diff}%` : `${diff}%`}
                              </span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ) : (
                  <div className="alert alert-neutral">
                    Distribution histogram data for <code>{selectedFeature}</code> is not available in precomputed payload.
                  </div>
                )}

                {/* Summary Statistics Comparison Grid */}
                <div className="dist-stats-grid">
                  <div className="stat-cohort-card">
                    <h4 className="stat-cohort-title text-cyan">Reference Baseline Summary</h4>
                    <div className="stat-cohort-details">
                      {activeFeatureResult.feature_type === 'numeric' && activeFeatureResult.reference_summary ? (
                        <>
                          <div className="stat-row">
                            <span>Mean:</span>
                            <strong>{activeFeatureResult.reference_summary.mean?.toFixed(2)}</strong>
                          </div>
                          <div className="stat-row">
                            <span>Std Dev:</span>
                            <strong>{activeFeatureResult.reference_summary.std?.toFixed(2)}</strong>
                          </div>
                          <div className="stat-row">
                            <span>Median:</span>
                            <strong>{activeFeatureResult.reference_summary.median?.toFixed(2)}</strong>
                          </div>
                          <div className="stat-row">
                            <span>IQR (25% - 75%):</span>
                            <strong>
                              {activeFeatureResult.reference_summary.q25?.toFixed(1)} - {activeFeatureResult.reference_summary.q75?.toFixed(1)}
                            </strong>
                          </div>
                        </>
                      ) : (
                        <div className="stat-cat-list">
                          {activeFeatureResult.reference_summary?.proportions &&
                            Object.entries(activeFeatureResult.reference_summary.proportions).map(([k, v]) => (
                              <div key={k} className="stat-row">
                                <span>{k}:</span>
                                <strong>{(v * 100).toFixed(1)}%</strong>
                              </div>
                            ))}
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="stat-cohort-card">
                    <h4 className="stat-cohort-title text-purple">Current Evaluation Cohort Summary</h4>
                    <div className="stat-cohort-details">
                      {activeFeatureResult.feature_type === 'numeric' && activeFeatureResult.current_summary ? (
                        <>
                          <div className="stat-row">
                            <span>Mean:</span>
                            <strong className={activeFeatureResult.drift_detected ? 'text-amber' : ''}>
                              {activeFeatureResult.current_summary.mean?.toFixed(2)}
                            </strong>
                          </div>
                          <div className="stat-row">
                            <span>Std Dev:</span>
                            <strong>{activeFeatureResult.current_summary.std?.toFixed(2)}</strong>
                          </div>
                          <div className="stat-row">
                            <span>Median:</span>
                            <strong>{activeFeatureResult.current_summary.median?.toFixed(2)}</strong>
                          </div>
                          <div className="stat-row">
                            <span>IQR (25% - 75%):</span>
                            <strong>
                              {activeFeatureResult.current_summary.q25?.toFixed(1)} - {activeFeatureResult.current_summary.q75?.toFixed(1)}
                            </strong>
                          </div>
                        </>
                      ) : (
                        <div className="stat-cat-list">
                          {activeFeatureResult.current_summary?.proportions &&
                            Object.entries(activeFeatureResult.current_summary.proportions).map(([k, v]) => (
                              <div key={k} className="stat-row">
                                <span>{k}:</span>
                                <strong>{(v * 100).toFixed(1)}%</strong>
                              </div>
                            ))}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* 5b. Counterfactual Prediction Drift Table */}
          {report.prediction_drift && report.prediction_drift.length > 0 && (
            <div className="card drift-table-card" style={{ marginTop: '1.5rem' }}>
              <div className="card-header">
                <div className="title-with-icon">
                  <div className="icon-badge icon-purple">
                    <Activity size={18} />
                  </div>
                  <div>
                    <h3 className="card-title">Counterfactual Model Prediction Drift (P0, P10, P20)</h3>
                    <p className="card-subtitle">
                      Distributional stability of Double Machine Learning predicted purchase probabilities across discount tiers.
                    </p>
                  </div>
                </div>
              </div>

              <div className="table-responsive">
                <table className="custom-table drift-table">
                  <thead>
                    <tr>
                      <th>Treatment Probability Arm</th>
                      <th>Test Method</th>
                      <th>KS Statistic (D)</th>
                      <th>P-Value</th>
                      <th>PSI</th>
                      <th>Severity</th>
                      <th>Operational Assessment</th>
                    </tr>
                  </thead>
                  <tbody>
                    {report.prediction_drift.map((pDrift) => (
                      <tr key={pDrift.feature_name}>
                        <td>
                          <strong>{pDrift.feature_name.replace('predicted_purchase_probability_', 'Tier $')}</strong>
                        </td>
                        <td>{pDrift.test_name}</td>
                        <td>{pDrift.test_statistic}</td>
                        <td>{pDrift.p_value}</td>
                        <td><code>{pDrift.psi?.toFixed(4) || 'N/A'}</code></td>
                        <td>{getSeverityBadge(pDrift.drift_severity || pDrift.severity)}</td>
                        <td className="action-cell">
                          <span className="action-cell-text">{pDrift.recommended_action}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* 5c. Historical Surveillance Logs */}
          {driftHistory && driftHistory.length > 0 && (
            <div className="card drift-table-card" style={{ marginTop: '1.5rem' }}>
              <div className="card-header">
                <div className="title-with-icon">
                  <div className="icon-badge icon-cyan">
                    <FileCheck2 size={18} />
                  </div>
                  <div>
                    <h3 className="card-title">Historical Drift Surveillance Logs</h3>
                    <p className="card-subtitle">
                      Audit log of recent statistical drift evaluations and population surveillance runs.
                    </p>
                  </div>
                </div>
              </div>

              <div className="table-responsive">
                <table className="custom-table drift-table">
                  <thead>
                    <tr>
                      <th>Timestamp (UTC)</th>
                      <th>Dataset / Reference</th>
                      <th>Overall Status</th>
                      <th>Total Features</th>
                      <th>Drifted Count</th>
                      <th>Severe Count</th>
                    </tr>
                  </thead>
                  <tbody>
                    {driftHistory.slice(-5).reverse().map((h) => (
                      <tr key={h.run_id || h.timestamp}>
                        <td>{new Date(h.timestamp).toLocaleString()}</td>
                        <td><code>{h.dataset_name}</code></td>
                        <td>{getStatusBadge(h.overall_status).label}</td>
                        <td>{h.total_features_evaluated} evaluated</td>
                        <td className={h.drifted_features_count > 0 ? 'text-amber font-bold' : ''}>{h.drifted_features_count} drifted</td>
                        <td className={h.severe_drift_count > 0 ? 'text-red font-bold' : ''}>{h.severe_drift_count} severe</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* 6. Methodological Notice Callout */}
          <div className="card methodological-callout-card">
            <div className="card-header">
              <div className="title-with-icon">
                <div className="icon-badge icon-amber">
                  <Info size={18} />
                </div>
                <div>
                  <h4 className="card-title">Causal Methodological Guidance: Covariate Shift vs Concept Drift</h4>
                  <p className="card-subtitle">
                    Why statistical feature drift does not automatically invalidate Double Machine Learning estimates.
                  </p>
                </div>
              </div>
            </div>
            <div className="card-content-padded">
              <p className="methodological-body-text">
                {report.methodological_notice}
              </p>
              <div className="methodological-points-grid">
                <div className="methodological-point">
                  <strong>Covariate Shift (P(X) Drift):</strong>
                  <p>
                    A shift in customer demographics (e.g. older users or changed acquisition channels) alters the marginal density of features. Under structural unconfoundedness, Conditional Average Treatment Effects (\tau(X)) remain mathematically unbiased within the overlapping support.
                  </p>
                </div>
                <div className="methodological-point">
                  <strong>Concept Drift (P(Y | X, T) Shift):</strong>
                  <p>
                    Occurs only when the underlying causal relationship between discounts and purchase decisions fundamentally shifts (e.g. macroeconomic shocks or severe brand devaluation). This requires live counterfactual holdout testing to confirm.
                  </p>
                </div>
                <div className="methodological-point">
                  <strong>Recommended Remediation:</strong>
                  <p>
                    Re-calibrate propensity scores e(X) and check for common support overlap degradation. <strong>Do not automatically trigger unmonitored model retraining</strong>, as automated retraining on non-stationary unverified data risks policy instability.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

