import React, { useState, useEffect } from 'react';
import { ShieldCheck, TrendingUp, AlertTriangle, RefreshCw, Info, HelpCircle } from 'lucide-react';
import { fetchPolicyEvaluationSummary, evaluatePolicies } from '../services/v2Service';

export default function PolicyEvaluationTab() {
  const [data, setData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isLiveApi, setIsLiveApi] = useState(false);
  const [bootstrapCount, setBootstrapCount] = useState(500);
  const [isReevaluating, setIsReevaluating] = useState(false);

  const loadData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await fetchPolicyEvaluationSummary();
      setData(res);
      setIsLiveApi(Boolean(res.isLiveApi));
    } catch (err) {
      console.error('Failed to load policy evaluation data:', err);
      setError(err.message || 'Unable to fetch policy evaluation results');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleReevaluate = async () => {
    setIsReevaluating(true);
    try {
      const res = await evaluatePolicies({ n_bootstraps: bootstrapCount, min_propensity_clip: 0.01 });
      setData(res);
      setIsLiveApi(Boolean(res.isLiveApi));
    } catch (err) {
      console.error('Re-evaluation error:', err);
    } finally {
      setIsReevaluating(false);
    }
  };

  if (isLoading) {
    return (
      <div className="card" style={{ padding: '3rem', textAlign: 'center', color: '#94a3b8' }}>
        <RefreshCw className="animate-spin" size={32} style={{ margin: '0 auto 1rem' }} />
        <p>Loading Doubly Robust Policy Evaluation results...</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="card" style={{ padding: '2rem', borderLeft: '4px solid #ef4444', color: '#f87171' }}>
        <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', margin: '0 0 0.5rem' }}>
          <AlertTriangle size={20} /> Failed to Load Policy Evaluation Data
        </h3>
        <p style={{ margin: '0 0 1rem', color: '#94a3b8' }}>{error}</p>
        <button type="button" className="action-btn" onClick={loadData}>Retry</button>
      </div>
    );
  }

  const policies = data.policies || {};
  const meta = data.metadata || {};
  const canonical = data.canonical_milestone3_benchmarks || {};

  return (
    <div className="space-y-6">
      {/* 1. Header Banner & Live Indicator */}
      <div className="card" style={{ padding: '1.5rem', background: '#1e293b', border: '1px solid #334155' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
              <h2 style={{ fontSize: '1.4rem', fontWeight: 700, margin: 0, color: '#f8fafc' }}>
                Doubly Robust Off-Policy Evaluation (OPE)
              </h2>
              <span className={`tab-badge ${isLiveApi ? 'tab-badge-highlight' : ''}`}>
                {isLiveApi ? '● Live Backend API' : 'Static Fallback Mode'}
              </span>
            </div>
            <p style={{ margin: 0, color: '#94a3b8', fontSize: '0.9rem', maxWidth: '850px' }}>
              Rigorous off-policy counterfactual benchmarking comparing personalized DML + MILP against organic
              and heuristic baselines using Inverse Probability Weighting (IPW) and the Doubly Robust (DR) estimator with 95% bootstrap confidence intervals.
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <label htmlFor="bootstrap-select" style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
              Bootstraps:
            </label>
            <select
              id="bootstrap-select"
              value={bootstrapCount}
              onChange={(e) => setBootstrapCount(Number(e.target.value))}
              style={{
                background: '#0f172a',
                color: '#e2e8f0',
                border: '1px solid #475569',
                padding: '0.35rem 0.5rem',
                borderRadius: '6px',
                fontSize: '0.85rem',
              }}
            >
              <option value={100}>100 (Fast)</option>
              <option value={500}>500 (Standard)</option>
              <option value={1000}>1,000 (High Precision)</option>
            </select>
            <button
              type="button"
              className="action-btn"
              onClick={handleReevaluate}
              disabled={isReevaluating}
              style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', padding: '0.4rem 0.75rem', fontSize: '0.85rem' }}
            >
              <RefreshCw size={14} className={isReevaluating ? 'animate-spin' : ''} />
              {isReevaluating ? 'Evaluating...' : 'Re-Evaluate'}
            </button>
          </div>
        </div>

        {/* Evaluation Metadata Bar */}
        <div style={{ display: 'flex', gap: '1.5rem', marginTop: '1.25rem', paddingTop: '1rem', borderTop: '1px solid #334155', fontSize: '0.82rem', color: '#94a3b8', flexWrap: 'wrap' }}>
          <div><strong>Cohort Size:</strong> {meta.evaluation_sample_size || 6005} customers</div>
          <div><strong>Treatment Arm Contrasts:</strong> $0 (control), $10, $20</div>
          <div><strong>Propensity Clip:</strong> {meta.min_propensity_clip || 0.01}</div>
          <div><strong>DML Prescriptive Lift:</strong> +${canonical.dml_milp_incremental_profit?.toLocaleString() || '10,199.87'} (+{canonical.dml_milp_incremental_profit_to_cost_pct || '204.05'}% ROI)</div>
        </div>
      </div>

      {/* 2. Core Policy Comparison Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
        {Object.entries(policies).map(([key, pol]) => {
          const isUnavailable = !pol.is_available;
          const econ = pol.economic_metrics || {};
          const conv = pol.conversion_metrics || {};
          const diag = pol.diagnostics || {};
          const isDML = key === 'dml_milp';

          return (
            <div
              key={key}
              className="card"
              style={{
                padding: '1.25rem',
                border: isDML ? '2px solid #3b82f6' : '1px solid #334155',
                background: isUnavailable ? '#0f172a' : '#1e293b',
                opacity: isUnavailable ? 0.75 : 1.0,
                position: 'relative',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
                <h3 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 600, color: isDML ? '#60a5fa' : '#f1f5f9' }}>
                  {pol.policy_name}
                </h3>
                {isUnavailable ? (
                  <span className="tab-badge" style={{ background: '#334155', color: '#94a3b8' }}>
                    Unavailable (Future)
                  </span>
                ) : isDML ? (
                  <span className="tab-badge tab-badge-highlight">Optimal Policy</span>
                ) : (
                  <span className="tab-badge" style={{ background: '#334155', color: '#cbd5e1' }}>
                    Baseline
                  </span>
                )}
              </div>

              <p style={{ fontSize: '0.82rem', color: '#94a3b8', margin: '0 0 1rem', minHeight: '38px' }}>
                {pol.description}
              </p>

              {isUnavailable ? (
                <div style={{ padding: '1.5rem 1rem', background: '#1e293b', borderRadius: '6px', textAlign: 'center', color: '#94a3b8', fontSize: '0.85rem' }}>
                  <HelpCircle size={24} style={{ margin: '0 auto 0.5rem', color: '#64748b' }} />
                  <p style={{ margin: 0 }}>Reserved for future Uncertainty-Aware Policy (Phase 5)</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {/* DR Expected Profit */}
                  <div style={{ background: '#0f172a', padding: '0.75rem', borderRadius: '6px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                      <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>DR Expected Profit:</span>
                      <span style={{ fontSize: '1.15rem', fontWeight: 700, color: '#38bdf8' }}>
                        ${Math.round(econ.dr_expected_profit || 0).toLocaleString()}
                      </span>
                    </div>
                    {econ.dr_profit_ci_95 && (
                      <div style={{ fontSize: '0.75rem', color: '#64748b', textAlign: 'right', marginTop: '0.2rem' }}>
                        95% CI: [${Math.round(econ.dr_profit_ci_95[0]).toLocaleString()} – ${Math.round(econ.dr_profit_ci_95[1]).toLocaleString()}]
                      </div>
                    )}
                  </div>

                  {/* Incremental Profit & Cost */}
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', fontSize: '0.82rem' }}>
                    <div style={{ background: '#0f172a', padding: '0.5rem', borderRadius: '6px' }}>
                      <div style={{ color: '#94a3b8', fontSize: '0.75rem' }}>Incremental Profit:</div>
                      <div style={{ fontWeight: 600, color: econ.incremental_profit_vs_no_discount > 0 ? '#4ade80' : '#94a3b8', marginTop: '0.2rem' }}>
                        {econ.incremental_profit_vs_no_discount ? `+$${Math.round(econ.incremental_profit_vs_no_discount).toLocaleString()}` : '$0.00'}
                      </div>
                    </div>

                    <div style={{ background: '#0f172a', padding: '0.5rem', borderRadius: '6px' }}>
                      <div style={{ color: '#94a3b8', fontSize: '0.75rem' }}>Expected Cost:</div>
                      <div style={{ fontWeight: 600, color: '#f87171', marginTop: '0.2rem' }}>
                        ${Math.round(econ.expected_promotional_cost || 0).toLocaleString()}
                      </div>
                    </div>
                  </div>

                  {/* Profit-to-Cost ROI & DR Conversion */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', padding: '0.25rem 0' }}>
                    <span style={{ color: '#94a3b8' }}>DR Conversion Rate:</span>
                    <span style={{ fontWeight: 600, color: '#e2e8f0' }}>
                      {((conv.dr_conversion_rate || 0) * 100).toFixed(2)}%
                    </span>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', padding: '0.25rem 0' }}>
                    <span style={{ color: '#94a3b8' }}>Profit-to-Cost ROI:</span>
                    <span style={{ fontWeight: 600, color: econ.incremental_profit_to_cost_ratio ? '#4ade80' : '#64748b' }}>
                      {econ.incremental_profit_to_cost_ratio ? `${(econ.incremental_profit_to_cost_ratio * 100).toFixed(1)}%` : 'N/A'}
                    </span>
                  </div>

                  {/* Effective Sample Size Diagnostic */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', color: '#64748b', borderTop: '1px solid #334155', paddingTop: '0.5rem' }}>
                    <span>Effective Sample Size (ESS):</span>
                    <span>{Math.round(diag.effective_sample_size || 0).toLocaleString()}</span>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* 3. Detailed Comparative OPE Table */}
      <div className="card" style={{ padding: '1.5rem', background: '#1e293b', border: '1px solid #334155' }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 600, margin: '0 0 1rem', color: '#f8fafc' }}>
          Head-to-Head Policy Evaluation Matrix
        </h3>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem', textAlign: 'left' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #334155', color: '#94a3b8' }}>
                <th style={{ padding: '0.75rem 0.5rem' }}>Policy Strategy</th>
                <th style={{ padding: '0.75rem 0.5rem' }}>Status</th>
                <th style={{ padding: '0.75rem 0.5rem' }}>DR Profit ($)</th>
                <th style={{ padding: '0.75rem 0.5rem' }}>95% Bootstrap CI</th>
                <th style={{ padding: '0.75rem 0.5rem' }}>Expected Cost</th>
                <th style={{ padding: '0.75rem 0.5rem' }}>Incr. Profit vs Control</th>
                <th style={{ padding: '0.75rem 0.5rem' }}>Incr. Profit / Cost</th>
                <th style={{ padding: '0.75rem 0.5rem' }}>DR Conversion</th>
                <th style={{ padding: '0.75rem 0.5rem' }}>Kish ESS</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(policies).map(([key, pol]) => {
                const econ = pol.economic_metrics || {};
                const conv = pol.conversion_metrics || {};
                const diag = pol.diagnostics || {};
                const isDML = key === 'dml_milp';

                return (
                  <tr
                    key={key}
                    style={{
                      borderBottom: '1px solid #334155',
                      background: isDML ? 'rgba(59, 130, 246, 0.08)' : 'transparent',
                    }}
                  >
                    <td style={{ padding: '0.75rem 0.5rem', fontWeight: 600, color: isDML ? '#60a5fa' : '#f1f5f9' }}>
                      {pol.policy_name}
                    </td>
                    <td style={{ padding: '0.75rem 0.5rem' }}>
                      <span className={`tab-badge ${pol.is_available ? 'tab-badge-highlight' : ''}`}>
                        {pol.status}
                      </span>
                    </td>
                    <td style={{ padding: '0.75rem 0.5rem', fontWeight: 600, color: '#38bdf8' }}>
                      {econ.dr_expected_profit ? `$${Math.round(econ.dr_expected_profit).toLocaleString()}` : '—'}
                    </td>
                    <td style={{ padding: '0.75rem 0.5rem', color: '#94a3b8' }}>
                      {econ.dr_profit_ci_95
                        ? `[$${Math.round(econ.dr_profit_ci_95[0]).toLocaleString()}, $${Math.round(econ.dr_profit_ci_95[1]).toLocaleString()}]`
                        : '—'}
                    </td>
                    <td style={{ padding: '0.75rem 0.5rem', color: '#f87171' }}>
                      {econ.expected_promotional_cost !== undefined
                        ? `$${Math.round(econ.expected_promotional_cost).toLocaleString()}`
                        : '—'}
                    </td>
                    <td style={{ padding: '0.75rem 0.5rem', color: econ.incremental_profit_vs_no_discount > 0 ? '#4ade80' : '#94a3b8' }}>
                      {econ.incremental_profit_vs_no_discount !== undefined
                        ? `$${Math.round(econ.incremental_profit_vs_no_discount).toLocaleString()}`
                        : '—'}
                    </td>
                    <td style={{ padding: '0.75rem 0.5rem', fontWeight: 600, color: econ.incremental_profit_to_cost_ratio ? '#4ade80' : '#64748b' }}>
                      {econ.incremental_profit_to_cost_ratio ? `${(econ.incremental_profit_to_cost_ratio * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td style={{ padding: '0.75rem 0.5rem', color: '#e2e8f0' }}>
                      {conv.dr_conversion_rate ? `${(conv.dr_conversion_rate * 100).toFixed(2)}%` : '—'}
                    </td>
                    <td style={{ padding: '0.75rem 0.5rem', color: '#94a3b8' }}>
                      {diag.effective_sample_size ? Math.round(diag.effective_sample_size).toLocaleString() : '—'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Methodological Cash Budget Disclaimer */}
        <div style={{ marginTop: '1.25rem', padding: '0.75rem', background: '#0f172a', borderRadius: '6px', fontSize: '0.8rem', color: '#94a3b8', display: 'flex', gap: '0.5rem', alignItems: 'flex-start' }}>
          <Info size={16} style={{ flexShrink: 0, marginTop: '2px', color: '#38bdf8' }} />
          <span>
            <strong>Budget Modeling Notice:</strong> Promotional expenditure figures represent ex-ante mathematical expectations calculated
            from model-predicted redemption probabilities. Realized redemption expenditure depends on actual customer purchase decisions and exhibits finite-sample variance.
          </span>
        </div>
      </div>
    </div>
  );
}
