import React, { useState, useEffect } from 'react';
import { Users, AlertTriangle, CheckCircle, Scale, Info, RefreshCw } from 'lucide-react';
import { fetchFairnessDiagnostics, fetchOptimizationBenchmarks } from '../services/v2Service';

export default function FairnessAnalysisTab() {
  const [fairnessData, setFairnessData] = useState(null);
  const [benchmarksData, setBenchmarksData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isLiveApi, setIsLiveApi] = useState(false);

  const loadData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [fRes, bRes] = await Promise.all([
        fetchFairnessDiagnostics(),
        fetchOptimizationBenchmarks(),
      ]);
      setFairnessData(fRes);
      setBenchmarksData(bRes);
      setIsLiveApi(Boolean(fRes.isLiveApi));
    } catch (err) {
      console.error('Failed to load fairness data:', err);
      setError(err.message || 'Unable to fetch customer segment fairness diagnostics');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  if (isLoading) {
    return (
      <div className="card" style={{ padding: '3rem', textAlign: 'center', color: '#94a3b8' }}>
        <RefreshCw className="animate-spin" size={32} style={{ margin: '0 auto 1rem' }} />
        <p>Loading Customer Segment Fairness Diagnostics...</p>
      </div>
    );
  }

  if (error || !fairnessData) {
    return (
      <div className="card" style={{ padding: '2rem', borderLeft: '4px solid #ef4444', color: '#f87171' }}>
        <h3 style={{ margin: '0 0 0.5rem' }}>Failed to Load Fairness Diagnostics</h3>
        <p style={{ margin: '0 0 1rem', color: '#94a3b8' }}>{error}</p>
        <button type="button" className="action-btn" onClick={loadData}>Retry</button>
      </div>
    );
  }

  const segments = fairnessData.segments || {};
  const regimes = benchmarksData?.core_regimes || {};
  const r1 = regimes.regime1_original_milp || {};
  const r3 = regimes.regime3_fairness_constrained_milp || {};

  return (
    <div className="space-y-6">
      {/* 1. Header Banner */}
      <div className="card" style={{ padding: '1.5rem', background: '#1e293b', border: '1px solid #334155' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
              <h2 style={{ fontSize: '1.4rem', fontWeight: 700, margin: 0, color: '#f8fafc' }}>
                Customer Segment Fairness & Demographic Parity Diagnostics
              </h2>
              <span className={`tab-badge ${isLiveApi ? 'tab-badge-highlight' : ''}`}>
                {isLiveApi ? '● Live Backend API' : 'Static Fallback Mode'}
              </span>
            </div>
            <p style={{ margin: 0, color: '#94a3b8', fontSize: '0.9rem', maxWidth: '850px' }}>
              Audit promotional discount allocation equity across business-defined customer tiers (Budget, Regular, Premium, VIP).
              Ensures algorithmic optimization does not systematically abandon price-sensitive customer tiers while maintaining campaign profit targets.
            </p>
          </div>
        </div>

        {/* Business Attribute Notice */}
        <div style={{ marginTop: '1rem', padding: '0.6rem 0.85rem', background: '#0f172a', borderRadius: '6px', fontSize: '0.8rem', color: '#94a3b8', display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <Scale size={16} style={{ color: '#38bdf8', flexShrink: 0 }} />
          <span>
            <strong>Compliance Policy:</strong> Fairness analysis is strictly performed across commercial account tiers and purchasing profiles. Sensitive demographic variables (race, gender, age) are intentionally excluded from modeling and optimization.
          </span>
        </div>
      </div>

      {/* 2. Fairness Health Scorecards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
        {/* Parity Gap */}
        <div className="card" style={{ padding: '1.25rem', background: '#1e293b', border: '1px solid #334155' }}>
          <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.4rem' }}>Demographic Parity Gap</div>
          <div style={{ fontSize: '1.6rem', fontWeight: 700, color: '#facc15' }}>
            {fairnessData.demographic_parity_gap_pct ? `${fairnessData.demographic_parity_gap_pct.toFixed(1)}%` : '18.1%'}
          </div>
          <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '0.3rem' }}>
            Max rate (27.0%) − Min rate (8.9%)
          </div>
        </div>

        {/* Minimum Parity Ratio */}
        <div className="card" style={{ padding: '1.25rem', background: '#1e293b', border: '1px solid #334155' }}>
          <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.4rem' }}>Lowest Segment Parity Ratio</div>
          <div style={{ fontSize: '1.6rem', fontWeight: 700, color: '#f87171' }}>
            {fairnessData.min_parity_ratio ? fairnessData.min_parity_ratio.toFixed(3) : '0.448'}
          </div>
          <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '0.3rem' }}>
            Disadvantaged tier: Budget segment (&lt; 0.80 standard threshold)
          </div>
        </div>

        {/* Overall Allocation Rate */}
        <div className="card" style={{ padding: '1.25rem', background: '#1e293b', border: '1px solid #334155' }}>
          <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.4rem' }}>Cohort-Wide Allocation Rate</div>
          <div style={{ fontSize: '1.6rem', fontWeight: 700, color: '#38bdf8' }}>
            {fairnessData.overall_allocation_rate_pct ? `${fairnessData.overall_allocation_rate_pct.toFixed(2)}%` : '19.82%'}
          </div>
          <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '0.3rem' }}>
            1,190 / 6,005 customers awarded $10 or $20
          </div>
        </div>

        {/* Constrained vs Unconstrained Tradeoff */}
        <div className="card" style={{ padding: '1.25rem', background: '#1e293b', border: '1px solid #334155' }}>
          <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.4rem' }}>Fairness Constraint Cost</div>
          <div style={{ fontSize: '1.6rem', fontWeight: 700, color: '#4ade80' }}>
            -$47.81 <span style={{ fontSize: '0.85rem', fontWeight: 400, color: '#94a3b8' }}>(-0.05%)</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '0.3rem' }}>
            Profit tradeoff for enforcing 15% budget floor
          </div>
        </div>
      </div>

      {/* 3. Segment Breakdown Table */}
      <div className="card" style={{ padding: '1.5rem', background: '#1e293b', border: '1px solid #334155' }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 600, margin: '0 0 1rem', color: '#f8fafc' }}>
          Segment-Level Allocation & Benefit Gap Matrix
        </h3>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem', textAlign: 'left' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #334155', color: '#94a3b8' }}>
                <th style={{ padding: '0.75rem 0.5rem' }}>Customer Segment</th>
                <th style={{ padding: '0.75rem 0.5rem' }}>Cohort Size</th>
                <th style={{ padding: '0.75rem 0.5rem' }}>Allocated ($10/$20)</th>
                <th style={{ padding: '0.75rem 0.5rem' }}>Allocation Rate (%)</th>
                <th style={{ padding: '0.75rem 0.5rem' }}>Avg Discount ($)</th>
                <th style={{ padding: '0.75rem 0.5rem' }}>Parity Ratio vs Cohort</th>
                <th style={{ padding: '0.75rem 0.5rem' }}>Avg Incr. Profit ($)</th>
                <th style={{ padding: '0.75rem 0.5rem' }}>Compliance Status</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(segments).map(([segName, m]) => {
                const isDisadvantaged = m.parity_ratio_vs_overall < 0.80;

                return (
                  <tr key={segName} style={{ borderBottom: '1px solid #334155' }}>
                    <td style={{ padding: '0.75rem 0.5rem', fontWeight: 600, textTransform: 'capitalize', color: '#f1f5f9' }}>
                      {m.segment_name || segName}
                    </td>
                    <td style={{ padding: '0.75rem 0.5rem', color: '#94a3b8' }}>
                      {m.sample_size?.toLocaleString()}
                    </td>
                    <td style={{ padding: '0.75rem 0.5rem', fontWeight: 600, color: '#e2e8f0' }}>
                      {m.allocation_count?.toLocaleString()}
                    </td>
                    <td style={{ padding: '0.75rem 0.5rem', fontWeight: 600, color: isDisadvantaged ? '#f87171' : '#38bdf8' }}>
                      {m.allocation_rate_pct?.toFixed(2)}%
                    </td>
                    <td style={{ padding: '0.75rem 0.5rem', color: '#cbd5e1' }}>
                      ${m.average_discount_amount?.toFixed(2)}
                    </td>
                    <td style={{ padding: '0.75rem 0.5rem', fontWeight: 600, color: isDisadvantaged ? '#f87171' : '#4ade80' }}>
                      {m.parity_ratio_vs_overall?.toFixed(3)}
                    </td>
                    <td style={{ padding: '0.75rem 0.5rem', color: '#4ade80' }}>
                      ${m.mean_incremental_profit?.toFixed(2)}
                    </td>
                    <td style={{ padding: '0.75rem 0.5rem' }}>
                      {isDisadvantaged ? (
                        <span className="tab-badge" style={{ background: 'rgba(239, 68, 68, 0.2)', color: '#f87171', border: '1px solid #ef4444' }}>
                          ⚠️ Disadvantaged (Ratio &lt; 0.8)
                        </span>
                      ) : (
                        <span className="tab-badge" style={{ background: 'rgba(34, 197, 94, 0.2)', color: '#4ade80', border: '1px solid #22c55e' }}>
                          ✓ Compliant
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* 4. Side-by-Side: Unconstrained vs Fairness-Constrained MILP */}
      <div className="card" style={{ padding: '1.5rem', background: '#1e293b', border: '1px solid #334155' }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 600, margin: '0 0 1rem', color: '#f8fafc' }}>
          Intervention Impact: Unconstrained vs Fairness-Constrained Allocation (15% Budget Floor)
        </h3>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.5rem' }}>
          {/* Regime 1 (Original) */}
          <div style={{ background: '#0f172a', padding: '1.25rem', borderRadius: '8px', border: '1px solid #334155' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
              <h4 style={{ margin: 0, color: '#f8fafc', fontSize: '1rem' }}>Regime 1: Unconstrained Baseline MILP</h4>
              <span className="tab-badge" style={{ background: '#334155', color: '#cbd5e1' }}>Unconstrained</span>
            </div>

            <div className="space-y-2" style={{ fontSize: '0.85rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#94a3b8' }}>Nominal Expected Profit:</span>
                <span style={{ fontWeight: 600, color: '#38bdf8' }}>
                  ${r1.financial_kpis?.nominal_expected_profit?.toLocaleString() || '100,636.58'}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#94a3b8' }}>Budget Segment Allocation:</span>
                <span style={{ fontWeight: 600, color: '#f87171' }}>8.88% (157 / 1,769 customers)</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#94a3b8' }}>Budget Segment Parity Ratio:</span>
                <span style={{ fontWeight: 600, color: '#f87171' }}>0.448 (Severe Disparity)</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#94a3b8' }}>Expected Budget Cost:</span>
                <span style={{ color: '#cbd5e1' }}>${r1.financial_kpis?.total_expected_cost?.toLocaleString() || '4,998.79'}</span>
              </div>
            </div>
          </div>

          {/* Regime 3 (Fairness Constrained) */}
          <div style={{ background: '#0f172a', padding: '1.25rem', borderRadius: '8px', border: '1px solid #22c55e' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
              <h4 style={{ margin: 0, color: '#4ade80', fontSize: '1rem' }}>Regime 3: Fairness-Constrained (15% Floor)</h4>
              <span className="tab-badge" style={{ background: 'rgba(34, 197, 94, 0.2)', color: '#4ade80' }}>Fairness Active</span>
            </div>

            <div className="space-y-2" style={{ fontSize: '0.85rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#94a3b8' }}>Nominal Expected Profit:</span>
                <span style={{ fontWeight: 600, color: '#38bdf8' }}>
                  ${r3.financial_kpis?.nominal_expected_profit?.toLocaleString() || '100,588.77'}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#94a3b8' }}>Budget Segment Allocation:</span>
                <span style={{ fontWeight: 600, color: '#4ade80' }}>15.00% (266 / 1,769 customers) [+69%]</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#94a3b8' }}>Budget Segment Parity Ratio:</span>
                <span style={{ fontWeight: 600, color: '#4ade80' }}>0.739 (Protected Floor)</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#94a3b8' }}>Expected Budget Cost:</span>
                <span style={{ color: '#cbd5e1' }}>${r3.financial_kpis?.total_expected_cost?.toLocaleString() || '4,997.37'}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
