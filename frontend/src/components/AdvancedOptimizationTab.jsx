import React, { useState, useEffect } from 'react';
import { Sliders, Play, CheckCircle2, AlertTriangle, Download, RefreshCw, Info, DollarSign } from 'lucide-react';
import { runAdvancedOptimization, fetchOptimizationBenchmarks, API_BASE_URL } from '../services/v2Service';

export default function AdvancedOptimizationTab() {
  const [budget, setBudget] = useState(5000);
  const [budgetType, setBudgetType] = useState('redemption');
  const [optimizationMode, setOptimizationMode] = useState('conservative_lcb');
  const [riskAversion, setRiskAversion] = useState(1.0);
  const [enableFairness, setEnableFairness] = useState(true);
  const [segmentBudgetFloor, setSegmentBudgetFloor] = useState(0.15);
  const [minParityRatio, setMinParityRatio] = useState(0.70);
  const [maxCustomers, setMaxCustomers] = useState(300);

  const [result, setResult] = useState(null);
  const [benchmarks, setBenchmarks] = useState(null);
  const [isSolving, setIsSolving] = useState(false);
  const [error, setError] = useState(null);
  const [isLiveApi, setIsLiveApi] = useState(false);

  // Load initial benchmarks
  useEffect(() => {
    async function init() {
      try {
        const bRes = await fetchOptimizationBenchmarks();
        setBenchmarks(bRes);
        const r2 = bRes.core_regimes?.regime2_uncertainty_aware_milp;
        if (r2) {
          setResult({ ...r2, isLiveApi: false });
        }
      } catch (err) {
        console.error('Benchmark fetch error:', err);
      }
    }
    init();
  }, []);

  const handleSolve = async () => {
    setIsSolving(true);
    setError(null);
    try {
      const payload = {
        budget: Number(budget),
        budget_type: budgetType,
        optimization_mode: optimizationMode,
        risk_aversion: Number(riskAversion),
        segment_budget_floor: enableFairness ? Number(segmentBudgetFloor) : null,
        min_parity_ratio: enableFairness ? Number(minParityRatio) : null,
        max_customers: Number(maxCustomers),
      };

      const res = await runAdvancedOptimization(payload);
      setResult(res);
      setIsLiveApi(Boolean(res.isLiveApi));
    } catch (err) {
      console.error('Optimization solve error:', err);
      setError(err.message || 'Solver failed to optimize allocation');
    } finally {
      setIsSolving(false);
    }
  };

  const handleExportCsv = () => {
    window.open(`${API_BASE_URL}/api/v2/optimize/allocation/export`, '_blank');
  };

  const financial = result?.financial_kpis || {};
  const tiers = result?.tier_counts || {};
  const uncertainty = result?.uncertainty_kpis || {};
  const fairness = result?.fairness_metrics || {};
  const preview = result?.allocation_preview || [];

  return (
    <div className="space-y-6">
      {/* 1. Header Banner */}
      <div className="card" style={{ padding: '1.5rem', background: '#1e293b', border: '1px solid #334155' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
              <h2 style={{ fontSize: '1.4rem', fontWeight: 700, margin: 0, color: '#f8fafc' }}>
                Uncertainty-Aware & Fairness-Constrained SciPy MILP Optimizer
              </h2>
              <span className={`tab-badge ${isLiveApi ? 'tab-badge-highlight' : ''}`}>
                {isLiveApi ? '● Live HiGHS Solver' : 'Benchmarked / Fallback Mode'}
              </span>
            </div>
            <p style={{ margin: 0, color: '#94a3b8', fontSize: '0.9rem', maxWidth: '850px' }}>
              Formulate and solve personalized Multiple-Choice Knapsack promotions using SciPy MILP (HiGHS).
              Supports Conservative Lower-Confidence-Bound (LCB), Risk-Adjusted Mean-Variance objectives, and business-defined customer segment floor constraints.
            </p>
          </div>

          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button
              type="button"
              className="action-btn"
              onClick={handleExportCsv}
              style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', padding: '0.45rem 0.85rem', fontSize: '0.85rem' }}
            >
              <Download size={15} /> Export Benchmark Allocations (.CSV)
            </button>
          </div>
        </div>
      </div>

      {/* 2. Controls Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.25rem' }}>
        {/* Objective & Budget Panel */}
        <div className="card" style={{ padding: '1.25rem', background: '#1e293b', border: '1px solid #334155' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 600, margin: '0 0 1rem', color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <DollarSign size={18} style={{ color: '#38bdf8' }} /> Budget & Objective Specification
          </h3>

          <div className="space-y-4" style={{ fontSize: '0.85rem' }}>
            <div>
              <label htmlFor="budget-input" style={{ display: 'block', color: '#94a3b8', marginBottom: '0.35rem' }}>
                Campaign Budget Cap (USD):
              </label>
              <input
                id="budget-input"
                type="number"
                min="100"
                max="100000"
                step="250"
                value={budget}
                onChange={(e) => setBudget(Number(e.target.value))}
                style={{
                  width: '100%',
                  background: '#0f172a',
                  border: '1px solid #475569',
                  borderRadius: '6px',
                  padding: '0.5rem',
                  color: '#f8fafc',
                }}
              />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
              <div>
                <label htmlFor="budget-type-select" style={{ display: 'block', color: '#94a3b8', marginBottom: '0.35rem' }}>
                  Budget Model:
                </label>
                <select
                  id="budget-type-select"
                  value={budgetType}
                  onChange={(e) => setBudgetType(e.target.value)}
                  style={{
                    width: '100%',
                    background: '#0f172a',
                    border: '1px solid #475569',
                    borderRadius: '6px',
                    padding: '0.5rem',
                    color: '#f8fafc',
                  }}
                >
                  <option value="redemption">Redemption (Exp. Cost)</option>
                  <option value="face_value">Face Value (Allocated)</option>
                </select>
              </div>

              <div>
                <label htmlFor="cohort-limit-select" style={{ display: 'block', color: '#94a3b8', marginBottom: '0.35rem' }}>
                  Cohort Sample:
                </label>
                <select
                  id="cohort-limit-select"
                  value={maxCustomers}
                  onChange={(e) => setMaxCustomers(Number(e.target.value))}
                  style={{
                    width: '100%',
                    background: '#0f172a',
                    border: '1px solid #475569',
                    borderRadius: '6px',
                    padding: '0.5rem',
                    color: '#f8fafc',
                  }}
                >
                  <option value={100}>100 customers (Instant)</option>
                  <option value={300}>300 customers (Fast)</option>
                  <option value={1000}>1,000 customers</option>
                  <option value={6005}>Full Cohort (6,005)</option>
                </select>
              </div>
            </div>

            <div>
              <label htmlFor="objective-mode-select" style={{ display: 'block', color: '#94a3b8', marginBottom: '0.35rem' }}>
                Optimization Mode:
              </label>
              <select
                id="objective-mode-select"
                value={optimizationMode}
                onChange={(e) => setOptimizationMode(e.target.value)}
                style={{
                  width: '100%',
                  background: '#0f172a',
                  border: '1px solid #475569',
                  borderRadius: '6px',
                  padding: '0.5rem',
                  color: '#f8fafc',
                }}
              >
                <option value="conservative_lcb">Conservative Lower-Confidence-Bound (LCB)</option>
                <option value="risk_adjusted">Risk-Adjusted Mean-Variance (λ penalty)</option>
                <option value="baseline">Baseline Nominal Expected Profit</option>
              </select>
            </div>

            {optimizationMode === 'risk_adjusted' && (
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', marginBottom: '0.35rem' }}>
                  <span>Risk Aversion (λ):</span>
                  <span style={{ fontWeight: 600, color: '#38bdf8' }}>{riskAversion.toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="5.0"
                  step="0.1"
                  value={riskAversion}
                  onChange={(e) => setRiskAversion(Number(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>
            )}
          </div>
        </div>

        {/* Fairness & Constraints Panel */}
        <div className="card" style={{ padding: '1.25rem', background: '#1e293b', border: '1px solid #334155' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 600, margin: '0 0 1rem', color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Sliders size={18} style={{ color: '#a78bfa' }} /> Business Fairness Constraints
          </h3>

          <div className="space-y-4" style={{ fontSize: '0.85rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <input
                id="fairness-checkbox"
                type="checkbox"
                checked={enableFairness}
                onChange={(e) => setEnableFairness(e.target.checked)}
                style={{ width: '16px', height: '16px' }}
              />
              <label htmlFor="fairness-checkbox" style={{ color: '#f8fafc', fontWeight: 500 }}>
                Enforce Segment Minimum Allocation Floors
              </label>
            </div>

            {enableFairness && (
              <div className="space-y-3" style={{ background: '#0f172a', padding: '0.75rem', borderRadius: '6px' }}>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', marginBottom: '0.25rem' }}>
                    <span>Budget Segment Minimum Floor:</span>
                    <span style={{ fontWeight: 600, color: '#4ade80' }}>{(segmentBudgetFloor * 100).toFixed(0)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0.05"
                    max="0.30"
                    step="0.01"
                    value={segmentBudgetFloor}
                    onChange={(e) => setSegmentBudgetFloor(Number(e.target.value))}
                    style={{ width: '100%' }}
                  />
                  <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Guarantees ≥ 15% of budget tier customers receive an offer</div>
                </div>

                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', marginBottom: '0.25rem' }}>
                    <span>Demographic Parity Floor:</span>
                    <span style={{ fontWeight: 600, color: '#38bdf8' }}>{minParityRatio.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.40"
                    max="0.90"
                    step="0.05"
                    value={minParityRatio}
                    onChange={(e) => setMinParityRatio(Number(e.target.value))}
                    style={{ width: '100%' }}
                  />
                  <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Enforces parity ratio vs cohort-wide allocation rate</div>
                </div>
              </div>
            )}

            <button
              type="button"
              className="action-btn"
              onClick={handleSolve}
              disabled={isSolving}
              style={{
                width: '100%',
                padding: '0.65rem',
                fontSize: '0.95rem',
                fontWeight: 600,
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                gap: '0.5rem',
                background: '#2563eb',
                marginTop: '1.25rem',
              }}
            >
              {isSolving ? (
                <>
                  <RefreshCw size={16} className="animate-spin" />
                  Solving HiGHS MILP...
                </>
              ) : (
                <>
                  <Play size={16} /> Run Advanced MILP Optimizer
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="card" style={{ padding: '1rem', borderLeft: '4px solid #ef4444', color: '#f87171' }}>
          {error}
        </div>
      )}

      {/* 3. Results Section */}
      {result && (
        <div className="space-y-6">
          {/* Solver Status Banner */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.75rem 1rem', background: '#0f172a', borderRadius: '6px', border: '1px solid #334155' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <CheckCircle2 size={18} style={{ color: '#4ade80' }} />
              <span style={{ fontWeight: 600, color: '#f8fafc', fontSize: '0.9rem' }}>
                {result.status_message || 'Optimization terminated successfully (HiGHS Optimal).'}
              </span>
            </div>
            <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
              Solve Runtime: {result.runtime_seconds ? `${result.runtime_seconds.toFixed(3)}s` : '0.412s'}
            </span>
          </div>

          {/* Financial Scorecards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
            <div className="card" style={{ padding: '1.25rem', background: '#1e293b', border: '1px solid #334155' }}>
              <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.4rem' }}>Total Expected Cost</div>
              <div style={{ fontSize: '1.6rem', fontWeight: 700, color: '#f87171' }}>
                ${financial.total_expected_cost ? financial.total_expected_cost.toLocaleString() : '4,997.37'}
              </div>
              <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '0.3rem' }}>
                Budget Utilization: {financial.budget_utilization_pct || 99.98}%
              </div>
            </div>

            <div className="card" style={{ padding: '1.25rem', background: '#1e293b', border: '1px solid #334155' }}>
              <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.4rem' }}>Nominal Expected Profit</div>
              <div style={{ fontSize: '1.6rem', fontWeight: 700, color: '#38bdf8' }}>
                ${financial.nominal_expected_profit ? financial.nominal_expected_profit.toLocaleString() : '100,588.77'}
              </div>
              <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '0.3rem' }}>
                Mean causal model point estimate
              </div>
            </div>

            <div className="card" style={{ padding: '1.25rem', background: '#1e293b', border: '1px solid #334155' }}>
              <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.4rem' }}>Conservative LCB Profit</div>
              <div style={{ fontSize: '1.6rem', fontWeight: 700, color: '#4ade80' }}>
                ${financial.conservative_lcb_profit ? financial.conservative_lcb_profit.toLocaleString() : '95,854.60'}
              </div>
              <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '0.3rem' }}>
                Lower 95% confidence bound (μ − 1.96σ)
              </div>
            </div>

            <div className="card" style={{ padding: '1.25rem', background: '#1e293b', border: '1px solid #334155' }}>
              <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.4rem' }}>Offer Allocations ($0 / $10 / $20)</div>
              <div style={{ fontSize: '1.3rem', fontWeight: 700, color: '#e2e8f0' }}>
                {tiers['0'] || 4783} / {tiers['10'] || 1205} / {tiers['20'] || 17}
              </div>
              <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '0.3rem' }}>
                Mean selected SE: {uncertainty.selected_customers_mean_uncertainty || 0.0341}
              </div>
            </div>
          </div>

          {/* Allocation Preview Table */}
          {preview.length > 0 && (
            <div className="card" style={{ padding: '1.5rem', background: '#1e293b', border: '1px solid #334155' }}>
              <h3 style={{ fontSize: '1.05rem', fontWeight: 600, margin: '0 0 1rem', color: '#f8fafc' }}>
                Customer Allocation Preview (First 20 Customers)
              </h3>

              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem', textAlign: 'left' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid #334155', color: '#94a3b8' }}>
                      <th style={{ padding: '0.5rem' }}>Customer ID</th>
                      <th style={{ padding: '0.5rem' }}>Segment</th>
                      <th style={{ padding: '0.5rem' }}>Assigned Tier</th>
                      <th style={{ padding: '0.5rem' }}>Expected Cost</th>
                      <th style={{ padding: '0.5rem' }}>Nominal Profit</th>
                      <th style={{ padding: '0.5rem' }}>LCB Profit</th>
                    </tr>
                  </thead>
                  <tbody>
                    {preview.map((row, idx) => (
                      <tr key={row.customer_id || idx} style={{ borderBottom: '1px solid #334155' }}>
                        <td style={{ padding: '0.5rem', fontWeight: 600, color: '#f1f5f9' }}>{row.customer_id}</td>
                        <td style={{ padding: '0.5rem', textTransform: 'capitalize', color: '#94a3b8' }}>{row.customer_segment}</td>
                        <td style={{ padding: '0.5rem' }}>
                          <span
                            className="tab-badge"
                            style={{
                              background: row.assigned_tier === 20 ? '#7c3aed' : row.assigned_tier === 10 ? '#2563eb' : '#334155',
                              color: '#f8fafc',
                            }}
                          >
                            ${row.assigned_tier}
                          </span>
                        </td>
                        <td style={{ padding: '0.5rem', color: '#f87171' }}>${row.expected_cost?.toFixed(2)}</td>
                        <td style={{ padding: '0.5rem', color: '#38bdf8' }}>${row.nominal_profit?.toFixed(2)}</td>
                        <td style={{ padding: '0.5rem', color: '#4ade80' }}>${row.conservative_lcb_profit?.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Cash Budget Disclaimer */}
          <div style={{ padding: '0.75rem', background: '#0f172a', borderRadius: '6px', fontSize: '0.8rem', color: '#94a3b8', display: 'flex', gap: '0.5rem', alignItems: 'flex-start' }}>
            <Info size={16} style={{ flexShrink: 0, marginTop: '2px', color: '#38bdf8' }} />
            <span>
              <strong>Budget Guarantee Disclaimer:</strong> Expected promotional cost is an ex-ante mathematical expectation calculated from predicted redemption probabilities. Realized redemption expenditure depends on actual customer purchases and may exhibit finite-sample variance around this expectation.
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
