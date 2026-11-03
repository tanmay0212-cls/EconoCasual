import React, { useState, useEffect, useMemo } from 'react';
import {
  FileText,
  DollarSign,
  TrendingUp,
  ShieldAlert,
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  Info,
  Clock,
  Hash,
  SlidersHorizontal,
  HelpCircle,
  Layers,
  ArrowRight,
} from 'lucide-react';
import { fetchOptimizationData } from '../services/optimizationService';
import { fetchDriftReport } from '../services/driftService';
import { generateExecutiveSummary } from '../services/summaryService';

export default function BusinessSummaryPanel({
  optimizationData: propOptimizationData = undefined,
  driftData: propDriftData = undefined,
  onNavigateToMatrix = null,
  onNavigateToDrift = null,
}) {
  const isControlled = propOptimizationData !== undefined || propDriftData !== undefined;
  const [optData, setOptData] = useState(propOptimizationData || null);
  const [driftPayload, setDriftPayload] = useState(propDriftData || null);
  const [isLoading, setIsLoading] = useState(!isControlled);
  const [expandedSectionIds, setExpandedSectionIds] = useState(new Set(['optimization_recommendation', 'budget_utilization']));
  const [error, setError] = useState(null);

  // Sync prop changes if controlled
  useEffect(() => {
    if (propOptimizationData !== undefined) setOptData(propOptimizationData);
    if (propDriftData !== undefined) setDriftPayload(propDriftData);
  }, [propOptimizationData, propDriftData]);

  // Load backend datasets if uncontrolled
  const loadData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [optRes, driftRes] = await Promise.all([
        propOptimizationData !== undefined ? Promise.resolve(propOptimizationData) : fetchOptimizationData().catch((err) => {
          console.warn('Could not load optimization data for summary:', err);
          return null;
        }),
        propDriftData !== undefined ? Promise.resolve(propDriftData) : fetchDriftReport('shifted').catch((err) => {
          console.warn('Could not load drift data for summary:', err);
          return null;
        }),
      ]);

      setOptData(optRes);
      setDriftPayload(driftRes);
    } catch (err) {
      setError(err.message || 'Failed to assemble executive summary metrics.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (!isControlled) {
      loadData();
    }
  }, [isControlled]);

  // Generate deterministic natural-language executive summary
  const summaryReport = useMemo(() => {
    return generateExecutiveSummary(optData || {}, driftPayload || {});
  }, [optData, driftPayload]);

  const toggleSection = (sectionId) => {
    setExpandedSectionIds((prev) => {
      const next = new Set(prev);
      if (next.has(sectionId)) {
        next.delete(sectionId);
      } else {
        next.add(sectionId);
      }
      return next;
    });
  };

  const getSectionIcon = (id) => {
    switch (id) {
      case 'optimization_recommendation':
        return <SlidersHorizontal size={18} className="text-cyan" />;
      case 'budget_utilization':
        return <DollarSign size={18} className="text-emerald" />;
      case 'customer_allocation':
        return <Layers size={18} className="text-purple" />;
      case 'revenue_profit_comparison':
        return <TrendingUp size={18} className="text-emerald" />;
      case 'drift_warning':
        return <ShieldAlert size={18} className="text-amber" />;
      case 'recommended_action':
        return <CheckCircle2 size={18} className="text-cyan" />;
      default:
        return <FileText size={18} className="text-muted" />;
    }
  };

  const getBadgeClass = (type) => {
    switch (type) {
      case 'danger':
        return 'badge-danger';
      case 'warning':
        return 'badge-warning';
      case 'success':
        return 'badge-success';
      case 'info':
      default:
        return 'badge-primary';
    }
  };

  return (
    <div className="summary-panel-container">
      {/* 1. Header Card with Run Metadata */}
      <div className="card summary-header-card">
        <div className="card-header-flex">
          <div className="title-with-icon">
            <div className="icon-badge icon-emerald">
              <FileText size={22} />
            </div>
            <div>
              <div className="summary-title-row">
                <h2 className="card-title">Executive Decision Briefing & Natural-Language Synthesis</h2>
                <span className="badge badge-success">Week 4 Deliverable</span>
              </div>
              <p className="card-subtitle">
                Deterministic business translation of causal pricing optimization and covariate drift telemetry for the Marketing Director.
              </p>
            </div>
          </div>

          <div className="summary-meta-actions">
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={loadData}
              disabled={isLoading}
              title="Refresh Briefing"
            >
              <RefreshCw size={14} className={isLoading ? 'spin-icon' : ''} />
              <span>Refresh Briefing</span>
            </button>
          </div>
        </div>

        {/* Run Metadata Strip */}
        <div className="summary-meta-strip">
          <div className="meta-item">
            <Hash size={14} className="text-cyan" />
            <span className="meta-label">Optimization Run ID:</span>
            <code className="meta-value">{summaryReport.runId}</code>
          </div>
          <div className="meta-item">
            <Clock size={14} className="text-muted" />
            <span className="meta-label">Generated Timestamp:</span>
            <span className="meta-value-text">{new Date(summaryReport.timestamp).toLocaleString()}</span>
          </div>
          <div className="meta-item">
            <Info size={14} className="text-emerald" />
            <span className="meta-label">Synthesis Engine:</span>
            <span className="meta-value-text">Deterministic Causal Accounting (No Paid LLM)</span>
          </div>
        </div>

        {/* Executive Takeaway Banner */}
        <div className="executive-takeaway-banner">
          <div className="takeaway-label">Key Executive Takeaway:</div>
          <p className="takeaway-text">{summaryReport.overallTakeaway}</p>
        </div>
      </div>

      {isLoading && (
        <div className="matrix-loading-state">
          <RefreshCw size={36} className="spin-icon text-cyan" />
          <p className="loading-text">Synthesizing executive briefing from causal optimization artifacts...</p>
        </div>
      )}

      {error && (
        <div className="alert alert-danger" style={{ margin: '1rem 0' }}>
          <AlertTriangle size={18} />
          <div>
            <strong>Error assembling executive summary:</strong> {error}
          </div>
        </div>
      )}

      {!isLoading && (
        <>
          {/* 2. Structured Sections Grid */}
          <div className="summary-sections-grid">
            {summaryReport.sections.map((section) => {
              const isExpanded = expandedSectionIds.has(section.id);
              return (
                <div key={section.id} className="card summary-section-card">
                  <div className="card-header summary-section-header">
                    <div className="title-with-icon">
                      <div className="icon-badge icon-neutral">{getSectionIcon(section.id)}</div>
                      <div>
                        <div className="section-title-wrap">
                          <h3 className="card-title">{section.title}</h3>
                          {section.badge && (
                            <span className={`badge ${getBadgeClass(section.badge.type)}`}>
                              {section.badge.label}
                            </span>
                          )}
                        </div>
                        <h4 className="section-headline">{section.headline}</h4>
                      </div>
                    </div>
                  </div>

                  <div className="summary-section-body">
                    {section.body.map((paragraph, pIdx) => (
                      <p key={pIdx} className="summary-paragraph">
                        {paragraph}
                      </p>
                    ))}

                    {/* Inspect Supporting Metrics Toggle */}
                    <div className="supporting-metrics-accordion">
                      <button
                        type="button"
                        className="btn-toggle-metrics"
                        onClick={() => toggleSection(section.id)}
                        aria-expanded={isExpanded}
                      >
                        <span>Inspect Supporting Metrics ({section.underlyingMetrics.length})</span>
                        {isExpanded ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
                      </button>

                      {isExpanded && (
                        <div className="metrics-drawer">
                          {section.underlyingMetrics.length > 0 ? (
                            <div className="metrics-drawer-grid">
                              {section.underlyingMetrics.map((m, mIdx) => (
                                <div key={mIdx} className="drawer-metric-item">
                                  <span className="drawer-metric-label">{m.label}</span>
                                  <strong className="drawer-metric-val">{m.value}</strong>
                                  {m.description && (
                                    <span className="drawer-metric-desc">{m.description}</span>
                                  )}
                                </div>
                              ))}
                            </div>
                          ) : (
                            <div className="drawer-empty-notice">
                              Underlying metrics unavailable for this section.
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* 3. Deep-Dive Links & Quick Navigation */}
          <div className="card summary-nav-shortcuts">
            <div className="card-header">
              <h4 className="card-title">Detailed Exploration Consoles</h4>
              <p className="card-subtitle">
                Access deep interactive customer records and granular statistical testing distributions.
              </p>
            </div>
            <div className="summary-shortcut-buttons">
              {onNavigateToMatrix && (
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={onNavigateToMatrix}
                >
                  <SlidersHorizontal size={16} />
                  <span>Inspect 3,000-Customer Allocation Matrix</span>
                  <ArrowRight size={14} />
                </button>
              )}
              {onNavigateToDrift && (
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={onNavigateToDrift}
                >
                  <ShieldAlert size={16} />
                  <span>Inspect Data Drift Histograms & Tests</span>
                  <ArrowRight size={14} />
                </button>
              )}
            </div>
          </div>

          {/* 4. Methodological Disclaimer Box */}
          <div className="card methodological-callout-card">
            <div className="card-header">
              <div className="title-with-icon">
                <div className="icon-badge icon-amber">
                  <Info size={18} />
                </div>
                <div>
                  <h4 className="card-title">Causal Accounting & Counterfactual Projections Notice</h4>
                  <p className="card-subtitle">
                    Standard governance principles for marketing directors reviewing model-derived expectations.
                  </p>
                </div>
              </div>
            </div>
            <div className="card-content-padded">
              <p className="methodological-body-text">
                {summaryReport.methodologicalNotice}
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
