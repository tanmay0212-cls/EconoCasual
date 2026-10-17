import React from 'react';
import { Users, CheckCircle2, DollarSign, TrendingUp, AlertTriangle } from 'lucide-react';

export default function MetricsCards({
  metrics,
  budget,
  currency,
  isValid,
  totalRows,
  validRows,
}) {
  const formatCurrency = (val) => {
    const num = Number(val);
    if (isNaN(num)) return `${currency}0`;
    return `${currency}${num.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;
  };

  const validPercentage = totalRows > 0 ? Math.round((validRows / totalRows) * 100) : 0;

  return (
    <div className="metrics-grid">
      {/* 1. Total Customers */}
      <div className="metric-card">
        <div className="metric-header">
          <span className="metric-title">Total Customers</span>
          <div className="metric-icon-wrap icon-blue">
            <Users size={18} />
          </div>
        </div>
        <div className="metric-value">
          {totalRows > 0 ? totalRows.toLocaleString() : '0'}
        </div>
        <div className="metric-caption">
          <span>Observed records in active dataset</span>
        </div>
      </div>

      {/* 2. Valid Records */}
      <div className="metric-card">
        <div className="metric-header">
          <span className="metric-title">Valid Records</span>
          <div className={`metric-icon-wrap ${isValid ? 'icon-emerald' : 'icon-amber'}`}>
            {isValid ? <CheckCircle2 size={18} /> : <AlertTriangle size={18} />}
          </div>
        </div>
        <div className="metric-value">
          {validRows > 0 ? validRows.toLocaleString() : '0'}
        </div>
        <div className="metric-caption">
          <span className={isValid ? 'text-emerald' : 'text-amber'}>
            {validPercentage}% pass causal schema validation
          </span>
        </div>
      </div>

      {/* 3. Configured Marketing Budget */}
      <div className="metric-card">
        <div className="metric-header">
          <span className="metric-title">Marketing Budget</span>
          <div className="metric-icon-wrap icon-purple">
            <DollarSign size={18} />
          </div>
        </div>
        <div className="metric-value">
          {formatCurrency(budget)}
        </div>
        <div className="metric-caption">
          <span>Configured spend cap ({currency})</span>
        </div>
      </div>

      {/* 4. Observed Baseline Conversion */}
      <div className="metric-card">
        <div className="metric-header">
          <span className="metric-title">Historical Conversion</span>
          <div className="metric-icon-wrap icon-cyan">
            <TrendingUp size={18} />
          </div>
        </div>
        <div className="metric-value">
          {metrics ? `${metrics.conversionRate.toFixed(1)}%` : '—'}
        </div>
        <div className="metric-caption text-subtle">
          <span>Observational baseline (pre-DML estimation)</span>
        </div>
      </div>
    </div>
  );
}
