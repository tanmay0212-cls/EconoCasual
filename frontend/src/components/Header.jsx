import React from 'react';
import { Database, Download, RefreshCw, Sparkles, FileText } from 'lucide-react';

export default function Header({
  datasetSource, // 'synthetic' | 'uploaded' | 'none'
  datasetName,
  totalRows,
  onLoadSynthetic,
  onDownloadSample,
  isLoading,
}) {
  return (
    <header className="header-container">
      <div className="header-branding">
        <div className="logo-badge">
          <Sparkles className="logo-icon" size={24} />
        </div>
        <div>
          <div className="title-row">
            <h1 className="header-title">EconoCausal</h1>
            <span className="internship-tag">Infotact Solutions • Week 1</span>
          </div>
          <p className="header-subtitle">Causal AI for Dynamic Pricing</p>
        </div>
      </div>

      <div className="header-actions">
        {/* Dataset Status Indicator */}
        <div className="status-indicator-card">
          <span className="status-label">Active Dataset:</span>
          {datasetSource === 'synthetic' ? (
            <div className="status-pill status-synthetic">
              <span className="status-dot dot-synthetic" />
              <Database size={14} className="pill-icon" />
              <span>Synthetic Sample Campaign (N={totalRows.toLocaleString()})</span>
            </div>
          ) : datasetSource === 'uploaded' ? (
            <div className="status-pill status-uploaded">
              <span className="status-dot dot-uploaded" />
              <FileText size={14} className="pill-icon" />
              <span>Uploaded: {datasetName} ({totalRows.toLocaleString()} rows)</span>
            </div>
          ) : (
            <div className="status-pill status-empty">
              <span className="status-dot dot-empty" />
              <span>No Dataset Active</span>
            </div>
          )}
        </div>

        {/* Action Buttons */}
        <div className="button-group">
          <button
            type="button"
            className="btn btn-secondary"
            onClick={onLoadSynthetic}
            disabled={isLoading}
            title="Reload included reproducible synthetic campaign dataset (N=2,500)"
          >
            <RefreshCw size={15} className={isLoading ? 'spinning' : ''} />
            <span>Load Sample Data</span>
          </button>

          <button
            type="button"
            className="btn btn-primary"
            onClick={onDownloadSample}
            title="Download synthetic retail marketing CSV file to your local computer"
          >
            <Download size={15} />
            <span>Download Sample CSV</span>
          </button>
        </div>
      </div>
    </header>
  );
}
