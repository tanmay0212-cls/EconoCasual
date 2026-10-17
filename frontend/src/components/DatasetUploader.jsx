import React, { useRef, useState } from 'react';
import { UploadCloud, FileSpreadsheet, AlertCircle, CheckCircle, HelpCircle, Download, RefreshCw, Eye } from 'lucide-react';
import { VARIABLE_ROLES, REQUIRED_COLUMNS } from '../utils/csvValidator';

export default function DatasetUploader({
  validationResult,
  onFileUpload,
  onLoadSynthetic,
  onDownloadSample,
  isLoading,
  fileName,
}) {
  const fileInputRef = useRef(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [showMissingDetails, setShowMissingDetails] = useState(false);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      onFileUpload(e.target.files[0]);
      e.target.value = ''; // Reset file input so re-uploading same file triggers change
    }
  };

  const {
    isValid = false,
    totalRows = 0,
    validRows = 0,
    columns = [],
    missingValues = {},
    errors = [],
    warnings = [],
    preview = [],
  } = validationResult || {};

  const totalMissingValues = Object.values(missingValues).reduce((a, b) => a + b, 0);

  return (
    <div className="card uploader-card">
      <div className="card-header">
        <div>
          <h2 className="card-title">Historical Campaign Dataset Ingestion</h2>
          <p className="card-subtitle">
            Upload retail campaign observations or explore the pre-packaged reproducible synthetic dataset.
            Data is parsed entirely client-side in your browser (no external API transmission).
          </p>
        </div>
      </div>

      {/* Drag & Drop Upload Zone */}
      <div
        className={`dropzone ${isDragOver ? 'dropzone-active' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept=".csv"
          style={{ display: 'none' }}
        />
        <div className="dropzone-content">
          <div className="dropzone-icon-wrap">
            <UploadCloud size={36} className="dropzone-icon" />
          </div>
          <p className="dropzone-title">
            Drag & drop your campaign CSV file here, or <span className="highlight-text">browse files</span>
          </p>
          <p className="dropzone-caption">
            Accepted format: <code>.csv</code> with required causal columns (minimum 10 rows).
          </p>
          <div className="dropzone-quick-actions" onClick={(e) => e.stopPropagation()}>
            <button
              type="button"
              className="btn btn-sm btn-outline"
              onClick={onLoadSynthetic}
              disabled={isLoading}
            >
              <RefreshCw size={13} className={isLoading ? 'spinning' : ''} />
              <span>Load Included Synthetic Dataset (2,500 rows)</span>
            </button>
            <button
              type="button"
              className="btn btn-sm btn-outline"
              onClick={onDownloadSample}
            >
              <Download size={13} />
              <span>Download Template CSV</span>
            </button>
          </div>
        </div>
      </div>

      {/* Validation Status & Error Reporting */}
      {errors.length > 0 && (
        <div className="alert alert-error">
          <div className="alert-header">
            <AlertCircle size={18} />
            <strong>Dataset Schema Validation Failed</strong>
          </div>
          <ul className="alert-list">
            {errors.map((err, i) => (
              <li key={i}>{err}</li>
            ))}
          </ul>
        </div>
      )}

      {warnings.length > 0 && (
        <div className="alert alert-warning">
          <div className="alert-header">
            <HelpCircle size={18} />
            <strong>Dataset Quality Notice</strong>
          </div>
          <ul className="alert-list">
            {warnings.map((warn, i) => (
              <li key={i}>{warn}</li>
            ))}
          </ul>
        </div>
      )}

      {isValid && (
        <div className="alert alert-success">
          <div className="alert-header">
            <CheckCircle size={18} />
            <strong>Schema Validated Successfully</strong>
          </div>
          <p>
            All required treatment, outcome, and confounder columns were recognized and satisfy causal integrity rules.
          </p>
        </div>
      )}

      {/* Dataset Summary Telemetry */}
      {totalRows > 0 && (
        <div className="dataset-telemetry-panel">
          <div className="telemetry-bar">
            <div className="telemetry-item">
              <span className="telemetry-label">Active File</span>
              <span className="telemetry-val file-badge">{fileName || 'synthetic_retail_campaign.csv'}</span>
            </div>
            <div className="telemetry-item">
              <span className="telemetry-label">Total Observations</span>
              <span className="telemetry-val">{totalRows.toLocaleString()} rows</span>
            </div>
            <div className="telemetry-item">
              <span className="telemetry-label">Causal Columns</span>
              <span className="telemetry-val">{columns.length} columns detected</span>
            </div>
            <div className="telemetry-item">
              <span className="telemetry-label">Missing Values</span>
              <span className={`telemetry-val ${totalMissingValues > 0 ? 'text-amber' : 'text-emerald'}`}>
                {totalMissingValues === 0 ? '0 (Clean)' : `${totalMissingValues} missing`}
              </span>
            </div>
            <button
              type="button"
              className="btn btn-xs btn-subtle"
              onClick={() => setShowMissingDetails(!showMissingDetails)}
            >
              <Eye size={12} />
              <span>{showMissingDetails ? 'Hide Missing Value Counts' : 'Inspect Column Missing Counts'}</span>
            </button>
          </div>

          {/* Missing value counts per column */}
          {showMissingDetails && (
            <div className="missing-counts-grid">
              {REQUIRED_COLUMNS.map((col) => {
                const count = missingValues[col] || 0;
                const role = VARIABLE_ROLES[col];
                return (
                  <div key={col} className="missing-card">
                    <div className="missing-col-header">
                      <span className="col-name">{col}</span>
                      <span className="col-role-badge" style={{ backgroundColor: `${role?.color}20`, color: role?.color }}>
                        {role?.label || 'Feature'}
                      </span>
                    </div>
                    <div className="missing-count-row">
                      <span>Missing:</span>
                      <span className={count > 0 ? 'count-bad' : 'count-good'}>
                        {count.toLocaleString()} ({totalRows > 0 ? ((count / totalRows) * 100).toFixed(1) : 0}%)
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Preview Table of First 10 Rows */}
      {preview.length > 0 && (
        <div className="preview-container">
          <div className="preview-header">
            <div className="preview-title-row">
              <FileSpreadsheet size={16} className="text-cyan" />
              <h3 className="preview-title">Dataset Preview (First 10 Observations)</h3>
            </div>
            <span className="preview-caption">Color-coded by causal taxonomy</span>
          </div>

          <div className="table-wrapper">
            <table className="preview-table">
              <thead>
                <tr>
                  {REQUIRED_COLUMNS.map((col) => {
                    const role = VARIABLE_ROLES[col];
                    return (
                      <th key={col} className="table-th">
                        <div className="th-content">
                          <span className="th-name">{col}</span>
                          <span
                            className="th-role"
                            style={{ backgroundColor: `${role?.color}25`, color: role?.color, borderColor: `${role?.color}50` }}
                          >
                            {role?.label}
                          </span>
                        </div>
                      </th>
                    );
                  })}
                </tr>
              </thead>
              <tbody>
                {preview.map((row, idx) => (
                  <tr key={idx} className="table-tr">
                    {REQUIRED_COLUMNS.map((col) => (
                      <td key={col} className="table-td">
                        {col === 'discount_offered' || col === 'purchased' ? (
                          <span className={`binary-badge ${Number(row[col]) === 1 ? 'binary-true' : 'binary-false'}`}>
                            {row[col]}
                          </span>
                        ) : col === 'prior_spend' || col === 'revenue' || col === 'discount_amount' ? (
                          `$${Number(row[col]).toFixed(2)}`
                        ) : (
                          row[col] ?? '—'
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
