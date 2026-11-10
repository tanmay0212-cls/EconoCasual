import React, { useState, useEffect, useCallback } from 'react';
import Papa from 'papaparse';
import { SlidersHorizontal, BarChart3, Database, Activity, FileText, ShieldCheck, Scale, Sliders, Layers } from 'lucide-react';
import Header from './components/Header';
import MetricsCards from './components/MetricsCards';
import DatasetUploader from './components/DatasetUploader';
import BudgetSettings from './components/BudgetSettings';
import CausalDAGViewer from './components/CausalDAGViewer';
import FutureDeliverables from './components/FutureDeliverables';
import AllocationMatrix from './components/AllocationMatrix';
import DataDriftViewer from './components/DataDriftViewer';
import BusinessSummaryPanel from './components/BusinessSummaryPanel';
import PolicyEvaluationTab from './components/PolicyEvaluationTab';
import UncertaintyAnalysisTab from './components/UncertaintyAnalysisTab';
import FairnessAnalysisTab from './components/FairnessAnalysisTab';
import AdvancedOptimizationTab from './components/AdvancedOptimizationTab';
import { validateDatasetRows } from './utils/csvValidator';

// Lazy-load Plotly-heavy MarketingTelemetry to prevent eager loading on initial page render
const MarketingTelemetry = React.lazy(() => import('./components/MarketingTelemetry'));

export default function App() {
  const [activeTab, setActiveTab] = useState('summary'); // 'summary' | 'matrix' | 'telemetry' | 'drift'
  const [datasetSource, setDatasetSource] = useState('none'); // 'synthetic' | 'uploaded' | 'none'
  const [datasetName, setDatasetName] = useState('');
  const [validationResult, setValidationResult] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  // Budget settings state
  const [budget, setBudget] = useState(5000);
  const [budgetType, setBudgetType] = useState('redemption');
  const [currency, setCurrency] = useState('USD');
  const [discountTiers, setDiscountTiers] = useState([10, 20]);

  // Load included reproducible synthetic dataset
  const loadSyntheticData = useCallback(async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/sample_retail_campaign.csv');
      if (!response.ok) {
        throw new Error(`Failed to load sample CSV: status ${response.status}`);
      }
      const csvText = await response.text();

      Papa.parse(csvText, {
        header: true,
        dynamicTyping: false,
        skipEmptyLines: true,
        complete: (results) => {
          const validated = validateDatasetRows(results.data);
          setValidationResult(validated);
          setDatasetSource('synthetic');
          setDatasetName('synthetic_retail_campaign.csv');
          setIsLoading(false);
        },
        error: (error) => {
          console.error('CSV parse error:', error);
          setIsLoading(false);
        },
      });
    } catch (err) {
      console.error('Error fetching sample CSV:', err);
      setIsLoading(false);
    }
  }, []);

  // Initial load
  useEffect(() => {
    loadSyntheticData();
  }, [loadSyntheticData]);

  // Handle user uploaded CSV
  const handleFileUpload = async (file) => {
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.csv')) {
      setValidationResult({
        isValid: false,
        totalRows: 0,
        validRows: 0,
        columns: [],
        missingValues: {},
        errors: [`Invalid file format: "${file.name}". Only .csv files are supported. Please upload a comma-separated values (.csv) file.`],
        warnings: [],
        preview: [],
        metrics: null,
      });
      setDatasetSource('uploaded');
      setDatasetName(file.name);
      return;
    }

    setIsLoading(true);

    // 1. Attempt upload to FastAPI backend POST /api/upload
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData,
      });
      if (res.ok) {
        const uploadData = await res.json();
        setValidationResult({
          isValid: uploadData.validation_passed,
          totalRows: uploadData.total_rows,
          validRows: uploadData.valid_rows,
          columns: uploadData.columns_detected,
          missingValues: uploadData.missing_values_by_column || {},
          errors: uploadData.errors || [],
          warnings: uploadData.warnings || [],
          preview: uploadData.preview || [],
          metrics: null,
          quarantinedColumns: uploadData.quarantined_columns || [],
          isLiveApi: true,
        });
        setDatasetSource('uploaded');
        setDatasetName(file.name);
        setIsLoading(false);
        return;
      }
    } catch {
      // Backend unreachable or offline; fall back to client-side validation
    }

    // 2. Client-side PapaParse fallback
    Papa.parse(file, {
      header: true,
      dynamicTyping: false,
      skipEmptyLines: true,
      complete: (results) => {
        const validated = validateDatasetRows(results.data);
        setValidationResult(validated);
        setDatasetSource('uploaded');
        setDatasetName(file.name);
        setIsLoading(false);
      },
      error: (error) => {
        setValidationResult({
          isValid: false,
          totalRows: 0,
          validRows: 0,
          columns: [],
          missingValues: {},
          errors: [`Failed to parse CSV file: ${error.message}`],
          warnings: [],
          preview: [],
          metrics: null,
        });
        setDatasetSource('uploaded');
        setDatasetName(file.name);
        setIsLoading(false);
      },
    });
  };

  // Trigger sample CSV download
  const handleDownloadSample = async () => {
    try {
      const response = await fetch('/sample_retail_campaign.csv');
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'sample_retail_campaign.csv');
      document.body.appendChild(link);
      link.click();
      link.parentNode.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      alert('Could not download sample CSV: ' + err.message);
    }
  };

  const totalRows = validationResult?.totalRows || 0;
  const validRows = validationResult?.validRows || 0;
  const isValid = validationResult?.isValid || false;
  const metrics = validationResult?.metrics || null;
  const previewRows = validationResult?.preview || [];

  return (
    <div className="app-container">
      {/* 1. Header with Live Status Indicator */}
      <Header
        datasetSource={datasetSource}
        datasetName={datasetName}
        totalRows={totalRows}
        onLoadSynthetic={loadSyntheticData}
        onDownloadSample={handleDownloadSample}
        isLoading={isLoading}
      />

      {/* 2. Primary Navigation Bar */}
      <div className="tab-navigation-container">
        <nav className="tab-navigation-bar" aria-label="Main Navigation">
          <button
            type="button"
            className={`nav-tab-btn ${activeTab === 'summary' ? 'active' : ''}`}
            onClick={() => setActiveTab('summary')}
          >
            <FileText size={17} />
            <span>Executive Summary</span>
            <span className="tab-badge tab-badge-highlight">Week 4</span>
          </button>

          <button
            type="button"
            className={`nav-tab-btn ${activeTab === 'matrix' ? 'active' : ''}`}
            onClick={() => setActiveTab('matrix')}
          >
            <SlidersHorizontal size={17} />
            <span>Prescriptive Allocation Matrix</span>
            <span className="tab-badge tab-badge-highlight">Week 3</span>
          </button>

          <button
            type="button"
            className={`nav-tab-btn ${activeTab === 'drift' ? 'active' : ''}`}
            onClick={() => setActiveTab('drift')}
          >
            <Activity size={17} />
            <span>Data Drift & Monitoring</span>
            <span className="tab-badge tab-badge-highlight">Week 4</span>
          </button>

          <button
            type="button"
            className={`nav-tab-btn ${activeTab === 'telemetry' ? 'active' : ''}`}
            onClick={() => setActiveTab('telemetry')}
          >
            <BarChart3 size={17} />
            <span>Causal DAG & Telemetry</span>
            <span className="tab-badge">Week 1 & 2</span>
          </button>

          <button
            type="button"
            className={`nav-tab-btn ${activeTab === 'policy_eval' ? 'active' : ''}`}
            onClick={() => setActiveTab('policy_eval')}
          >
            <ShieldCheck size={17} />
            <span>Policy Evaluation</span>
            <span className="tab-badge tab-badge-highlight">Phase 1</span>
          </button>

          <button
            type="button"
            className={`nav-tab-btn ${activeTab === 'uncertainty' ? 'active' : ''}`}
            onClick={() => setActiveTab('uncertainty')}
          >
            <Layers size={17} />
            <span>Uncertainty Analysis</span>
            <span className="tab-badge tab-badge-highlight">Phase 2</span>
          </button>

          <button
            type="button"
            className={`nav-tab-btn ${activeTab === 'fairness' ? 'active' : ''}`}
            onClick={() => setActiveTab('fairness')}
          >
            <Scale size={17} />
            <span>Fairness Diagnostics</span>
            <span className="tab-badge tab-badge-highlight">Phase 3</span>
          </button>

          <button
            type="button"
            className={`nav-tab-btn ${activeTab === 'advanced_opt' ? 'active' : ''}`}
            onClick={() => setActiveTab('advanced_opt')}
          >
            <Sliders size={17} />
            <span>Advanced MILP Optimizer</span>
            <span className="tab-badge tab-badge-highlight">Phase 3/4</span>
          </button>
        </nav>
      </div>

      <main className="main-content">
        {/* TAB 1: WEEK 4 NATURAL-LANGUAGE EXECUTIVE SUMMARY */}
        {activeTab === 'summary' && (
          <BusinessSummaryPanel
            onNavigateToMatrix={() => setActiveTab('matrix')}
            onNavigateToDrift={() => setActiveTab('drift')}
          />
        )}

        {/* TAB 2: WEEK 3 PRESCRIPTIVE ALLOCATION MATRIX */}
        {activeTab === 'matrix' && (
          <AllocationMatrix initialBudget={budget} budgetType={budgetType} />
        )}

        {/* TAB 3: WEEK 4 AUTOMATED DATA DRIFT & COVARIATE SHIFT MONITORING */}
        {activeTab === 'drift' && (
          <DataDriftViewer />
        )}

        {/* TAB 4: PHASE 1 DOUBLY ROBUST POLICY EVALUATION */}
        {activeTab === 'policy_eval' && (
          <PolicyEvaluationTab />
        )}

        {/* TAB 5: PHASE 2 UNCERTAINTY QUANTIFICATION & CI COVERAGE */}
        {activeTab === 'uncertainty' && (
          <UncertaintyAnalysisTab />
        )}

        {/* TAB 6: PHASE 3 CUSTOMER SEGMENT FAIRNESS & DEMOGRAPHIC PARITY */}
        {activeTab === 'fairness' && (
          <FairnessAnalysisTab />
        )}

        {/* TAB 7: PHASE 3 & 4 UNCERTAINTY-AWARE & FAIRNESS-CONSTRAINED MILP */}
        {activeTab === 'advanced_opt' && (
          <AdvancedOptimizationTab />
        )}

        {/* TAB 2: WEEK 1 & 2 EXPLORATORY TELEMETRY & CAUSAL DAG */}
        {activeTab === 'telemetry' && (
          <>
            {/* Top Telemetry Metrics Cards */}
            <MetricsCards
              metrics={metrics}
              budget={budget}
              currency={currency}
              isValid={isValid}
              totalRows={totalRows}
              validRows={validRows}
            />

            {/* Dataset Upload & Causal Schema Preview */}
            <DatasetUploader
              validationResult={validationResult}
              onFileUpload={handleFileUpload}
              onLoadSynthetic={loadSyntheticData}
              onDownloadSample={handleDownloadSample}
              isLoading={isLoading}
              fileName={datasetName}
            />

            {/* Marketing Budget & Pricing Bounds Settings */}
            <BudgetSettings
              budget={budget}
              onBudgetChange={setBudget}
              budgetType={budgetType}
              onBudgetTypeChange={setBudgetType}
              currency={currency}
              onCurrencyChange={setCurrency}
              discountTiers={discountTiers}
              onDiscountTiersChange={setDiscountTiers}
              totalCustomers={totalRows}
            />

            {/* Causal Directed Acyclic Graph (DAG) Specification */}
            <CausalDAGViewer />

            {/* Exploratory Marketing Telemetry (Plotly Charts) - Lazy Loaded */}
            <React.Suspense
              fallback={
                <div className="card telemetry-card" style={{ padding: '2rem', textAlign: 'center', color: '#94a3b8' }}>
                  <p>Loading interactive Plotly telemetry visualizations...</p>
                </div>
              }
            >
              <MarketingTelemetry
                previewRows={previewRows}
                metrics={metrics}
                totalRows={totalRows}
              />
            </React.Suspense>

            {/* Reserved / Completed Project Deliverables */}
            <FutureDeliverables
              onNavigateToMatrix={() => setActiveTab('matrix')}
              onNavigateToDrift={() => setActiveTab('drift')}
            />
          </>
        )}
      </main>

      {/* Footer with Methodological Disclaimers */}
      <footer className="footer-container">
        <div className="footer-content">
          <p className="footer-title">
            EconoCausal Platform • Infotact Solutions Internship Project (Weeks 1 - 3 Complete)
          </p>
          <p className="footer-disclaimer">
            Methodological Notice: Causal DAGs and back-door adjustment sets formalize structural domain assumptions.
            Causal treatment effects are estimated via Double Machine Learning (EconML LinearDRLearner).
            Prescriptive discount allocations are optimized via SciPy Mixed-Integer Linear Programming (MILP) under marketing budget constraints.
          </p>
        </div>
      </footer>
    </div>
  );
}
