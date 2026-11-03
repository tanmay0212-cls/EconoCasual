import React, { useState, useEffect, useCallback } from 'react';
import Papa from 'papaparse';
import { SlidersHorizontal, BarChart3, Database } from 'lucide-react';
import Header from './components/Header';
import MetricsCards from './components/MetricsCards';
import DatasetUploader from './components/DatasetUploader';
import BudgetSettings from './components/BudgetSettings';
import CausalDAGViewer from './components/CausalDAGViewer';
import MarketingTelemetry from './components/MarketingTelemetry';
import FutureDeliverables from './components/FutureDeliverables';
import AllocationMatrix from './components/AllocationMatrix';
import { validateDatasetRows } from './utils/csvValidator';

export default function App() {
  const [activeTab, setActiveTab] = useState('matrix'); // 'matrix' | 'telemetry'
  const [datasetSource, setDatasetSource] = useState('none'); // 'synthetic' | 'uploaded' | 'none'
  const [datasetName, setDatasetName] = useState('');
  const [validationResult, setValidationResult] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  // Budget settings state
  const [budget, setBudget] = useState(5000);
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
  const handleFileUpload = (file) => {
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
            className={`nav-tab-btn ${activeTab === 'matrix' ? 'active' : ''}`}
            onClick={() => setActiveTab('matrix')}
          >
            <SlidersHorizontal size={17} />
            <span>Prescriptive Allocation Matrix</span>
            <span className="tab-badge tab-badge-highlight">Week 3</span>
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
        </nav>
      </div>

      <main className="main-content">
        {/* TAB 1: WEEK 3 PRESCRIPTIVE ALLOCATION MATRIX */}
        {activeTab === 'matrix' && (
          <AllocationMatrix initialBudget={budget} />
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
              currency={currency}
              onCurrencyChange={setCurrency}
              discountTiers={discountTiers}
              onDiscountTiersChange={setDiscountTiers}
              totalCustomers={totalRows}
            />

            {/* Causal Directed Acyclic Graph (DAG) Specification */}
            <CausalDAGViewer />

            {/* Exploratory Marketing Telemetry (Plotly Charts) */}
            <MarketingTelemetry
              previewRows={previewRows}
              metrics={metrics}
              totalRows={totalRows}
            />

            {/* Reserved / Completed Project Deliverables */}
            <FutureDeliverables onNavigateToMatrix={() => setActiveTab('matrix')} />
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
