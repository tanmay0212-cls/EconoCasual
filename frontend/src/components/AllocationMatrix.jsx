import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
  DollarSign,
  TrendingUp,
  Users,
  Award,
  Wallet,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Search,
  Filter,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
  Info,
  ShieldAlert,
  Sparkles,
  SlidersHorizontal,
  Layers,
} from 'lucide-react';
import { fetchOptimizationData, DEFAULT_BUDGET } from '../services/optimizationService';

export default function AllocationMatrix({ initialBudget = DEFAULT_BUDGET }) {
  // Input and Optimization State
  const [budgetInput, setBudgetInput] = useState(initialBudget);
  const [activeBudget, setActiveBudget] = useState(initialBudget);
  const [budgetError, setBudgetError] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Data State
  const [records, setRecords] = useState([]);
  const [kpis, setKpis] = useState(null);
  const [strategies, setStrategies] = useState([]);
  const [metadata, setMetadata] = useState(null);

  // Table Filter and Sort State
  const [searchQuery, setSearchQuery] = useState('');
  const [discountFilter, setDiscountFilter] = useState('all'); // 'all' | '0' | '10' | '20'
  const [sortField, setSortField] = useState('expected_net_profit');
  const [sortDirection, setSortDirection] = useState('desc'); // 'asc' | 'desc'
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);
  const [expandedCustomerId, setExpandedCustomerId] = useState(null);

  // Fetch optimization data
  const loadData = useCallback(async (budget) => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await fetchOptimizationData(budget);
      setRecords(data.records);
      setKpis(data.kpis);
      setStrategies(data.strategies);
      setMetadata(data.metadata);
      setActiveBudget(budget);
      setCurrentPage(1);
    } catch (err) {
      console.error('Failed to load optimization data:', err);
      setError(err.message || 'An error occurred while loading allocation data.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Initial load
  useEffect(() => {
    loadData(initialBudget);
  }, [loadData, initialBudget]);

  // Handle Budget input
  const handleBudgetChange = (e) => {
    const val = e.target.value;
    setBudgetInput(val);

    if (val === '') {
      setBudgetError('Budget cannot be empty.');
      return;
    }

    const num = Number(val);
    if (isNaN(num) || !isFinite(num)) {
      setBudgetError('Budget must be a valid number.');
      return;
    }

    if (num < 0) {
      setBudgetError('Budget cannot be negative.');
      return;
    }

    setBudgetError(null);
  };

  // Trigger optimization run
  const handleRunOptimization = (e) => {
    e?.preventDefault();
    const num = Number(budgetInput);
    if (isNaN(num) || num <= 0) {
      setBudgetError('Please enter a positive budget amount.');
      return;
    }
    loadData(num);
  };

  // Reset to default
  const handleResetBudget = () => {
    setBudgetInput(DEFAULT_BUDGET);
    setBudgetError(null);
    loadData(DEFAULT_BUDGET);
  };

  // Sorting handler
  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection((prev) => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
    setCurrentPage(1);
  };

  // Filter and sort records
  const filteredAndSortedRecords = useMemo(() => {
    let result = records;

    // Filter by discount tier
    if (discountFilter !== 'all') {
      const targetDiscount = Number(discountFilter);
      result = result.filter((r) => r.recommended_discount === targetDiscount);
    }

    // Filter by search query (customer ID)
    if (searchQuery.trim()) {
      const q = searchQuery.trim().toLowerCase();
      result = result.filter((r) => r.customer_id.toLowerCase().includes(q));
    }

    // Sort
    const sorted = [...result].sort((a, b) => {
      let valA = a[sortField];
      let valB = b[sortField];

      if (typeof valA === 'string') {
        valA = valA.toLowerCase();
        valB = valB.toLowerCase();
      }

      if (valA < valB) return sortDirection === 'asc' ? -1 : 1;
      if (valA > valB) return sortDirection === 'asc' ? 1 : -1;
      return 0;
    });

    return sorted;
  }, [records, discountFilter, searchQuery, sortField, sortDirection]);

  // Pagination calculation
  const totalItems = filteredAndSortedRecords.length;
  const totalPages = Math.max(1, Math.ceil(totalItems / pageSize));
  const validCurrentPage = Math.min(currentPage, totalPages);

  const paginatedRecords = useMemo(() => {
    const startIndex = (validCurrentPage - 1) * pageSize;
    return filteredAndSortedRecords.slice(startIndex, startIndex + pageSize);
  }, [filteredAndSortedRecords, validCurrentPage, pageSize]);

  // Formatters
  const formatUSD = (val) =>
    `$${Number(val || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

  const formatPct = (val) =>
    `${(Number(val || 0) * 100).toFixed(1)}%`;

  return (
    <div className="allocation-matrix-container">
      {/* 1. Header & Counterfactual Notice */}
      <div className="allocation-header-card">
        <div className="card-header-flex">
          <div className="title-with-icon">
            <div className="icon-badge icon-purple">
              <SlidersHorizontal size={22} />
            </div>
            <div>
              <div className="future-badge-row">
                <h2 className="card-title">Prescriptive Discount Allocation Matrix</h2>
                <span className="badge badge-success">Week 3 Deliverable</span>
              </div>
              <p className="card-subtitle">
                SciPy Mixed-Integer Linear Programming (MILP) on Multi-Treatment Causal Model (EconML DML)
              </p>
            </div>
          </div>

          <div className="solver-tag-wrap">
            <span className="solver-tag">
              <Sparkles size={13} className="text-emerald" />
              Solver: HiGHS (Global Optimal)
            </span>
            <span className="solver-tag">
              Population: 3,000 Customers
            </span>
          </div>
        </div>

        {/* Counterfactual Distinction Notice */}
        <div className="counterfactual-notice-banner">
          <ShieldAlert size={18} className="notice-icon text-amber" />
          <div className="notice-text">
            <strong>Methodological Notice — Counterfactual Projections:</strong>
            {' '}All revenue, cost, and profit figures are model-predicted expectations under the prescriptive policy,
            not historical realized accounting results. Promotional discount costs are incurred only when offered
            customers convert (Expected Discount Cost = discount amount × predicted purchase probability).
          </div>
        </div>

        {/* 2. Configurable Marketing Budget Input Form */}
        <form className="budget-control-bar" onSubmit={handleRunOptimization}>
          <div className="budget-input-group">
            <label htmlFor="budget-input" className="budget-label">
              Marketing Campaign Budget Cap ($ USD):
            </label>
            <div className="budget-input-wrapper">
              <DollarSign size={16} className="input-currency-icon" />
              <input
                id="budget-input"
                type="number"
                min="0"
                step="100"
                value={budgetInput}
                onChange={handleBudgetChange}
                className={`budget-input-field ${budgetError ? 'input-error' : ''}`}
                placeholder="5000"
                disabled={isLoading}
              />
            </div>
            {budgetError && <span className="input-error-msg">{budgetError}</span>}
          </div>

          <div className="budget-actions-row">
            <button
              type="submit"
              className="btn btn-primary"
              disabled={isLoading || Boolean(budgetError)}
            >
              <RefreshCw size={15} className={isLoading ? 'spinning' : ''} />
              <span>{isLoading ? 'Optimizing...' : 'Run Optimization'}</span>
            </button>

            <button
              type="button"
              className="btn btn-secondary"
              onClick={handleResetBudget}
              disabled={isLoading}
              title="Reset to benchmark default $5,000 budget"
            >
              Reset to $5,000
            </button>
          </div>
        </form>
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="matrix-loading-state">
          <RefreshCw size={32} className="spinning text-primary" />
          <p className="loading-text">Solving Mixed-Integer Knapsack Optimization across 3,000 customer profiles...</p>
        </div>
      )}

      {/* Error State */}
      {error && !isLoading && (
        <div className="alert alert-error">
          <AlertTriangle size={20} />
          <div>
            <strong>Optimization Engine Error:</strong> {error}
            <button type="button" className="btn btn-sm btn-secondary retry-btn" onClick={() => loadData(activeBudget)}>
              Retry
            </button>
          </div>
        </div>
      )}

      {/* Data Loaded Successfully */}
      {!isLoading && !error && kpis && (
        <>
          {/* 3. Eight Summary KPI Cards (Requirement 7) */}
          <div className="kpi-summary-grid">
            {/* Card 1: Total Customers */}
            <div className="kpi-card">
              <div className="kpi-header">
                <span className="kpi-title">Total Customers</span>
                <div className="kpi-icon-wrap icon-blue">
                  <Users size={16} />
                </div>
              </div>
              <div className="kpi-value">{kpis.total_customers.toLocaleString()}</div>
              <div className="kpi-caption">Held-out evaluation cohort</div>
            </div>

            {/* Card 2: Customers Receiving $0 */}
            <div className="kpi-card">
              <div className="kpi-header">
                <span className="kpi-title">No Discount ($0)</span>
                <div className="kpi-icon-wrap icon-slate">
                  <Award size={16} />
                </div>
              </div>
              <div className="kpi-value">{kpis.count_discount_0.toLocaleString()}</div>
              <div className="kpi-caption">
                {((kpis.count_discount_0 / kpis.total_customers) * 100).toFixed(1)}% (VIPs / Inelastic)
              </div>
            </div>

            {/* Card 3: Customers Receiving $10 */}
            <div className="kpi-card">
              <div className="kpi-header">
                <span className="kpi-title">$10 Discount</span>
                <div className="kpi-icon-wrap icon-emerald">
                  <Award size={16} />
                </div>
              </div>
              <div className="kpi-value">{kpis.count_discount_10.toLocaleString()}</div>
              <div className="kpi-caption">
                {((kpis.count_discount_10 / kpis.total_customers) * 100).toFixed(1)}% (High Uplift)
              </div>
            </div>

            {/* Card 4: Customers Receiving $20 */}
            <div className="kpi-card">
              <div className="kpi-header">
                <span className="kpi-title">$20 Discount</span>
                <div className="kpi-icon-wrap icon-purple">
                  <Award size={16} />
                </div>
              </div>
              <div className="kpi-value">{kpis.count_discount_20.toLocaleString()}</div>
              <div className="kpi-caption">
                {((kpis.count_discount_20 / kpis.total_customers) * 100).toFixed(1)}% (High Basket / Lift)
              </div>
            </div>

            {/* Card 5: Expected Gross Revenue */}
            <div className="kpi-card">
              <div className="kpi-header">
                <span className="kpi-title">Expected Revenue</span>
                <div className="kpi-icon-wrap icon-cyan">
                  <DollarSign size={16} />
                </div>
              </div>
              <div className="kpi-value">{formatUSD(kpis.expected_revenue)}</div>
              <div className="kpi-caption">Gross counterfactual sales</div>
            </div>

            {/* Card 6: Expected Discount Expenditure */}
            <div className="kpi-card">
              <div className="kpi-header">
                <span className="kpi-title">Discount Expenditure</span>
                <div className="kpi-icon-wrap icon-amber">
                  <Wallet size={16} />
                </div>
              </div>
              <div className="kpi-value">{formatUSD(kpis.expected_discount_cost)}</div>
              <div className="kpi-caption">
                {kpis.budget_utilization_pct}% of ${kpis.budget_cap.toLocaleString()} cap
              </div>
            </div>

            {/* Card 7: Expected Net Profit */}
            <div className="kpi-card highlight-profit">
              <div className="kpi-header">
                <span className="kpi-title">Expected Net Profit</span>
                <div className="kpi-icon-wrap icon-emerald">
                  <TrendingUp size={16} />
                </div>
              </div>
              <div className="kpi-value text-emerald">{formatUSD(kpis.expected_net_profit)}</div>
              <div className="kpi-caption text-emerald">
                +$2,066.60 lift vs baseline
              </div>
            </div>

            {/* Card 8: Remaining Budget */}
            <div className="kpi-card">
              <div className="kpi-header">
                <span className="kpi-title">Remaining Budget</span>
                <div className="kpi-icon-wrap icon-blue">
                  <DollarSign size={16} />
                </div>
              </div>
              <div className="kpi-value">{formatUSD(kpis.remaining_budget)}</div>
              <div className="kpi-caption">Unallocated budget reserve</div>
            </div>
          </div>

          {/* 4. Strategy Benchmark Comparison Section (Requirement 8) */}
          <div className="card strategy-comparison-card">
            <div className="card-header">
              <div className="title-with-icon">
                <div className="icon-badge icon-cyan">
                  <Layers size={18} />
                </div>
                <div>
                  <h3 className="card-title">Strategy Benchmark Comparison (Fair Population Evaluation)</h3>
                  <p className="card-subtitle">
                    Evaluating all 3,000 customers under identical economic pricing and product margins.
                  </p>
                </div>
              </div>
            </div>

            {/* Strategy Comparison Visual Bars & Cards */}
            <div className="strategy-grid">
              {strategies.map((strat, idx) => (
                <div
                  key={idx}
                  className={`strategy-item-box ${strat.short_name === 'Optimized Policy' ? 'strategy-winner' : ''} ${!strat.compliant ? 'strategy-noncompliant' : ''}`}
                >
                  <div className="strategy-item-header">
                    <span className="strategy-name">{strat.name}</span>
                    {strat.compliant ? (
                      <span className="badge badge-success">
                        <CheckCircle2 size={12} /> Compliant
                      </span>
                    ) : (
                      <span className="badge badge-error">
                        <XCircle size={12} /> Over Budget
                      </span>
                    )}
                  </div>

                  <div className="strategy-metrics">
                    <div className="strat-metric-row">
                      <span className="strat-metric-label">Discount Cost:</span>
                      <span className={`strat-metric-val ${!strat.compliant ? 'text-red font-bold' : ''}`}>
                        {formatUSD(strat.cost)}
                        <span className="strat-pct-sub"> ({strat.budget_util_pct?.toFixed(0) || Math.round((strat.cost / activeBudget) * 100)}% of cap)</span>
                      </span>
                    </div>

                    <div className="strat-metric-row">
                      <span className="strat-metric-label">Expected Revenue:</span>
                      <span className="strat-metric-val">{formatUSD(strat.revenue)}</span>
                    </div>

                    <div className="strat-metric-row strat-metric-profit">
                      <span className="strat-metric-label">Expected Net Profit:</span>
                      <span className="strat-metric-val strat-profit-val">{formatUSD(strat.profit)}</span>
                    </div>

                    <div className="strat-metric-row">
                      <span className="strat-metric-label">Profit Lift vs $0:</span>
                      <span className={`strat-metric-val ${strat.profit_lift > 0 ? 'text-emerald' : strat.profit_lift < 0 ? 'text-red' : 'text-slate'}`}>
                        {strat.profit_lift > 0 ? `+${formatUSD(strat.profit_lift)}` : formatUSD(strat.profit_lift)}
                        {strat.profit_lift_pct ? ` (${strat.profit_lift_pct > 0 ? '+' : ''}${strat.profit_lift_pct}%)` : ''}
                      </span>
                    </div>
                  </div>

                  {/* Relative Profit Bar */}
                  <div className="strat-bar-container">
                    <div
                      className="strat-bar-fill"
                      style={{
                        width: `${Math.min(100, Math.max(10, (strat.profit / 55000) * 100))}%`,
                        backgroundColor: strat.color || '#38bdf8',
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>

            <div className="strategy-callout">
              <Info size={15} className="text-primary" />
              <span>
                <strong>Executive Takeaway:</strong> Blanket promotional policies ($10 or $20 to all customers)
                severely violate marketing budget caps by 325% to 714% and destroy net operating profits.
                The causal optimization engine identifies the 27.9% of customers where coupons drive profitable incremental conversions.
              </span>
            </div>
          </div>

          {/* 5. Customer Allocation Table Section (Requirements 6, 9, 10, 11) */}
          <div className="card table-card">
            <div className="card-header-table">
              <div className="table-header-left">
                <h3 className="card-title">Customer-Level Policy Allocation Matrix</h3>
                <span className="records-count-tag">
                  Showing {totalItems.toLocaleString()} of {records.length.toLocaleString()} Customers
                </span>
              </div>

              {/* Filter & Search Bar */}
              <div className="table-controls-wrap">
                {/* Search Bar */}
                <div className="search-box">
                  <Search size={15} className="search-icon" />
                  <input
                    type="text"
                    placeholder="Search Customer ID (e.g. OPT_012009)..."
                    value={searchQuery}
                    onChange={(e) => {
                      setSearchQuery(e.target.value);
                      setCurrentPage(1);
                    }}
                    className="search-input"
                  />
                  {searchQuery && (
                    <button
                      type="button"
                      className="search-clear-btn"
                      onClick={() => setSearchQuery('')}
                    >
                      ×
                    </button>
                  )}
                </div>

                {/* Tier Filter Tabs */}
                <div className="filter-pill-group">
                  <span className="filter-label">Discount:</span>
                  {[
                    { label: 'All Tiers', val: 'all' },
                    { label: '$0', val: '0' },
                    { label: '$10', val: '10' },
                    { label: '$20', val: '20' },
                  ].map((tier) => (
                    <button
                      key={tier.val}
                      type="button"
                      className={`filter-pill ${discountFilter === tier.val ? 'active' : ''}`}
                      onClick={() => {
                        setDiscountFilter(tier.val);
                        setCurrentPage(1);
                      }}
                    >
                      {tier.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Table Container */}
            <div className="matrix-table-wrapper">
              <table className="matrix-table">
                <thead>
                  <tr>
                    <th onClick={() => handleSort('customer_id')} className="sortable-th">
                      <div className="th-content">
                        <span>Customer ID</span>
                        {sortField === 'customer_id' ? (
                          sortDirection === 'asc' ? <ArrowUp size={14} /> : <ArrowDown size={14} />
                        ) : (
                          <ArrowUpDown size={14} className="sort-idle" />
                        )}
                      </div>
                    </th>

                    <th onClick={() => handleSort('recommended_discount')} className="sortable-th">
                      <div className="th-content">
                        <span>Recommended Discount</span>
                        {sortField === 'recommended_discount' ? (
                          sortDirection === 'asc' ? <ArrowUp size={14} /> : <ArrowDown size={14} />
                        ) : (
                          <ArrowUpDown size={14} className="sort-idle" />
                        )}
                      </div>
                    </th>

                    <th onClick={() => handleSort('predicted_purchase_probability')} className="sortable-th">
                      <div className="th-content">
                        <span>Predicted Purchase Prob</span>
                        {sortField === 'predicted_purchase_probability' ? (
                          sortDirection === 'asc' ? <ArrowUp size={14} /> : <ArrowDown size={14} />
                        ) : (
                          <ArrowUpDown size={14} className="sort-idle" />
                        )}
                      </div>
                    </th>

                    <th onClick={() => handleSort('expected_revenue')} className="sortable-th">
                      <div className="th-content">
                        <span>Expected Revenue</span>
                        {sortField === 'expected_revenue' ? (
                          sortDirection === 'asc' ? <ArrowUp size={14} /> : <ArrowDown size={14} />
                        ) : (
                          <ArrowUpDown size={14} className="sort-idle" />
                        )}
                      </div>
                    </th>

                    <th onClick={() => handleSort('expected_discount_cost')} className="sortable-th">
                      <div className="th-content">
                        <span>Expected Discount Cost</span>
                        {sortField === 'expected_discount_cost' ? (
                          sortDirection === 'asc' ? <ArrowUp size={14} /> : <ArrowDown size={14} />
                        ) : (
                          <ArrowUpDown size={14} className="sort-idle" />
                        )}
                      </div>
                    </th>

                    <th onClick={() => handleSort('expected_net_profit')} className="sortable-th">
                      <div className="th-content">
                        <span>Expected Net Profit</span>
                        {sortField === 'expected_net_profit' ? (
                          sortDirection === 'asc' ? <ArrowUp size={14} /> : <ArrowDown size={14} />
                        ) : (
                          <ArrowUpDown size={14} className="sort-idle" />
                        )}
                      </div>
                    </th>

                    <th>
                      <span>Causal Uplift Details</span>
                    </th>
                  </tr>
                </thead>

                <tbody>
                  {paginatedRecords.length === 0 ? (
                    <tr>
                      <td colSpan={7} className="empty-table-cell">
                        <div className="empty-filter-state">
                          <Filter size={28} className="empty-icon" />
                          <p className="empty-title">No customers match the current filter criteria</p>
                          <p className="empty-sub">
                            Try adjusting your search query or switching the discount filter to 'All Tiers'.
                          </p>
                          <button
                            type="button"
                            className="btn btn-secondary btn-sm"
                            onClick={() => {
                              setSearchQuery('');
                              setDiscountFilter('all');
                            }}
                          >
                            Clear Filters
                          </button>
                        </div>
                      </td>
                    </tr>
                  ) : (
                    paginatedRecords.map((row) => {
                      const isExpanded = expandedCustomerId === row.customer_id;
                      const disc = row.recommended_discount;

                      return (
                        <React.Fragment key={row.customer_id}>
                          <tr className={`matrix-row ${disc === 10 ? 'row-highlight-10' : disc === 20 ? 'row-highlight-20' : ''}`}>
                            <td className="font-mono text-primary font-semibold">
                              {row.customer_id}
                            </td>

                            <td>
                              {disc === 0 && (
                                <span className="discount-badge badge-discount-0">$0 (Standard)</span>
                              )}
                              {disc === 10 && (
                                <span className="discount-badge badge-discount-10">$10 Off</span>
                              )}
                              {disc === 20 && (
                                <span className="discount-badge badge-discount-20">$20 Off</span>
                              )}
                            </td>

                            <td>
                              <span className="prob-value font-mono">
                                {formatPct(row.predicted_purchase_probability)}
                              </span>
                              <div className="prob-bar-track">
                                <div
                                  className="prob-bar-fill"
                                  style={{ width: `${Math.round(row.predicted_purchase_probability * 100)}%` }}
                                />
                              </div>
                            </td>

                            <td className="font-mono">
                              {formatUSD(row.expected_revenue)}
                            </td>

                            <td className="font-mono">
                              <span className={disc > 0 ? 'text-amber' : 'text-slate'}>
                                {formatUSD(row.expected_discount_cost)}
                              </span>
                            </td>

                            <td className="font-mono font-semibold text-emerald">
                              {formatUSD(row.expected_net_profit)}
                            </td>

                            <td>
                              <button
                                type="button"
                                className="btn-table-expand"
                                onClick={() =>
                                  setExpandedCustomerId(isExpanded ? null : row.customer_id)
                                }
                                title="Inspect counterfactual probabilities across $0, $10, and $20"
                              >
                                {isExpanded ? 'Hide Uplift' : 'View Uplift'}
                              </button>
                            </td>
                          </tr>

                          {/* Expandable Counterfactual Uplift Row */}
                          {isExpanded && (
                            <tr className="expanded-details-row">
                              <td colSpan={7}>
                                <div className="expanded-details-box">
                                  <div className="counterfactual-title">
                                    <Sparkles size={14} className="text-cyan" />
                                    <span>Counterfactual Predictions across Discount Options for Customer {row.customer_id}:</span>
                                  </div>

                                  <div className="counterfactual-grid">
                                    <div className="cf-card">
                                      <span className="cf-label">P(Purchase | $0)</span>
                                      <span className="cf-val">{formatPct(row.predicted_purchase_probability_0)}</span>
                                      <span className="cf-sub">Baseline conversion</span>
                                    </div>

                                    <div className="cf-card">
                                      <span className="cf-label">P(Purchase | $10)</span>
                                      <span className="cf-val">{formatPct(row.predicted_purchase_probability_10)}</span>
                                      <span className="cf-lift text-emerald">
                                        Lift: +{formatPct(row.predicted_effect_10_vs_0)}
                                      </span>
                                    </div>

                                    <div className="cf-card">
                                      <span className="cf-label">P(Purchase | $20)</span>
                                      <span className="cf-val">{formatPct(row.predicted_purchase_probability_20)}</span>
                                      <span className="cf-lift text-purple">
                                        Lift: +{formatPct(row.predicted_effect_20_vs_0)}
                                      </span>
                                    </div>
                                  </div>
                                </div>
                              </td>
                            </tr>
                          )}
                        </React.Fragment>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>

            {/* Pagination Controls (Requirement 11) */}
            <div className="pagination-bar">
              <div className="pagination-info">
                <span>
                  Showing {totalItems === 0 ? 0 : (validCurrentPage - 1) * pageSize + 1} to{' '}
                  {Math.min(validCurrentPage * pageSize, totalItems)} of {totalItems.toLocaleString()} customers
                </span>
                <div className="page-size-selector">
                  <label htmlFor="page-size-select">Rows per page:</label>
                  <select
                    id="page-size-select"
                    value={pageSize}
                    onChange={(e) => {
                      setPageSize(Number(e.target.value));
                      setCurrentPage(1);
                    }}
                    className="page-size-dropdown"
                  >
                    <option value={25}>25</option>
                    <option value={50}>50</option>
                    <option value={100}>100</option>
                  </select>
                </div>
              </div>

              <div className="pagination-buttons">
                <button
                  type="button"
                  className="btn btn-secondary btn-pagination"
                  disabled={validCurrentPage <= 1}
                  onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                >
                  <ChevronLeft size={16} />
                  <span>Previous</span>
                </button>

                <div className="page-numbers-display">
                  Page <strong>{validCurrentPage}</strong> of <strong>{totalPages}</strong>
                </div>

                <button
                  type="button"
                  className="btn btn-secondary btn-pagination"
                  disabled={validCurrentPage >= totalPages}
                  onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                >
                  <span>Next</span>
                  <ChevronRight size={16} />
                </button>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
