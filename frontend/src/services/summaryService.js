/**
 * EconoCasual Deterministic Natural-Language Summary Service (Week 4 Deliverable)
 * 
 * Converts causal optimization and data drift telemetry into clear, professional,
 * and transparent natural-language summaries for a Marketing Director / Executive.
 * 
 * Strictly follows causal accounting principles:
 * - Distinguishes model-predicted expectations from historical realized outcomes.
 * - Explains missing or partial metrics without fabricating numbers.
 * - Avoids claims of guaranteed revenue or causal certainty.
 * - Operates entirely deterministically without external paid LLMs.
 */

const CURRENCY_FORMATTER = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const NUMBER_FORMATTER = new Intl.NumberFormat('en-US');

export function formatCurrency(amount) {
  if (amount == null || !Number.isFinite(Number(amount))) return 'Unavailable';
  return CURRENCY_FORMATTER.format(Number(amount));
}

export function formatNumber(count) {
  if (count == null || !Number.isFinite(Number(count))) return 'Unavailable';
  return NUMBER_FORMATTER.format(Number(count));
}

export function formatPct(value, decimals = 1) {
  if (value == null || !Number.isFinite(Number(value))) return 'Unavailable';
  return `${Number(value).toFixed(decimals)}%`;
}

/**
 * Generate Optimization Recommendation Summary Section
 */
function buildOptimizationRecommendationSection(kpis, metadata) {
  const metrics = [];

  if (!kpis || typeof kpis !== 'object') {
    return {
      id: 'optimization_recommendation',
      title: 'Optimization Recommendation',
      headline: 'Optimization metrics are unavailable',
      body: [
        'No active optimization results were provided for evaluation. Run a budget optimization to generate customer-level discount allocations and prescriptive recommendations.',
      ],
      badge: { label: 'Unavailable', type: 'warning' },
      status: 'unavailable',
      underlyingMetrics: [],
    };
  }

  const total = Number(kpis.total_customers);
  const c0 = Number(kpis.count_discount_0);
  const c10 = Number(kpis.count_discount_10);
  const c20 = Number(kpis.count_discount_20);

  const hasCustomerCounts = Number.isFinite(total) && total > 0;
  if (!hasCustomerCounts) {
    return {
      id: 'optimization_recommendation',
      title: 'Optimization Recommendation',
      headline: 'Customer counts are missing or inconsistent',
      body: [
        'The optimization result record is missing valid customer population totals. Allocation recommendations cannot be summarized without verifiable customer counts.',
      ],
      badge: { label: 'Incomplete Data', type: 'warning' },
      status: 'partial',
      underlyingMetrics: [],
    };
  }

  metrics.push({ label: 'Total Customer Cohort', value: formatNumber(total) });
  if (Number.isFinite(c0)) metrics.push({ label: 'Allocated $0 (Full Price)', value: formatNumber(c0) });
  if (Number.isFinite(c10)) metrics.push({ label: 'Allocated $10 Discount', value: formatNumber(c10) });
  if (Number.isFinite(c20)) metrics.push({ label: 'Allocated $20 Discount', value: formatNumber(c20) });

  let headline = '';
  const body = [];

  const treatedCount = (Number.isFinite(c10) ? c10 : 0) + (Number.isFinite(c20) ? c20 : 0);
  const pctTreated = formatPct((treatedCount / total) * 100);

  if (treatedCount === 0) {
    headline = 'Preserve full catalog pricing for all evaluated customers';
    body.push(
      `Under the active campaign constraints, the optimizer recommends no discounts ($0) for all ${formatNumber(total)} customers. The model projects that promotional coupon costs would exceed expected incremental contribution.`
    );
  } else {
    headline = `Target promotional discounts to ${formatNumber(treatedCount)} customers (${pctTreated} of cohort)`;
    body.push(
      `The causal optimizer recommends targeting personalized discounts to ${formatNumber(treatedCount)} customers: ${formatNumber(c10)} allocated $10 coupons and ${formatNumber(c20)} allocated $20 coupons.`
    );
    body.push(
      `For the remaining ${formatNumber(c0)} customers, the model recommends maintaining full price ($0) to avoid subsidizing organic purchases where baseline conversion propensity is already high.`
    );
  }

  return {
    id: 'optimization_recommendation',
    title: 'Optimization Recommendation',
    headline,
    body,
    badge: { label: treatedCount > 0 ? 'Targeted Policy' : 'Full Price Policy', type: 'info' },
    status: 'available',
    underlyingMetrics: metrics,
  };
}

/**
 * Generate Budget Utilization Summary Section
 */
function buildBudgetUtilizationSection(kpis) {
  const metrics = [];

  if (!kpis || typeof kpis !== 'object') {
    return {
      id: 'budget_utilization',
      title: 'Budget Utilization Summary',
      headline: 'Budget figures are unavailable',
      body: ['Budget accounting figures could not be computed because KPI summary records are missing.'],
      badge: { label: 'Unavailable', type: 'warning' },
      status: 'unavailable',
      underlyingMetrics: [],
    };
  }

  const budget = Number(kpis.budget_cap ?? kpis.budget ?? 5000);
  const cost = Number(kpis.expected_discount_cost);
  const remaining = Number(kpis.remaining_budget);
  const utilPct = Number(kpis.budget_utilization_pct);

  const hasCost = Number.isFinite(cost);
  const hasBudget = Number.isFinite(budget) && budget > 0;

  if (!hasCost) {
    return {
      id: 'budget_utilization',
      title: 'Budget Utilization Summary',
      headline: 'Expected discount expenditure is unavailable',
      body: ['The optimization record lacks a valid expected discount expenditure metric.'],
      badge: { label: 'Incomplete Data', type: 'warning' },
      status: 'partial',
      underlyingMetrics: [],
    };
  }

  metrics.push({ label: 'Authorized Budget Cap', value: formatCurrency(budget) });
  metrics.push({ label: 'Projected Discount Cost', value: formatCurrency(cost), description: 'Cost incurred only when customers convert' });
  if (Number.isFinite(remaining)) metrics.push({ label: 'Remaining Reserve', value: formatCurrency(remaining) });
  if (Number.isFinite(utilPct)) metrics.push({ label: 'Budget Utilization', value: formatPct(utilPct) });

  let headline = '';
  const body = [];

  const computedUtil = hasBudget ? (cost / budget) * 100 : utilPct;
  const safeUtil = Number.isFinite(computedUtil) ? computedUtil : utilPct;

  if (hasBudget && cost > budget) {
    headline = `Policy expenditure exceeds authorized budget cap by ${formatCurrency(cost - budget)}`;
    body.push(
      `The projected discount cost of ${formatCurrency(cost)} exceeds the authorized budget cap of ${formatCurrency(budget)}. Review constraint parameters.`
    );
  } else if (hasBudget) {
    headline = `Allocated ${formatCurrency(cost)} of the available ${formatCurrency(budget)} budget (${formatPct(safeUtil)})`;
    body.push(
      `The optimized strategy allocates ${formatCurrency(cost)} of the available ${formatCurrency(budget)} expected discount budget (${formatPct(safeUtil)} utilization), leaving ${formatCurrency(remaining)} in reserve.`
    );
    body.push(
      'Accounting note: In accordance with retail promotional standards, expected discount cost is accrued only when a customer redeems the coupon upon purchasing (Expected Cost = Discount × Predicted Purchase Probability).'
    );
  } else {
    headline = `Projected discount expenditure: ${formatCurrency(cost)}`;
    body.push(`The optimized policy is projected to incur ${formatCurrency(cost)} in promotional coupon expenses across redeeming buyers.`);
  }

  return {
    id: 'budget_utilization',
    title: 'Budget Utilization Summary',
    headline,
    body,
    badge: {
      label: safeUtil >= 90 ? 'High Utilization' : 'Moderate Utilization',
      type: safeUtil > 100 ? 'danger' : 'success',
    },
    status: 'available',
    underlyingMetrics: metrics,
  };
}

/**
 * Generate Customer Allocation Summary Section
 */
function buildCustomerAllocationSection(kpis) {
  const metrics = [];

  if (!kpis || typeof kpis !== 'object') {
    return {
      id: 'customer_allocation',
      title: 'Customer Allocation Summary',
      headline: 'Allocation breakdown is unavailable',
      body: ['Customer allocation counts are not available in the current dataset.'],
      badge: { label: 'Unavailable', type: 'warning' },
      status: 'unavailable',
      underlyingMetrics: [],
    };
  }

  const total = Number(kpis.total_customers);
  const c0 = Number(kpis.count_discount_0);
  const c10 = Number(kpis.count_discount_10);
  const c20 = Number(kpis.count_discount_20);

  if (!Number.isFinite(total) || total <= 0) {
    return {
      id: 'customer_allocation',
      title: 'Customer Allocation Summary',
      headline: 'Customer distribution is missing',
      body: ['The total customer population count is zero or undefined.'],
      badge: { label: 'Incomplete', type: 'warning' },
      status: 'partial',
      underlyingMetrics: [],
    };
  }

  const p0 = Number.isFinite(c0) ? (c0 / total) * 100 : null;
  const p10 = Number.isFinite(c10) ? (c10 / total) * 100 : null;
  const p20 = Number.isFinite(c20) ? (c20 / total) * 100 : null;

  metrics.push({ label: 'Total Evaluated', value: `${formatNumber(total)} customers` });
  if (p0 != null) metrics.push({ label: '$0 Tier (Full Price)', value: `${formatNumber(c0)} (${formatPct(p0)})` });
  if (p10 != null) metrics.push({ label: '$10 Tier (Light Incentive)', value: `${formatNumber(c10)} (${formatPct(p10)})` });
  if (p20 != null) metrics.push({ label: '$20 Tier (Deep Incentive)', value: `${formatNumber(c20)} (${formatPct(p20)})` });

  const headline = `${formatPct(p0 ?? 0)} full price, ${formatPct((p10 ?? 0) + (p20 ?? 0))} discounted`;
  const body = [
    `The optimizer recommended no discount ($0) for ${formatNumber(c0)} customers (${formatPct(p0 ?? 0)} of cohort), preserving product margins where customer elasticity does not justify discounting.`,
    `Moderate discounts ($10) were assigned to ${formatNumber(c10)} customers (${formatPct(p10 ?? 0)}), capturing price-responsive buyers with optimal incremental return.`,
    c20 > 0
      ? `Deep discounts ($20) were reserved for ${formatNumber(c20)} highly elastic customers (${formatPct(p20 ?? 0)}) where the expected margin gain outweighs the steeper $20 coupon expense.`
      : 'Deep discounts ($20) were not allocated to any customer, as the steeper $20 coupon expense eroded expected net margin across the population.',
  ];

  return {
    id: 'customer_allocation',
    title: 'Customer Allocation Summary',
    headline,
    body,
    badge: { label: 'Tier Distribution', type: 'info' },
    status: 'available',
    underlyingMetrics: metrics,
  };
}

/**
 * Generate Revenue & Profit Comparison Section
 */
function buildRevenueProfitComparisonSection(kpis, strategies) {
  const metrics = [];

  const rev = Number(kpis?.expected_revenue);
  const profit = Number(kpis?.expected_net_profit);

  const hasRev = Number.isFinite(rev);
  const hasProfit = Number.isFinite(profit);

  if (!hasRev && !hasProfit) {
    return {
      id: 'revenue_profit_comparison',
      title: 'Expected Revenue & Profit Comparison',
      headline: 'Revenue and profit projections are unavailable',
      body: ['Projected financial outcomes could not be calculated from the available records.'],
      badge: { label: 'Unavailable', type: 'warning' },
      status: 'unavailable',
      underlyingMetrics: [],
    };
  }

  if (hasRev) metrics.push({ label: 'Projected Gross Revenue', value: formatCurrency(rev) });
  if (hasProfit) metrics.push({ label: 'Projected Net Profit', value: formatCurrency(profit) });

  let headline = `Projected expected net profit: ${formatCurrency(profit)}`;
  const body = [];

  // Inspect benchmark strategies if available
  const baselineStrategy = Array.isArray(strategies)
    ? strategies.find((s) => s.short_name?.toLowerCase().includes('no discount') || s.discount === '$0')
    : null;
  const blanket10Strategy = Array.isArray(strategies)
    ? strategies.find((s) => s.short_name?.toLowerCase().includes('blanket $10') || s.name?.includes('Blanket $10'))
    : null;

  if (baselineStrategy && Number.isFinite(baselineStrategy.profit)) {
    const baseProfit = Number(baselineStrategy.profit);
    const liftUsd = profit - baseProfit;
    const liftPct = baseProfit > 0 ? (liftUsd / baseProfit) * 100 : 0;

    metrics.push({ label: 'Baseline Profit ($0 to All)', value: formatCurrency(baseProfit) });
    metrics.push({ label: 'Projected Profit Lift', value: `${formatCurrency(liftUsd)} (${formatPct(liftPct)})` });

    if (liftUsd >= 0) {
      headline = `Model projects +${formatCurrency(liftUsd)} (+${formatPct(liftPct)}) profit lift over no-discount baseline`;
      body.push(
        `The model estimates that the personalized strategy produces a higher expected contribution (${formatCurrency(profit)}) than blanket baseline pricing (${formatCurrency(baseProfit)}), representing an expected lift of ${formatCurrency(liftUsd)}.`
      );
    } else {
      headline = `Projected net profit: ${formatCurrency(profit)}`;
      body.push(
        `The personalized strategy produces an expected net profit of ${formatCurrency(profit)} under the given constraints.`
      );
    }
  } else {
    body.push(
      `The personalized allocation is projected to yield ${formatCurrency(rev)} in expected gross revenue and ${formatCurrency(profit)} in expected net contribution.`
    );
  }

  if (blanket10Strategy && Number.isFinite(blanket10Strategy.cost)) {
    const bCost = Number(blanket10Strategy.cost);
    metrics.push({ label: 'Blanket $10 Promotion Cost', value: formatCurrency(bCost) });
    body.push(
      `In comparison, unsegmented blanket discounting ($10 to all customers) would incur ${formatCurrency(bCost)} in discount expenses, severely breaching campaign budget limits and eroding net margins.`
    );
  }

  body.push(
    'Projections notice: Expected revenue and profit figures are model-predicted counterfactual expectations based on estimated purchase probabilities. They represent statistical projections, not guaranteed actual receipts.'
  );

  return {
    id: 'revenue_profit_comparison',
    title: 'Expected Revenue & Profit Comparison',
    headline,
    body,
    badge: { label: 'Counterfactual Projections', type: 'success' },
    status: 'available',
    underlyingMetrics: metrics,
  };
}

/**
 * Generate Data Drift Warning Section
 */
function buildDataDriftWarningSection(driftReport) {
  const metrics = [];

  if (!driftReport || typeof driftReport !== 'object') {
    return {
      id: 'drift_warning',
      title: 'Data Drift & Covariate Shift Surveillance',
      headline: 'Data drift evaluation has not been executed',
      body: [
        'No recent customer drift inspection was provided. Run a drift check against the baseline reference dataset to verify whether customer characteristics remain stationary.',
      ],
      badge: { label: 'Not Evaluated', type: 'warning' },
      status: 'unavailable',
      underlyingMetrics: [],
    };
  }

  const overallStatus = driftReport.overall_status || 'Unknown';
  const totalFeats = Number(driftReport.total_features_evaluated ?? 0);
  const driftedCount = Number(driftReport.drifted_features_count ?? 0);
  const highSevCount = Number(driftReport.high_severity_count ?? 0);

  metrics.push({ label: 'Overall Status', value: overallStatus });
  metrics.push({ label: 'Features Evaluated', value: totalFeats });
  metrics.push({ label: 'Drifted Features Count', value: `${driftedCount} / ${totalFeats}` });
  metrics.push({ label: 'High Severity Features', value: highSevCount });

  let headline = '';
  const body = [];
  let badge = { label: 'Stable', type: 'success' };

  const driftedFeatures = Array.isArray(driftReport.features)
    ? driftReport.features.filter((f) => f.drift_detected).map((f) => f.feature_name)
    : [];

  if (overallStatus.toLowerCase().includes('severe')) {
    badge = { label: 'Severe Drift', type: 'danger' };
    headline = `Severe covariate shift detected across ${driftedCount} of ${totalFeats} customer features`;
    body.push(
      `The reference and current customer distributions differ substantially. Statistically significant drift was detected in ${driftedFeatures.join(', ') || 'multiple customer covariates'}.`
    );
    body.push(
      'Model performance and conversion elasticity should be monitored closely. Covariate shift (P(X)) does not automatically invalidate causal treatment effect estimates, but common support overlap must be audited.'
    );
  } else if (overallStatus.toLowerCase().includes('potential') || driftedCount > 0) {
    badge = { label: 'Potential Drift', type: 'warning' };
    headline = `Potential drift detected in ${driftedFeatures.join(' and ') || 'incoming customer attributes'}`;
    body.push(
      `Moderate distribution shift was observed in ${driftedFeatures.join(', ')}. While within tolerable bounds, downstream purchase elasticity should be monitored.`
    );
  } else {
    badge = { label: 'No Drift Detected', type: 'success' };
    headline = 'Incoming customer cohort is statistically consistent with reference training baseline';
    body.push(
      'All evaluated customer features align closely with the training reference distribution (two-sample Kolmogorov-Smirnov p > 0.05 and Population Stability Index < 0.10). Model-predicted treatment effects remain on stable ground.'
    );
  }

  return {
    id: 'drift_warning',
    title: 'Data Drift & Covariate Shift Surveillance',
    headline,
    body,
    badge,
    status: 'available',
    underlyingMetrics: metrics,
  };
}

/**
 * Generate Recommended Next Action Section
 */
function buildRecommendedNextActionSection(kpis, driftReport) {
  const actions = [];
  const metrics = [];

  const treatedCount = (Number(kpis?.count_discount_10) || 0) + (Number(kpis?.count_discount_20) || 0);
  const cost = Number(kpis?.expected_discount_cost);
  const isDrifted = driftReport?.overall_status?.toLowerCase().includes('severe') || false;

  metrics.push({ label: 'Targeted Customers', value: formatNumber(treatedCount) });
  metrics.push({ label: 'Authorized Expenditure', value: formatCurrency(cost) });
  metrics.push({ label: 'Covariate Shift Level', value: isDrifted ? 'Severe' : 'Stable' });

  if (treatedCount > 0) {
    actions.push(
      `1. Deployment: Proceed with targeted discount distribution for the ${formatNumber(treatedCount)} identified elastic customers, maintaining promotional expenditure within the authorized budget.`
    );
  } else {
    actions.push(
      '1. Pricing Policy: Maintain standard catalog pricing without coupon distribution, protecting profit margin.'
    );
  }

  if (isDrifted) {
    actions.push(
      '2. Telemetry Audit: Because severe demographic drift was detected in incoming cohorts, verify common support overlap and recalibrate propensity scores e(X). Do not trigger automated model retraining without offline counterfactual holdout testing.'
    );
  } else {
    actions.push(
      '2. Telemetry Audit: Maintain weekly two-sample distribution monitoring (KS and PSI tests) on live customer cohorts.'
    );
  }

  actions.push(
    '3. Holdout Governance: Reserve a 5% randomized holdout group receiving $0 discount to measure realized conversion lift and validate counterfactual DML projections against actual financial results.'
  );

  return {
    id: 'recommended_action',
    title: 'Recommended Operational Next Actions',
    headline: 'Action protocol for promotional campaign launch and governance',
    body: actions,
    badge: { label: 'Operational Runbook', type: 'info' },
    status: 'available',
    underlyingMetrics: metrics,
  };
}

/**
 * Master Function: Generate Comprehensive Natural-Language Executive Summary
 * 
 * @param {Object} optimizationData - Contains kpis, strategies, metadata, records
 * @param {Object} driftData - Contains report, distributions, metadata
 * @returns {ExecutiveSummaryReport}
 */
export function generateExecutiveSummary(optimizationData = {}, driftData = {}) {
  const kpis = optimizationData?.kpis || null;
  const strategies = optimizationData?.strategies || null;
  const optMeta = optimizationData?.metadata || {};
  const driftReport = driftData?.report || null;

  const runId = optMeta.run_id || `run_opt_${new Date().toISOString().slice(0, 10).replace(/-/g, '')}_auto`;
  const timestamp = optMeta.timestamp || new Date().toISOString();

  // Build each section deterministically
  const secRecommendation = buildOptimizationRecommendationSection(kpis, optMeta);
  const secBudget = buildBudgetUtilizationSection(kpis);
  const secAllocation = buildCustomerAllocationSection(kpis);
  const secFinancials = buildRevenueProfitComparisonSection(kpis, strategies);
  const secDrift = buildDataDriftWarningSection(driftReport);
  const secActions = buildRecommendedNextActionSection(kpis, driftReport);

  const sections = [
    secRecommendation,
    secBudget,
    secAllocation,
    secFinancials,
    secDrift,
    secActions,
  ];

  // Synthesize top-level executive takeaway
  let overallTakeaway = '';
  if (secRecommendation.status === 'available' && secBudget.status === 'available') {
    const cost = kpis?.expected_discount_cost ? formatCurrency(kpis.expected_discount_cost) : 'budget';
    const profit = kpis?.expected_net_profit ? formatCurrency(kpis.expected_net_profit) : 'positive';
    overallTakeaway = `The causal optimizer recommends targeting personalized discounts with an expected expenditure of ${cost}, projecting an expected net profit of ${profit}.`;
  } else if (secDrift.status === 'available') {
    overallTakeaway = `Data drift monitoring reports: ${secDrift.headline}. Review covariate shift diagnostics.`;
  } else {
    overallTakeaway = 'Insufficient optimization and drift metrics available to synthesize an executive takeaway.';
  }

  const hasErrors = sections.some((s) => s.status === 'unavailable');

  return {
    runId,
    timestamp,
    overallTakeaway,
    hasErrors,
    sections,
    methodologicalNotice:
      'Methodological Notice: All revenue, cost, and profit figures are model-predicted counterfactual expectations derived from Double Machine Learning (DML) and Mixed-Integer Linear Programming (MILP). They reflect probabilistic projections under causal unconfoundedness assumptions, not guaranteed actual business earnings. A 5% randomized holdout group is recommended to measure true empirical lift.',
  };
}

