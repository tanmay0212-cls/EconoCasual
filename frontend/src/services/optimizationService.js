import Papa from 'papaparse';

/**
 * Service to load and manage the Week 3 Optimization & Allocation Matrix results.
 * Connects directly to the actual backend optimization artifacts:
 * - /allocation_matrix.csv (3,000 customers with counterfactual predictions and optimal MILP decisions)
 * - /optimization_metadata.json (strategy benchmarks, execution parameters, and model metadata)
 */

export const DEFAULT_BUDGET = 5000.0;

/**
 * Fetch and parse the allocation matrix CSV and metadata.
 * @param {number} targetBudget - Marketing budget cap in USD (defaults to $5,000.00).
 * @returns {Promise<{ records: Array, kpis: Object, strategies: Array, metadata: Object }>}
 */
export async function fetchOptimizationData(targetBudget = DEFAULT_BUDGET) {
  const budgetNum = Number(targetBudget) > 0 ? Number(targetBudget) : DEFAULT_BUDGET;

  // 1. Fetch CSV and metadata in parallel
  const [csvResponse, metaResponse] = await Promise.all([
    fetch('/allocation_matrix.csv'),
    fetch('/optimization_metadata.json').catch(() => null),
  ]);

  if (!csvResponse.ok) {
    throw new Error(`Failed to load allocation matrix CSV: HTTP ${csvResponse.status}`);
  }

  const csvText = await csvResponse.text();

  // 2. Parse CSV rows
  const parseResult = await new Promise((resolve, reject) => {
    Papa.parse(csvText, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: resolve,
      error: reject,
    });
  });

  const rawRows = parseResult.data || [];
  if (rawRows.length === 0) {
    throw new Error('Allocation matrix CSV is empty.');
  }

  let metadata = null;
  if (metaResponse && metaResponse.ok) {
    try {
      metadata = await metaResponse.json();
    } catch {
      metadata = null;
    }
  }

  // 3. Normalize records and ensure required numeric fields
  let processedRecords = rawRows.map((row) => {
    const p0 = Number(row.predicted_purchase_probability_0 ?? 0);
    const p10 = Number(row.predicted_purchase_probability_10 ?? 0);
    const p20 = Number(row.predicted_purchase_probability_20 ?? 0);
    const recDisc = Number(row.recommended_discount ?? 0);
    const cost = Number(row.expected_discount_cost ?? 0);
    const rev = Number(row.expected_revenue ?? 0);
    const profit = Number(row.expected_net_profit ?? 0);

    const activeProb = recDisc === 20 ? p20 : recDisc === 10 ? p10 : p0;

    return {
      customer_id: String(row.customer_id || ''),
      predicted_purchase_probability_0: p0,
      predicted_purchase_probability_10: p10,
      predicted_purchase_probability_20: p20,
      predicted_effect_10_vs_0: Number(row.predicted_effect_10_vs_0 ?? (p10 - p0)),
      predicted_effect_20_vs_0: Number(row.predicted_effect_20_vs_0 ?? (p20 - p0)),
      recommended_discount: recDisc,
      expected_discount_cost: cost,
      expected_revenue: rev,
      expected_net_profit: profit,
      predicted_purchase_probability: activeProb,
    };
  });

  // 4. If target budget is custom (not the default $5,000 backend run),
  // simulate budget-constrained knapsack re-allocation
  if (Math.abs(budgetNum - DEFAULT_BUDGET) > 0.01) {
    processedRecords = reallocateByBudget(processedRecords, budgetNum);
  }

  // 5. Calculate summary KPI metrics from records
  const kpis = calculateKPIs(processedRecords, budgetNum);

  // 6. Build strategy benchmark comparison
  const strategies = buildStrategyComparison(metadata, kpis, budgetNum);

  return {
    records: processedRecords,
    kpis,
    strategies,
    metadata: metadata || {
      evaluation_population: processedRecords.length,
      default_budget: DEFAULT_BUDGET,
      objective_type: 'net_profit',
      solver: 'scipy.optimize.milp (HiGHS)',
    },
  };
}

/**
 * Calculates aggregate KPIs for the current allocation records.
 */
export function calculateKPIs(records, budget) {
  let count0 = 0;
  let count10 = 0;
  let count20 = 0;
  let totalCost = 0;
  let totalRevenue = 0;
  let totalProfit = 0;

  for (const r of records) {
    if (r.recommended_discount === 20) count20 += 1;
    else if (r.recommended_discount === 10) count10 += 1;
    else count0 += 1;

    totalCost += r.expected_discount_cost;
    totalRevenue += r.expected_revenue;
    totalProfit += r.expected_net_profit;
  }

  const remainingBudget = Math.max(0, budget - totalCost);
  const budgetUtilizationPct = budget > 0 ? (totalCost / budget) * 100 : 0;

  return {
    total_customers: records.length,
    count_discount_0: count0,
    count_discount_10: count10,
    count_discount_20: count20,
    expected_revenue: Math.round(totalRevenue * 100) / 100,
    expected_discount_cost: Math.round(totalCost * 100) / 100,
    expected_net_profit: Math.round(totalProfit * 100) / 100,
    remaining_budget: Math.round(remainingBudget * 100) / 100,
    budget_utilization_pct: Math.round(budgetUtilizationPct * 10) / 10,
    budget_cap: budget,
  };
}

/**
 * Re-allocates discounts if user tests a non-standard budget in the UI.
 * Uses incremental profit-to-cost ratio knapsack heuristic.
 */
function reallocateByBudget(records, budget) {
  // Infer implied customer price and margin from baseline metrics
  // For $0: rev_0 = price * p0, profit_0 = price * margin * p0
  // cost_10 = 10 * p10, cost_20 = 20 * p20
  const candidates = records.map((r) => {
    // Reconstruct customer price and margin
    const p0 = Math.max(0.01, r.predicted_purchase_probability_0);
    const p10 = r.predicted_purchase_probability_10;
    const p20 = r.predicted_purchase_probability_20;

    // Use ratio from the default run
    const impliedPrice = r.recommended_discount === 0
      ? r.expected_revenue / p0
      : (r.recommended_discount === 10 ? (r.expected_revenue + 10 * p10) / p10 : (r.expected_revenue + 20 * p20) / p20);

    const price = Number.isFinite(impliedPrice) && impliedPrice > 10 ? impliedPrice : 85.0;
    const margin = 0.40; // baseline retail average gross margin

    const profit0 = price * margin * p0;
    const rev0 = price * p0;

    const cost10 = 10 * p10;
    const profit10 = (price * margin - 10) * p10;
    const rev10 = Math.max(0, price - 10) * p10;

    const cost20 = 20 * p20;
    const profit20 = (price * margin - 20) * p20;
    const rev20 = Math.max(0, price - 20) * p20;

    // Marginal gains
    const gain10 = profit10 - profit0;
    const roi10 = cost10 > 0 ? gain10 / cost10 : -999;

    const gain20_from_10 = profit20 - profit10;
    const incCost20 = cost20 - cost10;
    const roi20 = incCost20 > 0 ? gain20_from_10 / incCost20 : -999;

    return {
      record: r,
      p0,
      p10,
      p20,
      rev0,
      rev10,
      rev20,
      profit0,
      profit10,
      profit20,
      cost10,
      cost20,
      gain10,
      roi10,
      gain20_from_10,
      incCost20,
      roi20,
      assignedDiscount: 0,
      activeCost: 0,
      activeRev: rev0,
      activeProfit: profit0,
      activeProb: p0,
    };
  });

  // Sort candidates for $10 discount by ROI
  const eligible10 = candidates
    .filter((c) => c.gain10 > 0)
    .sort((a, b) => b.roi10 - a.roi10);

  let currentBudgetUsed = 0;

  for (const c of eligible10) {
    if (currentBudgetUsed + c.cost10 <= budget) {
      c.assignedDiscount = 10;
      c.activeCost = c.cost10;
      c.activeRev = c.rev10;
      c.activeProfit = c.profit10;
      c.activeProb = c.p10;
      currentBudgetUsed += c.cost10;
    }
  }

  // Then try upgrading from 10 to 20 if margin allows and ROI is positive
  const eligible20 = candidates
    .filter((c) => c.assignedDiscount === 10 && c.gain20_from_10 > 0)
    .sort((a, b) => b.roi20 - a.roi20);

  for (const c of eligible20) {
    if (currentBudgetUsed + c.incCost20 <= budget) {
      c.assignedDiscount = 20;
      c.activeCost = c.cost20;
      c.activeRev = c.rev20;
      c.activeProfit = c.profit20;
      c.activeProb = c.p20;
      currentBudgetUsed += c.incCost20;
    }
  }

  return candidates.map((c) => ({
    ...c.record,
    recommended_discount: c.assignedDiscount,
    expected_discount_cost: Math.round(c.activeCost * 100) / 100,
    expected_revenue: Math.round(c.activeRev * 100) / 100,
    expected_net_profit: Math.round(c.activeProfit * 100) / 100,
    predicted_purchase_probability: c.activeProb,
  }));
}

/**
 * Builds comparison strategies matching actual reports.
 */
function buildStrategyComparison(metadata, activeKPIs, budget) {
  if (metadata && metadata.strategies) {
    return metadata.strategies.map((s) => {
      if (s.short_name === 'Optimized Policy') {
        return {
          ...s,
          cost: activeKPIs.expected_discount_cost,
          revenue: activeKPIs.expected_revenue,
          profit: activeKPIs.expected_net_profit,
          budget_util_pct: activeKPIs.budget_utilization_pct,
          compliant: activeKPIs.expected_discount_cost <= budget,
          profit_lift: Math.round((activeKPIs.expected_net_profit - 51015.75) * 100) / 100,
          profit_lift_pct: Math.round(((activeKPIs.expected_net_profit - 51015.75) / 51015.75) * 1000) / 10,
        };
      }
      return {
        ...s,
        compliant: s.cost <= budget,
      };
    });
  }

  // Fallback if metadata json is missing
  return [
    {
      name: 'No Discount ($0 Baseline)',
      short_name: 'No Discount',
      discount: '$0',
      cost: 0.0,
      revenue: 106473.01,
      profit: 51015.75,
      compliant: true,
      profit_lift: 0.0,
      color: '#64748b',
    },
    {
      name: 'Blanket $10 Promotion',
      short_name: 'Blanket $10',
      discount: '$10 to All',
      cost: 16246.60,
      revenue: 137756.90,
      profit: 47828.63,
      compliant: 16246.60 <= budget,
      profit_lift: -3187.12,
      color: '#ef4444',
    },
    {
      name: 'Blanket $20 Promotion',
      short_name: 'Blanket $20',
      discount: '$20 to All',
      cost: 35705.84,
      revenue: 146789.04,
      profit: 31841.21,
      compliant: 35705.84 <= budget,
      profit_lift: -19174.54,
      color: '#f97316',
    },
    {
      name: 'Optimized Personalized (DML + SciPy)',
      short_name: 'Optimized Policy',
      discount: 'Personalized',
      cost: activeKPIs.expected_discount_cost,
      revenue: activeKPIs.expected_revenue,
      profit: activeKPIs.expected_net_profit,
      compliant: activeKPIs.expected_discount_cost <= budget,
      profit_lift: Math.round((activeKPIs.expected_net_profit - 51015.75) * 100) / 100,
      color: '#10b981',
    },
  ];
}

