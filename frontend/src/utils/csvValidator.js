/**
 * Schema and validation utilities for retail marketing campaign datasets.
 * Ensures strict causal variable taxonomy and data integrity.
 */

export const REQUIRED_COLUMNS = [
  'customer_id',
  'prior_purchases',
  'prior_spend',
  'engagement_score',
  'customer_tenure',
  'discount_offered',
  'discount_amount',
  'purchased',
  'revenue',
  'campaign_channel',
];

export const VARIABLE_ROLES = {
  discount_offered: {
    role: 'treatment',
    label: 'Treatment (T)',
    color: '#10b981',
    description: 'Promotional discount offered (binary: 0 or 1)',
  },
  purchased: {
    role: 'outcome',
    label: 'Primary Outcome (Y)',
    color: '#06b6d4',
    description: 'Customer purchase conversion (binary: 0 or 1)',
  },
  prior_purchases: {
    role: 'confounder',
    label: 'Confounder (W)',
    color: '#f59e0b',
    description: 'Pre-treatment purchase count',
  },
  prior_spend: {
    role: 'confounder',
    label: 'Confounder (W)',
    color: '#f59e0b',
    description: 'Pre-treatment cumulative spend',
  },
  engagement_score: {
    role: 'confounder',
    label: 'Confounder (W)',
    color: '#f59e0b',
    description: 'Pre-treatment digital activity score (0-100)',
  },
  customer_tenure: {
    role: 'confounder',
    label: 'Confounder (W)',
    color: '#f59e0b',
    description: 'Account tenure in months',
  },
  campaign_channel: {
    role: 'confounder',
    label: 'Confounder (W)',
    color: '#f59e0b',
    description: 'Marketing outreach delivery medium',
  },
  discount_amount: {
    role: 'post_treatment',
    label: 'Post-Treatment Mediator',
    color: '#a855f7',
    description: 'Realized dollar discount (excluded from confounders)',
  },
  revenue: {
    role: 'post_treatment',
    label: 'Downstream Terminal Outcome',
    color: '#ec4899',
    description: 'Realized basket spend post-discount (excluded from confounders)',
  },
  customer_id: {
    role: 'identifier',
    label: 'Identifier',
    color: '#94a3b8',
    description: 'Unique customer identifier code',
  },
};

/**
 * Validate parsed CSV rows against EconoCausal schema and domain rules.
 * @param {Array<Object>} rows - Array of row objects from PapaParse
 * @returns {Object} Validation summary, errors, warnings, metrics, and first 10 rows
 */
export function validateDatasetRows(rows) {
  if (!rows || !Array.isArray(rows) || rows.length === 0) {
    return {
      isValid: false,
      totalRows: 0,
      validRows: 0,
      columns: [],
      missingValues: {},
      errors: ['The uploaded file is empty or could not be parsed.'],
      warnings: [],
      preview: [],
      metrics: null,
    };
  }

  const columns = Object.keys(rows[0] || {});
  const errors = [];
  const warnings = [];

  // 1. Required column check
  const missingCols = REQUIRED_COLUMNS.filter((col) => !columns.includes(col));
  if (missingCols.length > 0) {
    errors.push(`Missing required columns: ${missingCols.join(', ')}`);
  }

  // 2. Minimum row count check
  if (rows.length < 10) {
    errors.push(`Dataset contains only ${rows.length} rows. A valid campaign dataset requires at least 10 rows.`);
  } else if (rows.length < 1000) {
    warnings.push(`Dataset contains ${rows.length} rows. Week 1 specifications recommend at least 1,000 records for causal modeling.`);
  }

  // 3. Missing values analysis
  const missingCounts = {};
  REQUIRED_COLUMNS.forEach((col) => {
    missingCounts[col] = 0;
  });

  let validRowCount = 0;
  let treatmentCount = 0;
  let purchaseCount = 0;
  let totalRevenue = 0;
  const channelCounts = {};

  rows.forEach((row) => {
    let rowHasMissing = false;

    REQUIRED_COLUMNS.forEach((col) => {
      const val = row[col];
      if (val === undefined || val === null || val === '' || String(val).trim() === '') {
        missingCounts[col] = (missingCounts[col] || 0) + 1;
        rowHasMissing = true;
      }
    });

    if (!rowHasMissing) {
      validRowCount++;

      // Metric calculations
      const tVal = Number(row.discount_offered);
      const yVal = Number(row.purchased);
      const revVal = Number(row.revenue);

      if (tVal === 1) treatmentCount++;
      if (yVal === 1) purchaseCount++;
      if (!isNaN(revVal)) totalRevenue += revVal;

      const ch = row.campaign_channel || 'unknown';
      channelCounts[ch] = (channelCounts[ch] || 0) + 1;
    }
  });

  // Check domain rules on valid rows
  if (missingCols.length === 0 && rows.length > 0) {
    // Binary checks
    const invalidTreatment = rows.some((r) => {
      const v = Number(r.discount_offered);
      return v !== 0 && v !== 1;
    });
    if (invalidTreatment) {
      errors.push('Column "discount_offered" must be strictly binary (0 or 1).');
    }

    const invalidPurchased = rows.some((r) => {
      const v = Number(r.purchased);
      return v !== 0 && v !== 1;
    });
    if (invalidPurchased) {
      errors.push('Column "purchased" must be strictly binary (0 or 1).');
    }

    // Non-negative checks
    if (rows.some((r) => Number(r.prior_purchases) < 0)) {
      errors.push('Column "prior_purchases" contains negative values.');
    }
    if (rows.some((r) => Number(r.prior_spend) < 0)) {
      errors.push('Column "prior_spend" contains negative values.');
    }
    if (rows.some((r) => Number(r.customer_tenure) < 0)) {
      errors.push('Column "customer_tenure" contains negative values.');
    }
    if (rows.some((r) => Number(r.discount_amount) < 0)) {
      errors.push('Column "discount_amount" contains negative values.');
    }
    if (rows.some((r) => Number(r.revenue) < 0)) {
      errors.push('Column "revenue" contains negative values.');
    }

    // Post-treatment consistency checks
    const nonZeroDiscountWithoutOffer = rows.filter(
      (r) => Number(r.discount_offered) === 0 && Number(r.discount_amount) > 0
    ).length;
    if (nonZeroDiscountWithoutOffer > 0) {
      errors.push(`Found ${nonZeroDiscountWithoutOffer} records where discount_offered=0 but discount_amount > 0.`);
    }

    const nonZeroRevenueWithoutPurchase = rows.filter(
      (r) => Number(r.purchased) === 0 && Number(r.revenue) > 0
    ).length;
    if (nonZeroRevenueWithoutPurchase > 0) {
      errors.push(`Found ${nonZeroRevenueWithoutPurchase} records where purchased=0 but revenue > 0.`);
    }
  }

  const isValid = errors.length === 0 && validRowCount > 0;

  const metrics = isValid
    ? {
        totalCustomers: rows.length,
        validRecords: validRowCount,
        treatmentRate: rows.length > 0 ? (treatmentCount / rows.length) * 100 : 0,
        conversionRate: rows.length > 0 ? (purchaseCount / rows.length) * 100 : 0,
        totalRevenue: Math.round(totalRevenue * 100) / 100,
        channelCounts,
      }
    : null;

  return {
    isValid,
    totalRows: rows.length,
    validRows: validRowCount,
    columns,
    missingValues: missingCounts,
    errors,
    warnings,
    preview: rows.slice(0, 10),
    metrics,
  };
}
