import { describe, it, expect } from 'vitest';
import { validateDatasetRows, REQUIRED_COLUMNS } from '../utils/csvValidator';

describe('CSV Validator & Schema Checks', () => {
  const sampleValidRows = [
    {
      customer_id: 'CUST_0001',
      prior_purchases: '5',
      prior_spend: '250.50',
      engagement_score: '65.2',
      customer_tenure: '12',
      discount_offered: '1',
      discount_amount: '15.0',
      purchased: '1',
      revenue: '89.50',
      campaign_channel: 'email',
    },
    {
      customer_id: 'CUST_0002',
      prior_purchases: '2',
      prior_spend: '80.00',
      engagement_score: '45.0',
      customer_tenure: '4',
      discount_offered: '0',
      discount_amount: '0.0',
      purchased: '0',
      revenue: '0.0',
      campaign_channel: 'push',
    },
  ];

  it('validates correct campaign dataset rows', () => {
    // Fill to 10 rows to meet minimum row requirement
    const tenRows = Array(10).fill(null).map((_, i) => ({
      ...sampleValidRows[i % 2],
      customer_id: `CUST_${i + 1}`,
    }));

    const result = validateDatasetRows(tenRows);
    expect(result.isValid).toBe(true);
    expect(result.errors.length).toBe(0);
    expect(result.totalRows).toBe(10);
    expect(result.validRows).toBe(10);
    expect(result.metrics).not.toBeNull();
    expect(result.metrics.treatmentRate).toBe(50);
  });

  it('rejects empty or null dataset', () => {
    const result = validateDatasetRows([]);
    expect(result.isValid).toBe(false);
    expect(result.errors.length).toBeGreaterThan(0);
    expect(result.errors[0]).toContain('empty');
  });

  it('detects missing required columns', () => {
    const corruptedRows = sampleValidRows.map((r) => {
      const copy = { ...r };
      delete copy.prior_spend;
      return copy;
    });

    const result = validateDatasetRows(corruptedRows);
    expect(result.isValid).toBe(false);
    expect(result.errors.some((e) => e.includes('prior_spend'))).toBe(true);
  });

  it('flags non-binary treatment or outcome values', () => {
    const badBinaryRows = Array(10).fill(null).map((_, i) => ({
      ...sampleValidRows[0],
      customer_id: `CUST_${i + 1}`,
      discount_offered: '5', // Invalid
    }));

    const result = validateDatasetRows(badBinaryRows);
    expect(result.isValid).toBe(false);
    expect(result.errors.some((e) => e.includes('strictly binary'))).toBe(true);
  });

  it('accurately counts missing values per column', () => {
    const rowsWithNulls = Array(10).fill(null).map((_, i) => ({
      ...sampleValidRows[0],
      customer_id: `CUST_${i + 1}`,
      engagement_score: i < 3 ? '' : '75.0', // 3 missing values
    }));

    const result = validateDatasetRows(rowsWithNulls);
    expect(result.missingValues.engagement_score).toBe(3);
  });
});
