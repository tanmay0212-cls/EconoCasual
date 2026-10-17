import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import DatasetUploader from '../components/DatasetUploader';

describe('DatasetUploader Component', () => {
  it('renders validation error messages clearly when schema validation fails', () => {
    const mockValidationResult = {
      isValid: false,
      totalRows: 250,
      validRows: 0,
      columns: ['customer_id', 'prior_purchases'],
      missingValues: { customer_id: 0, prior_spend: 250 },
      errors: ['Missing required columns: prior_spend, engagement_score', 'Column "discount_offered" must be strictly binary (0 or 1).'],
      warnings: ['Dataset contains only 250 rows.'],
      preview: [],
      metrics: null,
    };

    render(
      <DatasetUploader
        validationResult={mockValidationResult}
        onFileUpload={vi.fn()}
        onLoadSynthetic={vi.fn()}
        onDownloadSample={vi.fn()}
        isLoading={false}
        fileName="corrupted_campaign.csv"
      />
    );

    expect(screen.getByText(/Dataset Schema Validation Failed/i)).toBeDefined();
    expect(screen.getByText(/Missing required columns: prior_spend/i)).toBeDefined();
    expect(screen.getByText(/must be strictly binary/i)).toBeDefined();
    expect(screen.getByText(/Dataset Quality Notice/i)).toBeDefined();
  });

  it('renders preview table with first 10 rows when dataset is valid', () => {
    const mockValidationResult = {
      isValid: true,
      totalRows: 10,
      validRows: 10,
      columns: [
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
      ],
      missingValues: {},
      errors: [],
      warnings: [],
      preview: [
        {
          customer_id: 'CUST_00001',
          prior_purchases: 5,
          prior_spend: 150.0,
          engagement_score: 80.5,
          customer_tenure: 12,
          discount_offered: 1,
          discount_amount: 15.0,
          purchased: 1,
          revenue: 85.0,
          campaign_channel: 'email',
        },
      ],
      metrics: {
        totalCustomers: 10,
        validRecords: 10,
        treatmentRate: 50,
        conversionRate: 40,
        totalRevenue: 850,
        channelCounts: { email: 10 },
      },
    };

    render(
      <DatasetUploader
        validationResult={mockValidationResult}
        onFileUpload={vi.fn()}
        onLoadSynthetic={vi.fn()}
        onDownloadSample={vi.fn()}
        isLoading={false}
        fileName="sample_retail_campaign.csv"
      />
    );

    expect(screen.getByText(/Schema Validated Successfully/i)).toBeDefined();
    expect(screen.getByText(/Dataset Preview/i)).toBeDefined();
    expect(screen.getByText('CUST_00001')).toBeDefined();
  });
});
