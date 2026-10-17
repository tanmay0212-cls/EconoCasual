import React from 'react';
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import Header from '../components/Header';

describe('Header Component', () => {
  it('renders EconoCausal title and subtitle', () => {
    render(
      <Header
        datasetSource="none"
        datasetName=""
        totalRows={0}
        onLoadSynthetic={() => {}}
        onDownloadSample={() => {}}
        isLoading={false}
      />
    );

    expect(screen.getByText('EconoCausal')).toBeDefined();
    expect(screen.getByText('Causal AI for Dynamic Pricing')).toBeDefined();
    expect(screen.getByText('No Dataset Active')).toBeDefined();
  });

  it('displays synthetic sample data status indicator', () => {
    render(
      <Header
        datasetSource="synthetic"
        datasetName="synthetic_retail_campaign.csv"
        totalRows={2500}
        onLoadSynthetic={() => {}}
        onDownloadSample={() => {}}
        isLoading={false}
      />
    );

    expect(screen.getByText(/Synthetic Sample Campaign/i)).toBeDefined();
  });

  it('displays uploaded dataset status indicator', () => {
    render(
      <Header
        datasetSource="uploaded"
        datasetName="custom_campaign.csv"
        totalRows={1400}
        onLoadSynthetic={() => {}}
        onDownloadSample={() => {}}
        isLoading={false}
      />
    );

    expect(screen.getByText(/Uploaded: custom_campaign.csv/i)).toBeDefined();
  });
});
