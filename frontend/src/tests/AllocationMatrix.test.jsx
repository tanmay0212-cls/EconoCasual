import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import AllocationMatrix from '../components/AllocationMatrix';

const MOCK_CSV = `customer_id,predicted_purchase_probability_0,predicted_purchase_probability_10,predicted_purchase_probability_20,predicted_effect_10_vs_0,predicted_effect_20_vs_0,recommended_discount,expected_discount_cost,expected_revenue,expected_net_profit
OPT_012001,0.369754,0.520709,0.575259,0.150955,0.205505,0,0.0,31.49,10.05
OPT_012002,0.70823,0.83962,0.887738,0.13139,0.179508,0,0.0,31.27,12.67
OPT_012009,0.398235,0.564628,0.604475,0.166393,0.20624,10,5.65,58.37,23.95
OPT_012025,0.410000,0.620000,0.740000,0.210000,0.330000,20,14.80,62.16,28.40`;

const MOCK_METADATA = {
  evaluation_population: 3000,
  default_budget: 5000.0,
  objective_type: 'net_profit',
  solver: 'scipy.optimize.milp (HiGHS)',
  kpis: {
    total_customers: 3000,
    count_discount_0: 2162,
    count_discount_10: 836,
    count_discount_20: 2,
    expected_revenue: 121447.78,
    expected_discount_cost: 4604.20,
    expected_net_profit: 53082.35,
    remaining_budget: 395.80,
    budget_utilization_pct: 92.08,
  },
  strategies: [
    {
      name: 'No Discount ($0 Baseline)',
      short_name: 'No Discount',
      discount: '$0',
      cost: 0.0,
      revenue: 106473.01,
      profit: 51015.75,
      compliant: true,
      profit_lift: 0.0,
    },
    {
      name: 'Blanket $10 Promotion',
      short_name: 'Blanket $10',
      discount: '$10 to All',
      cost: 16246.60,
      revenue: 137756.90,
      profit: 47828.63,
      compliant: false,
      profit_lift: -3187.12,
    },
    {
      name: 'Blanket $20 Promotion',
      short_name: 'Blanket $20',
      discount: '$20 to All',
      cost: 35705.84,
      revenue: 146789.04,
      profit: 31841.21,
      compliant: false,
      profit_lift: -19174.54,
    },
    {
      name: 'Optimized Personalized (DML + SciPy)',
      short_name: 'Optimized Policy',
      discount: 'Personalized',
      cost: 4604.20,
      revenue: 121447.78,
      profit: 53082.35,
      compliant: true,
      profit_lift: 2066.60,
    },
  ],
};

describe('AllocationMatrix Component', () => {
  let originalFetch;

  beforeEach(() => {
    originalFetch = global.fetch;
    global.fetch = vi.fn((url) => {
      const urlStr = String(url);
      if (urlStr.includes('allocation_matrix.csv')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          text: () => Promise.resolve(MOCK_CSV),
        });
      }
      if (urlStr.includes('optimization_metadata.json')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve(MOCK_METADATA),
        });
      }
      return Promise.reject(new Error(`Unhandled URL: ${urlStr}`));
    });
  });

  afterEach(() => {
    global.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it('renders budget control input and initial $5,000 cap', async () => {
    render(<AllocationMatrix initialBudget={5000} />);

    expect(screen.getByText(/Prescriptive Discount Allocation Matrix/i)).toBeDefined();

    const input = screen.getByLabelText(/Marketing Campaign Budget Cap/i);
    expect(input).toBeDefined();
    expect(input.value).toBe('5000');

    await waitFor(() => {
      expect(screen.queryByText(/Solving Mixed-Integer/i)).toBeNull();
    });
  });

  it('displays counterfactual notice banner distinguishing expectations from realized revenue', async () => {
    render(<AllocationMatrix initialBudget={5000} />);

    await waitFor(() => {
      expect(screen.getByText(/Methodological Notice — Counterfactual Projections/i)).toBeDefined();
    });
  });

  it('renders all 8 summary KPI cards accurately', async () => {
    render(<AllocationMatrix initialBudget={5000} />);

    await waitFor(() => {
      expect(screen.queryByText(/Solving Mixed-Integer/i)).toBeNull();
    }, { timeout: 3000 });

    expect(screen.getByText('Total Customers')).toBeDefined();
    expect(screen.getByText('No Discount ($0)')).toBeDefined();
    expect(screen.getByText('$10 Discount')).toBeDefined();
    expect(screen.getByText('$20 Discount')).toBeDefined();
    expect(screen.getAllByText('Expected Revenue').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('Discount Expenditure')).toBeDefined();
    expect(screen.getAllByText('Expected Net Profit').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('Remaining Budget')).toBeDefined();
  });

  it('renders strategy benchmark comparison cards with compliance status', async () => {
    render(<AllocationMatrix initialBudget={5000} />);

    await waitFor(() => {
      expect(screen.queryByText(/Solving Mixed-Integer/i)).toBeNull();
    }, { timeout: 3000 });

    expect(screen.getByText(/Strategy Benchmark Comparison/i)).toBeDefined();
    expect(screen.getByText('No Discount ($0 Baseline)')).toBeDefined();
    expect(screen.getByText('Blanket $10 Promotion')).toBeDefined();
    expect(screen.getByText('Blanket $20 Promotion')).toBeDefined();
    expect(screen.getByText('Optimized Personalized (DML + SciPy)')).toBeDefined();

    const overBudgetBadges = screen.getAllByText(/Over Budget/i);
    expect(overBudgetBadges.length).toBeGreaterThanOrEqual(2);
  });

  it('renders customer allocation table with discount badges and allows search filtering', async () => {
    render(<AllocationMatrix initialBudget={5000} />);

    await waitFor(() => {
      expect(screen.getByText('OPT_012001')).toBeDefined();
      expect(screen.getByText('OPT_012009')).toBeDefined();
    });

    const searchInput = screen.getByPlaceholderText(/Search Customer ID/i);
    fireEvent.change(searchInput, { target: { value: 'OPT_012009' } });

    expect(screen.getByText('OPT_012009')).toBeDefined();
    expect(screen.queryByText('OPT_012001')).toBeNull();
  });

  it('allows filtering by discount tier buttons', async () => {
    render(<AllocationMatrix initialBudget={5000} />);

    await waitFor(() => {
      expect(screen.getByText('OPT_012001')).toBeDefined();
    });

    const pill10 = screen.getByRole('button', { name: '$10' });
    fireEvent.click(pill10);

    expect(screen.getByText('OPT_012009')).toBeDefined();
    expect(screen.queryByText('OPT_012001')).toBeNull();
  });

  it('expands counterfactual uplift details on button click', async () => {
    render(<AllocationMatrix initialBudget={5000} />);

    await waitFor(() => {
      expect(screen.getByText('OPT_012009')).toBeDefined();
    });

    const expandButtons = screen.getAllByText('View Uplift');
    fireEvent.click(expandButtons[0]);

    await waitFor(() => {
      expect(screen.getByText(/Counterfactual Predictions across Discount Options/i)).toBeDefined();
      expect(screen.getByText(/P\(Purchase \| \$0\)/i)).toBeDefined();
      expect(screen.getByText(/P\(Purchase \| \$10\)/i)).toBeDefined();
      expect(screen.getByText(/P\(Purchase \| \$20\)/i)).toBeDefined();
    });
  });

  it('handles and displays error state when data fails to load', async () => {
    global.fetch = vi.fn(() => Promise.reject(new Error('Network failure')));

    render(<AllocationMatrix initialBudget={5000} />);

    await waitFor(() => {
      expect(screen.getByText(/Optimization Engine Error/i)).toBeDefined();
      expect(screen.getByText(/Network failure/i)).toBeDefined();
    });
  });
});
