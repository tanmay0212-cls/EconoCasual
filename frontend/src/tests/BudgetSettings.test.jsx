import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import BudgetSettings from '../components/BudgetSettings';

describe('BudgetSettings Component', () => {
  it('renders initial budget value and discount options', () => {
    const handleBudgetChange = vi.fn();
    const handleCurrencyChange = vi.fn();
    const handleDiscountTiersChange = vi.fn();

    render(
      <BudgetSettings
        budget={5000}
        onBudgetChange={handleBudgetChange}
        currency="USD"
        onCurrencyChange={handleCurrencyChange}
        discountTiers={[10, 20]}
        onDiscountTiersChange={handleDiscountTiersChange}
        totalCustomers={2500}
      />
    );

    // Verify budget input displays 5000
    const input = screen.getByPlaceholderText('e.g. 5000');
    expect(input.value).toBe('5000');

    // Verify discount tiers $10 and $20 are rendered
    expect(screen.getByText('10')).toBeDefined();
    expect(screen.getByText('20')).toBeDefined();

    // Verify capacity simulation is present
    expect(screen.getByText(/Constraint Simulation/i)).toBeDefined();
  });

  it('validates and disallows negative budget values', () => {
    const handleBudgetChange = vi.fn();

    render(
      <BudgetSettings
        budget={5000}
        onBudgetChange={handleBudgetChange}
        currency="USD"
        onCurrencyChange={vi.fn()}
        discountTiers={[10, 20]}
        onDiscountTiersChange={vi.fn()}
        totalCustomers={2500}
      />
    );

    const input = screen.getByPlaceholderText('e.g. 5000');
    fireEvent.change(input, { target: { value: '-200' } });

    expect(screen.getByText(/cannot be negative/i)).toBeDefined();
  });
});
