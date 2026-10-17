import React, { useState } from 'react';
import { DollarSign, Sliders, AlertCircle, Plus, Trash2, ShieldCheck, Info } from 'lucide-react';

const CURRENCIES = [
  { code: 'USD', symbol: '$', name: 'US Dollar ($)' },
  { code: 'EUR', symbol: '€', name: 'Euro (€)' },
  { code: 'GBP', symbol: '£', name: 'British Pound (£)' },
  { code: 'INR', symbol: '₹', name: 'Indian Rupee (₹)' },
  { code: 'CAD', symbol: 'C$', name: 'Canadian Dollar (C$)' },
];

export default function BudgetSettings({
  budget,
  onBudgetChange,
  currency,
  onCurrencyChange,
  discountTiers,
  onDiscountTiersChange,
  totalCustomers,
}) {
  const [newTierAmount, setNewTierAmount] = useState('');
  const [budgetError, setBudgetError] = useState(null);

  // Validate budget input
  const handleBudgetInput = (e) => {
    const rawVal = e.target.value;
    if (rawVal === '') {
      onBudgetChange(0);
      setBudgetError('Budget cannot be empty.');
      return;
    }

    const parsed = Number(rawVal);
    if (isNaN(parsed) || !isFinite(parsed)) {
      setBudgetError('Budget must be a finite numeric value.');
      return;
    }

    if (parsed < 0) {
      setBudgetError('Budget cannot be negative.');
      return;
    }

    setBudgetError(null);
    onBudgetChange(parsed);
  };

  // Add discount tier
  const handleAddTier = (e) => {
    e.preventDefault();
    const amount = parseFloat(newTierAmount);
    if (isNaN(amount) || amount <= 0) {
      alert('Please enter a positive numeric discount amount.');
      return;
    }
    if (discountTiers.includes(amount)) {
      alert('This discount tier already exists.');
      return;
    }

    const updated = [...discountTiers, amount].sort((a, b) => a - b);
    onDiscountTiersChange(updated);
    setNewTierAmount('');
  };

  // Remove discount tier
  const handleRemoveTier = (tierToRemove) => {
    if (discountTiers.length <= 1) {
      alert('At least one discount tier must be configured.');
      return;
    }
    const updated = discountTiers.filter((t) => t !== tierToRemove);
    onDiscountTiersChange(updated);
  };

  const activeSymbol = CURRENCIES.find((c) => c.code === currency)?.symbol || '$';

  return (
    <div className="card budget-card">
      <div className="card-header">
        <div className="title-with-icon">
          <div className="icon-badge icon-purple">
            <Sliders size={20} />
          </div>
          <div>
            <h2 className="card-title">Marketing Campaign Budget & Pricing Bounds</h2>
            <p className="card-subtitle">
              Configure campaign spending constraints and discrete promotional discount tiers for downstream optimization.
            </p>
          </div>
        </div>
      </div>

      <div className="budget-settings-grid">
        {/* Total Budget Configuration */}
        <div className="setting-box">
          <label className="form-label">
            <span>Total Campaign Budget Cap</span>
            <span className="label-badge">Required</span>
          </label>
          <div className="input-group">
            <span className="input-prefix">{activeSymbol}</span>
            <input
              type="number"
              className={`form-input ${budgetError ? 'input-error' : ''}`}
              value={budget}
              onChange={handleBudgetInput}
              min="0"
              step="500"
              placeholder="e.g. 5000"
            />
          </div>
          {budgetError ? (
            <div className="field-error">
              <AlertCircle size={13} />
              <span>{budgetError}</span>
            </div>
          ) : (
            <span className="field-help">
              Strict finite budget constraint for promotional distribution. Initial default: {activeSymbol}5,000.
            </span>
          )}
        </div>

        {/* Currency Selector */}
        <div className="setting-box">
          <label className="form-label">
            <span>Campaign Currency</span>
          </label>
          <select
            className="form-select"
            value={currency}
            onChange={(e) => onCurrencyChange(e.target.value)}
          >
            {CURRENCIES.map((curr) => (
              <option key={curr.code} value={curr.code}>
                {curr.name}
              </option>
            ))}
          </select>
          <span className="field-help">Applied across all budget and pricing metrics.</span>
        </div>
      </div>

      {/* Configurable Discount Tiers */}
      <div className="discount-tiers-section">
        <label className="form-label">
          <span>Configurable Promotional Discount Options</span>
          <span className="field-help-inline">(Configured face values eligible for customer targeting)</span>
        </label>

        <div className="tiers-pill-list">
          {discountTiers.map((tier) => (
            <div key={tier} className="tier-pill">
              <span className="tier-symbol">{activeSymbol}</span>
              <span className="tier-val">{tier}</span>
              <button
                type="button"
                className="tier-remove-btn"
                onClick={() => handleRemoveTier(tier)}
                title={`Remove ${activeSymbol}${tier} tier`}
              >
                <Trash2 size={13} />
              </button>
            </div>
          ))}

          {/* Add custom tier form */}
          <form onSubmit={handleAddTier} className="add-tier-form">
            <div className="add-tier-input-wrap">
              <span className="input-prefix-sm">{activeSymbol}</span>
              <input
                type="number"
                className="add-tier-input"
                placeholder="Custom (e.g. 15)"
                value={newTierAmount}
                onChange={(e) => setNewTierAmount(e.target.value)}
                min="1"
                step="5"
              />
            </div>
            <button type="submit" className="btn btn-xs btn-primary">
              <Plus size={13} />
              <span>Add Tier</span>
            </button>
          </form>
        </div>
      </div>

      {/* Capacity & Constraint Feasibility Simulation */}
      <div className="budget-feasibility-card">
        <div className="feasibility-header">
          <ShieldCheck size={16} className="text-emerald" />
          <span className="feasibility-title">Marketing Budget Constraint Simulation</span>
        </div>
        <div className="feasibility-grid">
          {discountTiers.map((tier) => {
            const maxCoupons = tier > 0 ? Math.floor(budget / tier) : 0;
            const populationCovered =
              totalCustomers > 0 ? Math.min(100, Math.round((maxCoupons / totalCustomers) * 100)) : 0;

            return (
              <div key={tier} className="feasibility-item">
                <span className="feasibility-label">Tier {activeSymbol}{tier} Max Capacity:</span>
                <span className="feasibility-value">{maxCoupons.toLocaleString()} recipients</span>
                <span className="feasibility-caption">
                  (~{populationCovered}% of {totalCustomers.toLocaleString()} active customer base)
                </span>
              </div>
            );
          })}
        </div>
        <div className="feasibility-note">
          <Info size={13} className="text-cyan" />
          <span>
            Optimization Guardrail: In Week 3, the dynamic pricing policy optimizer will solve the constrained knapsack
            allocation problem to assign these discount tiers without exceeding the {activeSymbol}{budget.toLocaleString()} cap.
          </span>
        </div>
      </div>
    </div>
  );
}
