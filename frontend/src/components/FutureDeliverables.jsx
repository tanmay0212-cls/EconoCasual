import React from 'react';
import { Lock, Construction, Sparkles, SlidersHorizontal, BarChart2, ShieldAlert } from 'lucide-react';

export default function FutureDeliverables() {
  return (
    <div className="future-sections-grid">
      {/* Reserved Section 1: Future Uplift Charts (Week 2) */}
      <div className="card placeholder-card">
        <div className="card-header">
          <div className="title-with-icon">
            <div className="icon-badge icon-amber">
              <Construction size={18} />
            </div>
            <div>
              <div className="future-badge-row">
                <h3 className="card-title">Causal Uplift & Treatment Effect Curves</h3>
                <span className="badge badge-pending">Week 2 Deliverable</span>
              </div>
              <p className="card-subtitle">
                Conditional Average Treatment Effect (CATE) & Heterogeneous Elasticity
              </p>
            </div>
          </div>
        </div>

        <div className="placeholder-content">
          <div className="placeholder-illustration">
            <div className="wireframe-chart">
              <div className="wireframe-bar bar-1" />
              <div className="wireframe-bar bar-2" />
              <div className="wireframe-bar bar-3" />
              <div className="wireframe-bar bar-4" />
              <div className="wireframe-line" />
            </div>
            <div className="wireframe-lock-overlay">
              <Lock size={20} className="text-amber" />
              <span className="lock-text">Double Machine Learning (DML) Model Not Trained Yet</span>
            </div>
          </div>

          <div className="alert alert-neutral">
            <div className="alert-header">
              <ShieldAlert size={14} className="text-amber" />
              <strong>Methodological Notice (Week 1 Scope)</strong>
            </div>
            <p>
              Causal uplift estimates are <strong>not yet calculated</strong>. In Week 2, we will implement
              Double Machine Learning (Chernozhukov et al.) with 5-fold cross-fitting and residualization
              to recover unconfounded customer-level treatment effects.
            </p>
          </div>
        </div>
      </div>

      {/* Reserved Section 2: Future Discount Allocation Results (Week 3) */}
      <div className="card placeholder-card">
        <div className="card-header">
          <div className="title-with-icon">
            <div className="icon-badge icon-purple">
              <SlidersHorizontal size={18} />
            </div>
            <div>
              <div className="future-badge-row">
                <h3 className="card-title">Budget-Constrained Discount Allocation</h3>
                <span className="badge badge-pending">Week 3 Deliverable</span>
              </div>
              <p className="card-subtitle">
                Optimal Knapsack Policy under Marketing Budget Constraints
              </p>
            </div>
          </div>
        </div>

        <div className="placeholder-content">
          <div className="placeholder-illustration">
            <div className="wireframe-allocation-table">
              <div className="wireframe-row row-header" />
              <div className="wireframe-row" />
              <div className="wireframe-row" />
              <div className="wireframe-row" />
            </div>
            <div className="wireframe-lock-overlay">
              <Lock size={20} className="text-purple" />
              <span className="lock-text">Constrained Policy Optimizer Inactive</span>
            </div>
          </div>

          <div className="alert alert-neutral">
            <div className="alert-header">
              <ShieldAlert size={14} className="text-purple" />
              <strong>Optimization Notice (Week 1 Scope)</strong>
            </div>
            <p>
              Automated customer discount allocation and expected ROI optimizations are
              <strong> not yet active</strong>. In Week 3, integer linear programming will maximize incremental
              conversions subject to the configured campaign budget cap.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
