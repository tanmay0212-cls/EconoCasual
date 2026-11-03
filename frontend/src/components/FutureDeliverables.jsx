import React from 'react';
import { Lock, Construction, SlidersHorizontal, ShieldAlert, CheckCircle2, ArrowRight } from 'lucide-react';

export default function FutureDeliverables({ onNavigateToMatrix }) {
  return (
    <div className="future-sections-grid">
      {/* Section 1: Causal Uplift Curves (Week 2 Model Trained) */}
      <div className="card placeholder-card">
        <div className="card-header">
          <div className="title-with-icon">
            <div className="icon-badge icon-emerald">
              <CheckCircle2 size={18} />
            </div>
            <div>
              <div className="future-badge-row">
                <h3 className="card-title">Causal Multi-Tier Uplift Estimation</h3>
                <span className="badge badge-success">Week 2 Completed</span>
              </div>
              <p className="card-subtitle">
                Conditional Average Treatment Effect (CATE) & Multi-Valued Elasticity (EconML LinearDRLearner)
              </p>
            </div>
          </div>
        </div>

        <div className="placeholder-content">
          <div className="alert alert-neutral">
            <div className="alert-header">
              <CheckCircle2 size={14} className="text-emerald" />
              <strong>Causal DML Estimation Active</strong>
            </div>
            <p>
              Multi-treatment DML model (<code>models/multitreatment_model.joblib</code>) trained with 5-fold cross-fitting.
              Counterfactual purchase probabilities estimated for $0, $10, and $20 discount tiers across all 3,000 customers.
            </p>
          </div>
        </div>
      </div>

      {/* Section 2: Budget-Constrained Discount Allocation Results (Week 3 Completed) */}
      <div className="card placeholder-card">
        <div className="card-header">
          <div className="title-with-icon">
            <div className="icon-badge icon-purple">
              <SlidersHorizontal size={18} />
            </div>
            <div>
              <div className="future-badge-row">
                <h3 className="card-title">Budget-Constrained Discount Allocation</h3>
                <span className="badge badge-success">Week 3 Completed</span>
              </div>
              <p className="card-subtitle">
                Optimal Knapsack Policy under Strict Marketing Budget Constraints (SciPy MILP / HiGHS)
              </p>
            </div>
          </div>
        </div>

        <div className="placeholder-content">
          <div className="alert alert-neutral">
            <div className="alert-header">
              <CheckCircle2 size={14} className="text-emerald" />
              <strong>Prescriptive Allocation Matrix Available</strong>
            </div>
            <p>
              Integer linear programming solved across 3,000 customers. 836 customers allocated $10 coupons,
              2 allocated $20 coupons, achieving $53,082.35 expected net profit within the $5,000 budget cap.
            </p>
          </div>

          {onNavigateToMatrix && (
            <div style={{ marginTop: '1rem' }}>
              <button
                type="button"
                className="btn btn-primary"
                onClick={onNavigateToMatrix}
                style={{ width: '100%', justifyContent: 'center' }}
              >
                <span>View Allocation Matrix & Run Optimization</span>
                <ArrowRight size={16} />
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
