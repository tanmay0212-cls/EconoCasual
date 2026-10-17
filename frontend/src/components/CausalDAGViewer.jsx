import React, { useState } from 'react';
import { GitFork, Info, AlertTriangle, ShieldAlert, ArrowRight } from 'lucide-react';

const DAG_NODES = [
  // Confounders (Column 1 - Left)
  { id: 'prior_purchases', label: 'Prior Purchases', x: 80, y: 70, role: 'confounder', desc: 'Pre-treatment order frequency. Confounder: targeted for loyalty discounts & has higher organic conversion.' },
  { id: 'prior_spend', label: 'Prior Spend ($)', x: 80, y: 140, role: 'confounder', desc: 'Pre-treatment lifetime spend. Confounder: VIP discount tiers & greater purchasing power.' },
  { id: 'engagement_score', label: 'Engagement Score', x: 80, y: 210, role: 'confounder', desc: 'Digital activity index (0-100). Confounder: high app usage drives coupon exposure and conversion.' },
  { id: 'customer_tenure', label: 'Customer Tenure', x: 80, y: 280, role: 'confounder', desc: 'Account age in months. Confounder: loyalty retention targeting and habituated buying.' },
  { id: 'campaign_channel', label: 'Campaign Channel', x: 80, y: 350, role: 'confounder', desc: 'Outreach medium (email, push, sms, social). Confounder: coupon delivery rate & channel baseline response.' },

  // Treatment (Column 2 - Mid Left)
  { id: 'discount_offered', label: 'Discount Offered (T)', x: 380, y: 150, role: 'treatment', desc: 'Marketing intervention: binary assignment of promotional discount code (0 = standard, 1 = discount).' },

  // Post-Treatment Mediator (Column 2 - Mid Lower)
  { id: 'discount_amount', label: 'Discount Amount', x: 380, y: 320, role: 'post_treatment', desc: 'Realized dollar value. Post-treatment mediator: strictly EXCLUDED as a confounder to avoid mediator bias.' },

  // Primary Outcome (Column 3 - Right)
  { id: 'purchased', label: 'Purchased (Y)', x: 680, y: 150, role: 'outcome', desc: 'Primary conversion outcome: binary purchase conversion during campaign (0 or 1).' },

  // Downstream Terminal Outcome (Column 3 - Right Lower)
  { id: 'revenue', label: 'Revenue ($)', x: 680, y: 320, role: 'downstream_outcome', desc: 'Realized basket spend post-discount. Downstream outcome: strictly EXCLUDED from backdoor adjustment.' },
];

const DAG_EDGES = [
  // Confounders to Treatment
  { from: 'prior_purchases', to: 'discount_offered', type: 'confounding' },
  { from: 'prior_spend', to: 'discount_offered', type: 'confounding' },
  { from: 'engagement_score', to: 'discount_offered', type: 'confounding' },
  { from: 'customer_tenure', to: 'discount_offered', type: 'confounding' },
  { from: 'campaign_channel', to: 'discount_offered', type: 'confounding' },

  // Confounders to Outcome
  { from: 'prior_purchases', to: 'purchased', type: 'confounding' },
  { from: 'prior_spend', to: 'purchased', type: 'confounding' },
  { from: 'engagement_score', to: 'purchased', type: 'confounding' },
  { from: 'customer_tenure', to: 'purchased', type: 'confounding' },
  { from: 'campaign_channel', to: 'purchased', type: 'confounding' },

  // Causal Treatment to Outcome
  { from: 'discount_offered', to: 'purchased', type: 'causal_effect' },

  // Post-treatment downstream links
  { from: 'discount_offered', to: 'discount_amount', type: 'post_treatment' },
  { from: 'purchased', to: 'revenue', type: 'downstream' },
  { from: 'discount_amount', to: 'revenue', type: 'downstream' },
];

export default function CausalDAGViewer() {
  const [selectedNode, setSelectedNode] = useState(DAG_NODES[5]); // Default selected: Treatment

  const getNodeColor = (role) => {
    switch (role) {
      case 'treatment':
        return '#10b981'; // Emerald
      case 'outcome':
        return '#06b6d4'; // Cyan
      case 'confounder':
        return '#f59e0b'; // Amber
      case 'post_treatment':
        return '#a855f7'; // Purple
      case 'downstream_outcome':
        return '#ec4899'; // Pink
      default:
        return '#94a3b8';
    }
  };

  return (
    <div className="card dag-card">
      <div className="card-header">
        <div className="title-with-icon">
          <div className="icon-badge icon-emerald">
            <GitFork size={20} />
          </div>
          <div>
            <h2 className="card-title">Causal Directed Acyclic Graph (DAG) Specification</h2>
            <p className="card-subtitle">
              Formalization of structural domain relationships. Identifies required Back-Door conditioning sets
              while strictly excluding post-treatment colliders and mediators.
            </p>
          </div>
        </div>
      </div>

      {/* Methodological Alert */}
      <div className="alert alert-info">
        <div className="alert-header">
          <Info size={16} />
          <strong>Causal Methodology Guardrail (Week 1 Scope)</strong>
        </div>
        <p>
          The Causal DAG reflects structural hypotheses based on retail marketing dynamics.
          <strong> The graph does not mechanically prove causality</strong>; it explicitly defines the minimal
          adjustment set required for identification. Numerical treatment effect estimation via
          Double Machine Learning (DML) will be conducted in Week 2.
        </p>
      </div>

      {/* SVG Diagram Canvas */}
      <div className="dag-canvas-container">
        <svg viewBox="0 0 820 420" className="dag-svg">
          <defs>
            <marker
              id="arrowhead-confounding"
              markerWidth="8"
              markerHeight="6"
              refX="7"
              refY="3"
              orient="auto"
            >
              <polygon points="0 0, 8 3, 0 6" fill="#f59e0b80" />
            </marker>
            <marker
              id="arrowhead-causal"
              markerWidth="10"
              markerHeight="8"
              refX="9"
              refY="4"
              orient="auto"
            >
              <polygon points="0 0, 10 4, 0 8" fill="#10b981" />
            </marker>
            <marker
              id="arrowhead-post"
              markerWidth="8"
              markerHeight="6"
              refX="7"
              refY="3"
              orient="auto"
            >
              <polygon points="0 0, 8 3, 0 6" fill="#a855f780" />
            </marker>
          </defs>

          {/* Render Directed Edges */}
          {DAG_EDGES.map((edge, idx) => {
            const sourceNode = DAG_NODES.find((n) => n.id === edge.from);
            const targetNode = DAG_NODES.find((n) => n.id === edge.to);
            if (!sourceNode || !targetNode) return null;

            const isCausal = edge.type === 'causal_effect';
            const isPost = edge.type === 'post_treatment' || edge.type === 'downstream';

            const strokeColor = isCausal ? '#10b981' : isPost ? '#a855f760' : '#f59e0b40';
            const markerId = isCausal ? 'url(#arrowhead-causal)' : isPost ? 'url(#arrowhead-post)' : 'url(#arrowhead-confounding)';
            const strokeWidth = isCausal ? 3 : 1.5;

            // Curved paths for cleaner visual routing
            const dx = targetNode.x - sourceNode.x;
            const dy = targetNode.y - sourceNode.y;
            const curve = isCausal ? 0 : dy * 0.2;
            const midX = (sourceNode.x + targetNode.x) / 2;
            const midY = (sourceNode.y + targetNode.y) / 2 + curve;

            return (
              <path
                key={idx}
                d={`M ${sourceNode.x + 65} ${sourceNode.y} Q ${midX} ${midY} ${targetNode.x - 70} ${targetNode.y}`}
                fill="none"
                stroke={strokeColor}
                strokeWidth={strokeWidth}
                markerEnd={markerId}
                className={isCausal ? 'edge-causal-glow' : ''}
              />
            );
          })}

          {/* Render Nodes */}
          {DAG_NODES.map((node) => {
            const isSelected = selectedNode?.id === node.id;
            const color = getNodeColor(node.role);

            return (
              <g
                key={node.id}
                transform={`translate(${node.x}, ${node.y})`}
                onClick={() => setSelectedNode(node)}
                className="dag-node-group"
                style={{ cursor: 'pointer' }}
              >
                {/* Node Box */}
                <rect
                  x="-70"
                  y="-22"
                  width="140"
                  height="44"
                  rx="8"
                  fill="#111827"
                  stroke={isSelected ? '#ffffff' : color}
                  strokeWidth={isSelected ? 2.5 : 1.5}
                  className="dag-node-rect"
                />
                {/* Role Dot */}
                <circle cx="-52" cy="0" r="5" fill={color} />
                {/* Node Label */}
                <text
                  x="-38"
                  y="4"
                  fill="#f1f5f9"
                  fontSize="11.5"
                  fontWeight="600"
                  fontFamily="Inter, system-ui, sans-serif"
                >
                  {node.label}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Legend & Interactive Node Inspector */}
      <div className="dag-footer-grid">
        {/* Legend */}
        <div className="dag-legend">
          <span className="legend-title">Causal Taxonomy Legend:</span>
          <div className="legend-items">
            <span className="legend-item"><span className="legend-dot" style={{ backgroundColor: '#10b981' }} /> Treatment (T)</span>
            <span className="legend-item"><span className="legend-dot" style={{ backgroundColor: '#06b6d4' }} /> Primary Outcome (Y)</span>
            <span className="legend-item"><span className="legend-dot" style={{ backgroundColor: '#f59e0b' }} /> Observed Confounder (W)</span>
            <span className="legend-item"><span className="legend-dot" style={{ backgroundColor: '#a855f7' }} /> Post-Treatment Mediator</span>
            <span className="legend-item"><span className="legend-dot" style={{ backgroundColor: '#ec4899' }} /> Terminal Outcome</span>
          </div>
        </div>

        {/* Selected Node Details */}
        {selectedNode && (
          <div className="node-detail-card" style={{ borderColor: `${getNodeColor(selectedNode.role)}60` }}>
            <div className="node-detail-header">
              <span className="detail-name">{selectedNode.label}</span>
              <span className="detail-role" style={{ backgroundColor: `${getNodeColor(selectedNode.role)}25`, color: getNodeColor(selectedNode.role) }}>
                {selectedNode.role.toUpperCase()}
              </span>
            </div>
            <p className="detail-desc">{selectedNode.desc}</p>
          </div>
        )}
      </div>
    </div>
  );
}
