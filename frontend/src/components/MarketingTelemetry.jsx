import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { BarChart3, PieChart, Info } from 'lucide-react';

export default function MarketingTelemetry({ previewRows, metrics, totalRows }) {
  // Aggregate channel distribution for Plotly bar chart
  const channelData = useMemo(() => {
    if (!metrics || !metrics.channelCounts) {
      return { channels: [], counts: [] };
    }
    const channels = Object.keys(metrics.channelCounts);
    const counts = Object.values(metrics.channelCounts);
    return { channels, counts };
  }, [metrics]);

  // Extract engagement vs spend distribution from preview/metrics
  const scatterData = useMemo(() => {
    if (!previewRows || previewRows.length === 0) return { x: [], y: [], text: [], colors: [] };
    const x = [];
    const y = [];
    const text = [];
    const colors = [];

    previewRows.forEach((r) => {
      x.push(Number(r.engagement_score));
      y.push(Number(r.prior_spend));
      text.push(`Customer: ${r.customer_id}<br>Channel: ${r.campaign_channel}<br>Tenure: ${r.customer_tenure} mo`);
      colors.push(Number(r.discount_offered) === 1 ? '#10b981' : '#64748b');
    });

    return { x, y, text, colors };
  }, [previewRows]);

  if (!metrics || totalRows === 0) {
    return null;
  }

  return (
    <div className="card telemetry-card">
      <div className="card-header">
        <div className="title-with-icon">
          <div className="icon-badge icon-cyan">
            <BarChart3 size={20} />
          </div>
          <div>
            <h2 className="card-title">Exploratory Marketing Telemetry (Observed Baseline)</h2>
            <p className="card-subtitle">
              Interactive distribution analysis of active campaign observations powered by Plotly.
            </p>
          </div>
        </div>
      </div>

      <div className="charts-grid">
        {/* Chart 1: Campaign Channel Distribution */}
        <div className="chart-box">
          <div className="chart-title-bar">
            <h3 className="chart-title">Customer Distribution by Campaign Channel</h3>
            <span className="chart-badge">Observational Volume</span>
          </div>
          <Plot
            data={[
              {
                x: channelData.channels.map((c) => c.toUpperCase()),
                y: channelData.counts,
                type: 'bar',
                marker: {
                  color: ['#06b6d4', '#10b981', '#f59e0b', '#a855f7'],
                  opacity: 0.85,
                  line: { color: '#ffffff', width: 1 },
                },
                hoverinfo: 'x+y',
              },
            ]}
            layout={{
              autosize: true,
              margin: { l: 45, r: 20, t: 20, b: 40 },
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              font: { color: '#94a3b8', family: 'Inter, sans-serif' },
              xaxis: { gridcolor: '#1e293b' },
              yaxis: { gridcolor: '#1e293b', title: 'Customers' },
            }}
            useResizeHandler
            style={{ width: '100%', height: '260px' }}
            config={{ displayModeBar: false, responsive: true }}
          />
        </div>

        {/* Chart 2: Confounder Scatter: Engagement vs Prior Spend */}
        <div className="chart-box">
          <div className="chart-title-bar">
            <h3 className="chart-title">Pre-Treatment Confounder Space (Spend vs. Engagement)</h3>
            <span className="chart-badge">Sample Subset</span>
          </div>
          <Plot
            data={[
              {
                x: scatterData.x,
                y: scatterData.y,
                text: scatterData.text,
                mode: 'markers',
                type: 'scatter',
                marker: {
                  size: 9,
                  color: scatterData.colors,
                  opacity: 0.8,
                  line: { color: '#ffffff', width: 0.5 },
                },
                hoverinfo: 'text+x+y',
              },
            ]}
            layout={{
              autosize: true,
              margin: { l: 50, r: 20, t: 20, b: 45 },
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              font: { color: '#94a3b8', family: 'Inter, sans-serif' },
              xaxis: { gridcolor: '#1e293b', title: 'Engagement Score (0-100)' },
              yaxis: { gridcolor: '#1e293b', title: 'Prior Spend ($)' },
            }}
            useResizeHandler
            style={{ width: '100%', height: '260px' }}
            config={{ displayModeBar: false, responsive: true }}
          />
        </div>
      </div>

      <div className="chart-caption-row">
        <Info size={13} className="text-subtle" />
        <span>
          Visualizing pre-treatment confounding distributions: Green dots denote customers assigned promotional discounts;
          Grey dots denote unassigned baseline controls.
        </span>
      </div>
    </div>
  );
}
