"""Uplift modeling, Qini curve evaluation, and visualization layer for EconoCausal.

This module provides metrics and interactive Plotly visualizations to evaluate
treatment-effect ranking performance on held-out evaluation datasets.

METRICS & DEFINITIONS:
1. Treatment-Effect Ranking:
   Customers are sorted in descending order of predicted Conditional Average
   Treatment Effect: tau(X_(1)) >= tau(X_(2)) >= ... >= tau(X_(N)).
   Customers at the top are predicted to have the highest promotional elasticity.

2. Cumulative Uplift Curve:
   For the top k targeted customers (fraction p = k / N):
     Uplift(k) = (y_{T, k} / n_{T, k}) - (y_{C, k} / n_{C, k})
   Measures the observed conversion rate difference between treated and control
   subgroups among the top k customers.

3. Qini Curve (Radcliffe, 2007):
   Cumulative incremental buyers among the top k customers scaled for group size:
     Q(k) = y_{T, k} - y_{C, k} * (n_{T, k} / n_{C, k})
   The random targeting baseline is a straight line connecting (0, 0) to (N, Q(N)).

4. Qini Score (Area Between Curves):
   The area between the model's Qini curve and the random baseline line,
   integrated over the population fraction p in [0, 1].

5. Observational Data Confounding Disclaimer:
   In observational data, naive sample averages (y_T / n_T) - (y_C / n_C) reflect
   both causal treatment lift AND selection confounding bias. While DML eliminates
   confounding in the model's beta parameters, non-experimental uplift curves remain
   subject to observational distribution imbalances unless propensity-weighted.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .dml import EconoCausalDML, DEFAULT_TREATMENT_NAMES, DEFAULT_OUTCOME_NAMES


def resolve_eval_columns(
    df: pd.DataFrame,
    treatment_col: Optional[str] = None,
    outcome_col: Optional[str] = None,
) -> Tuple[str, str]:
    """Resolve treatment and outcome column names from DataFrame."""
    if treatment_col is not None:
        if treatment_col not in df.columns:
            raise ValueError(f"Specified treatment column '{treatment_col}' not found in DataFrame.")
        res_t = treatment_col
    else:
        matches = [c for c in DEFAULT_TREATMENT_NAMES if c in df.columns]
        if not matches:
            raise ValueError(f"Treatment column not found. Expected one of {DEFAULT_TREATMENT_NAMES}")
        res_t = matches[0]

    if outcome_col is not None:
        if outcome_col not in df.columns:
            raise ValueError(f"Specified outcome column '{outcome_col}' not found in DataFrame.")
        res_y = outcome_col
    else:
        matches = [c for c in DEFAULT_OUTCOME_NAMES if c in df.columns]
        if not matches:
            raise ValueError(f"Outcome column not found. Expected one of {DEFAULT_OUTCOME_NAMES}")
        res_y = matches[0]

    return res_t, res_y


def compute_ranking_table(
    df_eval: pd.DataFrame,
    model: Optional[EconoCausalDML] = None,
    estimated_effects: Optional[np.ndarray] = None,
    treatment_col: Optional[str] = None,
    outcome_col: Optional[str] = None,
) -> pd.DataFrame:
    """Produce ranked customer evaluation table sorted descending by estimated treatment effect.

    Args:
        df_eval: Held-out evaluation DataFrame.
        model: Optional fitted EconoCausalDML model (used if estimated_effects is None).
        estimated_effects: Optional pre-computed 1D array of CATE estimates.
        treatment_col: Optional treatment column name.
        outcome_col: Optional outcome column name.

    Returns:
        DataFrame sorted descending by estimated_effect with rank and percentile columns.
    """
    if df_eval is None or len(df_eval) == 0:
        raise ValueError("Input evaluation DataFrame is empty or None.")

    t_col, y_col = resolve_eval_columns(df_eval, treatment_col, outcome_col)

    if estimated_effects is not None:
        if len(estimated_effects) != len(df_eval):
            raise ValueError(
                f"Length mismatch: {len(estimated_effects)} effects for {len(df_eval)} evaluation rows."
            )
        effects = np.asarray(estimated_effects).flatten()
    elif model is not None:
        effects = model.effect(df_eval)
    else:
        raise ValueError("Must provide either a fitted model or pre-computed estimated_effects.")

    if not np.all(np.isfinite(effects)):
        raise ValueError("Non-finite values detected in estimated treatment effects.")

    ranked_df = df_eval.copy()
    ranked_df["estimated_effect"] = effects
    ranked_df["estimated_effect_percentage_points"] = effects * 100.0

    # Sort strictly descending by predicted causal effect
    ranked_df = ranked_df.sort_values(by="estimated_effect", ascending=False).reset_index(drop=True)

    n_total = len(ranked_df)
    ranked_df["rank"] = np.arange(1, n_total + 1)
    ranked_df["percentile"] = (ranked_df["rank"] / n_total) * 100.0

    return ranked_df


def compute_uplift_and_qini_metrics(
    ranked_df: pd.DataFrame,
    treatment_col: Optional[str] = None,
    outcome_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute cumulative uplift curve, Qini curve, and summary evaluation metrics.

    Args:
        ranked_df: DataFrame sorted descending by estimated_effect.
        treatment_col: Optional treatment column name.
        outcome_col: Optional outcome column name.

    Returns:
        Dictionary containing curve series arrays, decile summaries, and scalar metrics.
    """
    t_col, y_col = resolve_eval_columns(ranked_df, treatment_col, outcome_col)

    N = len(ranked_df)
    if N < 10:
        raise ValueError(f"Insufficient sample size for uplift curve evaluation: received {N} rows, minimum 10.")

    T = ranked_df[t_col].to_numpy().astype(int)
    Y = ranked_df[y_col].to_numpy().astype(float)

    # Validate treatment and outcome domains
    if not set(np.unique(T)).issubset({0, 1}):
        raise ValueError(f"Treatment column '{t_col}' must be binary (0 or 1), found: {set(np.unique(T))}")
    if not set(np.unique(Y)).issubset({0, 1}):
        raise ValueError(f"Outcome column '{y_col}' must be binary (0 or 1), found: {set(np.unique(Y))}")

    total_treated = int(np.sum(T))
    total_control = int(N - total_treated)

    if total_treated == 0:
        raise ValueError("Evaluation set contains zero treated customers. Cannot compute uplift or Qini.")
    if total_control == 0:
        raise ValueError("Evaluation set contains zero control customers. Cannot compute uplift or Qini.")

    cum_n = np.arange(1, N + 1)
    cum_t = np.cumsum(T)
    cum_c = cum_n - cum_t

    cum_y_t = np.cumsum(Y * T)
    cum_y_c = np.cumsum(Y * (1.0 - T))

    # Avoid divide-by-zero for early points
    rate_t = np.where(cum_t > 0, cum_y_t / np.maximum(cum_t, 1), 0.0)
    rate_c = np.where(cum_c > 0, cum_y_c / np.maximum(cum_c, 1), 0.0)
    cum_uplift = rate_t - rate_c

    # Baseline overall uplift (Diff-in-means at 100%)
    total_y_t = int(cum_y_t[-1])
    total_y_c = int(cum_y_c[-1])
    baseline_uplift = (total_y_t / total_treated) - (total_y_c / total_control)

    # Qini Curve: Q(k) = y_T(k) - y_C(k) * (n_T(k) / n_C(k))
    # When n_C(k) == 0, scaled control buyers = 0
    scaling_ratio = np.where(cum_c > 0, cum_t / np.maximum(cum_c, 1), 0.0)
    qini_curve = cum_y_t - (cum_y_c * scaling_ratio)

    # Random baseline connects (0, 0) to (N, Q(N))
    final_qini = float(qini_curve[-1])
    qini_random = (cum_n / N) * final_qini

    # Prepend origin (0, 0) for area calculations
    pop_fractions = np.concatenate([[0.0], cum_n / N])
    pop_percent = pop_fractions * 100.0
    qini_plot_vals = np.concatenate([[0.0], qini_curve])
    qini_rand_plot_vals = np.concatenate([[0.0], qini_random])
    uplift_plot_vals = np.concatenate([[cum_uplift[0]], cum_uplift])

    # Qini Score: Area Between Model Curve and Random Line over p in [0, 1]
    area_model = float(np.trapezoid(qini_plot_vals, pop_fractions))
    area_random = float(np.trapezoid(qini_rand_plot_vals, pop_fractions))
    qini_score = float(area_model - area_random)

    # Area Under the Uplift Curve (AUUC)
    auuc = float(np.trapezoid(uplift_plot_vals, pop_fractions))

    # Decile breakdown table
    deciles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    decile_rows = []
    for d in deciles:
        k = int(np.round(d * N))
        idx = k - 1
        decile_rows.append(
            {
                "decile": int(d * 10),
                "population_pct": round(d * 100.0, 1),
                "customers_targeted": k,
                "treated_count": int(cum_t[idx]),
                "control_count": int(cum_c[idx]),
                "treated_conversions": int(cum_y_t[idx]),
                "control_conversions": int(cum_y_c[idx]),
                "treated_conversion_rate": round(float(rate_t[idx]), 4),
                "control_conversion_rate": round(float(rate_c[idx]), 4),
                "cumulative_uplift": round(float(cum_uplift[idx]), 4),
                "cumulative_uplift_percentage_points": round(float(cum_uplift[idx]) * 100.0, 2),
                "qini_value": round(float(qini_curve[idx]), 2),
                "qini_random_value": round(float(qini_random[idx]), 2),
            }
        )

    decile_df = pd.DataFrame(decile_rows)

    # Identify peak cumulative uplift
    peak_idx = int(np.argmax(cum_uplift))
    peak_uplift = float(cum_uplift[peak_idx])
    peak_k = int(cum_n[peak_idx])
    peak_pct = round((peak_k / N) * 100.0, 1)

    return {
        "n_evaluation_customers": N,
        "n_treated_total": total_treated,
        "n_control_total": total_control,
        "total_treated_conversions": total_y_t,
        "total_control_conversions": total_y_c,
        "baseline_overall_uplift": round(baseline_uplift, 4),
        "baseline_overall_uplift_percentage_points": round(baseline_uplift * 100.0, 2),
        "final_qini_value": round(final_qini, 4),
        "qini_score_area_between_curves": round(qini_score, 4),
        "area_under_qini_curve": round(area_model, 4),
        "area_under_random_qini": round(area_random, 4),
        "area_under_uplift_curve_auuc": round(auuc, 4),
        "peak_uplift": round(peak_uplift, 4),
        "peak_uplift_percentage_points": round(peak_uplift * 100.0, 2),
        "peak_targeting_depth_customers": peak_k,
        "peak_targeting_depth_pct": peak_pct,
        "curve_series": {
            "pop_percent": pop_percent.tolist(),
            "qini_model": qini_plot_vals.tolist(),
            "qini_random": qini_rand_plot_vals.tolist(),
            "cumulative_uplift": uplift_plot_vals.tolist(),
            "baseline_uplift_line": [baseline_uplift] * len(pop_percent),
        },
        "decile_table": decile_df,
    }


def plot_uplift_curve(curve_data: Dict[str, Any], output_path: Optional[str] = None) -> go.Figure:
    """Create interactive Plotly cumulative uplift chart.

    Args:
        curve_data: Output dictionary from compute_uplift_and_qini_metrics().
        output_path: Optional HTML file path to save standalone interactive figure.

    Returns:
        Plotly Figure object.
    """
    series = curve_data["curve_series"]
    x = series["pop_percent"][1:]  # Skip 0% for uplift rate display
    y_uplift = [val * 100.0 for val in series["cumulative_uplift"][1:]]
    y_baseline = [curve_data["baseline_overall_uplift_percentage_points"]] * len(x)

    fig = go.Figure()

    # Model cumulative uplift curve
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_uplift,
            mode="lines",
            name="LinearDML Model Ranking",
            line=dict(color="#00D26A", width=3),
            hovertemplate=(
                "<b>Top %{x:.1f}% Targeted</b><br>"
                + "Cumulative Uplift: <b>+%{y:.2f} pp</b><extra></extra>"
            ),
        )
    )

    # Random targeting baseline
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_baseline,
            mode="lines",
            name=f"Random Targeting (ATE: +{curve_data['baseline_overall_uplift_percentage_points']:.2f} pp)",
            line=dict(color="#94A3B8", width=2, dash="dash"),
            hovertemplate="Random Targeting Baseline: +%{y:.2f} pp<extra></extra>",
        )
    )

    # Annotate peak uplift
    peak_x = curve_data["peak_targeting_depth_pct"]
    peak_y = curve_data["peak_uplift_percentage_points"]
    fig.add_annotation(
        x=peak_x,
        y=peak_y,
        text=f"Peak Lift: +{peak_y:.1f} pp (Top {peak_x}%)",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#00D26A",
        ax=40,
        ay=-30,
        font=dict(size=12, color="#00D26A"),
        bgcolor="rgba(15, 23, 42, 0.85)",
        bordercolor="#00D26A",
        borderwidth=1,
    )

    fig.update_layout(
        title=dict(
            text="<b>Cumulative Uplift Curve (Held-Out Evaluation Set)</b><br>"
            + "<sup>Observed conversion rate difference (Treated vs. Control) as progressively lower-ranked customers are included</sup>",
            font=dict(size=18, color="#F8FAFC"),
        ),
        xaxis=dict(
            title="<b>Percentage of Customers Targeted (Ranked by Model CATE)</b>",
            ticksuffix="%",
            range=[0, 102],
            gridcolor="#334155",
            zerolinecolor="#475569",
            color="#94A3B8",
        ),
        yaxis=dict(
            title="<b>Cumulative Conversion Uplift (Percentage Points)</b>",
            ticksuffix=" pp",
            gridcolor="#334155",
            zerolinecolor="#475569",
            color="#94A3B8",
        ),
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#F8FAFC"),
        ),
        hoverlabel=dict(bgcolor="#0F172A", font_size=13, font_color="#F8FAFC"),
        margin=dict(l=60, r=40, t=100, b=60),
    )

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.write_html(output_path, include_plotlyjs="cdn")

    return fig


def plot_qini_curve(curve_data: Dict[str, Any], output_path: Optional[str] = None) -> go.Figure:
    """Create interactive Plotly Qini curve chart with shaded area between curves.

    Args:
        curve_data: Output dictionary from compute_uplift_and_qini_metrics().
        output_path: Optional HTML file path to save standalone interactive figure.

    Returns:
        Plotly Figure object.
    """
    series = curve_data["curve_series"]
    x = series["pop_percent"]
    y_qini = series["qini_model"]
    y_random = series["qini_random"]

    fig = go.Figure()

    # Random baseline trace (lower boundary of filled area)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_random,
            mode="lines",
            name=f"Random Baseline (Qini at 100%: {curve_data['final_qini_value']:.1f})",
            line=dict(color="#94A3B8", width=2, dash="dash"),
            hovertemplate="Random Baseline: %{y:.2f} incremental buyers<extra></extra>",
        )
    )

    # Model Qini curve trace (with shaded fill down to random line)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_qini,
            mode="lines",
            name="LinearDML Model Qini Curve",
            fill="tonexty",
            fillcolor="rgba(0, 210, 106, 0.15)",
            line=dict(color="#00D26A", width=3),
            hovertemplate=(
                "<b>Top %{x:.1f}% Targeted</b><br>"
                + "Cumulative Incremental Buyers: <b>%{y:.2f}</b><extra></extra>"
            ),
        )
    )

    # Annotate Qini Area Score
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.05,
        y=0.92,
        text=(
            f"<b>Qini Score (Area Between Curves): {curve_data['qini_score_area_between_curves']:+.2f}</b><br>"
            + f"Peak Targeting Depth: Top {curve_data['peak_targeting_depth_pct']}%"
        ),
        showarrow=False,
        font=dict(size=13, color="#00D26A"),
        bgcolor="rgba(15, 23, 42, 0.85)",
        bordercolor="#00D26A",
        borderwidth=1,
        align="left",
    )

    fig.update_layout(
        title=dict(
            text="<b>Qini Curve (Radcliffe Cumulative Incremental Gain)</b><br>"
            + "<sup>Measures total incremental converted buyers achieved over random promotional coupon targeting</sup>",
            font=dict(size=18, color="#F8FAFC"),
        ),
        xaxis=dict(
            title="<b>Percentage of Customers Targeted (Ranked by Model CATE)</b>",
            ticksuffix="%",
            range=[0, 102],
            gridcolor="#334155",
            zerolinecolor="#475569",
            color="#94A3B8",
        ),
        yaxis=dict(
            title="<b>Cumulative Incremental Purchasers Q(k)</b>",
            gridcolor="#334155",
            zerolinecolor="#475569",
            color="#94A3B8",
        ),
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#F8FAFC"),
        ),
        hoverlabel=dict(bgcolor="#0F172A", font_size=13, font_color="#F8FAFC"),
        margin=dict(l=60, r=40, t=100, b=60),
    )

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.write_html(output_path, include_plotlyjs="cdn")

    return fig


def plot_effect_distribution(
    ranked_df: pd.DataFrame,
    ate_val: float = 0.1725,
    ci_bounds: Tuple[float, float] = (0.1296, 0.2153),
    output_path: Optional[str] = None,
) -> go.Figure:
    """Create interactive Plotly histogram showing customer treatment effect dispersion.

    Args:
        ranked_df: Ranked DataFrame with 'estimated_effect_percentage_points'.
        ate_val: Mean ATE estimate.
        ci_bounds: Tuple of (ci_lower, ci_upper) in raw scale.
        output_path: Optional HTML file path to save standalone interactive figure.

    Returns:
        Plotly Figure object.
    """
    effects_pp = ranked_df["estimated_effect_percentage_points"].to_numpy()

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=effects_pp,
            nbinsx=30,
            name="Customer CATE Estimates",
            marker=dict(color="#38BDF8", line=dict(color="#0284C7", width=1)),
            opacity=0.85,
            hovertemplate="Conversion Lift: %{x:.1f} pp<br>Customer Count: %{y}<extra></extra>",
        )
    )

    # ATE mean vertical line
    fig.add_vline(
        x=ate_val * 100.0,
        line_width=3,
        line_dash="solid",
        line_color="#00D26A",
        annotation_text=f"Mean ATE: +{ate_val*100:.1f} pp",
        annotation_position="top right",
        annotation_font=dict(color="#00D26A", size=12),
    )

    # 95% Confidence Interval shaded region
    fig.add_vrect(
        x0=ci_bounds[0] * 100.0,
        x1=ci_bounds[1] * 100.0,
        fillcolor="rgba(0, 210, 106, 0.12)",
        line_width=0,
        annotation_text="95% ATE CI",
        annotation_position="bottom right",
        annotation_font=dict(color="#00D26A", size=11),
    )

    # Median vertical line
    med_val = float(np.median(effects_pp))
    fig.add_vline(
        x=med_val,
        line_width=2,
        line_dash="dot",
        line_color="#FBBF24",
        annotation_text=f"Median: +{med_val:.1f} pp",
        annotation_position="top left",
        annotation_font=dict(color="#FBBF24", size=11),
    )

    fig.update_layout(
        title=dict(
            text="<b>Distribution of Estimated Customer Treatment Effects (CATE)</b><br>"
            + "<sup>Predicted conversion rate lift across the 500 held-out evaluation customers</sup>",
            font=dict(size=18, color="#F8FAFC"),
        ),
        xaxis=dict(
            title="<b>Estimated Purchase Probability Lift (Percentage Points)</b>",
            ticksuffix=" pp",
            gridcolor="#334155",
            zerolinecolor="#475569",
            color="#94A3B8",
        ),
        yaxis=dict(
            title="<b>Number of Customers</b>",
            gridcolor="#334155",
            zerolinecolor="#475569",
            color="#94A3B8",
        ),
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        legend=dict(font=dict(color="#F8FAFC")),
        hoverlabel=dict(bgcolor="#0F172A", font_size=13, font_color="#F8FAFC"),
        margin=dict(l=60, r=40, t=100, b=60),
    )

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.write_html(output_path, include_plotlyjs="cdn")

    return fig
