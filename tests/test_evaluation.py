"""Unit tests for uplift and Qini curve evaluation layer in EconoCausal."""

import os
import tempfile
from typing import Tuple
import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

from src.data.generator import generate_synthetic_campaign_data
from src.causal.dml import EconoCausalDML
from src.causal.evaluation import (
    compute_ranking_table,
    compute_uplift_and_qini_metrics,
    plot_uplift_curve,
    plot_qini_curve,
    plot_effect_distribution,
)


@pytest.fixture(scope="module")
def eval_setup() -> Tuple[pd.DataFrame, EconoCausalDML]:
    """Train a small reproducible DML model and provide evaluation DataFrame."""
    df = generate_synthetic_campaign_data(n_samples=1000, random_seed=42)
    train_df, eval_df = train_test_split(df, test_size=0.20, random_state=42)

    model = EconoCausalDML(
        model_type="linear",
        base_estimator="random_forest",
        n_estimators=30,
        max_depth=3,
        cv=3,
        random_state=42,
    )
    model.fit(train_df)
    return eval_df, model


def test_ranking_table_sorting_and_percentiles(eval_setup):
    """Verify compute_ranking_table sorts strictly descending and sets rank and percentile."""
    eval_df, model = eval_setup

    ranked_df = compute_ranking_table(eval_df, model=model)

    assert len(ranked_df) == len(eval_df)
    assert "estimated_effect" in ranked_df.columns
    assert "rank" in ranked_df.columns
    assert "percentile" in ranked_df.columns

    # Verify descending monotonicity
    effects = ranked_df["estimated_effect"].to_numpy()
    assert np.all(effects[:-1] >= effects[1:])

    # Verify rank is 1 to N
    assert list(ranked_df["rank"]) == list(range(1, len(eval_df) + 1))
    assert np.isclose(ranked_df["percentile"].iloc[-1], 100.0)


def test_uplift_and_qini_metrics_computation(eval_setup):
    """Verify compute_uplift_and_qini_metrics outputs all required scalar metrics and deciles."""
    eval_df, model = eval_setup
    ranked_df = compute_ranking_table(eval_df, model=model)

    metrics = compute_uplift_and_qini_metrics(ranked_df)

    assert "qini_score_area_between_curves" in metrics
    assert "final_qini_value" in metrics
    assert "baseline_overall_uplift" in metrics
    assert "area_under_uplift_curve_auuc" in metrics
    assert "peak_uplift" in metrics
    assert "curve_series" in metrics
    assert "decile_table" in metrics

    assert np.isfinite(metrics["qini_score_area_between_curves"])
    assert np.isfinite(metrics["baseline_overall_uplift"])

    # Decile table validation
    decile_df = metrics["decile_table"]
    assert len(decile_df) == 10
    assert list(decile_df["decile"]) == list(range(1, 11))
    assert list(decile_df["population_pct"]) == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]


def test_all_treated_or_all_control_edge_cases():
    """Verify evaluation raises ValueError when evaluation subset lacks treated or control units."""
    # All treated
    df_all_t = pd.DataFrame(
        {
            "discount_offered": [1] * 20,
            "purchased": [0, 1] * 10,
            "estimated_effect": np.linspace(0.1, 0.3, 20),
        }
    )
    with pytest.raises(ValueError, match="zero control customers"):
        compute_uplift_and_qini_metrics(df_all_t)

    # All control
    df_all_c = pd.DataFrame(
        {
            "discount_offered": [0] * 20,
            "purchased": [0, 1] * 10,
            "estimated_effect": np.linspace(0.1, 0.3, 20),
        }
    )
    with pytest.raises(ValueError, match="zero treated customers"):
        compute_uplift_and_qini_metrics(df_all_c)


def test_small_sample_size_edge_case():
    """Verify evaluation flags insufficient evaluation sample sizes."""
    df_small = pd.DataFrame(
        {
            "discount_offered": [0, 1, 0, 1],
            "purchased": [0, 0, 1, 1],
            "estimated_effect": [0.1, 0.2, 0.3, 0.4],
        }
    )
    with pytest.raises(ValueError, match="Insufficient sample size"):
        compute_uplift_and_qini_metrics(df_small)


def test_non_binary_values_edge_case():
    """Verify evaluation rejects non-binary treatment or outcome columns."""
    df_bad_t = pd.DataFrame(
        {
            "discount_offered": [0, 1, 2] * 5,
            "purchased": [0, 1, 0] * 5,
            "estimated_effect": np.linspace(0.1, 0.3, 15),
        }
    )
    with pytest.raises(ValueError, match="Treatment column.*must be binary"):
        compute_uplift_and_qini_metrics(df_bad_t)


def test_empty_dataframe_and_non_finite_effects(eval_setup):
    """Verify compute_ranking_table rejects empty inputs and non-finite effect arrays."""
    eval_df, model = eval_setup

    # Empty DataFrame
    with pytest.raises(ValueError, match="Input evaluation DataFrame is empty or None"):
        compute_ranking_table(pd.DataFrame(), model=model)

    # Non-finite effects
    bad_effects = np.array([0.1] * (len(eval_df) - 1) + [np.nan])
    with pytest.raises(ValueError, match="Non-finite values detected in estimated treatment effects"):
        compute_ranking_table(eval_df, estimated_effects=bad_effects)


def test_tied_effects_and_custom_column_names():
    """Verify evaluation handles tied effects and custom column names without crashing."""
    n = 30
    df = pd.DataFrame(
        {
            "my_treatment": [0, 1] * 15,
            "my_outcome": [0, 0, 1, 1] * 7 + [0, 1],
            "estimated_effect": [0.15] * n,  # Exact ties across all rows
        }
    )

    ranked_df = compute_ranking_table(
        df,
        estimated_effects=df["estimated_effect"].values,
        treatment_col="my_treatment",
        outcome_col="my_outcome",
    )
    assert len(ranked_df) == n
    assert list(ranked_df["rank"]) == list(range(1, n + 1))

    metrics = compute_uplift_and_qini_metrics(
        ranked_df,
        treatment_col="my_treatment",
        outcome_col="my_outcome",
    )
    assert np.isfinite(metrics["qini_score_area_between_curves"])
    assert np.isfinite(metrics["baseline_overall_uplift"])
    assert len(metrics["decile_table"]) == 10



def test_plotly_figures_export(eval_setup):
    """Verify Plotly figure creation functions and standalone HTML exports."""
    eval_df, model = eval_setup
    ranked_df = compute_ranking_table(eval_df, model=model)
    curve_data = compute_uplift_and_qini_metrics(ranked_df)

    with tempfile.TemporaryDirectory() as tmp_dir:
        uplift_file = os.path.join(tmp_dir, "uplift.html")
        qini_file = os.path.join(tmp_dir, "qini.html")
        dist_file = os.path.join(tmp_dir, "dist.html")

        fig1 = plot_uplift_curve(curve_data, output_path=uplift_file)
        fig2 = plot_qini_curve(curve_data, output_path=qini_file)
        fig3 = plot_effect_distribution(ranked_df, output_path=dist_file)

        assert isinstance(fig1, go.Figure)
        assert isinstance(fig2, go.Figure)
        assert isinstance(fig3, go.Figure)

        assert os.path.exists(uplift_file)
        assert os.path.exists(qini_file)
        assert os.path.exists(dist_file)

        assert os.path.getsize(uplift_file) > 1000
        assert os.path.getsize(qini_file) > 1000
        assert os.path.getsize(dist_file) > 1000
