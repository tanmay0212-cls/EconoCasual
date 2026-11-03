"""Automated unit test suite for EconoCausal data drift detection engine."""

import pytest
import numpy as np
import pandas as pd

from src.monitoring.drift_detector import (
    EconoCausalDriftDetector,
    DriftSeverity,
    OverallDriftStatus,
    DriftReport,
)


@pytest.fixture
def reference_dataset() -> pd.DataFrame:
    """Generate reproducible baseline reference customer population (N=1,000)."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "customer_id": [f"REF_{i:04d}" for i in range(n)],
        "age": np.clip(np.random.normal(loc=40.0, scale=10.0, size=n), 18.0, 75.0).round(1),
        "loyalty_score": np.clip(np.random.normal(loc=50.0, scale=15.0, size=n), 0.0, 100.0).round(2),
        "previous_purchases": np.random.poisson(lam=4, size=n),
        "customer_segment": np.random.choice(["budget_seeker", "casual_shopper", "regular_buyer", "vip_loyalist"], size=n, p=[0.25, 0.30, 0.25, 0.20]),
        "campaign_channel": np.random.choice(["email", "push", "sms", "social"], size=n, p=[0.40, 0.25, 0.15, 0.20]),
        "product_price": np.clip(np.random.normal(loc=85.0, scale=12.0, size=n), 30.0, 150.0).round(2),
        "gross_margin_rate": np.clip(np.random.normal(loc=0.40, scale=0.05, size=n), 0.15, 0.65).round(3),
    })


# ── Requirement 1: Identical Reference and Current Distributions ──────────────

def test_identical_distributions(reference_dataset: pd.DataFrame):
    """When current data is identical to reference, detector must report NO DRIFT."""
    detector = EconoCausalDriftDetector(reference_data=reference_dataset)
    current_data = reference_dataset.copy()

    report = detector.detect_drift(current_data)

    assert report.drift_detected is False
    assert report.drifted_features_count == 0
    assert report.overall_status == OverallDriftStatus.NO_DRIFT
    assert report.schema_validation.is_valid is True

    for feat in report.features:
        assert feat.drift_detected is False
        assert feat.severity == DriftSeverity.NONE
        if feat.feature_type == "numeric":
            assert feat.test_statistic == 0.0
            assert feat.p_value >= 0.99
            assert feat.psi == 0.0
        elif feat.feature_type == "categorical":
            assert feat.p_value >= 0.99
            assert feat.tvd == 0.0


# ── Requirement 2: Deliberately Shifted Numerical Feature ─────────────────────

def test_deliberately_shifted_numerical_feature(reference_dataset: pd.DataFrame):
    """Drastically shifting 'age' (+20 years) must be detected with high PSI and KS p < 0.001."""
    detector = EconoCausalDriftDetector(reference_data=reference_dataset)
    current_data = reference_dataset.copy()

    # Shift age: from ~40 to ~65
    current_data["age"] = np.clip(current_data["age"] + 25.0, 18.0, 95.0)

    report = detector.detect_drift(current_data)

    assert report.drift_detected is True
    assert report.drifted_features_count >= 1

    age_feat = next(f for f in report.features if f.feature_name == "age")
    assert age_feat.drift_detected is True
    assert age_feat.severity == DriftSeverity.HIGH
    assert age_feat.p_value < 0.001
    assert age_feat.psi >= 0.25
    assert age_feat.wasserstein_distance >= 20.0


# ── Requirement 3: Changed Categorical Distribution ──────────────────────────

def test_changed_categorical_distribution(reference_dataset: pd.DataFrame):
    """Distorting 'customer_segment' proportions must be detected via Chi-Square and TVD."""
    detector = EconoCausalDriftDetector(reference_data=reference_dataset)
    current_data = reference_dataset.copy()

    # Distort segments: 90% budget_seeker (baseline was 25%)
    n = len(current_data)
    current_data["customer_segment"] = np.random.choice(
        ["budget_seeker", "casual_shopper", "regular_buyer", "vip_loyalist"],
        size=n,
        p=[0.90, 0.05, 0.03, 0.02],
    )

    report = detector.detect_drift(current_data)

    assert report.drift_detected is True
    seg_feat = next(f for f in report.features if f.feature_name == "customer_segment")
    assert seg_feat.drift_detected is True
    assert seg_feat.severity == DriftSeverity.HIGH
    assert seg_feat.p_value < 0.001
    assert seg_feat.tvd >= 0.25


# ── Requirement 4: Missing Required Columns ───────────────────────────────────

def test_missing_required_columns(reference_dataset: pd.DataFrame):
    """Missing required model feature ('loyalty_score') must fail schema validation."""
    detector = EconoCausalDriftDetector(reference_data=reference_dataset)
    bad_current = reference_dataset.drop(columns=["loyalty_score"])

    report = detector.detect_drift(bad_current)

    assert report.schema_validation.is_valid is False
    assert "loyalty_score" in report.schema_validation.missing_required_features
    assert report.overall_status == OverallDriftStatus.SEVERE_DRIFT
    assert "schema validation failed" in report.summary_message.lower()


# ── Requirement 5: New and Disappearing Categorical Values ────────────────────

def test_new_and_disappearing_categorical_values(reference_dataset: pd.DataFrame):
    """Detect newly emerging categories ('luxury_whale') and disappearing categories."""
    detector = EconoCausalDriftDetector(reference_data=reference_dataset)
    current_data = reference_dataset.copy()

    # Replace 'vip_loyalist' with new unseen category 'luxury_whale'
    current_data["customer_segment"] = current_data["customer_segment"].replace({
        "vip_loyalist": "luxury_whale"
    })

    report = detector.detect_drift(current_data)

    seg_feat = next(f for f in report.features if f.feature_name == "customer_segment")
    assert seg_feat.drift_detected is True
    assert "luxury_whale" in seg_feat.new_categories
    assert "vip_loyalist" in seg_feat.missing_categories
    assert seg_feat.severity == DriftSeverity.HIGH


# ── Requirement 6: Small Sample Sizes ─────────────────────────────────────────

def test_small_sample_size_handling(reference_dataset: pd.DataFrame):
    """Small sample sizes (N=8) must degrade gracefully with informational messages."""
    detector = EconoCausalDriftDetector(reference_data=reference_dataset)
    tiny_current = reference_dataset.head(8).copy()

    report = detector.detect_drift(tiny_current)

    assert report.current_sample_size == 8
    # Should not crash or throw unhandled exceptions
    assert isinstance(report, DriftReport)
    for feat in report.features:
        assert isinstance(feat.recommended_action, str)


# ── Requirement 7: Missing Values and Invalid Data (Schema vs Drift) ──────────

def test_missing_values_and_type_integrity(reference_dataset: pd.DataFrame):
    """Ensure missing values and type errors are flagged in schema validation distinctly from drift."""
    detector = EconoCausalDriftDetector(reference_data=reference_dataset)
    corrupted_data = reference_dataset.copy()

    # Inject 15% nulls into required feature 'age'
    corrupted_data.loc[0:150, "age"] = np.nan

    report = detector.detect_drift(corrupted_data)

    assert report.schema_validation.is_valid is False
    assert "age" in report.schema_validation.missing_value_rates
    assert report.schema_validation.missing_value_rates["age"] >= 0.10


def test_out_of_range_numerical_values(reference_dataset: pd.DataFrame):
    """Out-of-range numerical values (e.g. negative age or loyalty > 100) are flagged in schema warnings."""
    detector = EconoCausalDriftDetector(reference_data=reference_dataset)
    bad_bounds_data = reference_dataset.copy()

    bad_bounds_data.loc[0, "age"] = 5.0      # age < 18
    bad_bounds_data.loc[1, "loyalty_score"] = 250.0  # loyalty > 100

    schema_res = detector.validate_schema(bad_bounds_data)

    assert "age" in schema_res.out_of_range_features
    assert "loyalty_score" in schema_res.out_of_range_features


# ── Additional: Methodological Principles & Thresholds ────────────────────────

def test_methodological_notice_included(reference_dataset: pd.DataFrame):
    """Report must include methodological distinction between covariate drift and concept drift."""
    detector = EconoCausalDriftDetector(reference_data=reference_dataset)
    report = detector.detect_drift(reference_dataset)

    assert "Covariate drift" in report.methodological_notice
    assert "concept drift" in report.methodological_notice
    assert "retrain" in report.methodological_notice.lower()


def test_report_dict_serialization(reference_dataset: pd.DataFrame):
    """DriftReport.to_dict() must produce JSON-serializable dictionary."""
    detector = EconoCausalDriftDetector(reference_data=reference_dataset)
    report = detector.detect_drift(reference_dataset)
    data = report.to_dict()

    assert isinstance(data, dict)
    assert "overall_status" in data
    assert "features" in data
    assert len(data["features"]) >= 5

