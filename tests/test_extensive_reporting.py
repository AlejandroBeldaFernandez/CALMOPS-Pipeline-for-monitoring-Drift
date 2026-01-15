"""
Test Suite for Extensive Reporting (YData + Plotly + Dashboard)
"""

import os
import shutil
import pytest
import pandas as pd
import numpy as np

from calmops.data_generators.Synthetic.SyntheticReporter import SyntheticReporter
from calmops.data_generators.Clinic.ClinicReporter import ClinicReporter
from calmops.data_generators.Real.RealReporter import RealReporter
from calmops.reports.ExternalReporter import ExternalReporter
from calmops.reports.PlotlyReporter import PlotlyReporter
from calmops.data_generators.DriftInjection.DriftInjector import DriftInjector
from calmops.data_generators.Dynamics.ScenarioInjector import ScenarioInjector

TEST_OUTPUT_DIR = "tests_output/extensive_reporting"


@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    yield


# =============================================================================
# PlotlyReporter Tests
# =============================================================================


def test_plotly_density_plots():
    """Test PlotlyReporter.generate_density_plots."""
    print("\n[TEST] PlotlyReporter - Density Plots...")
    df = pd.DataFrame(
        {
            "A": np.random.normal(0, 1, 100),
            "B": np.random.normal(5, 2, 100),
            "Target": np.random.choice([0, 1], 100),
        }
    )

    out_dir = os.path.join(TEST_OUTPUT_DIR, "plotly_density")
    os.makedirs(out_dir, exist_ok=True)

    path = PlotlyReporter.generate_density_plots(df, out_dir, color_col="Target")
    assert path is not None
    assert os.path.exists(path)


def test_plotly_dimensionality():
    """Test PlotlyReporter.generate_dimensionality_plot (PCA + UMAP combined)."""
    print("\n[TEST] PlotlyReporter - Dimensionality (PCA + UMAP)...")
    df = pd.DataFrame(
        {
            "A": np.random.normal(0, 1, 50),
            "B": np.random.normal(5, 2, 50),
            "C": np.random.normal(10, 3, 50),
        }
    )

    out_dir = os.path.join(TEST_OUTPUT_DIR, "plotly_dimensionality")
    os.makedirs(out_dir, exist_ok=True)

    path = PlotlyReporter.generate_dimensionality_plot(df, out_dir)
    assert path is not None
    assert os.path.exists(path)
    assert "dimensionality_plot.html" in path


def test_plotly_sdv_scores():
    """Test PlotlyReporter.generate_sdv_scores_card."""
    print("\n[TEST] PlotlyReporter - SDV Scores Card...")
    out_dir = os.path.join(TEST_OUTPUT_DIR, "plotly_sdv_scores")
    os.makedirs(out_dir, exist_ok=True)

    path = PlotlyReporter.generate_sdv_scores_card(
        overall_score=0.85, weighted_score=0.78, output_dir=out_dir
    )
    assert path is not None
    assert os.path.exists(path)


def test_plotly_sdv_evolution():
    """Test PlotlyReporter.generate_sdv_evolution_plot."""
    print("\n[TEST] PlotlyReporter - SDV Evolution...")
    scores = [
        {"overall": 0.85, "weighted": 0.80},
        {"overall": 0.87, "weighted": 0.82},
        {"overall": 0.90, "weighted": 0.85},
    ]

    out_dir = os.path.join(TEST_OUTPUT_DIR, "plotly_sdv_evolution")
    os.makedirs(out_dir, exist_ok=True)

    path = PlotlyReporter.generate_sdv_evolution_plot(
        scores, out_dir, x_labels=["Block 1", "Block 2", "Block 3"]
    )
    assert path is not None
    assert os.path.exists(path)


def test_plotly_drift_analysis():
    """Test PlotlyReporter.generate_drift_analysis."""
    print("\n[TEST] PlotlyReporter - Drift Analysis...")
    original = pd.DataFrame(
        {
            "A": np.random.normal(0, 1, 100),
            "B": np.random.normal(5, 2, 100),
        }
    )
    drifted = pd.DataFrame(
        {
            "A": np.random.normal(0.5, 1.2, 100),  # Shifted
            "B": np.random.normal(5, 2, 100),  # Same
        }
    )

    out_dir = os.path.join(TEST_OUTPUT_DIR, "plotly_drift_analysis")
    os.makedirs(out_dir, exist_ok=True)

    path = PlotlyReporter.generate_drift_analysis(original, drifted, out_dir)
    assert path is not None
    assert os.path.exists(path)
    assert "drift_analysis.html" in path


# =============================================================================
# ExternalReporter Tests
# =============================================================================


def test_external_reporter_profile():
    """Test ExternalReporter.generate_profile."""
    print("\n[TEST] ExternalReporter - Profile...")
    df = pd.DataFrame(
        {
            "A": np.random.normal(0, 1, 100),
            "B": np.random.choice(["X", "Y"], 100),
        }
    )

    out_dir = os.path.join(TEST_OUTPUT_DIR, "external_profile")
    os.makedirs(out_dir, exist_ok=True)

    p_path = ExternalReporter.generate_profile(df, out_dir, title="Profile Test")
    assert p_path is not None
    assert os.path.exists(p_path)


def test_external_reporter_comparison():
    """Test ExternalReporter.generate_comparison."""
    print("\n[TEST] ExternalReporter - Comparison...")
    df1 = pd.DataFrame({"A": np.random.normal(0, 1, 100)})
    df2 = pd.DataFrame({"A": np.random.normal(0.5, 1.2, 100)})

    out_dir = os.path.join(TEST_OUTPUT_DIR, "external_comparison")
    os.makedirs(out_dir, exist_ok=True)

    c_path = ExternalReporter.generate_comparison(
        ref_df=df1, curr_df=df2, output_dir=out_dir
    )
    assert c_path is not None
    assert os.path.exists(c_path)


# =============================================================================
# SyntheticReporter Tests
# =============================================================================


def test_synthetic_reporter_basic():
    """Test SyntheticReporter basic integration."""
    print("\n[TEST] SyntheticReporter - Basic...")
    df = pd.DataFrame(
        {
            "Age": np.random.randint(20, 80, 50),
            "Salary": np.random.normal(50000, 15000, 50),
        }
    )

    out_dir = os.path.join(TEST_OUTPUT_DIR, "synthetic_basic")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    reporter = SyntheticReporter(verbose=True)
    reporter.generate_report(
        synthetic_df=df, generator_name="TestSynthetic", output_dir=out_dir
    )

    assert os.path.exists(os.path.join(out_dir, "index.html"))
    assert os.path.exists(os.path.join(out_dir, "generated_profile.html"))
    assert os.path.exists(os.path.join(out_dir, "density_plots.html"))
    assert os.path.exists(os.path.join(out_dir, "dimensionality_plot.html"))


def test_synthetic_reporter_blocks():
    """Test SyntheticReporter with per-block reports."""
    print("\n[TEST] SyntheticReporter - Blocks...")
    df = pd.DataFrame({"block": np.repeat([1, 2], 30), "Value": np.random.randn(60)})

    out_dir = os.path.join(TEST_OUTPUT_DIR, "synthetic_blocks")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    reporter = SyntheticReporter(verbose=True)
    reporter.generate_report(
        synthetic_df=df,
        generator_name="TestBlocks",
        output_dir=out_dir,
        block_column="block",
        per_block_external_reports=True,
    )

    assert os.path.exists(os.path.join(out_dir, "generated_profile.html"))
    assert os.path.exists(os.path.join(out_dir, "block_1_plots", "profile_report.html"))
    assert os.path.exists(os.path.join(out_dir, "block_2_plots", "profile_report.html"))


# =============================================================================
# ClinicReporter Tests
# =============================================================================


def test_clinic_reporter_basic():
    """Test ClinicReporter basic functionality."""
    print("\n[TEST] ClinicReporter - Basic...")
    df = pd.DataFrame(
        {
            "Gene_A": np.random.normal(100, 20, 50),
            "Gene_B": np.random.normal(50, 10, 50),
        }
    )

    out_dir = os.path.join(TEST_OUTPUT_DIR, "clinic_basic")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    reporter = ClinicReporter(verbose=True)
    reporter.generate_report(
        synthetic_df=df, generator_name="TestClinic", output_dir=out_dir
    )

    assert os.path.exists(os.path.join(out_dir, "generated_profile.html"))
    assert os.path.exists(os.path.join(out_dir, "index.html"))


# =============================================================================
# RealReporter Tests
# =============================================================================


def test_real_reporter_comprehensive():
    """Test RealReporter.generate_comprehensive_report."""
    print("\n[TEST] RealReporter - Comprehensive Report...")
    real_df = pd.DataFrame(
        {
            "Feature1": np.random.normal(0, 1, 100),
            "Feature2": np.random.normal(5, 2, 100),
            "Target": np.random.choice([0, 1], 100),
        }
    )
    synthetic_df = pd.DataFrame(
        {
            "Feature1": np.random.normal(0.1, 1.1, 100),
            "Feature2": np.random.normal(5.2, 2.1, 100),
            "Target": np.random.choice([0, 1], 100),
        }
    )

    out_dir = os.path.join(TEST_OUTPUT_DIR, "real_comprehensive")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    reporter = RealReporter(verbose=True)
    reporter.generate_comprehensive_report(
        real_df=real_df,
        synthetic_df=synthetic_df,
        generator_name="TestRealReporter",
        output_dir=out_dir,
        target_column="Target",
    )

    assert os.path.exists(os.path.join(out_dir, "comparison_report.html"))
    assert os.path.exists(os.path.join(out_dir, "generated_profile.html"))
    assert os.path.exists(os.path.join(out_dir, "sdv_scores.html"))
    assert os.path.exists(os.path.join(out_dir, "density_plots.html"))
    assert os.path.exists(os.path.join(out_dir, "dimensionality_plot.html"))
    assert os.path.exists(os.path.join(out_dir, "drift_results.json"))
    assert os.path.exists(os.path.join(out_dir, "index.html"))


def test_real_reporter_with_blocks():
    """Test RealReporter with block column and SDV evolution."""
    print("\n[TEST] RealReporter - With Blocks (SDV Evolution)...")
    real_df = pd.DataFrame(
        {
            "block": np.repeat([1, 2, 3], 40),
            "Value": np.random.randn(120),
            "Target": np.random.choice([0, 1], 120),
        }
    )
    synthetic_df = pd.DataFrame(
        {
            "block": np.repeat([1, 2, 3], 40),
            "Value": np.random.randn(120) * 1.1,
            "Target": np.random.choice([0, 1], 120),
        }
    )

    out_dir = os.path.join(TEST_OUTPUT_DIR, "real_blocks")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    reporter = RealReporter(verbose=True)
    reporter.generate_comprehensive_report(
        real_df=real_df,
        synthetic_df=synthetic_df,
        generator_name="TestRealBlocks",
        output_dir=out_dir,
        target_column="Target",
        block_column="block",
    )

    assert os.path.exists(os.path.join(out_dir, "comparison_report.html"))
    assert os.path.exists(os.path.join(out_dir, "sdv_evolution.html"))
    assert os.path.exists(os.path.join(out_dir, "sdv_scores.html"))
    assert os.path.exists(os.path.join(out_dir, "drift_results.json"))


# =============================================================================
# DriftInjector Tests
# =============================================================================


def test_drift_injector_auto_report():
    """Test DriftInjector with auto_report=True."""
    print("\n[TEST] DriftInjector - Auto Report...")
    dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "Value": np.random.randn(100),
            "Category": np.random.choice(["A", "B"], 100),
        }
    )

    out_dir = os.path.join(TEST_OUTPUT_DIR, "drift_auto_report")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    injector = DriftInjector(
        original_df=df,
        output_dir=out_dir,
        generator_name="TestDrift",
        time_col="timestamp",
    )

    df_drifted = injector.inject_feature_drift_gradual(
        df=df,
        feature_cols=["Value"],
        drift_type="scale",
        drift_magnitude=2.0,
        center=50,
        width=10,
        resample_rule="2h",
        auto_report=True,
    )

    assert df_drifted is not None
    assert os.path.exists(os.path.join(out_dir, "comparison_report.html"))
    assert os.path.exists(os.path.join(out_dir, "drift_results.json"))


# =============================================================================
# ScenarioInjector Tests
# =============================================================================


def test_scenario_injector_auto_report():
    """Test ScenarioInjector with auto_report=True."""
    print("\n[TEST] ScenarioInjector - Auto Report...")
    df = pd.DataFrame(
        {
            "Feature1": np.random.normal(0, 1, 100),
            "Feature2": np.random.normal(5, 2, 100),
        }
    )

    out_dir = os.path.join(TEST_OUTPUT_DIR, "scenario_auto_report")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    injector = ScenarioInjector(seed=42)

    evolution_config = {
        "Feature1": {"type": "linear", "slope": 0.1, "intercept": 0.0},
    }

    df_modified = injector.evolve_features(
        df=df,
        evolution_config=evolution_config,
        output_dir=out_dir,
        auto_report=True,
        generator_name="TestScenario",
    )

    assert df_modified is not None
    assert os.path.exists(os.path.join(out_dir, "comparison_report.html"))


def test_project_to_future_period():
    """Test ScenarioInjector.project_to_future_period."""
    print("\n[TEST] ScenarioInjector - project_to_future_period...")
    df = pd.DataFrame(
        {
            "block": np.repeat([1, 2, 3], 30),
            "price": np.random.normal(100, 10, 90),
            "sales": np.random.normal(50, 5, 90),
        }
    )

    out_dir = os.path.join(TEST_OUTPUT_DIR, "project_future")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    injector = ScenarioInjector(seed=42)

    trend_config = {
        "price": {"type": "linear", "slope": 3.0},  # +3% per period
    }

    result = injector.project_to_future_period(
        df=df,
        periods=2,
        trend_config=trend_config,
        block_col="block",
        base_strategy="last_period",
        output_dir=out_dir,
        auto_report=True,
    )

    assert result is not None
    assert len(result) > len(df)
    # Should have blocks 1, 2, 3, 4, 5
    assert result["block"].max() == 5
    assert os.path.exists(os.path.join(out_dir, "drift_analysis.html"))


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    import sys

    print("Running comprehensive reporter tests...\n")

    tests = [
        test_plotly_density_plots,
        test_plotly_dimensionality,
        test_plotly_sdv_scores,
        test_plotly_sdv_evolution,
        test_plotly_drift_analysis,
        test_external_reporter_profile,
        test_external_reporter_comparison,
        test_synthetic_reporter_basic,
        test_synthetic_reporter_blocks,
        test_clinic_reporter_basic,
        test_real_reporter_comprehensive,
        test_real_reporter_with_blocks,
        test_drift_injector_auto_report,
        test_scenario_injector_auto_report,
        test_project_to_future_period,
    ]

    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    failed = []

    for test in tests:
        try:
            test()
            print(f"  [OK] {test.__name__}")
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed.append(test.__name__)

    print(f"\n{'=' * 60}")
    if failed:
        print(f"FAILED: {len(failed)}/{len(tests)}")
        sys.exit(1)
    else:
        print(f"ALL {len(tests)} TESTS PASSED")
        sys.exit(0)
