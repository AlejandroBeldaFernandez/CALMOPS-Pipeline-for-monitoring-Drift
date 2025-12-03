# dashboard_real.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import pickle
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.get_logger().setLevel('ERROR')

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(layout="wide")

# --- Auto-reloading Logic ---
REPORT_FILE = ".drift_report.pkl"
REFRESH_INTERVAL_SECONDS = 30

def check_for_updates():
    """Checks if the report file has been updated and reruns the app if it has."""
    if not os.path.exists(REPORT_FILE):
        return

    try:
        current_mtime = os.path.getmtime(REPORT_FILE)

        if "last_mtime" not in st.session_state:
            st.session_state.last_mtime = current_mtime

        if current_mtime > st.session_state.last_mtime:
            st.session_state.last_mtime = current_mtime
            st.toast("Report file updated. Refreshing dashboard...")
            # A short sleep helps ensure the toast message is rendered.
            time.sleep(1)
            st.rerun()
    except FileNotFoundError:
        if "last_mtime" in st.session_state:
            del st.session_state.last_mtime
    except Exception as e:
        st.warning(f"Could not check for file updates: {e}")

check_for_updates()
# --- End of Auto-reloading Logic ---

st.title("🔬 Drift Report Visualizer and Data Analysis")

# Add a manual refresh button as a fallback
if st.button("Manually Reload Data"):
    st.rerun()

# Define the path for the temporary report file
REPORT_FILE_PATH = ".drift_report.pkl"

def display_drift_report(report):
    """Helper function to display the drift analysis report contents."""
    st.header("Drift Analysis Results")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Drift", "Temporal Evolution", "PCA Analysis", "Label Shift", "Concept Drift"])

    with tab1:
        st.subheader("Feature Distribution Comparison")
        data_drift_plots = report.get("data_drift_plots", {})
        if data_drift_plots:
            for feature, fig in data_drift_plots.items():
                st.markdown(f"#### Feature: `{feature}`")
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("No data drift plots were generated.")

    with tab2:
        st.subheader("Feature Evolution Over Time")
        feature_evolution_plots = report.get("feature_evolution_plots", {})
        if feature_evolution_plots:
            for feature, fig in feature_evolution_plots.items():
                st.markdown(f"#### Feature: `{feature}`")
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("No temporal evolution plots were generated (requires a datetime column).")

    with tab3:
        st.subheader("Data Structure Comparison (PCA)")
        pca_plot = report.get("pca_comparison_plot")
        if pca_plot:
            st.pyplot(pca_plot)
            plt.close(pca_plot)
        else:
            st.info("No PCA comparison plot was generated.")

    with tab4:
        st.subheader("Target Distribution Comparison")
        label_shift_plot = report.get("label_shift_plot")
        if label_shift_plot:
            st.pyplot(label_shift_plot)
            plt.close(label_shift_plot)
        else:
            st.info("No label drift plot was generated.")

    with tab5:
        st.subheader("Feature-Target Relationship Analysis")
        concept_drift_plots = report.get("concept_drift_plots", {})
        if concept_drift_plots:
            for feature, fig in concept_drift_plots.items():
                st.markdown(f"#### Feature: `{feature}`")
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("No concept drift plots were generated.")

def display_comprehensive_report(report):
    """Helper function to display the comprehensive analysis report contents."""
    st.header("Comprehensive Comparative Analysis (Before vs. After Drift)")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Basic Info", "Statistical Analysis", "Feature Analysis", "SDV Quality", "Visualizations", "Scaling Evaluation"])

    with tab1:
        st.subheader("Basic Dataset Information")
        basic_info = report.get("basic_info", {})
        if basic_info:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Dataset (Before Drift)**")
                st.json(basic_info.get("real_dataset", {}))
            with col2:
                st.markdown("**Current Dataset (After Drift)**")
                st.json(basic_info.get("synthetic_dataset", {}))
        else:
            st.info("No basic information available.")

    with tab2:
        st.subheader("Detailed Statistical Analysis")
        stats_analysis = report.get("statistical_analysis", {})
        if stats_analysis:
            st.markdown("**Numeric Feature Summary**")
            if "numeric_stats" in stats_analysis and "real_summary" in stats_analysis["numeric_stats"]:
                st.markdown("***Original Dataset***")
                st.dataframe(pd.DataFrame(stats_analysis["numeric_stats"]["real_summary"]))
                st.markdown("***Current Dataset***")
                st.dataframe(pd.DataFrame(stats_analysis["numeric_stats"]["synthetic_summary"]))
            
            st.markdown("**Statistical Tests (Numeric)**")
            if "numeric_stats" in stats_analysis and "statistical_tests" in stats_analysis["numeric_stats"]:
                st.dataframe(pd.DataFrame(stats_analysis["numeric_stats"]["statistical_tests"]).T)

            st.markdown("**Categorical Feature Summary**")
            if "categorical_stats" in stats_analysis:
                st.json(stats_analysis["categorical_stats"])
        else:
            st.info("No statistical analysis available.")

    with tab3:
        st.subheader("Feature Analysis")
        feature_analysis = report.get("feature_analysis", {})
        if feature_analysis:
            st.markdown("**Correlation Analysis**")
            st.json(feature_analysis.get("correlation", {}))
            st.markdown("**Feature Range Analysis**")
            st.dataframe(pd.DataFrame(feature_analysis.get("feature_ranges", {})).T)
        else:
            st.info("No feature analysis available.")

    with tab4:
        st.subheader("Data Quality Evaluation (SDV)")
        sdv_quality = report.get("sdv_quality", {})
        if sdv_quality:
            st.json(sdv_quality)
        else:
            st.info("No SDV quality evaluation available.")

    with tab5:
        st.subheader("Additional Visualizations")
        visualizations = report.get("visualizations", {})
        if visualizations:
            for name, fig in visualizations.items():
                st.markdown(f"#### {name.replace('_', ' ').title()}")
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("No additional visualizations available.")

    with tab6:
        st.subheader("Generator Scaling Evaluation")
        scaling_plot = report.get("scaling_plot")
        if scaling_plot:
            st.pyplot(scaling_plot)
            plt.close(scaling_plot)
        else:
            st.info("No scaling evaluation plot was generated. This may require passing a `RealGeneratorClass` to the reporter.")

# --- Main Logic ---
if os.path.exists(REPORT_FILE_PATH):
    with st.spinner("Loading pre-calculated report..."):
        try:
            with open(REPORT_FILE_PATH, 'rb') as f:
                combined_report = pickle.load(f)
            
            # Display both reports
            display_drift_report(combined_report.get("drift_analysis", {}))
            st.divider()
            display_comprehensive_report(combined_report.get("comprehensive_analysis", {}))

        except Exception as e:
            st.error(f"Error loading the report file: {e}")
            st.info(f"Please run the 'analysis_runner.py' script again to generate the report.")

else:
    st.warning("No report found.")
    st.info("Please run the `analysis_runner.py` script to generate an analysis to be displayed here.")

# --- Polling mechanism for auto-refresh ---
# This part of the script will execute after the page is rendered.
# We sleep for the desired interval and then trigger a rerun to check for file updates again.
time.sleep(REFRESH_INTERVAL_SECONDS)
st.rerun()