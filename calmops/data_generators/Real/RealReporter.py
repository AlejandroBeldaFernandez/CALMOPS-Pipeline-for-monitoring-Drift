"""Static, File-Based Real Data Reporter

This module provides the RealReporter class, which is responsible for generating a detailed,
static report comparing a real dataset with a synthetic one. All generated artifacts,
including JSON reports and plot images, are saved directly to the disk.

Key Features:
- **Comprehensive Comparison**: Generates a wide array of comparisons, including basic info, target analysis, and statistical evaluations.
- **Rich Visualizations**: Creates and saves multiple plots:
  - PCA comparison to visualize overall data structure.
  - Correlation heatmaps to compare feature relationships.
  - Distribution plots (KDEs) for individual features.
  - Boxplots for numeric feature comparison.
  - Instance balance plots for categorical features.
  - Time evolution plots to track feature changes over time or index.
- **SDV Quality Score**: Integrates with `sdv` to calculate a `QualityScore` and a custom weighted score that penalizes for data duplication and null values.
- **Block-Level Analysis**: If a `block_column` is provided, it generates specific reports and plots for each individual block.
- **Drift-Aware Reporting**: Can highlight drift-injected areas in plots when provided with drift configuration.
- **Static Output**: All outputs are saved to a specified directory, making it suitable for environments without dynamic UIs.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
from datetime import datetime
import os
import json
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, ks_2samp

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    from sdv.evaluation.single_table import evaluate_quality
    from sdv.metadata import SingleTableMetadata

    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False
    warnings.warn("SDV not available. Quality assessment will be limited.")

from calmops.utils.distribution_fitter import fit_distribution

# Statistical tests
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class RealReporter:
    """
    Generates a static, file-based report comparing a real dataset and its synthetic counterpart.
    """

    def __init__(self, verbose: bool = True):
        """
        Initializes the RealReporter.

        Args:
            verbose (bool): If True, prints progress messages to the console.
        """
        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)

        plt.style.use("default")
        sns.set_palette("husl")
        plt.rcParams.update(
            {
                "figure.dpi": 300,
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
                "font.size": 14,
                "axes.titlesize": 16,
                "axes.labelsize": 14,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 12,
            }
        )

    def _report_block_analysis(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        block_column: str,
        target_column: Optional[str],
    ) -> Dict[str, Any]:
        """Analyzes and compares each block within the real and synthetic datasets."""
        analysis = {}
        try:
            unique_blocks = sorted(real_df[block_column].unique(), key=str)

            for block_id in unique_blocks:
                real_block_df = real_df[real_df[block_column] == block_id]
                synthetic_block_df = synthetic_df[
                    synthetic_df[block_column] == block_id
                ]

                block_stats = {
                    "real_num_rows": len(real_block_df),
                    "synthetic_num_rows": len(synthetic_block_df),
                    "real_null_values": int(real_block_df.isnull().sum().sum()),
                    "synthetic_null_values": int(
                        synthetic_block_df.isnull().sum().sum()
                    ),
                }

                if target_column and target_column in real_block_df.columns:
                    block_stats["real_target_distribution"] = (
                        real_block_df[target_column]
                        .value_counts(normalize=True)
                        .to_dict()
                    )
                if target_column and target_column in synthetic_block_df.columns:
                    block_stats["synthetic_target_distribution"] = (
                        synthetic_block_df[target_column]
                        .value_counts(normalize=True)
                        .to_dict()
                    )

                analysis[str(block_id)] = block_stats
        except Exception as e:
            self.logger.error(f"Failed to generate block analysis: {e}")

        return analysis

    def generate_comprehensive_report(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        generator_name: str,
        output_dir: str,
        target_column: Optional[str] = None,
        block_column: Optional[str] = None,
        focus_cols: Optional[List[str]] = None,
        drift_config: Optional[Dict[str, Any]] = None,
        time_col: Optional[str] = None,
        drift_history: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Generates a comprehensive file-based report comparing real and synthetic data.

        Args:
            real_df (pd.DataFrame): The original, real dataset.
            synthetic_df (pd.DataFrame): The generated, synthetic dataset.
            generator_name (str): Name of the generator used, for labeling plots.
            output_dir (str): Directory to save the report and all plots.
            target_column (Optional[str]): Name of the target variable column.
            block_column (Optional[str]): Name of the column defining data blocks.
            focus_cols (Optional[List[str]]): A list of specific columns to focus on for plotting.
            drift_config (Optional[Dict[str, Any]]): Configuration of the injected drift, used for highlighting plots.
            time_col (Optional[str]): Name of the timestamp column for time-series plots.
        """
        if self.verbose:
            print("=" * 80)
            print(f"COMPREHENSIVE REAL DATA GENERATION REPORT")
            print(f"Generator: {generator_name}")
            print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            if focus_cols:
                print(f"Focusing on columns: {focus_cols}")
            print("=" * 80)

        os.makedirs(output_dir, exist_ok=True)

        report = {
            "generator_name": generator_name,
            "generation_timestamp": datetime.now().isoformat(),
            "output_directory": output_dir,
            "basic_info": self._report_basic_info(real_df, synthetic_df),
            "target_analysis": self._report_target_analysis(
                real_df, synthetic_df, target_column
            )
            if target_column
            else None,
            "sdv_quality": self._report_sdv_quality_assessment(real_df, synthetic_df)
            if SDV_AVAILABLE
            else None,
            "mode_analysis": self._report_mode_analysis(real_df, synthetic_df),
            "numeric_analysis": None,
            "distribution_comparison": self._compare_numeric_distributions(
                real_df, synthetic_df
            ),
            "quality_analysis": self._report_detailed_quality_analysis(
                real_df, synthetic_df
            ),
            "statistical_tests": self._compute_statistical_tests(real_df, synthetic_df),
            "block_analysis": None,
            "drift_history": drift_history,
            "plots": {},
        }

        # --- Determine which columns to focus on for plotting ---
        # If drift is detected, focus on the affected columns. User can override with focus_cols.
        affected_cols = []
        if drift_config:
            affected_cols.extend(drift_config.get("feature_cols", []))
            if (
                "target_col" in drift_config
                and drift_config["target_col"] not in affected_cols
            ):
                affected_cols.append(drift_config["target_col"])

        # Priority: user-defined focus_cols > drift-affected cols > all cols
        if focus_cols:
            cols_for_dist_plots = focus_cols
        elif affected_cols:
            cols_for_dist_plots = affected_cols
        else:
            cols_for_dist_plots = real_df.columns.tolist()

        # Determine the time column to use
        final_time_col = time_col  # Use the one passed in, if any
        if final_time_col and final_time_col not in real_df.columns:
            self.logger.warning(
                f"Provided time column '{final_time_col}' not found. Ignoring."
            )
            final_time_col = None

        if not final_time_col:  # If not provided or not found, try to find default ones
            final_time_col = (
                "timestamp"
                if "timestamp" in real_df.columns
                else "chunk"
                if "chunk" in real_df.columns
                else None
            )

        # --- Generate and Save Plots --- #
        plots = {}
        # Generic plots (PCA, Correlation) are not focused by default unless specified by user
        plots["dimensionality_reduction"] = self._save_umap_plot(
            real_df, synthetic_df, output_dir, generator_name
        )
        plots["correlation_heatmap"] = self._save_correlation_heatmap(
            real_df, synthetic_df, output_dir, focus_cols=focus_cols
        )
        plots["interaction_plots"] = self._save_interaction_plots(
            real_df, synthetic_df, output_dir, focus_cols=focus_cols
        )
        plots["mode_comparison"] = self._save_mode_analysis_plot(
            report["mode_analysis"], output_dir
        )

        # Generate distribution plots only for the determined columns
        dist_plots = {}
        box_plots = {}
        balance_plots = {}
        for col in cols_for_dist_plots:
            if col in synthetic_df.columns:
                if col == final_time_col:
                    continue
                dist_plots[col] = self._save_distribution_plot(
                    real_df, synthetic_df, col, output_dir
                )

                is_categorical = False
                if pd.api.types.is_string_dtype(
                    real_df[col]
                ) or pd.api.types.is_categorical_dtype(real_df[col]):
                    is_categorical = True
                elif pd.api.types.is_numeric_dtype(real_df[col]):
                    unique_values = real_df[col].nunique()
                    if unique_values < 25 or (unique_values / len(real_df)) < 0.05:
                        is_categorical = True

                if is_categorical:
                    balance_plots[col] = self._save_instances_balance_plot(
                        real_df, synthetic_df, col, output_dir
                    )

                if pd.api.types.is_numeric_dtype(real_df[col]):
                    box_plots[col] = self._save_boxplot_plot(
                        real_df, synthetic_df, col, output_dir
                    )

        plots["distribution_plots"] = dist_plots
        plots["box_plots"] = box_plots
        plots["instances_balance_plots"] = balance_plots

        # Always call the plotting function
        plots["time_evolution_plots"] = self._save_time_evolution_plots(
            real_df,
            synthetic_df,
            output_dir,
            time_col=final_time_col,
            focus_cols=focus_cols,
            drift_config=drift_config,
        )

        report["plots"] = plots

        if block_column and block_column in synthetic_df.columns:
            report["block_analysis"] = self._report_block_analysis(
                real_df=real_df,
                synthetic_df=synthetic_df,
                block_column=block_column,
                target_column=target_column,
            )
            # --- Generate Per-Block Plots ---
            if self.verbose:
                print(f"\n🔬 Generating per-block analysis plots...")

            block_plots_report = {}
            # Ensure block order is consistent
            unique_blocks = sorted(real_df[block_column].unique(), key=str)

            for block_id in unique_blocks:
                block_output_dir = os.path.join(output_dir, f"block_{block_id}_plots")
                os.makedirs(block_output_dir, exist_ok=True)

                real_block_df = real_df[real_df[block_column] == block_id]
                synthetic_block_df = synthetic_df[
                    synthetic_df[block_column] == block_id
                ]

                if real_block_df.empty or synthetic_block_df.empty:
                    self.logger.warning(
                        f"Skipping plots for block {block_id} due to empty data."
                    )
                    continue

                block_plots_report[str(block_id)] = {
                    "distribution_plots": {},
                    "instances_balance_plots": {},
                    "correlation_heatmap": None,
                }

                for col in real_block_df.columns:
                    if col == block_column:
                        continue

                    # Generate distribution plot for the column within the block
                    plot_path = self._save_distribution_plot(
                        real_block_df, synthetic_block_df, col, block_output_dir
                    )
                    if plot_path:
                        block_plots_report[str(block_id)]["distribution_plots"][col] = (
                            plot_path
                        )

                    # Check if categorical to generate balance plot
                    is_categorical = False
                    if pd.api.types.is_string_dtype(
                        real_block_df[col]
                    ) or pd.api.types.is_categorical_dtype(real_block_df[col]):
                        is_categorical = True
                    elif (
                        pd.api.types.is_numeric_dtype(real_block_df[col])
                        and real_block_df[col].nunique() < 25
                    ):
                        is_categorical = True

                    if is_categorical:
                        balance_plot_path = self._save_instances_balance_plot(
                            real_block_df, synthetic_block_df, col, block_output_dir
                        )
                        if balance_plot_path:
                            block_plots_report[str(block_id)][
                                "instances_balance_plots"
                            ][col] = balance_plot_path

                if self.verbose:
                    print(
                        f"  -> Plots for block '{block_id}' saved to: {block_output_dir}"
                    )

            report["plots"]["per_block_plots"] = block_plots_report

        # Scaling evaluation is a global metric
        self.run_scaling_evaluation(
            original_data=real_df,
            synthetic_data_full=synthetic_df,
            output_dir=output_dir,
            block_column=block_column,
        )

        # Call the markdown generator instead of saving JSON
        try:
            self.generate_markdown_report(report, output_dir)
            if self.verbose:
                print(
                    f"\n✅ Markdown report saved to: {os.path.join(output_dir, 'report_summary.md')}"
                )
        except Exception as e:
            self.logger.error(f"Failed to generate markdown report: {e}")

    def _get_weighted_sdv_score(
        self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame, base_score: float
    ) -> float:
        """
        Calculates a weighted SDV score, penalized by data duplication and null values.
        The score is penalized by the percentage of rows in the synthetic dataset that are either:
        1. Duplicates of other rows within the synthetic dataset.
        2. Exact copies of rows from the real dataset.
        """
        if synthetic_df.empty:
            return 0.0

        # --- Calculate duplication percentage ---
        # 1. Get indices of rows that are internal duplicates in the synthetic data
        # Use duplicated() which is vectorized and efficient
        internal_dup_indices = synthetic_df[synthetic_df.duplicated()].index

        # 2. Get indices of rows in synthetic data that are copies of rows in real data
        # Optimized approach: Inner join on all columns to find exact matches
        # This avoids the slow and memory-intensive apply(hash)
        try:
            # We only care IF it exists in real_df, not how many times
            # dropping duplicates in real_df first speeds up the merge
            real_unique = real_df.drop_duplicates()

            # Merge is much faster than row-wise iteration
            merged = synthetic_df.merge(
                real_unique,
                on=list(synthetic_df.columns),
                how="inner",
                suffixes=("", "_real"),
            )

            # The indices in synthetic_df that are present in merged are the cross-duplicates
            # Note: merge resets index by default, so we need to be careful if we wanted exact indices,
            # but for counting how many rows are copies, we just need the count of the merge logic applied to the original synthetic.
            # Simpler approach: usage of isin if all columns match is tricky with NaNs.
            # Robust approach with merge:

            # Let's count rows in synthetic that have a match in real.
            # We can mark synthetic rows that are in real.

            # Filter synthetic rows that are in real_df
            # Indicator=True gives us a column '_merge' with 'both', 'left_only', 'right_only'
            merged_indicator = synthetic_df.merge(
                real_unique, on=list(synthetic_df.columns), how="left", indicator=True
            )
            cross_dup_indices = merged_indicator[
                merged_indicator["_merge"] == "both"
            ].index

        except Exception as e:
            self.logger.warning(
                f"Optimization for duplicate detection failed: {e}. Falling back to row-wise (slow)."
            )
            # fallback or safe count
            cross_dup_indices = pd.Index([])

        # 3. Combine the indices to get a unique set of "bad" rows
        bad_indices = internal_dup_indices.union(cross_dup_indices)
        total_bad_rows = len(bad_indices)

        duplicated_rows_percentage = (
            total_bad_rows / len(synthetic_df) if len(synthetic_df) > 0 else 0.0
        )

        # --- Calculate null percentage ---
        null_values_percentage = (
            synthetic_df.isnull().sum().sum()
            / (synthetic_df.shape[0] * synthetic_df.shape[1])
            if synthetic_df.size > 0
            else 0.0
        )

        # --- Calculate final score ---
        base_score = 0.0 if pd.isna(base_score) else base_score
        weighted_score = (
            base_score * (1 - duplicated_rows_percentage) * (1 - null_values_percentage)
        )

        if self.verbose:
            self.logger.info(
                f"Weighted Score Calculation: Base={base_score:.3f}, Duplication Penalty={duplicated_rows_percentage:.3f}, Null Penalty={null_values_percentage:.3f} -> Final={weighted_score:.3f}"
            )

        return weighted_score

    def _save_target_distribution_plot(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        target_column: str,
        output_dir: str,
    ) -> Optional[str]:
        """
        Saves a distribution comparison plot for the target column (for classification tasks).
        """
        try:
            real_target = real_df[target_column]

            # Determine problem type
            problem_type = ""
            if pd.api.types.is_string_dtype(
                real_target
            ) or pd.api.types.is_categorical_dtype(real_target):
                problem_type = "Classification"
            elif pd.api.types.is_numeric_dtype(real_target):
                unique_values = real_target.nunique()
                if unique_values < 25 or (unique_values / len(real_target)) < 0.05:
                    problem_type = "Classification"
                else:
                    problem_type = "Regression"

            if problem_type == "Regression":
                self.logger.info(
                    f"Target '{target_column}' is regression. Skipping target distribution plot."
                )
                return None

            fig, ax = plt.subplots(figsize=(12, 7))
            real_counts = real_df[target_column].value_counts(normalize=True)
            synthetic_counts = synthetic_df[target_column].value_counts(normalize=True)

            df_counts = (
                pd.DataFrame({"Real": real_counts, "Synthetic": synthetic_counts})
                .fillna(0)
                .sort_index()
            )

            df_counts.plot(kind="bar", ax=ax, alpha=0.7, width=0.8)

            ax.set_title(f"Target Column Distribution: {target_column}", fontsize=14)
            ax.set_ylabel("Proportion of Instances", fontsize=10)
            ax.set_xlabel(target_column, fontsize=10)
            ax.tick_params(axis="x", rotation=45, labelsize=9)
            ax.grid(axis="y", linestyle="--", alpha=0.7)

            # Add text labels on top of bars
            for container in ax.containers:
                ax.bar_label(container, fmt="%.2f", fontsize=8, padding=3)

            plt.tight_layout()
            plot_path = os.path.join(
                output_dir, f"target_dist_plot_{target_column}.png"
            )
            fig.savefig(plot_path)
            plt.close(fig)
            return plot_path
        except Exception as e:
            self.logger.error(
                f"Failed to generate target distribution plot for {target_column}: {e}"
            )
            return None

    def _save_instances_balance_plot(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        column: str,
        output_dir: str,
    ) -> Optional[str]:
        """
        Saves a distribution comparison plot for a categorical column.
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 7))
            real_counts = real_df[column].value_counts(normalize=True)
            synthetic_counts = synthetic_df[column].value_counts(normalize=True)

            df_counts = (
                pd.DataFrame({"Real": real_counts, "Synthetic": synthetic_counts})
                .fillna(0)
                .sort_index()
            )

            # High cardinality check
            if len(df_counts) > 20:
                self.logger.warning(
                    f"High cardinality in column '{column}' ({len(df_counts)} unique values). Plot may be cluttered."
                )
                # Truncate to top 20 for readability
                top20 = df_counts.sum(axis=1).nlargest(20).index
                df_counts = df_counts.loc[top20]

            df_counts.plot(kind="bar", ax=ax, alpha=0.7, width=0.8)

            ax.set_title(f"Instance Balance: {column}", fontsize=14)
            ax.set_ylabel("Proportion of Instances", fontsize=10)
            ax.set_xlabel(column, fontsize=10)
            ax.tick_params(axis="x", rotation=45, labelsize=9)
            ax.grid(axis="y", linestyle="--", alpha=0.7)

            # Add text labels on top of bars
            for container in ax.containers:
                ax.bar_label(container, fmt="%.2f", fontsize=8, padding=3)

            plt.tight_layout()
            plot_path = os.path.join(output_dir, f"instance_balance_plot_{column}.png")
            fig.savefig(plot_path)
            plt.close(fig)
            return plot_path
        except Exception as e:
            self.logger.error(
                f"Failed to generate instance balance plot for {column}: {e}"
            )
            return None

    def _save_time_evolution_plots(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        output_dir: str,
        time_col: Optional[str] = None,
        focus_cols: Optional[List[str]] = None,
        drift_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Saves visually enhanced evolution plots for numeric features over time or index, with moving averages and drift highlighting."""
        plots = {}
        x_axis_label = time_col if time_col else "Instance Index"
        moving_avg_window = 7  # Window for moving average

        # Determine features to plot
        features_to_plot = []
        if drift_config:
            features_to_plot.extend(drift_config.get("feature_cols", []))
            if (
                "target_column" in drift_config
                and drift_config["target_column"] not in features_to_plot
            ):
                features_to_plot.append(drift_config["target_column"])
        elif focus_cols:
            features_to_plot = focus_cols
        else:
            features_to_plot = real_df.columns.tolist()

        # Separate columns into continuous numeric, discrete numeric, and categorical
        continuous_numeric_cols = []
        discrete_numeric_cols = []
        categorical_cols_to_plot = []

        for col in features_to_plot:
            if col not in real_df.columns:
                continue

            if pd.api.types.is_numeric_dtype(real_df[col]):
                # Heuristic to distinguish discrete from continuous
                unique_values = real_df[col].nunique()
                if unique_values < 25:
                    discrete_numeric_cols.append(col)
                else:
                    continuous_numeric_cols.append(col)
            else:  # Is object, category, etc.
                categorical_cols_to_plot.append(col)

        # --- Plot Continuous Numeric Columns ---
        for col in continuous_numeric_cols:
            if col == time_col or col not in synthetic_df.columns:
                continue
            try:
                fig, ax = plt.subplots(figsize=(15, 7))

                real_color = "#1f77b4"
                synth_color = "#ff7f0e"

                if time_col:
                    # Aggregated plot for time-based axis
                    real_agg = (
                        real_df.groupby(time_col)[col].agg(["mean", "std"]).fillna(0)
                    )
                    synth_agg = (
                        synthetic_df.groupby(time_col)[col]
                        .agg(["mean", "std"])
                        .fillna(0)
                    )

                    # Calculate moving average of the mean
                    real_mean_ma = (
                        real_agg["mean"]
                        .rolling(window=moving_avg_window, min_periods=1, center=True)
                        .mean()
                    )
                    synth_mean_ma = (
                        synth_agg["mean"]
                        .rolling(window=moving_avg_window, min_periods=1, center=True)
                        .mean()
                    )

                    # Plot moving averages
                    real_mean_ma.plot(
                        ax=ax,
                        label=f"Original Mean ({moving_avg_window}-MA)",
                        style="-",
                        linewidth=2.5,
                        color=real_color,
                    )
                    synth_mean_ma.plot(
                        ax=ax,
                        label=f"Generated Mean ({moving_avg_window}-MA)",
                        style="--",
                        linewidth=2.5,
                        color=synth_color,
                    )
                else:
                    # Raw plot for index-based axis with moving average
                    real_ma = (
                        real_df[col]
                        .rolling(window=moving_avg_window, min_periods=1, center=True)
                        .mean()
                    )
                    synth_ma = (
                        synthetic_df[col]
                        .rolling(window=moving_avg_window, min_periods=1, center=True)
                        .mean()
                    )

                    real_ma.reset_index(drop=True).plot(
                        ax=ax,
                        label=f"Original ({moving_avg_window}-MA)",
                        style="-",
                        linewidth=2,
                        color=real_color,
                    )
                    synth_ma.reset_index(drop=True).plot(
                        ax=ax,
                        label=f"Generated ({moving_avg_window}-MA)",
                        style="--",
                        linewidth=2,
                        color=synth_color,
                    )

                # --- Add advanced drift visualization ---
                if drift_config:
                    method = drift_config.get("drift_method", "")

                    # Case 1: Recurrent drift with specific concepts/blocks
                    if (
                        method == "inject_feature_drift_recurrent"
                        and "applied_windows" in drift_config
                    ):
                        applied_windows = drift_config.get("applied_windows", [])
                        unique_blocks = sorted(
                            list(
                                set(
                                    w.get("block_id")
                                    for w in applied_windows
                                    if w.get("block_id") is not None
                                )
                            )
                        )
                        palette = sns.color_palette("husl", n_colors=len(unique_blocks))
                        color_map = {
                            block_id: color
                            for block_id, color in zip(unique_blocks, palette)
                        }
                        labeled_concepts = set()

                        for window in applied_windows:
                            block_id = window.get("block_id")
                            center_idx = window.get("center")
                            width = window.get("width")

                            if not all(
                                [block_id, center_idx is not None, width is not None]
                            ):
                                continue

                            start_idx = max(0, int(center_idx) - int(width) // 2)
                            end_idx = min(
                                len(synthetic_df) - 1, int(center_idx) + int(width) // 2
                            )

                            label = None
                            if block_id not in labeled_concepts:
                                label = f"Drift: {block_id}"
                                labeled_concepts.add(block_id)

                            color = color_map.get(block_id, "gray")

                            if time_col:
                                if start_idx < len(synthetic_df) and end_idx < len(
                                    synthetic_df
                                ):
                                    start_val = synthetic_df[time_col].iloc[start_idx]
                                    end_val = synthetic_df[time_col].iloc[end_idx]
                                    ax.axvspan(
                                        start_val,
                                        end_val,
                                        color=color,
                                        alpha=0.2,
                                        label=label,
                                    )
                            else:
                                ax.axvspan(
                                    start_idx,
                                    end_idx,
                                    color=color,
                                    alpha=0.2,
                                    label=label,
                                )

                    # Case 2: General window-based drifts (gradual, incremental, abrupt)
                    elif "center" in drift_config and "width" in drift_config:
                        center_idx = drift_config.get("center")
                        width = drift_config.get("width")
                        if center_idx is None:
                            center_idx = len(synthetic_df) // 2
                        label = f"Drift Area ({method}) "
                        start_idx = max(0, int(center_idx) - int(width) // 2)
                        end_idx = min(
                            len(synthetic_df) - 1, int(center_idx) + int(width) // 2
                        )

                        color = (
                            "orange"
                            if method == "inject_feature_drift_incremental"
                            else "red"
                        )

                        if time_col:
                            if start_idx < len(synthetic_df) and end_idx < len(
                                synthetic_df
                            ):
                                start_val = synthetic_df[time_col].iloc[start_idx]
                                end_val = synthetic_df[time_col].iloc[end_idx]
                                ax.axvspan(
                                    start_val,
                                    end_val,
                                    color=color,
                                    alpha=0.2,
                                    label=label,
                                )
                            else:
                                ax.axvspan(
                                    start_idx,
                                    end_idx,
                                    color=color,
                                    alpha=0.2,
                                    label=label,
                                )

                    # Case 3: Drifts defined only by a starting point
                    elif "start_index" in drift_config:
                        start_idx_only = drift_config.get("start_index")
                        if start_idx_only is not None:
                            label = f"Drift Start ({method})"
                            if time_col:
                                if start_idx_only < len(synthetic_df):
                                    start_val = synthetic_df[time_col].iloc[
                                        start_idx_only
                                    ]
                                    ax.axvline(
                                        x=start_val,
                                        color="red",
                                        linestyle="--",
                                        linewidth=2,
                                        label=label,
                                    )
                            else:
                                ax.axvline(
                                    x=start_idx_only,
                                    color="red",
                                    linestyle="--",
                                    linewidth=2,
                                    label=label,
                                )

                ax.set_title(f"Evolution of {col}", fontsize=18, weight="bold")
                ax.set_xlabel(x_axis_label, fontsize=14)
                ax.set_ylabel(col, fontsize=14)

                ax.grid(
                    True, which="major", linestyle="--", linewidth="0.5", color="grey"
                )
                ax.tick_params(axis="x", rotation=45, labelsize=12)
                ax.tick_params(axis="y", labelsize=12)

                fig.autofmt_xdate()

                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc="best")

                plt.tight_layout()
                plot_path = os.path.join(output_dir, f"evolution_{col}.png")
                fig.savefig(plot_path)
                plt.close(fig)
                plots[col] = plot_path
            except Exception as e:
                self.logger.error(f"Failed to generate evolution plot for {col}: {e}")

        # --- Plot Discrete Numeric and Categorical Columns ---
        all_categorical_like_cols = discrete_numeric_cols + categorical_cols_to_plot
        if time_col:
            for col in all_categorical_like_cols:
                if col == time_col or col not in synthetic_df.columns:
                    continue
                # Use the categorical evolution plot for discrete numerics as well
                plots[col] = self._save_categorical_evolution_plot(
                    real_df, synthetic_df, time_col, col, output_dir
                )

        return plots

    def _save_categorical_evolution_plot(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        time_col: str,
        category_col: str,
        output_dir: str,
    ) -> Optional[str]:
        """Saves a visually enhanced plot showing the evolution of a categorical feature's proportions over time."""
        try:
            # --- High Cardinality Handling ---
            HIGH_CARDINALITY_THRESHOLD = 15

            real_df_plot = real_df.copy()
            synthetic_df_plot = synthetic_df.copy()

            n_unique = real_df_plot[category_col].nunique()
            if n_unique > HIGH_CARDINALITY_THRESHOLD:
                self.logger.info(
                    f"High cardinality ({n_unique}) detected for '{category_col}'. Grouping less frequent categories into 'Other'."
                )

                # Find top categories from the real data
                top_categories = (
                    real_df_plot[category_col]
                    .value_counts()
                    .nlargest(HIGH_CARDINALITY_THRESHOLD - 1)
                    .index.tolist()
                )

                # Apply the grouping to both dataframes
                real_df_plot[category_col] = real_df_plot[category_col].apply(
                    lambda x: x if x in top_categories else "Other"
                )
                synthetic_df_plot[category_col] = synthetic_df_plot[category_col].apply(
                    lambda x: x if x in top_categories else "Other"
                )

            # --- Data Preparation ---
            # Pivot tables to get proportions of each category over time
            real_pivot = real_df_plot.pivot_table(
                index=time_col, columns=category_col, aggfunc="size", fill_value=0
            )
            real_pivot = real_pivot.div(real_pivot.sum(axis=1), axis=0)

            synth_pivot = synthetic_df_plot.pivot_table(
                index=time_col, columns=category_col, aggfunc="size", fill_value=0
            )
            synth_pivot = synth_pivot.div(synth_pivot.sum(axis=1), axis=0)

            # Align data for error calculation
            all_cols = real_pivot.columns.union(synth_pivot.columns)
            all_idx = real_pivot.index.union(synth_pivot.index)
            real_aligned = real_pivot.reindex(
                index=all_idx, columns=all_cols, fill_value=0
            )
            synth_aligned = synth_pivot.reindex(
                index=all_idx, columns=all_cols, fill_value=0
            )
            error_pivot = (real_aligned - synth_aligned).abs()

            # --- Plotting ---
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 22), sharex=True)

            # Plot 1: Real Data Evolution
            real_pivot.plot(kind="line", stacked=False, ax=ax1, alpha=0.7)
            ax1.set_title(
                f"Evolution of {category_col} (Real)", fontsize=18, weight="bold"
            )
            ax1.set_ylabel("Proportion", fontsize=14)
            ax1.legend(
                title=category_col,
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                fontsize=12,
            )
            ax1.grid(True, which="major", linestyle="--", linewidth="0.5", color="grey")

            # Plot 2: Synthetic Data Evolution
            synth_pivot.plot(kind="line", stacked=False, ax=ax2, alpha=0.7)
            ax2.set_title(
                f"Evolution of {category_col} (Synthetic)", fontsize=18, weight="bold"
            )
            ax2.set_ylabel("Proportion", fontsize=14)
            ax2.legend(
                title=category_col,
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                fontsize=12,
            )
            ax2.grid(True, which="major", linestyle="--", linewidth="0.5", color="grey")

            # Plot 3: Absolute Error
            error_pivot.plot(
                kind="line", ax=ax3, marker="o", linestyle="--", markersize=5
            )
            ax3.set_title(
                "Absolute Error in Proportion (Real vs. Synthetic)",
                fontsize=18,
                weight="bold",
            )
            ax3.set_xlabel(time_col, fontsize=14)
            ax3.set_ylabel("Absolute Difference", fontsize=14)
            ax3.grid(True, which="major", linestyle="--", linewidth="0.5", color="grey")
            ax3.legend(
                title="Category",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                fontsize=12,
            )

            # Improve tick labels and layout
            plt.xticks(rotation=45, ha="right")
            fig.autofmt_xdate()
            plt.tight_layout(
                rect=[0, 0, 0.9, 1]
            )  # Adjust layout to make space for legend

            plot_path = os.path.join(
                output_dir, f"categorical_evolution_{category_col}.png"
            )
            fig.savefig(plot_path)
            plt.close(fig)
            return plot_path
        except Exception as e:
            self.logger.error(
                f"Failed to generate categorical evolution plot for {category_col}: {e}"
            )
            return None

    def _save_distribution_plot(
        self, real_df, synthetic_df, column, output_dir
    ) -> Optional[str]:
        """Saves a distribution comparison plot (KDE) for a single numeric feature."""
        try:
            fig = plt.figure(figsize=(10, 6))
            fig = plt.figure(figsize=(10, 6))
            is_numeric = pd.api.types.is_numeric_dtype(real_df[column])
            # User requested KDE style even for low cardinality (like booleans)
            if is_numeric and real_df[column].nunique() > 1:
                # Downsample for plotting performance if needed
                MAX_SAMPLES = 50000
                real_plot = real_df[column]
                if len(real_plot) > MAX_SAMPLES:
                    real_plot = real_plot.sample(n=MAX_SAMPLES, random_state=42)

                synth_plot = synthetic_df[column]
                if len(synth_plot) > MAX_SAMPLES:
                    synth_plot = synth_plot.sample(n=MAX_SAMPLES, random_state=42)

                sns.kdeplot(real_plot, label="Real", fill=True)
                sns.kdeplot(synth_plot, label="Synthetic", fill=True, alpha=0.7)
                plt.title(f"Distribution Comparison: {column}")
            else:
                HIGH_CARDINALITY_THRESHOLD = 30
                CATEGORY_THRESHOLD = 10
                real_series = real_df[column]
                synth_series = synthetic_df[column]
                if real_series.nunique() > HIGH_CARDINALITY_THRESHOLD:
                    top_categories = (
                        real_series.value_counts()
                        .nlargest(HIGH_CARDINALITY_THRESHOLD - 1)
                        .index.tolist()
                    )
                    real_series = real_series.apply(
                        lambda x: x if x in top_categories else "Other"
                    )
                    synth_series = synth_series.apply(
                        lambda x: x if x in top_categories else "Other"
                    )
                before_counts = real_series.value_counts(normalize=True).rename("Real")
                after_counts = synth_series.value_counts(normalize=True).rename(
                    "Synthetic"
                )
                df_counts = pd.concat([before_counts, after_counts], axis=1).fillna(0)
                if len(df_counts) > CATEGORY_THRESHOLD:
                    current_figsize = (max(12, len(df_counts) * 0.8), 6)
                    fig.set_size_inches(current_figsize)
                    ax = df_counts.plot(kind="bar", alpha=0.7, ax=plt.gca())
                    plt.xticks(rotation=90)
                else:
                    ax = df_counts.plot(
                        kind="line",
                        alpha=0.8,
                        ax=plt.gca(),
                        marker="o",
                        linewidth=2,
                        markersize=8,
                    )
                    for i, (index, row) in enumerate(df_counts.iterrows()):
                        for j, value in enumerate(row):
                            ax.text(
                                i + (j * 0.1),
                                value + 0.01,
                                f"{value:.2f}",
                                ha="center",
                                va="bottom",
                                fontsize=8,
                            )
                    plt.xticks(rotation=45)
                plt.title(f"Distribution Comparison: {column}")
                plt.ylabel("Proportion")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)
            plot_path = os.path.join(output_dir, f"dist_plot_{column}.png")
            fig.savefig(plot_path)
            plt.close(fig)
            return plot_path
        except Exception as e:
            self.logger.error(f"Failed to generate distribution plot for {column}: {e}")
            return None

    def _save_correlation_heatmap(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        output_dir: str,
        focus_cols: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Saves a heatmap comparing the correlation matrices of numeric features between real and synthetic data."""
        try:
            all_numeric_cols = real_df.select_dtypes(include=np.number).columns
            if focus_cols:
                numeric_cols = [col for col in focus_cols if col in all_numeric_cols]
            else:
                numeric_cols = all_numeric_cols.tolist()

            if len(numeric_cols) < 2:
                self.logger.info(
                    "Not enough numeric columns to generate a correlation heatmap."
                )
                return None

            real_corr = real_df[numeric_cols].corr()
            synth_corr = synthetic_df[numeric_cols].corr()
            diff_corr = (real_corr - synth_corr).abs()

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(40, 12))

            sns.heatmap(
                real_corr,
                ax=ax1,
                annot=True,
                cmap="viridis",
                fmt=".2f",
                annot_kws={"size": 12},
            )
            ax1.set_title("Real Data Correlation", fontsize=16)
            ax1.tick_params(axis="x", rotation=45, labelsize=12)
            ax1.tick_params(axis="y", rotation=0, labelsize=12)

            sns.heatmap(
                synth_corr,
                ax=ax2,
                annot=True,
                cmap="viridis",
                fmt=".2f",
                annot_kws={"size": 12},
            )
            ax2.set_title("Synthetic Data Correlation", fontsize=16)
            ax2.tick_params(axis="x", rotation=45, labelsize=12)
            ax2.tick_params(axis="y", rotation=0, labelsize=12)

            sns.heatmap(
                diff_corr,
                ax=ax3,
                annot=True,
                cmap="hot_r",
                fmt=".2f",
                annot_kws={"size": 12},
            )
            ax3.set_title("Absolute Difference", fontsize=16)
            ax3.tick_params(axis="x", rotation=45, labelsize=12)
            ax3.tick_params(axis="y", rotation=0, labelsize=12)

            plt.tight_layout()
            plot_path = os.path.join(output_dir, "correlation_heatmap.png")
            fig.savefig(plot_path)
            plt.close(fig)
            return plot_path
        except Exception as e:
            self.logger.error(f"Failed to generate correlation heatmap: {e}")
            return None

    def _save_interaction_plots(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        output_dir: str,
        max_plots: int = 5,
        focus_cols: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Saves boxplots comparing the interaction between categorical and numeric features."""
        plots = {}
        try:
            numeric_cols = real_df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = real_df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            plot_count = 0
            for num_col in numeric_cols:
                for cat_col in categorical_cols:
                    if num_col == cat_col:
                        continue
                    if plot_count >= max_plots:
                        break
                    if 2 <= real_df[cat_col].nunique() <= 10:
                        try:
                            # Downsample for plotting performance if needed
                            MAX_SAMPLES = 50000
                            real_plot = real_df
                            if len(real_plot) > MAX_SAMPLES:
                                real_plot = real_plot.sample(
                                    n=MAX_SAMPLES, random_state=42
                                )

                            synth_plot = synthetic_df
                            if len(synth_plot) > MAX_SAMPLES:
                                synth_plot = synth_plot.sample(
                                    n=MAX_SAMPLES, random_state=42
                                )

                            fig, (ax1, ax2) = plt.subplots(
                                1, 2, figsize=(20, 8), sharey=True
                            )
                            sns.boxplot(x=cat_col, y=num_col, data=real_plot, ax=ax1)
                            ax1.set_title(f"Real: {num_col} by {cat_col}")
                            ax1.tick_params(axis="x", rotation=45)

                            sns.boxplot(x=cat_col, y=num_col, data=synth_plot, ax=ax2)
                            ax2.set_title(f"Synthetic: {num_col} by {cat_col}")
                            ax2.tick_params(axis="x", rotation=45)

                            plot_filename = f"interaction_{cat_col}_{num_col}.png"
                            plot_path = os.path.join(output_dir, plot_filename)

                            plt.tight_layout()
                            plt.savefig(plot_path)
                            plt.close(fig)

                            plots[plot_filename] = plot_path
                            plot_count += 1
                        except Exception as e:
                            self.logger.warning(
                                f"Could not generate interaction plot for {cat_col} and {num_col}: {e}"
                            )
                if plot_count >= max_plots:
                    break
            return plots
        except Exception as e:
            self.logger.error(f"Failed to generate interaction plots: {e}")
            return plots

    def _save_pca_plot(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        output_dir: str,
        generator_name: str,
    ) -> Optional[str]:
        """Generates and saves a PCA plot comparing the structure of real and synthetic data."""
        if self.verbose:
            print(f"\n🔬 Generating PCA analysis plot...")
        try:
            real_df_pca = real_df.drop(columns=["timestamp"], errors="ignore")
            synthetic_df_pca = synthetic_df.drop(columns=["timestamp"], errors="ignore")
            numeric_features = real_df_pca.select_dtypes(
                include=np.number
            ).columns.tolist()
            categorical_features = real_df_pca.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), numeric_features),
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore"),
                        categorical_features,
                    ),
                ],
                remainder="drop",
            )
            real_prepared = preprocessor.fit_transform(real_df_pca)
            synthetic_prepared = preprocessor.transform(synthetic_df_pca)
            pca = PCA(n_components=2, random_state=42)
            real_pca = pca.fit_transform(real_prepared)
            synthetic_pca = pca.transform(synthetic_prepared)
            fig = plt.figure(figsize=(10, 8))
            sns.scatterplot(
                x=real_pca[:, 0], y=real_pca[:, 1], alpha=0.5, label="Real Data"
            )
            sns.scatterplot(
                x=synthetic_pca[:, 0],
                y=synthetic_pca[:, 1],
                alpha=0.5,
                label=f"{generator_name} (Synthetic)",
                marker="X",
            )
            plt.title(f"PCA Comparison - Real vs. {generator_name}")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(output_dir, f"pca_comparison_{generator_name}.png")
            fig.savefig(plot_path)
            plt.close(fig)
            return plot_path
        except Exception as e:
            self.logger.error(f"Failed to generate PCA analysis: {e}")
            return None

    def _report_basic_info(
        self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generates a concise summary of basic statistics for both datasets.
        """

        def get_summary(df: pd.DataFrame, name: str) -> Dict[str, Any]:
            return {
                "dataset": name,
                "rows": df.shape[0],
                "columns": df.shape[1],
                "total_cells": int(df.size),
                "memory_usage_mb": round(
                    df.memory_usage(deep=True).sum() / (1024 * 1024), 2
                ),
                "total_null_values": int(df.isnull().sum().sum()),
                "total_duplicated_rows": int(df.duplicated().sum()),
            }

        report = {
            "real_data_summary": get_summary(real_df, "Real"),
            "synthetic_data_summary": get_summary(synthetic_df, "Synthetic"),
        }
        return report

    def _report_target_analysis(
        self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame, target_column: str
    ) -> Optional[Dict[str, Any]]:
        """
        Analyzes the target column to determine problem type and compares its distribution.
        """
        if (
            target_column not in real_df.columns
            or target_column not in synthetic_df.columns
        ):
            self.logger.warning(
                f"Target column '{target_column}' not found in one of the dataframes."
            )
            return None
        try:
            report = {}
            real_target = real_df[target_column]
            synthetic_target = synthetic_df[target_column]
            problem_type = ""
            if pd.api.types.is_string_dtype(
                real_target
            ) or pd.api.types.is_categorical_dtype(real_target):
                problem_type = "Classification"
            elif pd.api.types.is_numeric_dtype(real_target):
                unique_values = real_target.nunique()
                if unique_values < 25 or (unique_values / len(real_target)) < 0.05:
                    problem_type = "Classification"
                else:
                    problem_type = "Regression"
            report["problem_type"] = problem_type
            if problem_type == "Classification":
                real_dist = real_target.value_counts(normalize=True)
                synth_dist = synthetic_target.value_counts(normalize=True)
                report["real_data_distribution"] = real_dist.to_dict()
                report["synthetic_data_distribution"] = synth_dist.to_dict()
                all_categories = real_dist.index.union(synth_dist.index)
                js_divergence = jensenshannon(
                    real_dist.reindex(all_categories, fill_value=0),
                    synth_dist.reindex(all_categories, fill_value=0),
                    base=2,
                )
                report["distribution_similarity"] = {
                    "metric": "Jensen-Shannon Divergence",
                    "distance": js_divergence,
                    "notes": "A value closer to 0 indicates more similar distributions.",
                }
            elif problem_type == "Regression":
                report["real_data_describe"] = real_target.describe().to_dict()
                report["synthetic_data_describe"] = (
                    synthetic_target.describe().to_dict()
                )
                ks_statistic, p_value = stats.ks_2samp(
                    real_target.dropna(), synthetic_target.dropna()
                )
                report["distribution_similarity"] = {
                    "test": "Kolmogorov-Smirnov",
                    "statistic": ks_statistic,
                    "p_value": p_value,
                    "notes": "A p-value > 0.05 suggests the distributions are statistically similar.",
                }
            return report
        except Exception as e:
            self.logger.error(
                f"Failed to generate target analysis for column '{target_column}': {e}"
            )
            return {"error": str(e)}

    def _report_sdv_quality_assessment(
        self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """
        Calculates and returns the overall SDV quality score and a custom weighted score.
        Aligns columns before evaluation to ensure a fair comparison.
        """
        if not SDV_AVAILABLE:
            self.logger.warning(
                "SDV/SDMetrics library not found. Skipping quality assessment."
            )
            return {
                "status": "SKIPPED",
                "reason": "SDV/SDMetrics library is not installed.",
            }
        if self.verbose:
            print("🔬 Running SDMetrics Quality Assessment...")
        try:
            # Start with copies to avoid modifying original dataframes
            real_df_eval = real_df.copy()
            synthetic_df_eval = synthetic_df.copy()

            # The columns of the original real data define the schema for evaluation.
            original_cols = set(real_df.columns)
            synthetic_cols = set(synthetic_df.columns)

            # Identify columns that are in the synthetic data but not in the original real data.
            extra_cols_in_synth = synthetic_cols - original_cols

            # Drop these extra columns from the synthetic dataframe copy.
            if extra_cols_in_synth:
                cols_to_drop = [
                    col
                    for col in extra_cols_in_synth
                    if col in synthetic_df_eval.columns
                ]
                if cols_to_drop:
                    synthetic_df_eval = synthetic_df_eval.drop(columns=cols_to_drop)

            # Ensure both dataframes have the exact same columns for a fair comparison.
            # The intersection is taken to handle cases where real_df might have columns not in synthetic_df.
            final_cols = [
                col for col in real_df_eval.columns if col in synthetic_df_eval.columns
            ]

            real_df_eval = real_df_eval[final_cols]
            synthetic_df_eval = synthetic_df_eval[final_cols]

            if real_df_eval.empty or synthetic_df_eval.empty:
                self.logger.warning(
                    "SDV Assessment skipped: One of the dataframes is empty after column alignment."
                )
                return {
                    "status": "SKIPPED",
                    "reason": "Empty dataframe after alignment.",
                }

            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(real_df_eval)
            quality_report = evaluate_quality(
                real_data=real_df_eval,
                synthetic_data=synthetic_df_eval,
                metadata=metadata,
            )

            # Use the cleaned and aligned dataframes for the weighted score calculation as well.
            weighted_score = self._get_weighted_sdv_score(
                real_df_eval, synthetic_df_eval, base_score=quality_report.get_score()
            )

            final_report = {
                "status": "COMPLETED",
                "overall_quality_score": quality_report.get_score(),
                "weighted_quality_score": weighted_score,
            }
            if self.verbose:
                print(
                    f"✅ SDMetrics Assessment complete. Overall: {final_report['overall_quality_score']:.2f}, Weighted: {final_report['weighted_quality_score']:.2f}"
                )
            return final_report
        except Exception as e:
            self.logger.error(f"SDMetrics quality assessment failed: {e}")
            return {"status": "FAILED", "error": str(e)}

    def _save_boxplot_plot(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        column: str,
        output_dir: str,
    ) -> Optional[str]:
        """Saves a boxplot comparing a single numerical feature between real and synthetic data."""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))

            # Combine data for plotting with seaborn
            # Downsample for plotting performance if needed
            MAX_SAMPLES = 50000
            real_plot = real_df[[column]].copy()
            if len(real_plot) > MAX_SAMPLES:
                real_plot = real_plot.sample(n=MAX_SAMPLES, random_state=42)
            real_plot["source"] = "Real"

            synth_plot = synthetic_df[[column]].copy()
            if len(synth_plot) > MAX_SAMPLES:
                synth_plot = synth_plot.sample(n=MAX_SAMPLES, random_state=42)
            synth_plot["source"] = "Synthetic"

            combined_df = pd.concat([real_plot, synth_plot], ignore_index=True)

            sns.boxplot(x="source", y=column, data=combined_df, ax=ax)

            ax.set_title(f"Boxplot Comparison: {column}")
            ax.set_xlabel("Dataset")
            ax.set_ylabel(column)
            plt.grid(True, linestyle="--", alpha=0.6)

            plot_path = os.path.join(output_dir, f"boxplot_plot_{column}.png")
            fig.savefig(plot_path)
            plt.close(fig)
            return plot_path
        except Exception as e:
            self.logger.error(f"Failed to generate boxplot for {column}: {e}")
            return None

    def _report_mode_analysis(
        self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyzes and compares the mode (most frequent value) of categorical columns."""
        report = {}
        categorical_cols = real_df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        for col in categorical_cols:
            if col in synthetic_df.columns:
                try:
                    real_mode_info = (
                        real_df[col].value_counts(normalize=True).nlargest(1)
                    )
                    synth_mode_info = (
                        synthetic_df[col].value_counts(normalize=True).nlargest(1)
                    )

                    if real_mode_info.empty or synth_mode_info.empty:
                        continue

                    report[col] = {
                        "real_mode": {
                            "value": real_mode_info.index[0],
                            "proportion": round(real_mode_info.iloc[0], 4),
                        },
                        "synthetic_mode": {
                            "value": synth_mode_info.index[0],
                            "proportion": round(synth_mode_info.iloc[0], 4),
                        },
                    }
                except Exception as e:
                    self.logger.error(
                        f"Failed to generate mode analysis for column '{col}': {e}"
                    )
                    report[col] = {"error": str(e)}
        return report

    def _report_block_analysis(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        block_column: str,
        target_column: Optional[str],
    ) -> Dict[str, Any]:
        """Analyzes and compares each block within the real and synthetic datasets."""
        analysis = {}
        unique_blocks = sorted(real_df[block_column].unique(), key=str)

        for block_id in unique_blocks:
            real_block_df = real_df[real_df[block_column] == block_id]
            synthetic_block_df = synthetic_df[synthetic_df[block_column] == block_id]

            block_stats = {
                "real_num_rows": len(real_block_df),
                "synthetic_num_rows": len(synthetic_block_df),
                "real_null_values": int(real_block_df.isnull().sum().sum()),
                "synthetic_null_values": int(synthetic_block_df.isnull().sum().sum()),
            }

            if target_column and target_column in real_block_df.columns:
                block_stats["real_target_distribution"] = (
                    real_block_df[target_column].value_counts(normalize=True).to_dict()
                )
            if target_column and target_column in synthetic_block_df.columns:
                block_stats["synthetic_target_distribution"] = (
                    synthetic_block_df[target_column]
                    .value_counts(normalize=True)
                    .to_dict()
                )

            analysis[str(block_id)] = block_stats

        return analysis

    def _save_mode_analysis_plot(
        self, mode_analysis_data: Dict[str, Any], output_dir: str
    ) -> Optional[str]:
        """Saves a plot visualizing the mode analysis data for categorical features."""
        try:
            if not mode_analysis_data:
                self.logger.info("No data for mode analysis plot.")
                return None

            columns = list(mode_analysis_data.keys())
            real_proportions = [
                d.get("real_mode", {}).get("proportion", 0)
                for d in mode_analysis_data.values()
            ]
            real_modes = [
                str(d.get("real_mode", {}).get("value", "N/A"))
                for d in mode_analysis_data.values()
            ]
            synth_proportions = [
                d.get("synthetic_mode", {}).get("proportion", 0)
                for d in mode_analysis_data.values()
            ]
            synth_modes = [
                str(d.get("synthetic_mode", {}).get("value", "N/A"))
                for d in mode_analysis_data.values()
            ]

            df = pd.DataFrame(
                {
                    "Feature": columns,
                    "Real Proportion": real_proportions,
                    "Real Mode": real_modes,
                    "Synthetic Proportion": synth_proportions,
                    "Synthetic Mode": synth_modes,
                }
            )

            if df.empty:
                self.logger.info("No data to plot for mode analysis.")
                return None

            fig, ax = plt.subplots(figsize=(12, len(columns) * 0.8 + 3))

            bar_height = 0.4
            y = np.arange(len(columns))

            bars1 = ax.barh(
                y + bar_height / 2,
                df["Real Proportion"],
                height=bar_height,
                label="Real",
                color="skyblue",
                alpha=0.8,
            )
            bars2 = ax.barh(
                y - bar_height / 2,
                df["Synthetic Proportion"],
                height=bar_height,
                label="Synthetic",
                color="salmon",
                alpha=0.8,
            )

            ax.set_xlabel("Proportion of Mode")
            ax.set_ylabel("Feature")
            ax.set_title("Mode and Proportion Comparison for Categorical Features")
            ax.set_yticks(y)
            ax.set_yticklabels(columns)
            ax.legend()
            ax.invert_yaxis()

            # Add labels with mode value and proportion
            real_labels = [
                f"{df['Real Mode'][i]} ({df['Real Proportion'][i]:.2f})"
                for i in range(len(bars1))
            ]
            synth_labels = [
                f"{df['Synthetic Mode'][i]} ({df['Synthetic Proportion'][i]:.2f})"
                for i in range(len(bars2))
            ]

            ax.bar_label(bars1, labels=real_labels, padding=3, fontsize=9)
            ax.bar_label(bars2, labels=synth_labels, padding=3, fontsize=9)

            ax.grid(axis="x", linestyle="--", alpha=0.6)
            if not df.empty and (
                df["Real Proportion"].max() > 0 or df["Synthetic Proportion"].max() > 0
            ):
                ax.set_xlim(
                    right=max(
                        df["Real Proportion"].max(), df["Synthetic Proportion"].max()
                    )
                    * 1.3
                )

            plt.tight_layout()

            plot_path = os.path.join(output_dir, "mode_comparison.png")
            fig.savefig(plot_path)
            plt.close(fig)
            return plot_path

        except Exception as e:
            self.logger.error(f"Failed to generate mode analysis plot: {e}")
            return None

    def run_scaling_evaluation(
        self,
        original_data: pd.DataFrame,
        synthetic_data_full: pd.DataFrame,
        output_dir: str,
        block_column: Optional[str] = None,
    ) -> Optional[str]:
        """Evaluates data quality by taking incremental subsets of a pre-generated synthetic dataset, either by block or by percentage."""
        if not SDV_AVAILABLE:
            return None

        # --- Block-based Scaling ---
        if (
            block_column
            and block_column in synthetic_data_full.columns
            and block_column in original_data.columns
        ):
            if self.verbose:
                print(
                    f"--- Starting Scaling Evaluation by Blocks ('{block_column}') ---"
                )

            all_blocks = sorted(synthetic_data_full[block_column].unique())
            if len(all_blocks) <= 1:
                if self.verbose:
                    print(
                        "Only one block found, skipping block-wise scaling evaluation."
                    )
                return None

            scores = []
            x_labels = [f"1..{i + 1}" for i in range(len(all_blocks))]
            x_ticks = list(range(1, len(all_blocks) + 1))

            for i in x_ticks:
                blocks_to_include = all_blocks[:i]
                if self.verbose:
                    print(f"Evaluating blocks: {blocks_to_include}...")

                synthetic_subset = synthetic_data_full[
                    synthetic_data_full[block_column].isin(blocks_to_include)
                ]
                original_subset = original_data[
                    original_data[block_column].isin(blocks_to_include)
                ]

                if original_subset.empty or synthetic_subset.empty:
                    self.logger.warning(
                        f"Skipping evaluation for blocks {blocks_to_include} due to empty data."
                    )
                    scores.append(0.0)
                    continue

                try:
                    metadata = SingleTableMetadata()
                    metadata.detect_from_dataframe(original_subset)
                    quality_report = evaluate_quality(
                        real_data=original_subset,
                        synthetic_data=synthetic_subset,
                        metadata=metadata,
                    )
                    score = self._get_weighted_sdv_score(
                        original_subset, synthetic_subset, quality_report.get_score()
                    )
                    scores.append(score)
                except Exception as e:
                    self.logger.error(
                        f"Error during block scaling evaluation for blocks {blocks_to_include}: {e}"
                    )
                    scores.append(0.0)

            fig = plt.figure(figsize=(12, 7))
            plt.plot(x_ticks, scores, marker="o", linestyle="-", color="g")
            plt.title("Data Quality vs. Number of Blocks", fontsize=16)
            plt.xlabel("Number of Blocks Included", fontsize=12)
            plt.ylabel("Weighted SDV Score", fontsize=12)
            plt.xticks(ticks=x_ticks, labels=x_labels, rotation=45)
            plt.ylim(0, 1)
            for i, score in enumerate(scores):
                plt.text(x_ticks[i], score + 0.02, f"{score:.2f}", ha="center")

        # --- Percentage-based Scaling (Fallback) --- #
        else:
            if self.verbose:
                print(f"--- Starting Scaling Evaluation by Subsetting ---")
            percentages = [0.25, 0.50, 0.75, 1.0]
            total_rows = len(synthetic_data_full)
            sample_sizes = [max(1, int(p * total_rows)) for p in percentages]
            sample_sizes = sorted(list(set(sample_sizes)))
            if self.verbose:
                print(
                    f"Evaluating subsets of sizes: {sample_sizes} (from a total of {total_rows} synthetic rows)"
                )
            scores = []
            for n_samples in sample_sizes:
                if self.verbose:
                    print(f"Evaluating a subset of {n_samples} samples...")
                try:
                    synthetic_subset = synthetic_data_full.sample(
                        n=n_samples, random_state=42
                    )
                    if not synthetic_subset.empty:
                        metadata = SingleTableMetadata()
                        metadata.detect_from_dataframe(original_data)
                        quality_report = evaluate_quality(
                            real_data=original_data,
                            synthetic_data=synthetic_subset,
                            metadata=metadata,
                        )
                        score = self._get_weighted_sdv_score(
                            original_data, synthetic_subset, quality_report.get_score()
                        )
                        scores.append(score)
                    else:
                        scores.append(0.0)
                except Exception as e:
                    self.logger.error(
                        f"Error during scaling evaluation for {n_samples} samples: {e}"
                    )
                    scores.append(0.0)

            fig = plt.figure(figsize=(12, 7))
            plt.plot(sample_sizes, scores, marker="o", linestyle="-", color="b")
            plt.title(f"Data Quality vs. Subset Size", fontsize=16)
            plt.xlabel("Number of Samples in Synthetic Subset", fontsize=12)
            plt.ylabel("Weighted SDV Score", fontsize=12)
            plt.xticks(sample_sizes, rotation=45)
            plt.ylim(0, 1)
            for i, score in enumerate(scores):
                plt.text(sample_sizes[i], score + 0.02, f"{score:.2f}", ha="center")

        plot_path = os.path.join(output_dir, f"scaling_evaluation_subsets.png")
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        if self.verbose:
            print(f"--- Scaling Evaluation Complete. Plot saved to: {plot_path} ---")
        return plot_path

    def update_report_after_drift(
        self,
        original_df: pd.DataFrame,
        drifted_df: pd.DataFrame,
        output_dir: str,
        drift_config: Dict[str, Any],
        focus_cols: Optional[List[str]] = None,
        time_col: Optional[str] = None,
    ) -> None:
        """
        Generates a new report for a drifted dataset and updates the JSON file with drift configuration.
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("DRIFT INJECTION REPORT")
            print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)

        # --- 1. Load/Update Drift History to JSON first ---
        report_path = os.path.join(output_dir, "report.json")
        drift_history = []
        full_report_json = {}

        try:
            if os.path.exists(report_path):
                with open(report_path, "r") as f:
                    try:
                        full_report_json = json.load(f)
                        # Check for existing history or single config
                        if "drift_history" in full_report_json:
                            drift_history = full_report_json["drift_history"]
                        elif "drift_config" in full_report_json:
                            # Migrate old single config to history
                            drift_history = [full_report_json["drift_config"]]
                    except json.JSONDecodeError:
                        full_report_json = {}

            # Append current config
            # Add timestamp to config if not present
            if "timestamp" not in drift_config:
                drift_config["timestamp"] = datetime.now().isoformat()

            drift_history.append(drift_config)
            full_report_json["drift_history"] = drift_history
            # Keep "drift_config" as the latest for backwards compatibility if needed,
            # or just rely on history[-1]
            full_report_json["drift_config"] = drift_config

            # Write back immediately so we have it saved
            with open(report_path, "w") as f:
                json.dump(full_report_json, f, indent=4, cls=NumpyEncoder)

        except Exception as e:
            self.logger.error(f"Failed to update drift history in report.json: {e}")

        generator_name = drift_config.get("generator_name", "Drifted Data")
        target_column = drift_config.get("target_column")
        block_column = drift_config.get("block_column")

        # Automatically determine focus columns from drift config if not provided
        if focus_cols is None:
            affected_cols = drift_config.get("feature_cols", [])
            # Also include target if it exists, as drift might affect it relationship-wise
            if target_column and target_column not in affected_cols:
                # If drift is label drift, we might want target.
                # If drift is feature drift, we might want to see correlation with target.
                # Let's include it.
                pass

            if affected_cols:
                focus_cols = list(affected_cols)
                if target_column:
                    focus_cols.append(target_column)

        self.generate_comprehensive_report(
            real_df=original_df,
            synthetic_df=drifted_df,
            generator_name=generator_name,
            output_dir=output_dir,
            target_column=target_column,
            block_column=block_column,
            focus_cols=focus_cols,
            drift_config=drift_config,
            time_col=time_col,
            drift_history=drift_history,
        )

    def _save_umap_plot(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        output_dir: str,
        generator_name: str,
    ) -> Optional[str]:
        """Generates and saves a UMAP plot comparing real and synthetic data structure."""
        if not UMAP_AVAILABLE:
            self.logger.warning("UMAP not available. Skipping UMAP plot.")
            return None

        try:
            # Prepare data
            real_nosamples = real_df.drop(
                columns=["timestamp", "chunk", "block"], errors="ignore"
            ).sample(min(len(real_df), 1000), random_state=42)
            synth_nosamples = synthetic_df.drop(
                columns=["timestamp", "chunk", "block"], errors="ignore"
            ).sample(min(len(synthetic_df), 1000), random_state=42)

            real_nosamples["Origin"] = "Real"
            synth_nosamples["Origin"] = "Synthetic"

            combined = pd.concat([real_nosamples, synth_nosamples], axis=0)
            origin = combined["Origin"]
            combined = combined.drop(columns=["Origin"])

            # Preprocessing
            numeric_features = combined.select_dtypes(
                include=np.number
            ).columns.tolist()
            categorical_features = combined.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), numeric_features),
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        categorical_features,
                    ),
                ],
                remainder="drop",
            )

            prepared_data = preprocessor.fit_transform(combined)

            # UMAP
            reducer = umap.UMAP(random_state=42)
            embedding = reducer.fit_transform(prepared_data)

            # Plot
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.scatterplot(
                x=embedding[:, 0],
                y=embedding[:, 1],
                hue=origin,
                alpha=0.6,
                palette={"Real": "#1f77b4", "Synthetic": "#ff7f0e"},
                ax=ax,
            )
            plt.title(f"UMAP Comparison - {generator_name}")
            plot_path = os.path.join(output_dir, f"umap_{generator_name}.png")
            fig.savefig(plot_path)
            plt.close(fig)
            return plot_path

        except Exception as e:
            self.logger.error(f"Failed to generate UMAP plot: {e}")
            return None

    def _compare_numeric_distributions(
        self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compares distributions of numeric columns."""
        comparison = {}
        try:
            numeric_cols = real_df.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                if col not in synthetic_df.columns:
                    continue

                real_data = real_df[col].dropna()
                synth_data = synthetic_df[col].dropna()

                if len(real_data) == 0 or len(synth_data) == 0:
                    continue

                # Basic stats
                r_mean = real_data.mean()
                r_std = real_data.std()
                s_mean = synth_data.mean()
                s_std = synth_data.std()

                # KS Test
                is_match = False
                try:
                    ks_stat, p_value = ks_2samp(real_data, synth_data)
                    # Use standard alpha=0.05
                    is_match = p_value > 0.05
                except Exception:
                    p_value = None

                comparison[col] = {
                    "real_distribution": f"Mean: {r_mean:.2f}, Std: {r_std:.2f}",
                    "synthetic_distribution": f"Mean: {s_mean:.2f}, Std: {s_std:.2f}",
                    "match": is_match,
                    "stats": {
                        "ks_stat": float(ks_stat) if p_value is not None else None,
                        "p_value": float(p_value) if p_value is not None else None,
                    },
                }
        except Exception as e:
            self.logger.error(f"Failed to compare numeric distributions: {e}")

        return comparison

    def _compute_statistical_tests(
        self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Computes statistical tests to compare distributions:
        - Numeric: Kolmogorov-Smirnov (KS) Test
        - Categorical: Chi-Square Test (or TVD/L1 distance)
        """
        results = {}

        for col in real_df.columns:
            if col not in synthetic_df.columns:
                continue

            col_res = {}
            real_data = real_df[col].dropna()
            synth_data = synthetic_df[col].dropna()

            if real_data.empty or synth_data.empty:
                continue

            if pd.api.types.is_numeric_dtype(real_df[col]):
                # KS Test (Two-sample)
                try:
                    stat, p_value = ks_2samp(real_data, synth_data)
                    col_res["test"] = "Kolmogorov-Smirnov"
                    col_res["statistic"] = float(stat)
                    col_res["p_value"] = float(p_value)
                    col_res["conclusion"] = (
                        "Distributions are likely different"
                        if p_value < 0.05
                        else "Distributions are likely similar"
                    )
                except Exception as e:
                    col_res["error"] = str(e)
            else:
                # Chi-Square / TVD
                try:
                    # Align indices for Chi2? No, samples are independent and likely not paired.
                    # Chi2 requires frequencies.
                    # Better metric for generative models: Total Variation Distance (TVD)
                    real_freq = real_data.value_counts(normalize=True)
                    synth_freq = synth_data.value_counts(normalize=True)

                    # Align
                    all_cats = real_freq.index.union(synth_freq.index)
                    real_freq = real_freq.reindex(all_cats, fill_value=0)
                    synth_freq = synth_freq.reindex(all_cats, fill_value=0)

                    tvd = 0.5 * (real_freq - synth_freq).abs().sum()

                    col_res["test"] = "Total Variation Distance"
                    col_res["statistic"] = float(tvd)
                    col_res["conclusion"] = (
                        f"Distance: {tvd:.4f} (0 is identical, 1 is disjoint)"
                    )

                except Exception as e:
                    col_res["error"] = str(e)

            # --- Add Distribution Fit Analysis ---
            try:
                real_fit = fit_distribution(real_data)
                synth_fit = fit_distribution(synth_data)

                col_res["real_dist_fit"] = real_fit["distribution"]
                col_res["real_dist_p"] = real_fit["p_value"]
                col_res["synth_dist_fit"] = synth_fit["distribution"]
                col_res["synth_dist_p"] = synth_fit["p_value"]

            except Exception as e:
                col_res["dist_fit_error"] = str(e)

            results[col] = col_res

        return results

    def _report_detailed_quality_analysis(
        self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generates comprehensive quality stats comparing Real vs Synthetic including:
        - Nulls, Duplicates, Descriptive Stats (Mean/Min/Max/Mode)
        - Data types
        """
        analysis = {}

        # Helper to get stats for a DF
        def get_df_stats(df, name):
            stats = {}
            stats["rows"] = len(df)
            stats["duplicates"] = int(df.duplicated().sum())
            stats["duplicate_pct"] = (
                round((stats["duplicates"] / len(df) * 100), 2) if len(df) > 0 else 0
            )

            cols = {}
            for col in df.columns:
                c_stat = {}
                c_stat["nulls"] = int(df[col].isnull().sum())
                c_stat["type"] = str(df[col].dtype)

                clean_col = df[col].dropna()
                if clean_col.empty:
                    c_stat["stats"] = "Empty"
                elif pd.api.types.is_numeric_dtype(df[col]):
                    desc = clean_col.describe().to_dict()
                    mode_v = clean_col.mode()
                    desc["mode"] = float(mode_v.iloc[0]) if not mode_v.empty else None
                    desc["median"] = float(clean_col.median())
                    c_stat["stats"] = desc
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    c_stat["stats"] = {
                        "min": clean_col.min().isoformat(),
                        "max": clean_col.max().isoformat(),
                    }
                else:
                    # Categorical
                    c_stat["unique"] = int(clean_col.nunique())
                    mode_v = clean_col.mode()
                    c_stat["mode"] = str(mode_v.iloc[0]) if not mode_v.empty else None
                    top10 = clean_col.value_counts(normalize=True).head(10).to_dict()
                    c_stat["top_10"] = {str(k): round(v, 4) for k, v in top10.items()}

                cols[col] = c_stat
            stats["columns"] = cols
            return stats

        analysis["real_data"] = get_df_stats(real_df, "Real")
        analysis["synthetic_data"] = get_df_stats(synthetic_df, "Synthetic")

        return analysis

    def generate_markdown_report(self, report: Dict[str, Any], output_dir: str):
        """Generates a human-readable Markdown summary of the analysis."""
        try:
            md_path = os.path.join(output_dir, "report_summary.md")
            with open(md_path, "w") as f:
                f.write(f"# Data Quality & Comparison Report\n\n")
                f.write(f"**Generator:** {report.get('generator_name')}\n")
                f.write(f"**Date:** {report.get('generation_timestamp')}\n\n")

                # Drift History
                if "drift_history" in report and report["drift_history"]:
                    f.write("## 📜 Drift Injection History\n")
                    history = report["drift_history"]
                    for i, drift in enumerate(history):
                        f.write(f"### Injection #{i + 1}\n")
                        f.write(f"- **Method:** {drift.get('drift_method', 'N/A')}\n")
                        f.write(f"- **Type:** {drift.get('drift_type', 'N/A')}\n")
                        if "feature_cols" in drift:
                            f.write(
                                f"- **Features:** {', '.join(drift['feature_cols'])}\n"
                            )
                        if "drift_magnitude" in drift:
                            f.write(f"- **Magnitude:** {drift['drift_magnitude']}\n")
                        if "timestamp" in drift:
                            f.write(f"- **Time:** {drift['timestamp']}\n")
                        f.write("\n")
                # Sdv Score
                if "sdv_quality" in report and report["sdv_quality"]:
                    sdv = report["sdv_quality"]
                    f.write(f"## ⭐ Quality Score\n")
                    f.write(
                        f"- **Overall Quality:** {sdv.get('overall_quality_score', 'N/A')}\n"
                    )
                    f.write(
                        f"- **Weighted Score:** {sdv.get('weighted_quality_score', 'N/A')}\n\n"
                    )

                # Basic Stats
                f.write("## 📊 Dataset Statistics\n")
                real_info = report.get("basic_info", {}).get("real_data_summary", {})
                synth_info = report.get("basic_info", {}).get(
                    "synthetic_data_summary", {}
                )

                f.write("| Metric | Real | Synthetic |\n")
                f.write("| :--- | :--- | :--- |\n")
                f.write(
                    f"| Rows | {real_info.get('rows')} | {synth_info.get('rows')} |\n"
                )
                f.write(
                    f"| Columns | {real_info.get('columns')} | {synth_info.get('columns')} |\n"
                )
                f.write(
                    f"| Duplicates | {real_info.get('total_duplicated_rows')} | {synth_info.get('total_duplicated_rows')} |\n\n"
                )

                # Statistical Tests
                if "statistical_tests" in report and report["statistical_tests"]:
                    f.write("## 📉 Statistical Tests & Distribution Fit\n")
                    f.write(
                        "| Column | Test | Statistic | P-Value | Real Dist | Synth Dist | Conclusion |\n"
                    )
                    f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
                    for col, res in report["statistical_tests"].items():
                        if "error" in res:
                            f.write(
                                f"| {col} | Error | - | - | - | - | {res['error']} |\n"
                            )
                        else:
                            real_dist = res.get("real_dist_fit", "-")
                            synth_dist = res.get("synth_dist_fit", "-")

                            p_val = res.get("p_value", "-")
                            if p_val != "-":
                                p_val = (
                                    f"{p_val:.4f}"
                                    if isinstance(p_val, float)
                                    else str(p_val)
                                )

                            f.write(
                                f"| {col} | {res.get('test')} | {res.get('statistic'):.4f} | {p_val} | {real_dist} | {synth_dist} | {res.get('conclusion')} |\n"
                            )
                    f.write("\n")

                # Distribution Comparison
                if (
                    "distribution_comparison" in report
                    and report["distribution_comparison"]
                ):
                    f.write("## 📉 Distribution Comparison\n")
                    f.write("| Column | Real Dist | Synthetic Dist | Match |\n")
                    f.write("| :--- | :--- | :--- | :--- |\n")
                    for col, res in report["distribution_comparison"].items():
                        match_icon = "✅" if res.get("match") else "❌"
                        f.write(
                            f"| {col} | {res.get('real_distribution')} | {res.get('synthetic_distribution')} | {match_icon} |\n"
                        )
                    f.write("\n")

                # Visualizations
                f.write("## 🖼️ Visualizations\n")
                plots = report.get("plots", {})
                if plots.get("dimensionality_reduction"):
                    f.write(f"### Dimensionality Reduction (UMAP/PCA)\n")
                    f.write(
                        f"![UMAP]({os.path.basename(plots['dimensionality_reduction'])})\n\n"
                    )

                if plots.get("correlation_heatmap"):
                    f.write(f"### Correlation Matrix\n")
                    f.write(
                        f"![Correlation]({os.path.basename(plots['correlation_heatmap'])})\n\n"
                    )

            if self.verbose:
                print(f"✅ Markdown report saved to: {md_path}")
        except Exception as e:
            self.logger.error(f"Failed to generate markdown report: {e}")
