"""
Static, File-Based Synthetic Data Reporter

This module provides the SyntheticReporter class, designed to generate a detailed, static report
for a single synthetic dataset. It analyzes the dataset's properties and saves all artifacts,
including a JSON report and various plots, directly to the disk.

Key Features:
- **Comprehensive Analysis**: Generates a wide array of analyses, including basic info, target distribution, and mode analysis for categorical features.
- **Rich Visualizations**: Creates and saves multiple plots to visualize the dataset's characteristics:
  - PCA plot to visualize overall data structure.
  - Correlation heatmap for numeric features.
  - Distribution plots (KDEs or bar plots) for individual features.
  - Boxplots for numeric feature summaries.
  - Instance balance plots for categorical features.
  - Time evolution plots to track feature changes over time or index.
- **Block-Level Analysis**: If a `block_column` is provided, it generates specific reports and plots for each individual block.
- **Drift-Aware Reporting**: Can highlight drift-injected areas in plots when provided with drift configuration.
- **Static Output**: All outputs are saved to a specified directory, making it ideal for automated pipelines and environments without dynamic UIs.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch
except ImportError:
    plt = None
    sns = None
    Patch = None
import warnings
import logging
from datetime import datetime
import os
import json
from scipy import stats
from calmops.utils.distribution_fitter import fit_distribution

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

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


class SyntheticReporter:
    """
    Generates a static, file-based report analyzing the properties of a synthetic dataset.
    """

    def __init__(self, verbose: bool = True):
        """
        Initializes the SyntheticReporter.

        Args:
            verbose (bool): If True, prints progress messages to the console.
        """
        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)

        if plt:
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

    def generate_report(
        self,
        synthetic_df: pd.DataFrame,
        generator_name: str,
        output_dir: str,
        target_column: Optional[str] = None,
        block_column: Optional[str] = None,
        focus_cols: Optional[List[str]] = None,
        drift_config: Optional[Dict[str, Any]] = None,
        time_col: Optional[str] = None,
    ) -> None:
        """
        Generates a comprehensive file-based report for the synthetic dataset.

        Args:
            synthetic_df (pd.DataFrame): The generated, synthetic dataset.
            generator_name (str): Name of the generator used, for labeling plots.
            output_dir (str): Directory to save the report and all plots.
            target_column (Optional[str]): Name of the target variable column.
            block_column (Optional[str]): Name of the column defining data blocks.
            focus_cols (Optional[List[str]]): A list of specific columns to focus on for plotting.
            drift_config (Optional[Dict[str, Any]]): Configuration of any injected drift, used for highlighting plots.
            time_col (Optional[str]): Name of the timestamp column for time-series plots.
        """
        if self.verbose:
            print("=" * 80)
            print(f"COMPREHENSIVE SYNTHETIC DATA GENERATION REPORT")
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
            "basic_info": self._report_basic_info(synthetic_df),
            "target_analysis": self._report_target_analysis(synthetic_df, target_column)
            if target_column
            else None,
            "mode_analysis": self._report_mode_analysis(synthetic_df),
            "numeric_analysis": self._report_numeric_analysis(synthetic_df),
            "quality_analysis": self._report_detailed_quality_analysis(synthetic_df),
            "block_analysis": None,
            "plots": {},
        }

        cols_to_plot = focus_cols if focus_cols else synthetic_df.columns.tolist()

        final_time_col = (
            time_col
            if time_col and time_col in synthetic_df.columns
            else "timestamp"
            if "timestamp" in synthetic_df.columns
            else None
        )

        # --- Generate and Save Plots for the entire dataset --- #
        # --- Generate and Save Plots for the entire dataset --- #
        plots = {}
        if plt:
            plots["pca"] = self._save_pca_plot(synthetic_df, output_dir, generator_name)
            plots["correlation_heatmap"] = self._save_correlation_heatmap(
                synthetic_df, output_dir, focus_cols=focus_cols
            )
            plots["interaction_plots"] = self._save_interaction_plots(
                synthetic_df, output_dir, focus_cols=focus_cols
            )
            plots["mode_analysis"] = self._save_mode_analysis_plot(
                report["mode_analysis"], output_dir
            )
        else:
            self.logger.warning("Matplotlib not installed. Skipping all plots.")

        dist_plots = {}
        box_plots = {}
        balance_plots = {}

        if plt:
            for col in cols_to_plot:
                if col in synthetic_df.columns:
                    if col == final_time_col:
                        continue
                    dist_plots[col] = self._save_distribution_plot(
                        synthetic_df, col, output_dir
                    )

                is_categorical = (
                    pd.api.types.is_string_dtype(synthetic_df[col])
                    or pd.api.types.is_categorical_dtype(synthetic_df[col])
                    or (
                        pd.api.types.is_numeric_dtype(synthetic_df[col])
                        and synthetic_df[col].nunique() < 25
                    )
                )

                if is_categorical:
                    balance_plots[col] = self._save_instances_balance_plot(
                        synthetic_df, col, output_dir
                    )

                if pd.api.types.is_numeric_dtype(synthetic_df[col]):
                    box_plots[col] = self._save_boxplot_plot(
                        synthetic_df, col, output_dir
                    )

        plots["distribution_plots"] = dist_plots
        plots["box_plots"] = box_plots
        plots["instances_balance_plots"] = balance_plots

        if plt:
            plots["time_evolution_plots"] = self._save_time_evolution_plots(
                synthetic_df,
                output_dir,
                time_col=final_time_col,
                focus_cols=focus_cols,
                drift_config=drift_config,
            )
        else:
            plots["time_evolution_plots"] = {}

        report["plots"] = plots

        # --- Per-Block Analysis --- #
        if block_column and block_column in synthetic_df.columns:
            report["block_analysis"] = self._report_block_analysis(
                synthetic_df=synthetic_df,
                block_column=block_column,
                target_column=target_column,
            )

            if self.verbose:
                print(f"\n🔬 Generating per-block analysis plots...")

            block_plots_report = {}
            unique_blocks = sorted(synthetic_df[block_column].unique(), key=str)

            for block_id in unique_blocks:
                block_output_dir = os.path.join(output_dir, f"block_{block_id}_plots")
                os.makedirs(block_output_dir, exist_ok=True)

                block_df = synthetic_df[synthetic_df[block_column] == block_id]

                if block_df.empty:
                    self.logger.warning(
                        f"Skipping plots for block {block_id} due to empty data."
                    )
                    continue

                block_plots_report[str(block_id)] = {
                    "distribution_plots": {},
                    "instances_balance_plots": {},
                }

                if plt:
                    for col in block_df.columns:
                        if col == block_column:
                            continue

                        plot_path = self._save_distribution_plot(
                            block_df, col, block_output_dir
                        )
                        if plot_path:
                            block_plots_report[str(block_id)]["distribution_plots"][
                                col
                            ] = plot_path

                        is_categorical = (
                            pd.api.types.is_string_dtype(block_df[col])
                            or pd.api.types.is_categorical_dtype(block_df[col])
                            or (
                                pd.api.types.is_numeric_dtype(block_df[col])
                                and block_df[col].nunique() < 25
                            )
                        )

                        if is_categorical:
                            balance_plot_path = self._save_instances_balance_plot(
                                block_df, col, block_output_dir
                            )
                            if balance_plot_path:
                                block_plots_report[str(block_id)][
                                    "instances_balance_plots"
                                ][col] = balance_plot_path

                    if self.verbose:
                        print(
                            f"  -> Plots for block '{block_id}' saved to: {block_output_dir}"
                        )
                else:
                    if self.verbose:
                        print(
                            f"  -> Plots skipped for block '{block_id}' (Matplotlib missing)."
                        )

            report["plots"]["per_block_plots"] = block_plots_report

        # Call the markdown generator instead of saving JSON
        try:
            self.generate_markdown_report(report, output_dir)
            if self.verbose:
                print(
                    f"\n✅ Markdown report saved to: {os.path.join(output_dir, 'report_summary.md')}"
                )
        except Exception as e:
            self.logger.error(f"Failed to save Markdown report: {e}")

    def _report_block_analysis(
        self,
        synthetic_df: pd.DataFrame,
        block_column: str,
        target_column: Optional[str],
    ) -> Dict[str, Any]:
        """Analyzes and returns statistics for each block within the synthetic dataset."""
        analysis = {}
        unique_blocks = sorted(synthetic_df[block_column].unique(), key=str)

        for block_id in unique_blocks:
            block_df = synthetic_df[synthetic_df[block_column] == block_id]
            block_stats = {
                "num_rows": len(block_df),
                "null_values": int(block_df.isnull().sum().sum()),
            }
            if target_column and target_column in block_df.columns:
                block_stats["target_distribution"] = (
                    block_df[target_column].value_counts(normalize=True).to_dict()
                )

            analysis[str(block_id)] = block_stats

        return analysis

    def _save_instances_balance_plot(
        self, df: pd.DataFrame, column: str, output_dir: str
    ) -> Optional[str]:
        """Saves a bar plot showing the distribution of instances for a categorical column."""
        try:
            fig, ax = plt.subplots(figsize=(12, 7))
            counts = df[column].value_counts(normalize=True)

            df_counts = pd.DataFrame({"Proportion": counts}).fillna(0).sort_index()

            if len(df_counts) > 20:
                self.logger.warning(
                    f"High cardinality in column '{column}' ({len(df_counts)} unique values). Plot may be cluttered."
                )
                top20 = df_counts.nlargest(20, "Proportion").index
                df_counts = df_counts.loc[top20]

            df_counts.plot(kind="bar", ax=ax, alpha=0.7, width=0.8, legend=None)

            ax.set_title(f"Instance Balance: {column}", fontsize=14)
            ax.set_ylabel("Proportion of Instances", fontsize=10)
            ax.set_xlabel(column, fontsize=10)
            ax.tick_params(axis="x", rotation=45, labelsize=9)
            ax.grid(axis="y", linestyle="--", alpha=0.7)

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
        df: pd.DataFrame,
        output_dir: str,
        time_col: Optional[str] = None,
        focus_cols: Optional[List[str]] = None,
        drift_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Saves evolution plots for numeric and categorical features over time or index."""
        plots = {}
        x_axis_label = time_col if time_col else "Instance Index"
        moving_avg_window = 7

        features_to_plot = focus_cols if focus_cols else df.columns.tolist()

        # Separate columns into numeric and categorical
        numeric_cols = []
        categorical_cols = []
        for col in features_to_plot:
            if col == time_col or col == "block":
                continue
            # Treat discrete numerics as categorical for this plot
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 25:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

        # --- Plot Numeric Columns ---
        for col in numeric_cols:
            try:
                fig, ax = plt.subplots(figsize=(15, 7))
                color = "#1f77b4"

                if time_col:
                    agg = df.groupby(time_col)[col].agg(["mean", "std"]).fillna(0)
                    mean_ma = (
                        agg["mean"]
                        .rolling(window=moving_avg_window, min_periods=1, center=True)
                        .mean()
                    )
                    mean_ma.plot(
                        ax=ax,
                        label=f"Mean ({moving_avg_window}-MA)",
                        style="-",
                        linewidth=2.5,
                        color=color,
                    )

                else:
                    ma = (
                        df[col]
                        .rolling(window=moving_avg_window, min_periods=1, center=True)
                        .mean()
                    )
                    ma.reset_index(drop=True).plot(
                        ax=ax,
                        label=f"({moving_avg_window}-MA)",
                        style="-",
                        linewidth=2,
                        color=color,
                    )

                if drift_config:
                    pass

                ax.set_title(f"Evolution of {col}", fontsize=18, weight="bold")
                ax.set_xlabel(x_axis_label, fontsize=14)
                ax.set_ylabel(col, fontsize=14)
                ax.grid(
                    True, which="major", linestyle="--", linewidth="0.5", color="grey"
                )
                ax.tick_params(axis="x", rotation=45, labelsize=12)
                ax.tick_params(axis="y", labelsize=12)
                fig.autofmt_xdate()
                ax.legend(loc="best")
                plt.tight_layout()
                plot_path = os.path.join(output_dir, f"evolution_{col}.png")
                fig.savefig(plot_path)
                plt.close(fig)
                plots[col] = plot_path
            except Exception as e:
                self.logger.error(f"Failed to generate evolution plot for {col}: {e}")

        # --- Plot Categorical Columns ---
        if time_col:
            for col in categorical_cols:
                plots[col] = self._save_categorical_evolution_plot(
                    df, time_col, col, output_dir
                )

        return plots

    def _save_categorical_evolution_plot(
        self, df: pd.DataFrame, time_col: str, category_col: str, output_dir: str
    ) -> Optional[str]:
        """Saves a plot showing the evolution of a categorical feature's proportions over time."""
        try:
            HIGH_CARDINALITY_THRESHOLD = 15
            df_plot = df.copy()

            n_unique = df_plot[category_col].nunique()
            if n_unique > HIGH_CARDINALITY_THRESHOLD:
                self.logger.info(
                    f"High cardinality ({n_unique}) for '{category_col}', grouping less frequent categories."
                )
                top_categories = (
                    df_plot[category_col]
                    .value_counts()
                    .nlargest(HIGH_CARDINALITY_THRESHOLD - 1)
                    .index.tolist()
                )
                df_plot[category_col] = df_plot[category_col].apply(
                    lambda x: x if x in top_categories else "Other"
                )

            pivot = df_plot.pivot_table(
                index=time_col, columns=category_col, aggfunc="size", fill_value=0
            )
            pivot = pivot.div(pivot.sum(axis=1), axis=0)

            fig, ax = plt.subplots(
                figsize=(16, 8)
            )  # Increased figure size for better readability

            pivot.plot(kind="line", stacked=False, ax=ax, alpha=0.7)
            ax.set_title(f"Evolution of {category_col}", fontsize=18, weight="bold")
            ax.set_ylabel("Proportion", fontsize=14)
            ax.legend(
                title=category_col,
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                fontsize=12,
            )
            ax.grid(True, which="major", linestyle="--", linewidth="0.5", color="grey")

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
        self, df: pd.DataFrame, column: str, output_dir: str
    ) -> Optional[str]:
        """Saves a distribution plot (KDE or bar) for a single feature."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            if pd.api.types.is_numeric_dtype(df[column]):
                sns.kdeplot(df[column], label="Synthetic", fill=True, ax=ax)
                ax.set_title(f"Distribution: {column}")
            else:
                counts = df[column].value_counts(normalize=True).rename("Synthetic")
                df_counts = pd.DataFrame(counts)
                if len(df_counts) > 10:
                    df_counts.plot(kind="bar", alpha=0.7, ax=ax)
                    plt.xticks(rotation=90)
                else:
                    df_counts.plot(
                        kind="line",
                        alpha=0.8,
                        ax=ax,
                        marker="o",
                        linewidth=2,
                        markersize=8,
                    )
                    plt.xticks(rotation=45)
                ax.set_title(f"Distribution: {column}")
                ax.set_ylabel("Proportion")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.6)
            plot_path = os.path.join(output_dir, f"dist_plot_{column}.png")
            fig.savefig(plot_path)
            plt.close(fig)
            return plot_path
        except Exception as e:
            self.logger.error(f"Failed to generate distribution plot for {column}: {e}")
            return None

    def _save_correlation_heatmap(
        self, df: pd.DataFrame, output_dir: str, focus_cols: Optional[List[str]] = None
    ) -> Optional[str]:
        """Saves a heatmap of the correlation matrix for numeric features."""
        try:
            numeric_cols = (
                focus_cols
                if focus_cols
                else df.select_dtypes(include=np.number).columns.tolist()
            )
            if len(numeric_cols) < 2:
                return None

            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr, ax=ax, annot=True, cmap="viridis", fmt=".2f")
            ax.set_title("Correlation Matrix")
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
        df: pd.DataFrame,
        output_dir: str,
        max_plots: int = 5,
        focus_cols: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Saves boxplots showing interactions between categorical and numeric features."""
        plots = {}
        try:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            if focus_cols:
                numeric_cols = [c for c in numeric_cols if c in focus_cols]
                categorical_cols = [c for c in categorical_cols if c in focus_cols]

            plot_count = 0
            for num_col in numeric_cols:
                for cat_col in categorical_cols:
                    if num_col == cat_col or plot_count >= max_plots:
                        continue
                    if 2 <= df[cat_col].nunique() <= 10:
                        try:
                            fig, ax = plt.subplots(figsize=(12, 8))
                            sns.boxplot(x=cat_col, y=num_col, data=df, ax=ax)
                            ax.set_title(f"{num_col} by {cat_col}")
                            ax.tick_params(axis="x", rotation=45)
                            plt.tight_layout()
                            plot_filename = f"interaction_plot_{cat_col}_{num_col}.png"
                            plot_path = os.path.join(output_dir, plot_filename)
                            fig.savefig(plot_path)
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
        self, df: pd.DataFrame, output_dir: str, generator_name: str
    ) -> Optional[str]:
        """Generates and saves a PCA plot to visualize the data's structure."""
        try:
            df_pca = df.drop(columns=["timestamp"], errors="ignore")
            numeric_features = df_pca.select_dtypes(include=np.number).columns.tolist()
            categorical_features = df_pca.select_dtypes(
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

            prepared_data = preprocessor.fit_transform(df_pca)
            pca = PCA(n_components=2, random_state=42)
            pca_result = pca.fit_transform(prepared_data)

            fig = plt.figure(figsize=(10, 8))
            sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], alpha=0.5)
            plt.title(f"PCA of Synthetic Data - {generator_name}")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.grid(True)
            plot_path = os.path.join(output_dir, f"pca_{generator_name}.png")
            fig.savefig(plot_path)
            plt.close(fig)
            return plot_path
        except Exception as e:
            self.logger.error(f"Failed to generate PCA analysis: {e}")
            return None

    def _report_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generates a concise summary of the dataset's basic statistics."""
        return {
            "dataset": "Synthetic",
            "rows": df.shape[0],
            "columns": df.shape[1],
            "total_cells": int(df.size),
            "memory_usage_mb": round(
                df.memory_usage(deep=True).sum() / (1024 * 1024), 2
            ),
            "total_null_values": int(df.isnull().sum().sum()),
            "total_duplicated_rows": int(df.duplicated().sum()),
        }

    def _report_target_analysis(
        self, df: pd.DataFrame, target_column: str
    ) -> Optional[Dict[str, Any]]:
        """Analyzes the target column to determine problem type and distribution."""
        if target_column not in df.columns:
            return None
        try:
            report = {}
            target = df[target_column]

            problem_type = "Regression"
            if (
                pd.api.types.is_string_dtype(target)
                or pd.api.types.is_categorical_dtype(target)
                or target.nunique() < 25
            ):
                problem_type = "Classification"

            report["problem_type"] = problem_type
            if problem_type == "Classification":
                report["distribution"] = target.value_counts(normalize=True).to_dict()
            else:  # Regression
                report["description"] = target.describe().to_dict()
            return report
        except Exception as e:
            self.logger.error(
                f"Failed to generate target analysis for column '{target_column}': {e}"
            )
            return {"error": str(e)}

    def _save_boxplot_plot(
        self, df: pd.DataFrame, column: str, output_dir: str
    ) -> Optional[str]:
        """Saves a boxplot for a single numerical feature."""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(y=column, data=df, ax=ax)
            ax.set_title(f"Boxplot: {column}")
            ax.set_ylabel(column)
            plt.grid(True, linestyle="--", alpha=0.6)
            plot_path = os.path.join(output_dir, f"boxplot_plot_{column}.png")
            fig.savefig(plot_path)
            plt.close(fig)
            return plot_path
        except Exception as e:
            self.logger.error(f"Failed to generate boxplot for {column}: {e}")
            return None

    def _report_mode_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyzes the mode (most frequent value) of categorical columns."""
        report = {}
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        for col in categorical_cols:
            try:
                mode_info = df[col].value_counts(normalize=True).nlargest(1)
                if not mode_info.empty:
                    report[col] = {
                        "mode": {
                            "value": mode_info.index[0],
                            "proportion": round(mode_info.iloc[0], 4),
                        }
                    }
            except Exception as e:
                self.logger.error(
                    f"Failed to generate mode analysis for column '{col}': {e}"
                )
        return report

    def _detect_distribution(self, data: pd.Series) -> str:
        """
        Identifies the best fitting distribution using the fit_distribution utility.
        """
        try:
            fit_res = fit_distribution(data)
            dist_name = fit_res["distribution"]
            p_val = fit_res["p_value"]

            if dist_name == "Insufficient Data":
                return dist_name
            if dist_name in ["None", "Unknown"]:
                return "Unknown"

            return f"{dist_name} (p={p_val:.4f})"
        except Exception as e:
            return f"Error: {e}"

    def _report_numeric_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyzes numeric columns, including statistical properties and distribution type."""
        report = {}
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        for col in numeric_cols:
            try:
                col_data = df[col].dropna()
                if col_data.empty:
                    continue

                stats_dict = {
                    "mean": float(col_data.mean()),
                    "median": float(col_data.median()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                }

                # Mode
                mode_values = col_data.mode()
                if not mode_values.empty:
                    stats_dict["mode"] = float(mode_values.iloc[0])
                else:
                    stats_dict["mode"] = None

                # Distribution Fitting
                stats_dict["distribution_type"] = self._detect_distribution(col_data)

                report[col] = stats_dict
            except Exception as e:
                self.logger.error(
                    f"Failed to generate numeric analysis for column '{col}': {e}"
                )
        return report

    def _save_mode_analysis_plot(
        self, mode_analysis_data: Dict[str, Any], output_dir: str
    ) -> Optional[str]:
        """Saves a plot visualizing the mode analysis data."""
        try:
            if not mode_analysis_data:
                return None

            columns = list(mode_analysis_data.keys())
            proportions = [
                d.get("mode", {}).get("proportion", 0)
                for d in mode_analysis_data.values()
            ]
            modes = [
                str(d.get("mode", {}).get("value", "N/A"))
                for d in mode_analysis_data.values()
            ]

            df = pd.DataFrame(
                {"Feature": columns, "Proportion": proportions, "Mode": modes}
            )
            if df.empty:
                return None

            fig, ax = plt.subplots(figsize=(14, len(columns) * 0.6 + 3))
            bars = ax.barh(df["Feature"], df["Proportion"], color="skyblue", alpha=0.8)
            ax.set_xlabel("Proportion of Mode")
            ax.set_title("Mode and Proportion for Categorical Features")
            ax.invert_yaxis()

            labels = [
                f"{df['Mode'][i]} ({df['Proportion'][i]:.2f})" for i in range(len(bars))
            ]
            ax.bar_label(bars, labels=labels, padding=5, fontsize=9, color="blue")

            ax.grid(axis="x", linestyle="--", alpha=0.6)
            if not df.empty and df["Proportion"].max() > 0:
                ax.set_xlim(right=df["Proportion"].max() * 1.4)

            plt.tight_layout()
            plot_path = os.path.join(output_dir, "mode_comparison.png")
            fig.savefig(plot_path)
            plt.close(fig)
            return plot_path
        except Exception as e:
            self.logger.error(f"Failed to generate mode analysis plot: {e}")
            return None

    def _report_detailed_quality_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates a detailed quality analysis including:
        - Null values (count and percentage)
        - Duplicates (row level)
        - Descriptive Statistics (Mean, Median, Mode, Std, Min, Max, Quantiles) fro numeric
        - Date Ranges (Min, Max) for datetime columns
        - Categorical details (Top-K, Unique count, Mode)
        - Outliers (IQR method)
        - Data Types and Memory Usage info
        """
        analysis = {}

        # 1. General Data Info
        analysis["data_info"] = {
            "rows": len(df),
            "columns": len(df.columns),
            "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
            "dtypes": df.dtypes.apply(lambda x: str(x)).to_dict(),
        }

        # 2. Duplicates
        num_duplicates = int(df.duplicated().sum())
        analysis["duplicates"] = {
            "count": num_duplicates,
            "percentage": round((num_duplicates / len(df)) * 100, 2)
            if len(df) > 0
            else 0.0,
        }

        # 3. Column-wise Analysis
        col_analysis = {}
        for col in df.columns:
            col_stats = {}

            # General Info
            num_nulls = int(df[col].isnull().sum())
            col_stats["type"] = str(df[col].dtype)
            col_stats["nulls"] = {
                "count": num_nulls,
                "percentage": round((num_nulls / len(df)) * 100, 2)
                if len(df) > 0
                else 0.0,
            }

            # Numeric Statistics & Outliers
            if pd.api.types.is_numeric_dtype(
                df[col]
            ) and not pd.api.types.is_bool_dtype(df[col]):
                col_data = df[col].dropna()
                if not col_data.empty:
                    # Describe
                    desc = col_data.describe().to_dict()
                    col_stats["statistics"] = desc

                    # Explicit Stats (Mean, Median, Mode, Min, Max)
                    mode_vals = col_data.mode()
                    col_stats["statistics"].update(
                        {
                            "mode": float(mode_vals.iloc[0])
                            if not mode_vals.empty
                            else None,
                            "median": float(col_data.median()),
                            "min": float(col_data.min()),
                            "max": float(col_data.max()),
                        }
                    )

                    # Outliers (IQR)
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = col_data[
                        (col_data < lower_bound) | (col_data > upper_bound)
                    ]
                    col_stats["outliers"] = {
                        "count": int(len(outliers)),
                        "percentage": round((len(outliers) / len(col_data)) * 100, 2),
                        "iqr_bounds": {
                            "lower": float(lower_bound),
                            "upper": float(upper_bound),
                        },
                    }
                else:
                    col_stats["statistics"] = "Empty or All-Null"

            # Datetime Statistics
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                col_data = df[col].dropna()
                if not col_data.empty:
                    col_stats["date_range"] = {
                        "min": col_data.min().isoformat(),
                        "max": col_data.max().isoformat(),
                        "range_days": (col_data.max() - col_data.min()).days,
                    }

            # Categorical Statistics
            else:
                col_data = df[col].dropna()
                if not col_data.empty:
                    col_stats["unique_values"] = int(col_data.nunique())
                    mode_vals = col_data.mode()
                    col_stats["mode"] = (
                        str(mode_vals.iloc[0]) if not mode_vals.empty else None
                    )

                    # Top-K (Propotion of non-numeric)
                    top_k = 10
                    value_counts = col_data.value_counts(normalize=True).head(top_k)
                    col_stats["top_values"] = {
                        str(k): round(v, 4) for k, v in value_counts.items()
                    }

            col_analysis[col] = col_stats

        analysis["columns"] = col_analysis
        return analysis

    def generate_markdown_report(self, report: Dict[str, Any], output_dir: str):
        """Generates a human-readable Markdown summary of the analysis."""
        try:
            md_path = os.path.join(output_dir, "report_summary.md")
            with open(md_path, "w") as f:
                f.write(f"# Synthetic Data Quality Report\n\n")
                f.write(f"**Generator:** {report.get('generator_name')}\n")
                f.write(f"**Date:** {report.get('generation_timestamp')}\n\n")

                # Basic Stats
                f.write("## 📊 Dataset Information\n")
                basic = report.get("basic_info", {})
                f.write(f"- **Rows:** {basic.get('rows')}\n")
                f.write(f"- **Columns:** {basic.get('columns')}\n")
                f.write(
                    f"- **Duplicates:** {basic.get('total_duplicated_rows')} ({basic.get('percentage_duplicates')} %)\n\n"
                )

                # Quality Analysis Highlights
                if "quality_analysis" in report:
                    qa = report["quality_analysis"]
                    f.write("## ⭐ Quality Highlights\n")
                    f.write(
                        f"- **Null Values:** {qa.get('data_info', {}).get('total_nulls', 'N/A')}\n"
                    )
                    duplicates_info = qa.get("duplicates", {})
                    f.write(
                        f"- **Exact Duplicates:** {duplicates_info.get('count', 0)} ({duplicates_info.get('percentage', 0)}%)\n\n"
                    )

                # Numeric Statistics
                if "numeric_analysis" in report and report["numeric_analysis"]:
                    f.write("## 🔢 Numeric Statistics\n")
                    f.write(
                        "| Column | Mean | Median | Mode | Std Dev | Distribution |\n"
                    )
                    f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
                    numeric_stats = report["numeric_analysis"]
                    for col, stats in numeric_stats.items():
                        f.write(
                            f"| {col} | {stats.get('mean', 'N/A'):.4f} | {stats.get('median', 'N/A'):.4f} | {stats.get('mode', 'N/A')} | {stats.get('std', 0):.4f} | **{stats.get('distribution_type', 'N/A')}** |\n"
                        )
                    f.write("\n")

                # Visualizations
                f.write("## 🖼️ Visualizations\n")
                plots = report.get("plots", {})
                if plots.get("pca"):
                    f.write(f"### Dimensionality Reduction (PCA)\n")
                    f.write(f"![PCA]({os.path.basename(plots['pca'])})\n\n")

                if plots.get("correlation_heatmap"):
                    f.write(f"### Correlation Matrix\n")
                    f.write(
                        f"![Correlation]({os.path.basename(plots['correlation_heatmap'])})\n\n"
                    )

            # Save the detailed JSON report
            # report_path = os.path.join(output_dir, "report.json")
            # try:
            #     with open(report_path, "w") as f:
            #         json.dump(report, f, indent=4, cls=NumpyEncoder)
            #     if self.verbose:
            #         print(f"✅ JSON report saved to: {report_path}")
            # except Exception as e:
            #     self.logger.error(f"Failed to save JSON report: {e}")

            if self.verbose:
                print(f"✅ Markdown report saved to: {md_path}")
        except Exception as e:
            self.logger.error(f"Failed to generate markdown report: {e}")
