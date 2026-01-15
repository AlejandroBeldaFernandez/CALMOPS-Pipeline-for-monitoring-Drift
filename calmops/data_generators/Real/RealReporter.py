"""
Static, File-Based Real Data Reporter

This module provides the RealReporter class, which generates a detailed,
static report comparing a real dataset with a synthetic one.
Uses YData Profiling for analysis and Plotly for interactive visualizations.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union
import warnings
import logging
from datetime import datetime
import os
import json

from calmops.reports.ExternalReporter import ExternalReporter
from calmops.reports.PlotlyReporter import PlotlyReporter
from calmops.reports.LocalIndexGenerator import LocalIndexGenerator

# Try to import SDV for quality metrics
try:
    from sdv.evaluation.single_table import evaluate_quality
    from sdv.metadata import SingleTableMetadata

    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False
    warnings.warn("SDV not available. Quality assessment will be limited.")

logger = logging.getLogger("RealReporter")


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        return super().default(obj)


class RealReporter:
    """
    Generates a static, file-based report comparing a real dataset and its synthetic counterpart.
    Uses YData Profiling and Plotly for visualizations.
    """

    def __init__(self, verbose: bool = True):
        """
        Initializes the RealReporter.

        Args:
            verbose (bool): If True, prints progress messages to the console.
        """
        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)

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
        resample_rule: Optional[Union[str, int]] = None,
    ) -> None:
        """
        Generates a comprehensive file-based report comparing real and synthetic data.
        """
        if self.verbose:
            print("=" * 80)
            print("COMPREHENSIVE REAL DATA GENERATION REPORT")
            print(f"Generator: {generator_name}")
            print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)

        os.makedirs(output_dir, exist_ok=True)

        # Determine time column
        final_time_col = (
            time_col
            if time_col and time_col in real_df.columns
            else "timestamp"
            if "timestamp" in real_df.columns
            else None
        )

        # === Resampling Logic ===
        real_df_for_report = real_df.copy()
        synthetic_df_for_report = synthetic_df.copy()

        if resample_rule is not None:
            real_df_for_report = self._apply_resampling(
                real_df_for_report, final_time_col, block_column, resample_rule
            )
            synthetic_df_for_report = self._apply_resampling(
                synthetic_df_for_report, final_time_col, block_column, resample_rule
            )

        # === SDV Quality Assessment ===
        sdv_quality = self._assess_sdv_quality(
            real_df_for_report, synthetic_df_for_report
        )

        # Generate SDV Scores Card
        if "overall_quality_score" in sdv_quality:
            if self.verbose:
                print("\nGenerating SDV Scores Card...")
            PlotlyReporter.generate_sdv_scores_card(
                overall_score=sdv_quality["overall_quality_score"],
                weighted_score=sdv_quality["weighted_quality_score"],
                output_dir=output_dir,
            )

        # === SDV Quality by Block (for evolution plot) ===
        sdv_scores_by_block = []
        if block_column and block_column in real_df.columns:
            sdv_scores_by_block = self._calculate_sdv_by_block(
                real_df, synthetic_df, block_column
            )

            if sdv_scores_by_block:
                block_labels = [s["block"] for s in sdv_scores_by_block]
                PlotlyReporter.generate_sdv_evolution_plot(
                    scores=sdv_scores_by_block,
                    output_dir=output_dir,
                    x_labels=block_labels,
                )

        # === Save Results JSON ===
        results = {
            "generator_name": generator_name,
            "generation_timestamp": datetime.now().isoformat(),
            "real_rows": len(real_df),
            "synthetic_rows": len(synthetic_df),
            "sdv_quality": sdv_quality,
            "sdv_by_block": sdv_scores_by_block if sdv_scores_by_block else None,
            "compared_data_files": {
                "original": "real_data",
                "generated": "synthetic_data",
            },
        }

        results_path = os.path.join(output_dir, "drift_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

        # === Generate Plotly Visualizations ===
        if self.verbose:
            print("\nGenerating Plotly Visualizations...")

        # Density plots (on synthetic data)
        PlotlyReporter.generate_density_plots(
            df=synthetic_df_for_report,
            output_dir=output_dir,
            columns=focus_cols,
            color_col=target_column,
        )

        # Combined PCA + UMAP (real + synthetic merged)
        combined_df = pd.concat(
            [
                real_df_for_report.assign(_source="Real"),
                synthetic_df_for_report.assign(_source="Synthetic"),
            ],
            ignore_index=True,
        )
        PlotlyReporter.generate_dimensionality_plot(
            df=combined_df,
            output_dir=output_dir,
            color_col="_source",
        )

        # Drift Analysis (comparing real vs synthetic)
        PlotlyReporter.generate_drift_analysis(
            original_df=real_df_for_report,
            drifted_df=synthetic_df_for_report,
            output_dir=output_dir,
            columns=focus_cols,
        )

        # === Generate YData Reports ===
        if self.verbose:
            print("\nGenerating YData Reports...")

        ExternalReporter.generate_comparison(
            ref_df=real_df_for_report,
            curr_df=synthetic_df_for_report,
            output_dir=output_dir,
            ref_name="Original / Real",
            curr_name="Generated / Synthetic",
            time_col=final_time_col,
            block_col=block_column,
        )

        # Profile for GENERATED data (renamed to clarify)
        ExternalReporter.generate_profile(
            df=synthetic_df_for_report,
            output_dir=output_dir,
            filename="generated_profile.html",
            time_col=final_time_col,
            block_col=block_column,
            title="Generated Data Profile",
        )

        # === Generate Dashboard ===
        LocalIndexGenerator.create_index(output_dir)

        if self.verbose:
            print(f"\nReport generated at: {output_dir}")

    def update_report_after_drift(
        self,
        original_df: pd.DataFrame,
        drifted_df: pd.DataFrame,
        output_dir: str,
        drift_config: Optional[Dict[str, Any]] = None,
        time_col: Optional[str] = None,
        block_column: Optional[str] = None,
        resample_rule: Optional[Union[str, int]] = None,
    ) -> None:
        """
        Updates the report after drift injection.
        """
        generator_name = (
            drift_config.get("generator_name", "DriftInjector")
            if drift_config
            else "DriftInjector"
        )

        self.generate_comprehensive_report(
            real_df=original_df,
            synthetic_df=drifted_df,
            generator_name=generator_name,
            output_dir=output_dir,
            focus_cols=drift_config.get("feature_cols") if drift_config else None,
            drift_config=drift_config,
            time_col=time_col,
            block_column=block_column,
            resample_rule=resample_rule,
        )

    def _assess_sdv_quality(
        self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Assesses the quality of synthetic data using SDV metrics.
        """
        if not SDV_AVAILABLE:
            return {"error": "SDV not available"}

        try:
            if self.verbose:
                print("\nRunning SDMetrics Quality Assessment...")

            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(real_df)

            quality_report = evaluate_quality(
                real_data=real_df,
                synthetic_data=synthetic_df,
                metadata=metadata,
            )

            overall_score = quality_report.get_score()
            weighted_score = self._get_weighted_sdv_score(
                real_df, synthetic_df, overall_score
            )

            if self.verbose:
                print(
                    f"SDMetrics Assessment complete. Overall: {overall_score:.2f}, Weighted: {weighted_score:.2f}"
                )

            return {
                "overall_quality_score": round(overall_score, 4),
                "weighted_quality_score": round(weighted_score, 4),
            }

        except Exception as e:
            self.logger.error(f"SDV quality assessment failed: {e}")
            return {"error": str(e)}

    def _calculate_sdv_by_block(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        block_column: str,
    ) -> List[Dict[str, Any]]:
        """
        Calculates SDV quality scores for each block.
        """
        if not SDV_AVAILABLE:
            return []

        scores = []
        try:
            unique_blocks = sorted(real_df[block_column].unique(), key=str)

            for block_id in unique_blocks:
                real_block = real_df[real_df[block_column] == block_id]
                synth_block = synthetic_df[synthetic_df[block_column] == block_id]

                if real_block.empty or synth_block.empty:
                    continue

                try:
                    metadata = SingleTableMetadata()
                    metadata.detect_from_dataframe(real_block)

                    quality_report = evaluate_quality(
                        real_data=real_block,
                        synthetic_data=synth_block,
                        metadata=metadata,
                    )

                    overall = quality_report.get_score()
                    weighted = self._get_weighted_sdv_score(
                        real_block, synth_block, overall
                    )

                    scores.append(
                        {
                            "block": str(block_id),
                            "overall": round(overall, 4),
                            "weighted": round(weighted, 4),
                        }
                    )

                except Exception as e:
                    self.logger.warning(f"SDV failed for block {block_id}: {e}")

        except Exception as e:
            self.logger.error(f"Block-wise SDV calculation failed: {e}")

        return scores

    def _get_weighted_sdv_score(
        self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame, base_score: float
    ) -> float:
        """
        Calculates a weighted SDV score, penalized by data duplication and null values.
        """
        if synthetic_df.empty:
            return 0.0

        # Internal duplicates
        internal_dup_count = synthetic_df.duplicated().sum()

        # Cross duplicates
        try:
            real_unique = real_df.drop_duplicates()
            merged = synthetic_df.merge(
                real_unique, on=list(synthetic_df.columns), how="left", indicator=True
            )
            cross_dup_count = (merged["_merge"] == "both").sum()
        except Exception:
            cross_dup_count = 0

        total_bad = internal_dup_count + cross_dup_count
        duplication_penalty = (
            total_bad / len(synthetic_df) if len(synthetic_df) > 0 else 0.0
        )

        null_penalty = (
            synthetic_df.isnull().sum().sum() / synthetic_df.size
            if synthetic_df.size > 0
            else 0.0
        )

        base_score = 0.0 if pd.isna(base_score) else base_score
        weighted_score = base_score * (1 - duplication_penalty) * (1 - null_penalty)

        return weighted_score

    def _apply_resampling(
        self,
        df: pd.DataFrame,
        time_col: Optional[str],
        block_column: Optional[str],
        resample_rule: Union[str, int],
    ) -> pd.DataFrame:
        """
        Applies resampling/aggregation based on time or block column.
        """
        if self.verbose:
            print(f"Applying resampling rule: {resample_rule}")

        def agg_mode(x):
            m = x.mode()
            return m.iloc[0] if not m.empty else np.nan

        exclude_cols = [c for c in [time_col, block_column] if c]
        cols_to_agg = [c for c in df.columns if c not in exclude_cols]

        agg_dict = {}
        for col in cols_to_agg:
            if pd.api.types.is_numeric_dtype(df[col]):
                agg_dict[col] = "mean"
            else:
                agg_dict[col] = agg_mode

        if time_col and time_col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

            df = df.set_index(time_col)
            df = df.resample(resample_rule).agg(agg_dict).reset_index()
            df = df.dropna(how="all")

        elif (
            block_column
            and block_column in df.columns
            and isinstance(resample_rule, int)
        ):
            df["_block_group"] = df[block_column] // resample_rule
            df = df.groupby("_block_group").agg(agg_dict).reset_index(drop=True)

        return df
