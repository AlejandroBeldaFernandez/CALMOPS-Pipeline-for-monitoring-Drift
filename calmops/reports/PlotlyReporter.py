"""
PlotlyReporter Module
Generates interactive Plotly HTML plots for data visualization.
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import logging

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

logger = logging.getLogger("PlotlyReporter")


class PlotlyReporter:
    """
    Generates interactive Plotly HTML visualizations for data reports.
    """

    @staticmethod
    def generate_density_plots(
        df: pd.DataFrame,
        output_dir: str,
        columns: Optional[List[str]] = None,
        filename: str = "density_plots.html",
        color_col: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generates interactive density/distribution plots for numeric columns.
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Skipping density plots.")
            return None

        try:
            # Select numeric columns
            if columns:
                numeric_cols = [
                    c
                    for c in columns
                    if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
                ]
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            if not numeric_cols:
                logger.warning("No numeric columns found for density plots.")
                return None

            # Limit to 12 columns max
            numeric_cols = numeric_cols[:12]

            n_cols = min(3, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

            fig = make_subplots(
                rows=n_rows,
                cols=n_cols,
                subplot_titles=numeric_cols,
                vertical_spacing=0.08,
                horizontal_spacing=0.06,
            )

            for idx, col in enumerate(numeric_cols):
                row = idx // n_cols + 1
                col_pos = idx % n_cols + 1

                if color_col and color_col in df.columns:
                    for group_name in df[color_col].unique():
                        group_data = df[df[color_col] == group_name][col].dropna()
                        fig.add_trace(
                            go.Histogram(
                                x=group_data,
                                name=f"{col} ({group_name})",
                                opacity=0.7,
                                showlegend=(idx == 0),
                            ),
                            row=row,
                            col=col_pos,
                        )
                else:
                    fig.add_trace(
                        go.Histogram(
                            x=df[col].dropna(),
                            name=col,
                            opacity=0.7,
                            showlegend=False,
                        ),
                        row=row,
                        col=col_pos,
                    )

            fig.update_layout(
                title_text="Feature Distributions",
                height=300 * n_rows,
                showlegend=bool(color_col),
                barmode="overlay",
            )

            output_path = os.path.join(output_dir, filename)
            pio.write_html(fig, output_path, include_plotlyjs=True, full_html=True)

            return output_path

        except Exception as e:
            logger.error(f"Failed to generate density plots: {e}")
            return None

    @staticmethod
    def generate_dimensionality_plot(
        df: pd.DataFrame,
        output_dir: str,
        filename: str = "dimensionality_plot.html",
        color_col: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generates combined PCA + UMAP visualization in a single HTML file.
        """
        if not PLOTLY_AVAILABLE or not SKLEARN_AVAILABLE:
            logger.warning("Required libraries not available.")
            return None

        try:
            # Select numeric columns and drop NaN
            numeric_df = df.select_dtypes(include=[np.number]).dropna()

            if numeric_df.shape[1] < 2:
                logger.warning(
                    "Not enough numeric columns for dimensionality reduction."
                )
                return None

            if numeric_df.shape[0] < 10:
                logger.warning("Not enough samples for dimensionality reduction.")
                return None

            # Standardize
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df)

            # === PCA ===
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_data)

            pca_df = pd.DataFrame(
                pca_result,
                columns=["PC1", "PC2"],
                index=numeric_df.index,
            )

            # === UMAP (if available and enough samples) ===
            umap_df = None
            if UMAP_AVAILABLE and numeric_df.shape[0] >= 15:
                try:
                    reducer = umap.UMAP(
                        n_neighbors=min(15, numeric_df.shape[0] - 1),
                        min_dist=0.1,
                        n_components=2,
                        random_state=42,
                    )
                    umap_result = reducer.fit_transform(scaled_data)
                    umap_df = pd.DataFrame(
                        umap_result,
                        columns=["UMAP1", "UMAP2"],
                        index=numeric_df.index,
                    )
                except Exception as e:
                    logger.warning(f"UMAP failed: {e}")

            # Add color column if provided
            color_values = None
            if color_col and color_col in df.columns:
                color_values = df.loc[numeric_df.index, color_col].values

            # Create subplots
            if umap_df is not None:
                fig = make_subplots(
                    rows=1,
                    cols=2,
                    subplot_titles=[
                        f"PCA (Explained Variance: {sum(pca.explained_variance_ratio_):.1%})",
                        "UMAP",
                    ],
                    horizontal_spacing=0.1,
                )

                # PCA scatter
                if color_values is not None:
                    for val in np.unique(color_values):
                        mask = color_values == val
                        fig.add_trace(
                            go.Scatter(
                                x=pca_df.loc[mask, "PC1"],
                                y=pca_df.loc[mask, "PC2"],
                                mode="markers",
                                name=str(val),
                                legendgroup=str(val),
                                showlegend=True,
                            ),
                            row=1,
                            col=1,
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=umap_df.loc[mask, "UMAP1"],
                                y=umap_df.loc[mask, "UMAP2"],
                                mode="markers",
                                name=str(val),
                                legendgroup=str(val),
                                showlegend=False,
                            ),
                            row=1,
                            col=2,
                        )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=pca_df["PC1"],
                            y=pca_df["PC2"],
                            mode="markers",
                            showlegend=False,
                        ),
                        row=1,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=umap_df["UMAP1"],
                            y=umap_df["UMAP2"],
                            mode="markers",
                            showlegend=False,
                        ),
                        row=1,
                        col=2,
                    )

                fig.update_xaxes(title_text="PC1", row=1, col=1)
                fig.update_yaxes(title_text="PC2", row=1, col=1)
                fig.update_xaxes(title_text="UMAP1", row=1, col=2)
                fig.update_yaxes(title_text="UMAP2", row=1, col=2)

            else:
                # Only PCA
                fig = go.Figure()
                if color_values is not None:
                    for val in np.unique(color_values):
                        mask = color_values == val
                        fig.add_trace(
                            go.Scatter(
                                x=pca_df.loc[mask, "PC1"],
                                y=pca_df.loc[mask, "PC2"],
                                mode="markers",
                                name=str(val),
                            )
                        )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=pca_df["PC1"],
                            y=pca_df["PC2"],
                            mode="markers",
                            showlegend=False,
                        )
                    )

                fig.update_layout(
                    title=f"PCA Visualization (Explained Variance: {sum(pca.explained_variance_ratio_):.1%})",
                )

            fig.update_layout(height=500)

            output_path = os.path.join(output_dir, filename)
            pio.write_html(fig, output_path, include_plotlyjs=True, full_html=True)

            return output_path

        except Exception as e:
            logger.error(f"Failed to generate dimensionality plot: {e}")
            return None

    @staticmethod
    def generate_sdv_scores_card(
        overall_score: float,
        weighted_score: float,
        output_dir: str,
        filename: str = "sdv_scores.html",
    ) -> Optional[str]:
        """
        Generates a simple HTML card showing SDV scores.
        """
        if not PLOTLY_AVAILABLE:
            return None

        try:
            fig = go.Figure()

            # Overall score indicator
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=overall_score * 100,
                    title={"text": "Overall SDV Score"},
                    domain={"x": [0, 0.45], "y": [0, 1]},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "blue"},
                        "steps": [
                            {"range": [0, 50], "color": "#ffcccc"},
                            {"range": [50, 75], "color": "#ffffcc"},
                            {"range": [75, 100], "color": "#ccffcc"},
                        ],
                    },
                    number={"suffix": "%"},
                )
            )

            # Weighted score indicator
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=weighted_score * 100,
                    title={"text": "Weighted SDV Score"},
                    domain={"x": [0.55, 1], "y": [0, 1]},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "green"},
                        "steps": [
                            {"range": [0, 50], "color": "#ffcccc"},
                            {"range": [50, 75], "color": "#ffffcc"},
                            {"range": [75, 100], "color": "#ccffcc"},
                        ],
                    },
                    number={"suffix": "%"},
                )
            )

            fig.update_layout(
                title="SDV Quality Scores",
                height=350,
            )

            output_path = os.path.join(output_dir, filename)
            pio.write_html(fig, output_path, include_plotlyjs=True, full_html=True)

            return output_path

        except Exception as e:
            logger.error(f"Failed to generate SDV scores card: {e}")
            return None

    @staticmethod
    def generate_sdv_evolution_plot(
        scores: List[Dict[str, Any]],
        output_dir: str,
        filename: str = "sdv_evolution.html",
        x_labels: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Generates SDV quality evolution plot showing overall and weighted scores.
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Skipping SDV evolution plot.")
            return None

        try:
            if not scores:
                logger.warning("No scores provided for SDV evolution plot.")
                return None

            overall_scores = [s.get("overall", 0) for s in scores]
            weighted_scores = [s.get("weighted", 0) for s in scores]

            if x_labels is None:
                x_labels = [f"Block {i + 1}" for i in range(len(scores))]

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=overall_scores,
                    mode="lines+markers",
                    name="Overall SDV Score",
                    line=dict(color="blue", width=2),
                    marker=dict(size=10),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=weighted_scores,
                    mode="lines+markers",
                    name="Weighted SDV Score",
                    line=dict(color="green", width=2),
                    marker=dict(size=10),
                )
            )

            fig.update_layout(
                title="SDV Quality Evolution",
                xaxis_title="Block / Time Period",
                yaxis_title="Quality Score",
                yaxis=dict(range=[0, 1]),
                height=500,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                hovermode="x unified",
            )

            output_path = os.path.join(output_dir, filename)
            pio.write_html(fig, output_path, include_plotlyjs=True, full_html=True)

            return output_path

        except Exception as e:
            logger.error(f"Failed to generate SDV evolution plot: {e}")
            return None

    @staticmethod
    def generate_drift_analysis(
        original_df: pd.DataFrame,
        drifted_df: pd.DataFrame,
        output_dir: str,
        filename: str = "drift_analysis.html",
        columns: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Generates drift analysis visualizations comparing original and drifted data.

        Includes:
        - Overlay density plots per feature
        - JS Divergence bar chart
        - Feature change summary
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Skipping drift analysis.")
            return None

        try:
            from scipy.spatial.distance import jensenshannon

            # Select numeric columns
            if columns:
                numeric_cols = [
                    c
                    for c in columns
                    if c in original_df.columns
                    and c in drifted_df.columns
                    and pd.api.types.is_numeric_dtype(original_df[c])
                ]
            else:
                numeric_cols = [
                    c
                    for c in original_df.select_dtypes(include=[np.number]).columns
                    if c in drifted_df.columns
                ]

            if not numeric_cols:
                logger.warning("No numeric columns found for drift analysis.")
                return None

            # Limit to 12 columns
            numeric_cols = numeric_cols[:12]

            # Calculate JS Divergence for each column
            js_scores = {}
            for col in numeric_cols:
                try:
                    # Create histograms for JS divergence
                    orig_vals = original_df[col].dropna()
                    drift_vals = drifted_df[col].dropna()

                    if len(orig_vals) == 0 or len(drift_vals) == 0:
                        continue

                    # Use common bins
                    min_val = min(orig_vals.min(), drift_vals.min())
                    max_val = max(orig_vals.max(), drift_vals.max())
                    bins = np.linspace(min_val, max_val, 50)

                    hist_orig, _ = np.histogram(orig_vals, bins=bins, density=True)
                    hist_drift, _ = np.histogram(drift_vals, bins=bins, density=True)

                    # Add small epsilon to avoid division by zero
                    hist_orig = hist_orig + 1e-10
                    hist_drift = hist_drift + 1e-10

                    # Normalize
                    hist_orig = hist_orig / hist_orig.sum()
                    hist_drift = hist_drift / hist_drift.sum()

                    js_scores[col] = jensenshannon(hist_orig, hist_drift)
                except Exception:
                    js_scores[col] = 0.0

            # Sort by JS divergence
            sorted_cols = sorted(
                js_scores.keys(), key=lambda x: js_scores[x], reverse=True
            )

            # Create figure with subplots
            n_density_cols = min(6, len(sorted_cols))
            n_rows = 2 + ((n_density_cols + 2) // 3)  # JS bar + summary + density plots

            fig = make_subplots(
                rows=n_rows,
                cols=3,
                specs=[[{"colspan": 3}, None, None]]  # JS Divergence bar
                + [[{"colspan": 3}, None, None]]  # Summary stats
                + [
                    [{}, {}, {}] for _ in range((n_density_cols + 2) // 3)
                ],  # Density plots
                subplot_titles=[
                    "JS Divergence by Feature",
                    "Distribution Shift Summary",
                ]
                + sorted_cols[:n_density_cols],
                vertical_spacing=0.08,
                row_heights=[0.2, 0.15]
                + [0.65 / max(1, (n_density_cols + 2) // 3)]
                * ((n_density_cols + 2) // 3),
            )

            # 1. JS Divergence Bar Chart
            fig.add_trace(
                go.Bar(
                    x=sorted_cols,
                    y=[js_scores[c] for c in sorted_cols],
                    marker_color=[
                        "red"
                        if js_scores[c] > 0.1
                        else "orange"
                        if js_scores[c] > 0.05
                        else "green"
                        for c in sorted_cols
                    ],
                    showlegend=False,
                    hovertemplate="%{x}: %{y:.4f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            fig.update_yaxes(title_text="JS Divergence", row=1, col=1)

            # 2. Summary Stats Table (as text annotation)
            summary_text = "<b>Feature</b> | <b>Orig Mean</b> | <b>Drift Mean</b> | <b>Change %</b><br>"
            for col in sorted_cols[:5]:
                orig_mean = original_df[col].mean()
                drift_mean = drifted_df[col].mean()
                pct_change = (
                    ((drift_mean - orig_mean) / abs(orig_mean) * 100)
                    if orig_mean != 0
                    else 0
                )
                summary_text += f"{col} | {orig_mean:.2f} | {drift_mean:.2f} | {pct_change:+.1f}%<br>"

            fig.add_annotation(
                text=summary_text,
                xref="x2 domain",
                yref="y2 domain",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(family="monospace", size=12),
                align="left",
                row=2,
                col=1,
            )

            # 3. Overlay Density Plots
            for idx, col in enumerate(sorted_cols[:n_density_cols]):
                row = 3 + idx // 3
                col_pos = 1 + idx % 3

                # Original
                fig.add_trace(
                    go.Histogram(
                        x=original_df[col].dropna(),
                        name="Original",
                        opacity=0.6,
                        marker_color="blue",
                        showlegend=(idx == 0),
                        legendgroup="original",
                        histnorm="probability density",
                    ),
                    row=row,
                    col=col_pos,
                )

                # Drifted
                fig.add_trace(
                    go.Histogram(
                        x=drifted_df[col].dropna(),
                        name="Drifted",
                        opacity=0.6,
                        marker_color="red",
                        showlegend=(idx == 0),
                        legendgroup="drifted",
                        histnorm="probability density",
                    ),
                    row=row,
                    col=col_pos,
                )

            fig.update_layout(
                title_text="Drift Analysis: Original vs Drifted Data",
                height=200 + 250 * n_rows,
                barmode="overlay",
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            )

            output_path = os.path.join(output_dir, filename)
            pio.write_html(fig, output_path, include_plotlyjs=True, full_html=True)

            return output_path

        except Exception as e:
            logger.error(f"Failed to generate drift analysis: {e}")
            return None
