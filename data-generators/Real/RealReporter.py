import pandas as pd
from sdmetrics.reports.single_table import QualityReport
from sdv.metadata import SingleTableMetadata
import logging

# Setup logger to track events and errors
logger = logging.getLogger('RealReporterLogger')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class RealReporter:
    def _report_real_dataset(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        target_col: str,
        method: str,
        extra_info: dict = None
    ):
        """
        Generates and logs a report comparing the real and synthetic datasets.
        """
        logger.info("\n📊 === REAL DATASET REPORT ===")
        logger.info(f"Method: {method}")
        logger.info(f"Original samples: {len(real_df)} | Synthetic samples: {len(synthetic_df)}")

        # Check if the target column is present in both datasets
        if target_col in real_df.columns and target_col in synthetic_df.columns:
            logger.info("\n⚖️ Class distribution comparison:")
            real_dist = real_df[target_col].value_counts(normalize=True).to_dict()
            synth_dist = synthetic_df[target_col].value_counts(normalize=True).to_dict()
            for cls in sorted(set(real_dist) | set(synth_dist)):
                logger.info(f"  - Class {cls}: real={real_dist.get(cls, 0):.2%}, synth={synth_dist.get(cls, 0):.2%}")
        else:
            logger.warning(f"Target column '{target_col}' not found in one or both datasets")

        # Statistics for synthetic features
        logger.info("\n📈 Synthetic feature statistics:")
        synthetic_stats = synthetic_df.drop(columns=[target_col, "chunk"], errors="ignore").describe().T
        logger.info(synthetic_stats)

        # Statistics for real features
        logger.info("\n📈 Original feature statistics:")
        real_stats = real_df.drop(columns=[target_col], errors="ignore").describe().T
        logger.info(real_stats)

        # Filter columns that exist in both datasets
        common_cols = [col for col in synthetic_df.columns if col in real_df.columns]
        real_clean = real_df[common_cols].copy()
        synth_clean = synthetic_df[common_cols].copy()

        # Initialize metadata and generate quality report
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(real_clean)

        quality_report = QualityReport()
        quality_report.generate(
            real_data=real_clean,
            synthetic_data=synth_clean,
            metadata=metadata.to_dict()
        )

        # Log quality score and details
        logger.info(f"\n✅ Quality Score: {quality_report.get_score():.3f}")
        logger.info("📊 Quality details:")
        for prop in ["Column Shapes", "Column Pair Trends"]:
            logger.info(f"\n🔹 {prop}")
            logger.info(quality_report.get_details(prop).head())

        # Log extra information if available
        if extra_info:
            logger.info("\nℹ️ Extra information:")
            for key, val in extra_info.items():
                logger.info(f"  - {key}: {val}")

        logger.info("\n=== END OF REAL DATASET REPORT ===\n")
