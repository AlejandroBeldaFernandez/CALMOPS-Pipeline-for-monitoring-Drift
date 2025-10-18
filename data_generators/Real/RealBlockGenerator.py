"""
Real Block Data Generator with Enhanced Block-wise Processing

This module extends the RealGenerator to support block-based data generation and drift injection.
It is designed to work with datasets that are naturally partitioned into blocks (e.g., by time, customer ID, etc.)
or to dynamically create blocks from a continuous dataset.

Key Features:
- **Block-based Generation**: Synthesizes data for each block separately, preserving block-specific characteristics.
- **Dynamic Chunking**: Can automatically create blocks based on a fixed row count (`chunk_size`) or changes in a timestamp column (`chunk_by_timestamp`).
- **Scheduled Drift Injection**: Allows for complex drift scenarios to be scheduled across different blocks.
- **Comprehensive Reporting**: Generates detailed reports that include block-level statistics and comparisons, in addition to the overall dataset quality.
- **Timestamp Alignment**: Can inject timestamps that are aligned with the block structure.
"""

import os

import logging

import warnings

from pathlib import Path

from typing import Optional, Dict, Any, List, Union



import numpy as np

import pandas as pd



# Suppress common warnings for cleaner output

warnings.filterwarnings('ignore', category=UserWarning)

warnings.filterwarnings('ignore', category=FutureWarning)

warnings.filterwarnings('ignore', category=DeprecationWarning)



from .RealGenerator import RealGenerator

from .RealReporter import RealReporter





class RealBlockGenerator(RealGenerator):

    """
    An enhanced data generator that processes real datasets in a block-wise manner.
    It supports dynamic chunking, scheduled drift injection, and detailed block-level reporting.
    """



    def __init__(self,

                 original_data: pd.DataFrame,

                 method: str = 'cart',

                 target_column: Optional[str] = None,

                 block_column: Optional[str] = None,

                 chunk_size: Optional[int] = None,

                 chunk_by_timestamp: Optional[str] = None,

                 auto_report: bool = True,

                 random_state: int = 42,

                 verbose: bool = True,

                 model_params: Optional[Dict[str, Any]] = None):

        """
        Initializes the RealBlockGenerator.

        Args:
            original_data (pd.DataFrame): The full, original dataset.
            method (str): The synthesis method to use for each block (e.g., 'cart', 'ctgan').
            target_column (Optional[str]): Name of the target variable column.
            block_column (Optional[str]): Name of an existing column that defines the blocks.
            chunk_size (Optional[int]): If provided, creates blocks of this fixed size.
            chunk_by_timestamp (Optional[str]): If provided, creates blocks based on changes in this timestamp column.
            auto_report (bool): Whether to automatically generate a comprehensive report after processing all blocks.
            random_state (int): Random seed for reproducibility.
            verbose (bool): Enables verbose logging.
            model_params (Optional[Dict[str, Any]]): Dictionary of hyperparameters for the synthesis model.
        """

        # Create a copy to avoid modifying the original dataframe in place

        original_data = original_data.copy()



        # Call parent constructor

        super().__init__(

            original_data=original_data,

            method=method,

            target_column=target_column,

            auto_report=auto_report,

            random_state=random_state,

            model_params=model_params

        )



        self.model_params = model_params  # Store for block-level generators

        self.logger = logging.getLogger(self.__class__.__name__)

        if verbose:

            self.logger.setLevel(logging.INFO)



        # Create chunks dynamically if specified

        self.block_column = self._create_chunks(

            block_column=block_column,

            chunk_size=chunk_size,

            chunk_by_timestamp=chunk_by_timestamp

        )



        # Robust ordering in case of mixed types

        self.blocks = sorted(self.original_data[self.block_column].unique(), key=str)

        self.n_blocks = len(self.blocks)

        

        self.reporter = RealReporter()



        self.logger.info("RealBlockGenerator initialized with %d blocks", self.n_blocks)

        self.logger.info("Blocks are defined by column: '%s'", self.block_column)

        self.logger.info("Blocks: %s", self.blocks)

        self.logger.info(

            "Block sizes: %s",

            self.original_data[self.block_column].value_counts().sort_index().to_dict()

        )



    # --------------------------------------------------------------------------- #

    #                             INTERNAL HELPERS                                #

    # --------------------------------------------------------------------------- #



    def _create_chunks(self,

                       block_column: Optional[str],

                       chunk_size: Optional[int],

                       chunk_by_timestamp: Optional[str]) -> str:

        """
        Dynamically creates a chunk/block column in the original data based on the specified strategy.
        Returns the name of the block column to be used.
        """

        chunking_strategies = sum(p is not None for p in [block_column, chunk_size, chunk_by_timestamp])

        if chunking_strategies > 1:

            raise ValueError("Please specify only one chunking strategy: 'block_column', 'chunk_size', or 'chunk_by_timestamp'.")

        if chunking_strategies == 0:

            raise ValueError("A chunking strategy is required. Please provide 'block_column', 'chunk_size', or 'chunk_by_timestamp'.")



        if block_column:

            self.logger.info("Using existing column '%s' for blocks.", block_column)

            if block_column not in self.original_data.columns:

                raise ValueError(f"Block column '{block_column}' not found in dataset")

            return block_column



        new_chunk_col_name = 'chunk'

        if chunk_by_timestamp:

            self.logger.info("Creating chunks based on changes in timestamp column '%s'.", chunk_by_timestamp)

            if chunk_by_timestamp not in self.original_data.columns:

                raise ValueError(f"Timestamp column '{chunk_by_timestamp}' not found for chunking.")

            

            # Increment chunk ID every time the timestamp value changes

            self.original_data[new_chunk_col_name] = self.original_data[chunk_by_timestamp].ne(

                self.original_data[chunk_by_timestamp].shift()

            ).cumsum()



        elif chunk_size:

            self.logger.info("Creating chunks of fixed size %d.", chunk_size)

            if chunk_size <= 0:

                raise ValueError("'chunk_size' must be a positive integer.")

            

            # Create chunks of fixed size

            self.original_data[new_chunk_col_name] = np.arange(len(self.original_data)) // chunk_size

        

        return new_chunk_col_name



    def _generate_block_data(self,

                             block_id: Any,

                             output_dir: str,

                             n_samples: Optional[int] = None,

                             custom_distributions: Optional[Dict] = None) -> pd.DataFrame:

        """
        Generates synthetic data for a specific block by creating a temporary RealGenerator instance for that block.
        """

        block_data = self.original_data[self.original_data[self.block_column] == block_id].copy()

        block_data_no_block = block_data.drop(columns=[self.block_column])



        if len(block_data) == 0:

            raise ValueError(f"No data found for block: {block_id}")



        if n_samples is None:

            n_samples = len(block_data)



        self.logger.info("Generating %d samples for block %s using method '%s'", n_samples, str(block_id), self.method)



        # Create a new RealGenerator instance for this specific block

        block_generator = RealGenerator(

            original_data=block_data_no_block,

            method=self.method,

            target_column=self.target_column,

            auto_report=False,  # Reporting is done at the end for the whole dataset

            random_state=self.random_state,

            model_params=self.model_params # Pass down model params

        )



        # Synthesize data for the block

        synthetic_block = block_generator.synthesize(

            n_samples=n_samples,

            output_dir=output_dir,

            custom_distributions=custom_distributions

        )



        if synthetic_block is None:

            raise RuntimeError(f"Synthesis failed for block {block_id}")



        # Re-attach block column

        synthetic_block[self.block_column] = block_id



        return synthetic_block



    # --------------------------------------------------------------------------- #

    #                                 PUBLIC API                                  #

    # --------------------------------------------------------------------------- #



    def generate_block_dataset(self,

                               output_dir: str,

                               samples_per_block: Optional[Union[int, Dict[Any, int]]] = None,

                               drift_schedule: Optional[List[Dict[str, Any]]] = None,

                               custom_distributions: Optional[Dict] = None,

                               date_start: Optional[str] = None,

                               date_step: Optional[Dict[str, int]] = None,

                               date_col: str = "timestamp") -> pd.DataFrame:

        """
        Generates a complete synthetic dataset by processing each block and applying a drift schedule.

        Args:
            output_dir (str): Directory to save the report and any intermediate artifacts.
            samples_per_block (Optional[Union[int, Dict[Any, int]]]): Uniform number of samples for all blocks or a dictionary mapping each block_id to a specific number of samples.
            drift_schedule (Optional[List[Dict[str, Any]]]): A list of drift configurations to be applied sequentially to the entire dataset using DriftInjector.
            custom_distributions (Optional[Dict]): A dictionary to specify custom distributions for columns during synthesis.
            date_start (Optional[str]): Start date for block-aligned date injection.
            date_step (Optional[Dict[str, int]]): The time step for date injection (e.g., {'days': 1}).
            date_col (str): The name of the date column to be injected.

        Returns:
            pd.DataFrame: The complete synthetic dataset with all blocks and applied drift.
        """

        from data_generators.DriftInjection.DriftInjector import DriftInjector

        synthetic_blocks: List[pd.DataFrame] = []

        start_ts = pd.to_datetime(date_start) if date_start else None

        step_offset = pd.DateOffset(**(date_step or {"days": 1})) if start_ts else None



        for i, block_id in enumerate(self.blocks):

            # Determine samples for this block

            if samples_per_block is None:

                n_samples = len(self.original_data[self.original_data[self.block_column] == block_id])

            elif isinstance(samples_per_block, int):

                n_samples = samples_per_block

            else:

                n_samples = samples_per_block.get(

                    block_id,

                    len(self.original_data[self.original_data[self.block_column] == block_id])

                )



            # Generate block without drift

            synthetic_block = self._generate_block_data(

                block_id=block_id,

                output_dir=output_dir,

                n_samples=n_samples,

                custom_distributions=custom_distributions

            )

            

            # Inject block-aligned timestamp if specified

            if start_ts and step_offset:

                block_timestamp = start_ts + (step_offset * i)

                synthetic_block[date_col] = block_timestamp



            synthetic_blocks.append(synthetic_block)



            self.logger.info("Generated block %s: %d samples", str(block_id), len(synthetic_block))



        # Combine all synthetic blocks into a single dataset

        complete_dataset = pd.concat(synthetic_blocks, ignore_index=True)



        # Apply drift to the complete dataset based on the schedule

        if drift_schedule:

            self.logger.info("Applying drift schedule to the complete dataset.")

            

            drift_injector = DriftInjector(

                original_df=self.original_data,

                output_dir=output_dir,

                generator_name=f"RealBlockGenerator_{self.method}",

                target_column=self.target_column,

                block_column=self.block_column,

                random_state=self.random_state

            )



            complete_dataset = drift_injector.inject_multiple_types_of_drift(

                df=complete_dataset,

                schedule=drift_schedule

            )

            self.logger.info("Drift schedule applied successfully.")

        

        else:

            # Generate a report for the final dataset (no drift)

            self._generate_block_report(complete_dataset, output_dir=output_dir)



        self.logger.info("Complete synthetic dataset generated: %s", str(complete_dataset.shape))

        self.logger.info(

            "Final block distribution: %s",

            complete_dataset[self.block_column].value_counts().sort_index().to_dict()

        )



        # Always save the final dataset

        final_dataset_path = os.path.join(output_dir, f"complete_block_dataset_{self.method}.csv")

        self.save_block_dataset(complete_dataset, output_path=final_dataset_path)

        self.logger.info("Complete synthetic dataset saved to: %s", final_dataset_path)



        return complete_dataset



    def _generate_block_report(self, synthetic_dataset: pd.DataFrame, output_dir: str):

        """Generates a comprehensive report for the full block-based dataset at the dataset level."""

        if not self.auto_report:

            return

        try:

            os.makedirs(output_dir, exist_ok=True)



            self.reporter.generate_comprehensive_report(

                real_df=self.original_data,

                synthetic_df=synthetic_dataset,

                generator_name=f"RealBlockGenerator_{self.method}",

                output_dir=output_dir,

                target_column=self.target_column,

                block_column=self.block_column

            )

            self.logger.info("Block dataset report and visualizations saved to: %s", output_dir)

        except Exception as e:

            self.logger.error("Failed to generate block report: %s", e, exc_info=True)



    def analyze_block_statistics(self, synthetic_dataset: pd.DataFrame) -> Dict[str, Any]:

        """Analyzes and returns statistics for each block in the synthetic dataset compared to the original."""

        block_stats: Dict[Any, Any] = {}



        for block_id in self.blocks:

            original_block = self.original_data[self.original_data[self.block_column] == block_id]

            synthetic_block = synthetic_dataset[synthetic_dataset[self.block_column] == block_id]



            stats = {

                'original_size': len(original_block),

                'synthetic_size': len(synthetic_block),

                'original_target_dist': None,

                'synthetic_target_dist': None

            }



            # Target distribution if available

            if self.target_column and self.target_column in original_block.columns:

                stats['original_target_dist'] = original_block[self.target_column].value_counts().to_dict()

                stats['synthetic_target_dist'] = synthetic_block[self.target_column].value_counts().to_dict()



            # Numeric stats

            numeric_cols = original_block.select_dtypes(include=[np.number]).columns

            numeric_cols = [c for c in numeric_cols if c != self.block_column]

            if numeric_cols:

                stats['original_numeric_means'] = original_block[numeric_cols].mean().to_dict()

                stats['synthetic_numeric_means'] = synthetic_block[numeric_cols].mean().to_dict()

                stats['original_numeric_stds'] = original_block[numeric_cols].std().to_dict()

                stats['synthetic_numeric_stds'] = synthetic_block[numeric_cols].std().to_dict()



            block_stats[block_id] = stats



        return block_stats



    def get_block_info(self) -> Dict[str, Any]:

        """Returns detailed information about the blocks in the original dataset."""

        block_info: Dict[Any, Any] = {}



        for block_id in self.blocks:

            block_data = self.original_data[self.original_data[self.block_column] == block_id]

            info = {

                'size': len(block_data),

                'percentage': (len(block_data) / len(self.original_data) * 100.0) if len(self.original_data) else 0.0,

                'target_distribution': None,

                'feature_means': None,

                'missing_values': int(block_data.isnull().sum().sum())

            }



            if self.target_column and self.target_column in block_data.columns:

                info['target_distribution'] = block_data[self.target_column].value_counts().to_dict()



            numeric_cols = block_data.select_dtypes(include=[np.number]).columns

            numeric_cols = [c for c in numeric_cols if c != self.block_column]

            if numeric_cols:

                info['feature_means'] = block_data[numeric_cols].mean().to_dict()



            block_info[block_id] = info



        return block_info



    def save_block_dataset(self,

                           synthetic_dataset: pd.DataFrame,

                           output_path: str,

                           format: str = 'csv',

                           separate_blocks: bool = False) -> Union[str, List[str]]:

        """
        Saves the synthetic block dataset to one or more files.

        Args:
            synthetic_dataset (pd.DataFrame): The complete synthetic dataset.
            output_path (str): The output path (file or directory).
            format (str): The output format ('csv', 'parquet', 'excel').
            separate_blocks (bool): If True, saves each block as a separate file.

        Returns:
            Union[str, List[str]]: Path(s) to the saved file(s).
        """

        output = Path(output_path)



        if separate_blocks:

            # Prepare dir/name/ext

            if output.is_file():

                out_dir = output.parent

                base = output.stem

                ext = output.suffix

            else:

                out_dir = output

                base = f"synthetic_block_{self.method}"

                ext = f".{format}"



            out_dir.mkdir(parents=True, exist_ok=True)

            saved: List[str] = []



            for block_id in self.blocks:

                block_data = synthetic_dataset[synthetic_dataset[self.block_column] == block_id]

                block_path = out_dir / f"{base}_block_{block_id}{ext}"



                if format.lower() == 'csv':

                    block_data.to_csv(block_path, index=False)

                elif format.lower() == 'parquet':

                    block_data.to_parquet(block_path, index=False)

                elif format.lower() in ['xlsx', 'excel']:

                    block_data.to_excel(block_path, index=False)

                else:

                    raise ValueError(f"Unsupported format: {format}")



                self.logger.info("Block %s saved to: %s", str(block_id), str(block_path))

                saved.append(str(block_path))



            return saved



        # Save full dataset
        if format.lower() == 'csv':
            synthetic_dataset.to_csv(output, index=False)
        elif format.lower() == 'parquet':
            synthetic_dataset.to_parquet(output, index=False)
        elif format.lower() in ['xlsx', 'excel']:
            synthetic_dataset.to_excel(output, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        return str(output)