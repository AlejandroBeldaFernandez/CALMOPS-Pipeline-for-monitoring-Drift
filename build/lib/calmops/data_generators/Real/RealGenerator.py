#!/usr/bin/env python3
"""
Real Data Generator - Advanced data synthesis for real datasets.

This module provides the RealGenerator class, which serves as a powerful tool for
synthesizing data that mimics the characteristics of a real-world dataset. It integrates
several synthesis methods, from classic statistical approaches to modern deep learning models.

Key Features:
- **Multiple Synthesis Methods**: Supports a variety of methods including:
  - `cart`: FCS-like method using Decision Trees.
  - `rf`: FCS-like method using Random Forests.
  - `lgbm`: FCS-like method using LightGBM.
  - `gmm`: Gaussian Mixture Models (for numeric data).
  - `ctgan`, `tvae`, `copula`: Advanced deep learning models from the Synthetic Data Vault (SDV) library.
  - `datasynth`: Correlated attribute mode synthesis from the DataSynthesizer library.
  - `resample`: Simple resampling with replacement.
- **Conditional Synthesis**: Can generate data that follows custom-defined distributions for specified columns.
- **Target Balancing**: Automatically balances the distribution of the target variable.
- **Date Injection**: Capable of adding a timestamp column with configurable start dates and steps.
- **Comprehensive Reporting**: Automatically generates a detailed quality report comparing the synthetic data to the original, including visualizations and statistical metrics.
"""

import logging
import pandas as pd
import numpy as np
import warnings
from typing import Optional, Dict, Union, Any
from datetime import datetime
import os
import math
import tempfile

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# SDV for advanced synthesis
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, CopulaGANSynthesizer
from sdv.metadata import SingleTableMetadata
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator

# Model imports
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.mixture import GaussianMixture
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImblearnPipeline
import lightgbm as lgb

# Custom logger and reporter
from calmops.logger.logger import get_logger
from data_generators.Real.RealReporter import RealReporter

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.get_logger().setLevel("ERROR")


class RealGenerator:
    """
    A class for advanced data synthesis from a real dataset, offering multiple generation methods and detailed reporting.
    """

    def __init__(
        self,
        original_data: pd.DataFrame,
        method: str = "cart",
        target_column: Optional[str] = None,
        block_column: Optional[str] = None,
        auto_report: bool = True,
        logger: Optional[logging.Logger] = None,
        random_state: Optional[int] = None,
        balance_target: bool = False,
        model_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the RealGenerator.

        Args:
            original_data (pd.DataFrame): The real dataset to be synthesized.
            method (str): The synthesis method to use. See class docstring for options.
            target_column (Optional[str]): The name of the target variable column.
            block_column (Optional[str]): The name of the column defining data blocks.
            auto_report (bool): If True, automatically generates a quality report after synthesis.
            logger (Optional[logging.Logger]): An external logger instance. If None, a new one is created.
            random_state (Optional[int]): Seed for random number generation for reproducibility.
            balance_target (bool): If True, balances the distribution of the target column.
            model_params (Optional[Dict[str, Any]]): A dictionary of hyperparameters for the chosen synthesis model.
        """
        self.original_data = original_data
        self.method = method
        self.target_column = target_column
        self.block_column = block_column
        self.auto_report = auto_report
        self.logger = logger if logger else get_logger("RealGenerator")
        self.random_state = random_state
        self.balance_target = balance_target
        self.rng = np.random.default_rng(random_state)
        self.reporter = RealReporter()

        # --- Set default parameters ---
        defaults = {
            "cart_iterations": 10,
            "cart_min_samples_leaf": None,
            "rf_n_estimators": None,
            "rf_min_samples_leaf": None,
            "lgbm_n_estimators": None,
            "lgbm_learning_rate": None,
            "gmm_n_components": 5,
            "gmm_covariance_type": "full",
            "sdv_epochs": 300,
            "sdv_batch_size": 100,
            "ds_k": 5,
        }

        # --- Merge user-provided params with defaults ---
        params = defaults.copy()
        if model_params:
            params.update(model_params)

        # --- Store synthesis parameters as instance attributes ---
        self.cart_iterations = params["cart_iterations"]
        self.cart_min_samples_leaf = params["cart_min_samples_leaf"]
        self.rf_n_estimators = params["rf_n_estimators"]
        self.rf_min_samples_leaf = params["rf_min_samples_leaf"]
        self.lgbm_n_estimators = params["lgbm_n_estimators"]
        self.lgbm_learning_rate = params["lgbm_learning_rate"]
        self.gmm_n_components = params["gmm_n_components"]
        self.gmm_covariance_type = params["gmm_covariance_type"]
        self.sdv_epochs = params["sdv_epochs"]
        self.sdv_batch_size = params["sdv_batch_size"]
        self.ds_k = params["ds_k"]

        self.metadata = None
        self.synthesizer = None

        valid_methods = [
            "cart",
            "rf",
            "lgbm",
            "gmm",
            "ctgan",
            "tvae",
            "copula",
            "datasynth",
            "resample",
        ]
        if self.method not in valid_methods:
            raise ValueError(
                f"Unknown synthesis method '{self.method}'. Valid methods are: {valid_methods}"
            )

    def _build_metadata(self, data: pd.DataFrame) -> SingleTableMetadata:
        """Builds SDV metadata from a DataFrame."""
        self.logger.info("Building SDV metadata...")
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data)
        return metadata

    def _get_synthesizer(
        self,
    ) -> Union[CTGANSynthesizer, TVAESynthesizer, CopulaGANSynthesizer]:
        """Initializes and returns the appropriate SDV synthesizer based on the method."""
        self.metadata = self._build_metadata(self.original_data)
        if self.method == "ctgan":
            return CTGANSynthesizer(
                metadata=self.metadata,
                epochs=self.sdv_epochs,
                batch_size=self.sdv_batch_size,
                verbose=True,
            )
        elif self.method == "tvae":
            return TVAESynthesizer(
                metadata=self.metadata,
                epochs=self.sdv_epochs,
                batch_size=self.sdv_batch_size,
            )
        elif self.method == "copula":
            return CopulaGANSynthesizer(
                metadata=self.metadata,
                epochs=self.sdv_epochs,
                batch_size=self.sdv_batch_size,
                verbose=True,
            )
        else:
            raise ValueError(f"No SDV synthesizer for method '{self.method}'")

    def _validate_custom_distributions(
        self, custom_distributions: Dict, data: pd.DataFrame
    ) -> Dict:
        """Validates and normalizes custom distribution dictionaries."""
        if not isinstance(custom_distributions, dict):
            raise TypeError("custom_distributions must be a dictionary.")
        validated_distributions = custom_distributions.copy()
        for col, dist in validated_distributions.items():
            if col not in data.columns:
                raise ValueError(
                    f"Column '{col}' specified in custom_distributions does not exist in the dataset."
                )
            if not isinstance(dist, dict):
                raise TypeError(
                    f"The distribution for column '{col}' must be a dictionary."
                )
            if not dist:
                self.logger.warning(
                    f"Distribution for column '{col}' is empty. It will be ignored."
                )
                continue
            if any(p < 0 for p in dist.values()):
                raise ValueError(f"Proportions for column '{col}' cannot be negative.")
            total_proportion = sum(dist.values())
            if not math.isclose(total_proportion, 1.0):
                self.logger.warning(
                    f"Proportions for column '{col}' do not sum to 1.0 (sum={total_proportion}). They will be normalized."
                )
                validated_distributions[col] = {
                    k: v / total_proportion for k, v in dist.items()
                }
        return validated_distributions

    def _synthesize_sdv(
        self, n_samples: int, custom_distributions: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Synthesizes data using an SDV model, with support for conditional sampling."""
        self.logger.info(
            f"Starting conditional SDV synthesis with method: {self.method}..."
        )
        if self.synthesizer is None:
            self.synthesizer = self._get_synthesizer()
            self.synthesizer.fit(self.original_data)
        if not custom_distributions:
            self.logger.info(
                "No custom distributions provided. Generating samples unconditionally."
            )
            return self.synthesizer.sample(num_rows=n_samples)
        self.logger.info(
            f"Applying custom distributions via conditional sampling: {custom_distributions}"
        )
        all_synth_parts = []
        if len(custom_distributions.keys()) > 1:
            self.logger.warning(
                f"Multiple columns found in custom_distributions. Conditioning on first column: '{next(iter(custom_distributions))}'."
            )
        col_to_condition = (
            self.target_column
            if self.target_column in custom_distributions
            else next(iter(custom_distributions))
        )
        dist = custom_distributions[col_to_condition]
        remaining_samples = n_samples
        for value, proportion in dist.items():
            num_rows_for_val = int(n_samples * proportion)
            if num_rows_for_val > 0 and remaining_samples > 0:
                num_rows_for_val = min(num_rows_for_val, remaining_samples)
                self.logger.info(
                    f"Generating {num_rows_for_val} samples for '{col_to_condition}' = '{value}'"
                )
                try:
                    synth_part = self.synthesizer.sample(
                        num_rows=num_rows_for_val, conditions={col_to_condition: value}
                    )
                    all_synth_parts.append(synth_part)
                    remaining_samples -= len(synth_part)
                except Exception as e:
                    self.logger.warning(
                        f"Could not generate conditional samples for {col_to_condition}='{value}': {e}"
                    )
        if remaining_samples > 0:
            self.logger.info(
                f"Generating {remaining_samples} remaining samples unconditionally."
            )
            all_synth_parts.append(self.synthesizer.sample(num_rows=remaining_samples))
        if not all_synth_parts:
            raise RuntimeError("Conditional synthesis failed to generate any data.")
        return (
            pd.concat(all_synth_parts, ignore_index=True)
            .sample(frac=1, random_state=self.random_state)
            .reset_index(drop=True)
        )

    def _synthesize_resample(
        self, n_samples: int, custom_distributions: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Synthesizes data by resampling from the original dataset, with optional weighting."""
        self.logger.info("Starting synthesis by resampling...")
        if not custom_distributions:
            return self.original_data.sample(
                n=n_samples, replace=True, random_state=self.random_state
            )
        self.logger.info(
            f"Applying custom distributions via weighted resampling: {custom_distributions}"
        )
        self.logger.warning(
            "The 'resample' method with custom distributions changes proportions but does not generate new data."
        )
        col_to_condition = (
            self.target_column
            if self.target_column in custom_distributions
            else next(iter(custom_distributions))
        )
        dist = custom_distributions[col_to_condition]
        weights = pd.Series(0.0, index=self.original_data.index)
        for category, proportion in dist.items():
            weights[self.original_data[col_to_condition] == category] = proportion
        if weights.sum() == 0:
            self.logger.warning(
                "Weights are all zero. Falling back to uniform resampling."
            )
            return self.original_data.sample(
                n=n_samples, replace=True, random_state=self.random_state
            )
        return self.original_data.sample(
            n=n_samples, replace=True, random_state=self.random_state, weights=weights
        )

    def _synthesize_gmm(
        self, n_samples: int, custom_distributions: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Synthesizes data using Gaussian Mixture Models. Only supports numeric data."""
        self.logger.info("Starting GMM synthesis...")
        non_numeric_cols = self.original_data.select_dtypes(exclude=np.number).columns
        if not non_numeric_cols.empty:
            raise ValueError(
                f"The 'gmm' method only supports numeric data, but found non-numeric columns: {list(non_numeric_cols)}."
            )
        gmm = GaussianMixture(
            n_components=self.gmm_n_components,
            covariance_type=self.gmm_covariance_type,
            random_state=self.random_state,
        )
        gmm.fit(self.original_data)
        synth_data, _ = gmm.sample(n_samples)
        synth = pd.DataFrame(synth_data, columns=self.original_data.columns)

        # If the target is supposed to be classification, round the results
        if self.target_column and self.target_column in synth.columns:
            unique_values = self.original_data[self.target_column].nunique()
            if unique_values < 25 or (unique_values / len(self.original_data)) < 0.05:
                self.logger.info(
                    f"Rounding GMM results for target column '{self.target_column}' to nearest integer."
                )
                synth[self.target_column] = (
                    synth[self.target_column].round().astype(int)
                )

        if custom_distributions:
            self.logger.warning(
                "Applying custom distributions to GMM output via post-processing. This may break learned correlations."
            )
            col_to_condition = (
                self.target_column
                if self.target_column in custom_distributions
                else next(iter(custom_distributions))
            )
            dist = custom_distributions[col_to_condition]
            n_synth_samples = len(synth)
            new_values = []
            for value, proportion in dist.items():
                count = int(n_synth_samples * proportion)
                new_values.extend([value] * count)
            if len(new_values) < n_synth_samples:
                new_values.extend(
                    [list(dist.keys())[-1]] * (n_synth_samples - len(new_values))
                )
            self.rng.shuffle(new_values)
            synth[col_to_condition] = new_values[:n_synth_samples]
        return synth

    def _apply_resampling_strategy(self, X, y, custom_dist, n_samples):
        """Applies over/under-sampling to match a custom distribution before model training."""
        try:
            original_counts = y.value_counts().to_dict()
            target_total_size = n_samples
            target_counts = {
                k: int(v * target_total_size) for k, v in custom_dist.items()
            }
            oversampling_strategy = {
                k: v for k, v in target_counts.items() if v > original_counts.get(k, 0)
            }
            undersampling_strategy = {
                k: v for k, v in target_counts.items() if v < original_counts.get(k, 0)
            }
            steps = []
            if oversampling_strategy:
                steps.append(
                    (
                        "o",
                        RandomOverSampler(
                            sampling_strategy=oversampling_strategy,
                            random_state=self.random_state,
                        ),
                    )
                )
            if undersampling_strategy:
                steps.append(
                    (
                        "u",
                        RandomUnderSampler(
                            sampling_strategy=undersampling_strategy,
                            random_state=self.random_state,
                        ),
                    )
                )
            if not steps:
                return X, y
            pipeline = ImblearnPipeline(steps=steps)
            self.logger.info(
                f"Applying resampling pipeline to match distribution for column '{y.name}'."
            )
            X_res, y_res = pipeline.fit_resample(X, y)
            return X_res, y_res
        except Exception as e:
            self.logger.warning(
                f"Could not apply resampling strategy for column '{y.name}': {e}. Using original distribution."
            )
            return X, y

    def _synthesize_cart(
        self,
        data: pd.DataFrame,
        n_samples: int,
        custom_distributions: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """Synthesizes data using a Fully Conditional Specification (FCS) approach with Decision Trees."""
        self.logger.info("Starting CART (FCS-style) synthesis...")
        if custom_distributions:
            self.logger.warning(
                "For 'cart' method, custom distributions are handled by resampling the training data."
            )
        X_real = data.copy()
        for col in X_real.select_dtypes(include=["object"]).columns:
            X_real[col] = X_real[col].astype("category")
        # Sample with replacement and reset index immediately to prevent issues with duplicate indices.
        X_synth = X_real.sample(
            n=n_samples, replace=True, random_state=self.random_state
        ).reset_index(drop=True)
        try:
            for it in range(self.cart_iterations):
                self.logger.info("CART iteration %d/%d", it + 1, self.cart_iterations)
                for col in X_real.columns:
                    y_real_train = X_real[col]
                    Xr_real_train = X_real.drop(columns=col)
                    Xs_synth = X_synth.drop(columns=col)

                    # Determine if the column should be treated as a classification target
                    is_classification = False
                    if not pd.api.types.is_numeric_dtype(y_real_train):
                        is_classification = True
                    # Special heuristic for the main target column
                    elif col == self.target_column:
                        unique_values = y_real_train.nunique()
                        if (
                            unique_values < 25
                            or (unique_values / len(y_real_train)) < 0.05
                        ):
                            is_classification = True

                    model_params = {"random_state": self.random_state}
                    if self.cart_min_samples_leaf is not None:
                        model_params["min_samples_leaf"] = self.cart_min_samples_leaf

                    model = (
                        DecisionTreeClassifier(**model_params)
                        if is_classification
                        else DecisionTreeRegressor(**model_params)
                    )

                    y_to_fit, X_to_fit = (y_real_train, Xr_real_train)
                    if custom_distributions and col in custom_distributions:
                        X_to_fit, y_to_fit = self._apply_resampling_strategy(
                            Xr_real_train,
                            y_real_train,
                            custom_distributions[col],
                            n_samples,
                        )

                    # Encode categorical features for sklearn
                    X_to_fit = X_to_fit.copy()
                    Xs_synth_encoded = Xs_synth.copy()
                    for c in X_to_fit.select_dtypes(include=["category"]).columns:
                        X_to_fit[c] = X_to_fit[c].cat.codes
                        Xs_synth_encoded[c] = Xs_synth_encoded[c].cat.codes

                    try:
                        model.fit(X_to_fit, y_to_fit)
                    except ValueError as e:
                        if "Input contains NaN" in str(e):
                            raise ValueError(
                                f"The '{self.method}' method failed due to NaNs. Please pre-clean data."
                            ) from e
                        else:
                            raise e
                    y_synth = model.predict(Xs_synth_encoded)
                    X_synth[col] = y_synth
            return X_synth
        except Exception as e:
            self.logger.error(f"CART synthesis failed: {e}", exc_info=True)
            return None

    def _synthesize_rf(
        self,
        data: pd.DataFrame,
        n_samples: int,
        custom_distributions: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """Synthesizes data using a Fully Conditional Specification (FCS) approach with Random Forests."""
        self.logger.info("Starting Random Forest (FCS-style) synthesis...")
        if custom_distributions:
            self.logger.warning(
                "For 'rf' method, custom distributions are handled by resampling the training data."
            )
        X_real = data.copy()
        for col in X_real.select_dtypes(include=["object"]).columns:
            X_real[col] = X_real[col].astype("category")
        # Sample with replacement and reset index immediately to prevent issues with duplicate indices.
        X_synth = X_real.sample(
            n=n_samples, replace=True, random_state=self.random_state
        ).reset_index(drop=True)
        try:
            for it in range(self.cart_iterations):
                self.logger.info("RF iteration %d/%d", it + 1, self.cart_iterations)
                for col in X_real.columns:
                    y_real_train = X_real[col]
                    Xr_real_train = X_real.drop(columns=col)
                    Xs_synth = X_synth.drop(columns=col)

                    # Determine if the column should be treated as a classification target
                    is_classification = False
                    if not pd.api.types.is_numeric_dtype(y_real_train):
                        is_classification = True
                    # Special heuristic for the main target column
                    elif col == self.target_column:
                        unique_values = y_real_train.nunique()
                        if (
                            unique_values < 25
                            or (unique_values / len(y_real_train)) < 0.05
                        ):
                            is_classification = True

                    model_params = {"random_state": self.random_state, "n_jobs": 1}
                    if self.rf_n_estimators is not None:
                        model_params["n_estimators"] = self.rf_n_estimators
                    if self.rf_min_samples_leaf is not None:
                        model_params["min_samples_leaf"] = self.rf_min_samples_leaf

                    if is_classification:
                        model = RandomForestClassifier(**model_params)
                    else:
                        model = RandomForestRegressor(**model_params)

                    y_to_fit, X_to_fit = (y_real_train, Xr_real_train)
                    if custom_distributions and col in custom_distributions:
                        X_to_fit, y_to_fit = self._apply_resampling_strategy(
                            Xr_real_train,
                            y_real_train,
                            custom_distributions[col],
                            n_samples,
                        )

                    # Encode categorical features for sklearn
                    X_to_fit = X_to_fit.copy()
                    Xs_synth_encoded = Xs_synth.copy()
                    for c in X_to_fit.select_dtypes(include=["category"]).columns:
                        X_to_fit[c] = X_to_fit[c].cat.codes
                        Xs_synth_encoded[c] = Xs_synth_encoded[c].cat.codes

                    try:
                        model.fit(X_to_fit, y_to_fit)
                    except ValueError as e:
                        if "Input contains NaN" in str(e):
                            raise ValueError(
                                f"The '{self.method}' method failed due to NaNs. Please pre-clean data."
                            ) from e
                        else:
                            raise e
                    y_synth = model.predict(Xs_synth_encoded)
                    X_synth[col] = y_synth
            return X_synth
        except Exception as e:
            self.logger.error(f"RF synthesis failed: {e}", exc_info=True)
            return None

    def _synthesize_lgbm(
        self,
        data: pd.DataFrame,
        n_samples: int,
        custom_distributions: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """Synthesizes data using a Fully Conditional Specification (FCS) approach with LightGBM."""
        self.logger.info("Starting LightGBM (FCS-style) synthesis...")
        if custom_distributions:
            self.logger.warning(
                "For 'lgbm' method, custom distributions are handled by resampling the training data."
            )
        X_real = data.copy()
        # Sample with replacement and reset index immediately to prevent issues with duplicate indices.
        X_synth = X_real.sample(
            n=n_samples, replace=True, random_state=self.random_state
        ).reset_index(drop=True)

        # Ensure categorical columns have the same categories defined.
        cat_cols = X_real.select_dtypes(include="category").columns
        for col in cat_cols:
            X_synth[col] = pd.Categorical(
                X_synth[col], categories=X_real[col].cat.categories
            )

        try:
            for it in range(self.cart_iterations):
                self.logger.info("LGBM iteration %d/%d", it + 1, self.cart_iterations)
                for col in X_real.columns:
                    y_real_train = X_real[col]
                    Xr_real_train = X_real.drop(columns=col)
                    Xs_synth = X_synth.drop(columns=col)

                    # Determine if the column should be treated as a classification target
                    is_classification = False
                    if not pd.api.types.is_numeric_dtype(y_real_train):
                        is_classification = True
                    # Special heuristic for the main target column
                    elif col == self.target_column:
                        unique_values = y_real_train.nunique()
                        if (
                            unique_values < 25
                            or (unique_values / len(y_real_train)) < 0.05
                        ):
                            is_classification = True

                    model_params = {
                        "random_state": self.random_state,
                        "n_jobs": 1,
                        "verbose": -1,
                    }
                    if self.lgbm_n_estimators is not None:
                        model_params["n_estimators"] = self.lgbm_n_estimators
                    if self.lgbm_learning_rate is not None:
                        model_params["learning_rate"] = self.lgbm_learning_rate

                    if is_classification:
                        model = lgb.LGBMClassifier(**model_params)
                    else:
                        model = lgb.LGBMRegressor(**model_params)

                    y_to_fit, X_to_fit = (y_real_train, Xr_real_train)
                    if custom_distributions and col in custom_distributions:
                        X_to_fit, y_to_fit = self._apply_resampling_strategy(
                            Xr_real_train,
                            y_real_train,
                            custom_distributions[col],
                            n_samples,
                        )
                    try:
                        model.fit(X_to_fit, y_to_fit)
                    except ValueError as e:
                        if "Input contains NaN" in str(e):
                            raise ValueError(
                                f"The '{self.method}' method failed due to NaNs. Please pre-clean data."
                            ) from e
                        else:
                            raise e
                    y_synth = model.predict(Xs_synth)
                    if y_real_train.dtype.name == "category":
                        y_synth = pd.Categorical(
                            y_synth, categories=y_real_train.cat.categories
                        )
                    X_synth[col] = y_synth
            return X_synth
        except Exception as e:
            self.logger.error(f"LGBM synthesis failed: {e}", exc_info=True)
            return None

    def _synthesize_datasynth(
        self, n_samples: int, custom_distributions: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Synthesizes data using DataSynthesizer in correlated attribute mode.
        Uses a secure temporary directory to avoid issues with file paths.
        """
        self.logger.info("Starting DataSynthesizer synthesis...")

        # Use tempfile.TemporaryDirectory() to create a unique and secure directory.
        # The 'with' block ensures the directory and its contents are removed upon exit.
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_csv_path = os.path.join(temp_dir, "temp_data.csv")
            temp_description_file = os.path.join(temp_dir, "description.json")

            try:
                # Save the original data to the secure temporary location.
                self.original_data.to_csv(temp_csv_path, index=False)

                # Describe the dataset
                describer = DataDescriber()
                describer.describe_dataset_in_correlated_attribute_mode(
                    dataset_file=temp_csv_path, k=self.ds_k
                )

                # Save the DataDescriber object for the DataGenerator to read.
                describer.save_dataset_description_to_file(temp_description_file)

                # Generate the dataset
                generator = DataGenerator()
                synth = generator.generate_dataset_in_correlated_attribute_mode(
                    n_samples=n_samples, description_file=temp_description_file
                )

                # Apply custom distributions (Post-processing)
                if custom_distributions:
                    self.logger.warning(
                        "Applying custom distributions to DataSynthesizer output via post-processing."
                    )
                    col_to_condition = (
                        self.target_column
                        if self.target_column in custom_distributions
                        else next(iter(custom_distributions))
                    )
                    dist = custom_distributions[col_to_condition]
                    n_synth_samples = len(synth)

                    # Resample values to match custom distribution
                    new_values = []
                    for value, proportion in dist.items():
                        count = int(n_synth_samples * proportion)
                        new_values.extend([value] * count)

                    # Fill if necessary and shuffle
                    if len(new_values) < n_synth_samples:
                        new_values.extend(
                            [list(dist.keys())[-1]]
                            * (n_synth_samples - len(new_values))
                        )

                    self.rng.shuffle(new_values)
                    synth[col_to_condition] = new_values[:n_synth_samples]

                return synth

            except Exception as e:
                self.logger.error(f"Synthesis with method 'datasynth' failed: {e}")
                # The 'with' statement will handle cleanup, just re-raise the error.
                raise e

    def _inject_dates(
        self,
        df: pd.DataFrame,
        date_col: str,
        date_start: Optional[str],
        date_every: int,
        date_step: Optional[Dict[str, int]],
    ) -> pd.DataFrame:
        """Injects a date column into the DataFrame with specified frequency and step."""
        if date_start is None:
            return df
        if not isinstance(date_every, int) or date_every <= 0:
            raise ValueError(f"date_every must be a positive integer, got {date_every}")
        step = date_step or {"days": 1}
        valid_keys = {
            "years",
            "months",
            "weeks",
            "days",
            "hours",
            "minutes",
            "seconds",
            "microseconds",
            "nanoseconds",
        }
        if set(step.keys()) - valid_keys:
            raise ValueError(f"Invalid date_step keys: {set(step.keys()) - valid_keys}")
        try:
            start_ts = pd.to_datetime(date_start)
        except Exception as e:
            raise ValueError(f"Invalid date_start '{date_start}': {e}") from e
        total = len(df)
        if total == 0:
            df[date_col] = pd.Series(dtype="datetime64[ns]")
            return df
        periods = (total + date_every - 1) // date_every
        anchors = [start_ts + pd.DateOffset(**step) * i for i in range(periods)]
        series = (
            pd.Series(anchors).repeat(date_every).iloc[:total].reset_index(drop=True)
        )
        if date_col not in df.columns:
            df.insert(0, date_col, series)
        else:
            df[date_col] = series
        self.logger.info(
            f"[RealGenerator] Injected date column '{date_col}' starting at {start_ts}."
        )
        return df

    def synthesize(
        self,
        n_samples: int,
        output_dir: str,
        custom_distributions: Optional[Dict] = None,
        date_start: Optional[str] = None,
        date_every: int = 1,
        date_step: Optional[Dict[str, int]] = None,
        date_col: str = "timestamp",
        save_dataset: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        The main public method to generate synthetic data.

        Args:
            n_samples (int): The number of synthetic samples to generate.
            output_dir (str): Directory to save the report and dataset.
            custom_distributions (Optional[Dict]): A dictionary to specify custom distributions for columns.
            date_start (Optional[str]): Start date for date injection (e.g., '2023-01-01').
            date_every (int): Generate a new date every N rows.
            date_step (Optional[Dict[str, int]]): The time step for date injection (e.g., {'days': 1}).
            date_col (str): The name of the date column to be injected.
            save_dataset (bool): If True, saves the generated dataset to a CSV file.

        Returns:
            Optional[pd.DataFrame]: The generated synthetic DataFrame, or None if synthesis fails.
        """
        self.logger.info(
            f"Starting synthesis of {n_samples} samples using method '{self.method}'..."
        )
        if custom_distributions:
            custom_distributions = self._validate_custom_distributions(
                custom_distributions, self.original_data
            )
        if (
            self.balance_target
            and self.target_column
            and (
                custom_distributions is None
                or self.target_column not in custom_distributions
            )
        ):
            self.logger.info(
                f"'balance_target' is True. Generating balanced distribution for '{self.target_column}'."
            )
            target_classes = self.original_data[self.target_column].unique()
            custom_distributions = custom_distributions or {}
            custom_distributions[self.target_column] = {
                c: 1 / len(target_classes) for c in target_classes
            }
        try:
            synth = None
            if self.method in ["ctgan", "tvae", "copula"]:
                synth = self._synthesize_sdv(
                    n_samples, custom_distributions=custom_distributions
                )
            elif self.method == "resample":
                synth = self._synthesize_resample(
                    n_samples, custom_distributions=custom_distributions
                )
            elif self.method == "cart":
                synth = self._synthesize_cart(
                    self.original_data,
                    n_samples,
                    custom_distributions=custom_distributions,
                )
            elif self.method == "rf":
                synth = self._synthesize_rf(
                    self.original_data,
                    n_samples,
                    custom_distributions=custom_distributions,
                )
            elif self.method == "lgbm":
                synth = self._synthesize_lgbm(
                    self.original_data,
                    n_samples,
                    custom_distributions=custom_distributions,
                )
            elif self.method == "gmm":
                synth = self._synthesize_gmm(
                    n_samples, custom_distributions=custom_distributions
                )
            elif self.method == "datasynth":
                synth = self._synthesize_datasynth(
                    n_samples, custom_distributions=custom_distributions
                )

            if synth is not None:
                self.logger.info(f"Successfully synthesized {len(synth)} samples.")
                synth = self._inject_dates(
                    df=synth,
                    date_col=date_col,
                    date_start=date_start,
                    date_every=date_every,
                    date_step=date_step,
                )
                if self.auto_report:
                    self.reporter.generate_comprehensive_report(
                        real_df=self.original_data,
                        synthetic_df=synth,
                        generator_name=f"RealGenerator_{self.method}",
                        output_dir=output_dir,
                        target_column=self.target_column,
                        time_col=date_col,
                    )

                # Save the generated dataset for inspection
                if save_dataset:  # Only save if save_dataset is True
                    try:
                        save_path = os.path.join(
                            output_dir, f"synthetic_data_{self.method}.csv"
                        )
                        synth.to_csv(save_path, index=False)
                        self.logger.info(
                            f"Generated synthetic dataset saved to: {save_path}"
                        )
                    except Exception as e:
                        self.logger.error(f"Failed to save synthetic dataset: {e}")

                return synth
            else:
                self.logger.error(
                    f"Synthesis method '{self.method}' failed to generate data."
                )
                return None
        except Exception as e:
            self.logger.error(
                f"Synthesis with method '{self.method}' failed: {e}", exc_info=True
            )
            return None
