import os
import pandas as pd
import numpy as np
from typing import Literal, Optional, List, Dict, Any
from sklearn.mixture import GaussianMixture
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OrdinalEncoder
from collections import Counter
from .RealReporter import RealReporter
try:
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
    from sdv.metadata import SingleTableMetadata
    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False
    print("⚠️ SDV not installed. CTGAN and Copula methods will be unavailable.")


class RealGenerator(RealReporter):
    def __init__(self, dataset_path: str, target_col: str = "target"):
        """
        Initialize the RealGenerator with the dataset path and target column.
        Loads the dataset, separates predictors and target column.
        """
        self.dataset_path = dataset_path
        self.df = pd.read_csv(dataset_path)
        self.target_col = target_col

        # Ensure target column exists in the dataframe
        if self.target_col not in self.df.columns:
            raise ValueError(f"The dataset must contain a column named '{self.target_col}'.")

        # Separate features (X) and target (y)
        self.X = self.df.drop(columns=[self.target_col])
        self.y = self.df[self.target_col]

        # Impute missing values in the features
        self.X = self._impute_missing(self.X)

        if SDV_AVAILABLE:
            self.metadata = SingleTableMetadata()
            self.metadata.detect_from_dataframe(data=self.df)

    def _impute_missing(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in the dataframe, filling categorical columns with mode and numerical columns with mean.
        """
        X_imputed = pd.DataFrame()
        for col in X.columns:
            if X[col].dtype == "object":
                X_imputed[col] = X[col].fillna(X[col].mode()[0])
            else:
                X_imputed[col] = X[col].fillna(X[col].mean())
        return X_imputed

    def generate(
        self,
        output_path: str,
        filename: str,
        n_samples: int,
        method: Literal["resample", "smote", "gmm", "ctgan", "copula"] = "resample",
        random_state: Optional[int] = None,
        id_cols: Optional[List[str]] = None,
        method_params: Optional[Dict[str, Any]] = None,
        balance: bool = False
    ) -> str:
        """
        Generate synthetic data based on a real dataset.
        """
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, filename)
        method_params = method_params or {}

        X, y = self.X.copy(), self.y.copy()

        # Generate data based on the selected method
        if method == "resample":
            result = self._resample(X, y, n_samples, random_state)
        elif method == "smote":
            result = self._apply_smote(X, y, n_samples, random_state, method_params)
        elif method == "gmm":
            result = self._apply_gmm(X, y, n_samples, random_state, method_params)
        elif method == "ctgan":
            result = self._apply_ctgan(n_samples, method_params)
        elif method == "copula":
            result = self._apply_copula(n_samples, method_params)
        else:
            raise ValueError("Invalid method. Choose 'resample', 'smote', 'gmm', 'ctgan', or 'copula'.")

        # Balance the dataset if requested
        if balance:
            result = self._balance_classes(result, n_samples, random_state)

        # Fill missing values with the mode
        result = result.fillna(result.mode().iloc[0])

        # Save the generated data to CSV
        result.to_csv(full_path, index=False)
        print(f"✅ Generated {len(result)} samples using method='{method}' and saved at: {full_path}")

        # Report the dataset quality
        self._report_real_dataset(
            real_df=self.df,
            synthetic_df=result,
            target_col=self.target_col,
            method=method,
            extra_info={"Balance": balance}
        )

        return full_path

    def _resample(self, X: pd.DataFrame, y: pd.Series, n_samples: int, random_state: Optional[int]):
        """Resample data by sampling with replacement."""
        sampled_idx = X.sample(n=n_samples, replace=True, random_state=random_state).index
        return pd.concat([X.loc[sampled_idx], y.loc[sampled_idx]], axis=1)

    def _apply_smote(self, X: pd.DataFrame, y: pd.Series, n_samples: int, random_state: Optional[int], method_params: Dict[str, Any]):
        """Apply SMOTE to balance the classes."""
        cat_cols = X.select_dtypes(include=["object"]).columns
        num_cols = [c for c in X.columns if c not in cat_cols]

        enc = OrdinalEncoder()
        X_cat = enc.fit_transform(X[cat_cols]) if len(cat_cols) > 0 else np.empty((len(X), 0))
        X_num = X[num_cols].to_numpy()

        X_combined = np.hstack([X_num, X_cat])
        sm = SMOTE(random_state=random_state, **method_params)
        X_res, y_res = sm.fit_resample(X_combined, y)

        result = pd.DataFrame(X_res[:, :len(num_cols)], columns=num_cols)
        if len(cat_cols) > 0:
            decoded_cats = enc.inverse_transform(X_res[:, len(num_cols):])
            for i, col in enumerate(cat_cols):
                result[col] = decoded_cats[:, i]

        result[self.target_col] = y_res
        return result

    def _apply_gmm(self, X: pd.DataFrame, y: pd.Series, n_samples: int, random_state: Optional[int], method_params: Dict[str, Any]):
        """Apply Gaussian Mixture Model to generate synthetic data."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        cat_cols = X.select_dtypes(exclude=[np.number]).columns

        gmm = GaussianMixture(
            n_components=method_params.get("n_components", 10),
            covariance_type=method_params.get("covariance_type", "full"),
            random_state=random_state
        )
        gmm.fit(X[numeric_cols])

        X_new, labels = gmm.sample(n_samples)
        synthetic_X = pd.DataFrame(X_new, columns=numeric_cols)

        for col in cat_cols:
            synthetic_X[col] = np.random.choice(X[col], size=n_samples)

        synthetic_X[self.target_col] = np.random.choice(y, size=n_samples)
        return synthetic_X

    def _apply_ctgan(self, n_samples: int, method_params: Dict[str, Any]):
        """Generate synthetic data using CTGAN."""
        if not SDV_AVAILABLE:
            raise ImportError("SDV is not installed. Please install with `pip install sdv`.")

        synthesizer = CTGANSynthesizer(self.metadata, **method_params)
        synthesizer.fit(self.df)
        return synthesizer.sample(n_samples)

    def _apply_copula(self, n_samples: int, method_params: Dict[str, Any]):
        """Generate synthetic data using Gaussian Copula."""
        if not SDV_AVAILABLE:
            raise ImportError("SDV is not installed. Please install with `pip install sdv`.")

        synthesizer = GaussianCopulaSynthesizer(self.metadata, **method_params)
        synthesizer.fit(self.df)
        return synthesizer.sample(n_samples)

    def _balance_classes(self, result: pd.DataFrame, n_samples: int, random_state: Optional[int]):
        """Balance the classes in the dataset if requested."""
        classes = result[self.target_col].unique()
        n_classes = len(classes)
        target_per_class = n_samples // n_classes

        balanced_data = []
        for cls in classes:
            subset = result[result[self.target_col] == cls]
            if len(subset) < target_per_class:
                resampled = subset.sample(target_per_class, replace=True, random_state=random_state)
            else:
                resampled = subset.sample(target_per_class, replace=False, random_state=random_state)
            balanced_data.append(resampled)

        result = pd.concat(balanced_data)

        # Adjust if the resulting dataset is smaller than n_samples
        if len(result) < n_samples:
            diff = n_samples - len(result)
            extra = result.sample(diff, replace=True, random_state=random_state)
            result = pd.concat([result, extra])

        # Shuffle the dataset
        result = result.sample(frac=1, random_state=random_state).reset_index(drop=True)
        print(f"⚖️ Balanced dataset with {len(result)} samples: {dict(Counter(result[self.target_col]))}")

        return result
