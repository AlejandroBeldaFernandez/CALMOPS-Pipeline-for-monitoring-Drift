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
    print(" SDV not installed. CTGAN and Copula methods will be unavailable.")


class RealGenerator(RealReporter):
    def __init__(self, target_col: str = "target", df:pd.DataFrame = None, dataset_path: str  = None ):
        self.dataset_path = dataset_path
        if df is None: 
            self.df = pd.read_csv(dataset_path)
        else:
            self.df = df.copy()
        self.target_col = target_col

        if self.target_col not in self.df.columns:
            raise ValueError(f"The dataset must contain a column named '{self.target_col}'.")

        # Separate features and target
        self.X = self.df.drop(columns=[self.target_col])
        self.y = self.df[self.target_col]

        # Impute missing values
        self.X = self._impute_missing(self.X)

        if SDV_AVAILABLE:
            self.metadata = SingleTableMetadata()
            self.metadata.detect_from_dataframe(data=self.df)

    def _impute_missing(self, X: pd.DataFrame) -> pd.DataFrame:
        X_imputed = pd.DataFrame()
        for col in X.columns:
            if X[col].dtype == "object":
                X_imputed[col] = X[col].fillna(X[col].mode()[0])
            else:
                X_imputed[col] = X[col].fillna(X[col].mean())
        return X_imputed

    def _validate_params(self, output_path: str, filename: str, n_samples: int, method: str, id_cols: Optional[List[str]] = None, method_params: Optional[Dict[str, Any]] = None, random_state: Optional[int] = None):
        """
        Validates input parameters for generating synthetic data.
        Raises informative errors if any validation fails.
        """
        # Validate output path and filename
        if not output_path or not filename:
            raise ValueError("output_path and filename must be provided.")
        if not os.path.exists(output_path):
            raise ValueError(f"The output_path '{output_path}' does not exist.")

        
        if not os.access(output_path, os.W_OK):
            raise PermissionError(f"Output directory '{output_path}' is not writable.")

        # Validate n_samples
        if not isinstance(n_samples, int):
            raise ValueError("n_samples must be an integer.")
        if n_samples <= 0:
            raise ValueError("n_samples must be greater than 0.")

        # Validate method
        valid_methods = ["resample", "smote", "gmm", "ctgan", "copula"]
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Choose one of {valid_methods}.")

        # Validate id_cols
        if id_cols:
            missing_cols = [c for c in id_cols if c not in self.df.columns]
            if missing_cols:
                raise ValueError(f"ID columns not found in dataset: {missing_cols}")
            if self.target_col in id_cols:
                raise ValueError(f"target_col '{self.target_col}' cannot be included in id_cols.")

        # Validate method_params depending on method
        method_params = method_params or {}
        if method == "smote":
            # SMOTE requires at least 2 samples per class
            class_counts = self.y.value_counts()
            if (class_counts < 2).any():
                raise ValueError(f"SMOTE requires at least 2 samples per class. Found classes with counts: {class_counts[class_counts < 2].to_dict()}")
        elif method == "gmm":
            n_components = method_params.get("n_components", 10)
            if n_components > len(self.X):
                raise ValueError(f"gmm 'n_components' ({n_components}) cannot exceed the number of samples ({len(self.X)}).")

        # Validate random_state
        if random_state is not None and not isinstance(random_state, int):
            raise ValueError("random_state must be an integer or None.")


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
        self._validate_params(
        output_path=output_path,
        filename=filename,
        n_samples=n_samples,
        method=method,
        id_cols=id_cols,
        method_params=method_params,
        random_state=random_state
        )

        
        full_path = os.path.join(output_path, filename)
        method_params = method_params or {}

        # Exclude ID columns from synthesis
        X, y = self.X.copy(), self.y.copy()
        if id_cols:
            X_ids = X[id_cols]
            X = X.drop(columns=id_cols)
        else:
            X_ids = None

        # Generate synthetic data
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

        # Restore ID columns
        if X_ids is not None:
            for col in X_ids.columns:
                result[col] = np.random.choice(X_ids[col], size=len(result))

        # Balance classes if requested
        if balance:
            result = self._balance_classes(result, n_samples, random_state)

        # Fill missing values
        result = result.fillna(result.mode().iloc[0])

        # Save CSV
        result.to_csv(full_path, index=False)
        print(f" Generated {len(result)} samples using method='{method}' at: {full_path}")

        # Report dataset quality
        self._report_real_dataset(
            real_df=self.df,
            synthetic_df=result,
            target_col=self.target_col,
            method=method,
            extra_info={"Balance": balance}
        )

        return full_path

    # ---------------- GENERATION METHODS ---------------- #
    def _resample(self, X: pd.DataFrame, y: pd.Series, n_samples: int, random_state: Optional[int]):
        sampled_idx = X.sample(n=n_samples, replace=True, random_state=random_state).index
        return pd.concat([X.loc[sampled_idx], y.loc[sampled_idx]], axis=1)

    def _apply_smote(self, X: pd.DataFrame, y: pd.Series, n_samples: int, random_state: Optional[int], method_params: Dict[str, Any]):
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
        if not SDV_AVAILABLE:
            raise ImportError("SDV is not installed. Install with `pip install sdv`.")
        synthesizer = CTGANSynthesizer(self.metadata, **method_params)
        synthesizer.fit(self.df)
        return synthesizer.sample(n_samples)

    def _apply_copula(self, n_samples: int, method_params: Dict[str, Any]):
        if not SDV_AVAILABLE:
            raise ImportError("SDV is not installed. Install with `pip install sdv`.")
        synthesizer = GaussianCopulaSynthesizer(self.metadata, **method_params)
        synthesizer.fit(self.df)
        return synthesizer.sample(n_samples)

    # ---------------- BALANCING ---------------- #
    def _balance_classes(self, result: pd.DataFrame, n_samples: int, random_state: Optional[int]):
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

        if len(result) < n_samples:
            diff = n_samples - len(result)
            extra = result.sample(diff, replace=True, random_state=random_state)
            result = pd.concat([result, extra])

        result = result.sample(frac=1, random_state=random_state).reset_index(drop=True)
        print(f" Balanced dataset with {len(result)} samples: {dict(Counter(result[self.target_col]))}")
        return result