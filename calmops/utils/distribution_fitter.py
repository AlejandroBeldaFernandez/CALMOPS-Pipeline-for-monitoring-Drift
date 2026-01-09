import numpy as np
import pandas as pd
from scipy import stats
import warnings


def fit_distribution(data: pd.Series) -> dict:
    """
    Identifies the best fitting distribution for a given pandas Series.
    Supports both continuous and discrete data.

    Returns:
        dict: {
            "distribution": str (name),
            "params": tuple (fitted parameters),
            "p_value": float (goodness of fit p-value),
            "is_discrete": bool
        }
    """
    clean_data = data.dropna()
    if len(clean_data) < 10:
        return {
            "distribution": "Insufficient Data",
            "params": (),
            "p_value": 0.0,
            "is_discrete": False,
        }

    # 1. Determine if Discrete or Continuous
    # Heuristic: If numeric and few unique values relative to size, or integer type with few uniques
    n_unique = clean_data.nunique()
    is_discrete = False

    if pd.api.types.is_integer_dtype(clean_data) or (
        pd.api.types.is_float_dtype(clean_data)
        and n_unique < 20
        and (clean_data % 1 == 0).all()
    ):
        is_discrete = True

    start_dist = "None"
    best_p = 0.0
    best_params = ()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        if is_discrete:
            # Check Bernoulli (0/1)
            unique_vals = sorted(clean_data.unique())
            if set(unique_vals).issubset({0, 1}):
                p = clean_data.mean()
                return {
                    "distribution": "Bernoulli",
                    "params": (p,),
                    "p_value": 1.0,
                    "is_discrete": True,
                }

            # Check Poisson
            # Poisson lambda = mean
            # We use Chi-Square for goodness of fit
            mu = clean_data.mean()
            # Bin data for Chi-Square
            observed_counts = clean_data.value_counts().sort_index()
            expected_probs = stats.poisson.pmf(observed_counts.index, mu)
            expected_counts = expected_probs * len(clean_data)

            # Normalize to sum to len due to truncation of tail in observed
            expected_counts = expected_counts * (
                len(clean_data) / expected_counts.sum()
            )

            if len(observed_counts) >= 2:
                stat, p = stats.chisquare(observed_counts, f_exp=expected_counts)
                if p > best_p:
                    best_p = p
                    start_dist = "Poisson"
                    best_params = (mu,)

        else:
            # Continuous Distributions
            # Standardize for better fitting stability (sometimes required)
            # but scikit stats fit usually handles scale. We pass raw data.

            distributions = ["norm", "expon", "uniform", "beta", "gamma", "lognorm"]

            for dist_name in distributions:
                dist = getattr(stats, dist_name)

                # Constraints checks
                if dist_name == "lognorm" and (clean_data <= 0).any():
                    continue
                if dist_name == "beta" and (
                    (clean_data <= 0).any() or (clean_data >= 1).any()
                ):
                    # Beta is strictly [0,1], usually (0,1) for fit stability
                    continue
                if dist_name == "gamma" and (clean_data <= 0).any():
                    continue
                if dist_name == "expon" and (clean_data < 0).any():
                    continue

                try:
                    params = dist.fit(clean_data)

                    # KS Test
                    # args for kstest needs to be passed
                    stat, p = stats.kstest(clean_data, dist_name, args=params)

                    if p > best_p:
                        best_p = p
                        start_dist = dist_name
                        best_params = params

                except Exception:
                    continue

    return {
        "distribution": start_dist,
        "params": best_params,
        "p_value": best_p,
        "is_discrete": is_discrete,
    }
