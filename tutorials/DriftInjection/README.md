# `DriftInjector` Documentation

The `DriftInjector` is a flexible tool designed to inject various types of data drift into existing pandas DataFrames. It supports feature drift, label drift, and concept drift, with customizable profiles like abrupt, gradual, and incremental changes.

## Installation

The `DriftInjector` is part of the `calmops` package. Ensure `calmops` is installed.

## Basic Usage

```python
from calmops.data_generators.DriftInjection.DriftInjector import DriftInjector
import pandas as pd
import numpy as np

# Create a sample dataframe
df = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 1000),
    'feature2': np.random.normal(5, 2, 1000),
    'target': np.random.randint(0, 2, 1000)
})

# Initialize injector
injector = DriftInjector(df)

# Inject drift
drifted_df = injector.add_drift(
    drift_type='feature_shift',
    column='feature1',
    magnitude=2.0,
    start_idx=500,
    end_idx=1000
)
```

## Drift Types

### Feature Drift
- **Shift:** Adds a constant value to the feature.
- **Scale:** Multiplies the feature by a factor.
- **Noise:** Adds Gaussian noise.

### Label Drift
- **Flip:** Randomly flips labels for classification tasks.

### Concept Drift
- **Target Shift:** Changes the distribution of the target variable.

## Drift Profiles

- **Abrupt:** Sudden change at a specific point.
- **Gradual:** Change happens over a transition period.
- **Incremental:** Change increases linearly over time.

## Tutorial

For a complete example, run the `tutorial.py` script in this directory:

```bash
python tutorial.py
```
