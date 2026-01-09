# Data Generators

This module contains a set of data generators designed to create synthetic and realistic datasets to simulate various scenarios in MLOps, especially those involving data drift and concept drift.

## Structure

The data generator module is organized into the following folders:

-   `Clinic/`: Generator for synthetic clinical data, including demographic, gene expression, and protein data.
-   `DriftInjection/`: A tool for injecting a wide variety of drifts into existing datasets.
-   `Dynamics/`: A module for evolving features over time and constructing dynamic target variables.
-   `Real/`: Generators that synthesize data mimicking the characteristics of a real-world dataset.
-   `Synthetic/`: Generators based on the `river` library to create synthetic datasets with different types of drifts.

## Generators

### 1. `ClinicGenerator`

Located in `calmops/data_generators/Clinic/Clinic.py`, this generator specializes in the creation of synthetic clinical data.

**Key Features:**

-   **Multi-omics Data:** Generates demographic, gene expression (simulating RNA-Seq or Microarray), and protein data.
-   **Correlations:** Allows defining complex correlations between different variables, including internal group correlations (fixed or random ranges) and demographic-omics correlations.
-   **Target Variable:** Can generate a target variable (e.g., diagnosis) based on a linear combination of features.
-   **Longitudinal Simulation:** Capable of simulating data over time, for example, for longitudinal studies.
-   **Drift Injection:** Can simulate drifts by transitioning patients between groups (e.g., from "control" to "disease").

For detailed documentation and a tutorial, see the `Clinic/` directory.

### 2. `DriftInjector`

Located in `calmops/data_generators/DriftInjection/DriftInjector.py`, this is a powerful tool for introducing drifts into an existing pandas `DataFrame`.

**Key Features:**

-   **Drift Types:** Supports a wide variety of drifts, including:
    -   **Feature Drift:** Gaussian noise, shift, scale.
    -   **Label Drift:** Random label flipping.
    -   **Concept Drift:** Changes in the target distribution.
-   **Drift Profiles:** Allows simulating gradual, abrupt, incremental, and recurrent drifts.
-   **Flexibility:** Drift can be applied to the entire dataset, specific blocks, or row ranges.

### 3. `RealGenerator` and `RealBlockGenerator`

Located in `calmops/data_generators/Real/`, these generators create synthetic data that mimics a real dataset.

-   **`RealGenerator`**: Uses a variety of synthesis methods, from simple `resampling` to advanced `deep learning` models (via the `SDV` library).
-   **`RealBlockGenerator`**: Extends `RealGenerator` to work with block-structured data. It is ideal for:
    -   Generating data for each block separately.
    -   Dynamically creating blocks (by size or `timestamp`).
    -   Scheduling drifts across blocks.

### 4. `SyntheticGenerator` and `SyntheticBlockGenerator`

Located in `calmops/data_generators/Synthetic/`, these generators are based on the `river` library to create synthetic datasets.

-   **`GeneratorFactory`**: A factory class that standardizes the creation of `river` generators.
-   **`SyntheticGenerator`**: The main generator that uses `river` generators to create data with different types of drift (gradual, abrupt, etc.).
-   **`SyntheticBlockGenerator`**: A high-level abstraction for generating block-structured datasets, where each block can be generated with different parameters to simulate changes in the environment.

## Usage (Unified API)

All generators (`RealGenerator`, `SyntheticGenerator`, `ClinicGenerator`) now implement a unified `generate()` interface.

### Example: Synthetic Generator

```python
from calmops.data_generators.Synthetic import SyntheticGenerator
from calmops.data_generators.configs import DateConfig

from river.datasets import synth

# Initialize (Lightweight)
gen = SyntheticGenerator(random_state=42)
sea_stream = synth.SEA(variant=0)

# Generate with Drift Injection
df = gen.generate(
    generator_instance=sea_stream,
    n_samples=1000,
    drift_injection_config=[
        {"method": "inject_feature_drift", "params": {"feature_cols": ["col1"], "drift_magnitude": 0.5}}
    ],
    date_config=DateConfig(start_date="2024-01-01")
)
```

### Example: Real Generator

```python
from calmops.data_generators.Real import RealGenerator

gen = RealGenerator(data=original_df, method="cart")
df_new = gen.generate(n_samples=500)
```