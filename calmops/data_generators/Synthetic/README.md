# `SyntheticGenerator` Documentation

The `Synthetic` generator module provides tools to generate synthetic tabular datasets (batch) by consuming data streams from the `river` library. It supports simulating various types of concept drift.

## Generators

### 1. `SyntheticGenerator`
A wrapper around `river` datasets (e.g., SEA, AGRAWAL, STAGGER) that consumes the stream to generate a static dataset (pandas DataFrame or CSV) with configurable drift (abrupt, gradual, etc.).

### 2. `SyntheticBlockGenerator`
A high-level abstraction for generating block-structured datasets. It allows:
- Defining a sequence of blocks.
- Applying different generation methods or parameters per block.
- Simulating complex drift scenarios over time.

## Installation

Ensure `calmops` and `river` are installed.

## Basic Usage (`SyntheticGenerator`)

```python
from calmops.data_generators.Synthetic.SyntheticGenerator import SyntheticGenerator

# Initialize generator (e.g., SEA dataset)
generator = SyntheticGenerator(method="sea", seed=42)

# Generate data
data = generator.generate(n_samples=1000, method_params={'function': 1})
```

## Block Usage (`SyntheticBlockGenerator`)

```python
from calmops.data_generators.Synthetic.SyntheticBlockGenerator import SyntheticBlockGenerator

block_gen = SyntheticBlockGenerator()

# Generate blocks with different concepts
block_gen.generate_blocks_simple(
    output_path="synthetic_blocks",
    filename="data.csv",
    n_blocks=3,
    total_samples=3000,
    methods="sea",
    method_params=[
        {"function": 1}, # Block 1: Concept 1
        {"function": 2}, # Block 2: Concept 2 (Abrupt Drift)
        {"function": 1}  # Block 3: Concept 1 (Recurrent Drift)
    ]
)
```

## Tutorial

For a complete example, run the `tutorial.py` script in this directory:

```bash
python tutorial.py
```
