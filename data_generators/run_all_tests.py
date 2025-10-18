import os
import datetime
import pandas as pd
import numpy as np
import unittest

# Import all the necessary generators and injectors
from data_generators.Synthetic.SyntheticGenerator import SyntheticGenerator
from data_generators.Synthetic.SyntheticBlockGenerator import SyntheticBlockGenerator
from data_generators.Synthetic.GeneratorFactory import GeneratorFactory, GeneratorType, GeneratorConfig
from data_generators.Real.RealGenerator import RealGenerator
from data_generators.Real.RealBlockGenerator import RealBlockGenerator
from data_generators.DriftInjection.DriftInjector import DriftInjector

"""
Test Suite for Data Generators

This script contains a series of integration tests for the data generation and drift injection
modules in the CalmOps project. It is designed to be run as a standalone script to verify
the core functionalities of the data generators.

Each test function is decorated with `@run_test`, which handles the creation of a unique,
timestamped output directory for that test run, ensuring that artifacts from different
tests and different runs are kept separate.

Key Tests:
- `test_integration_01_drift_injection`: Verifies that `DriftInjector` works correctly with both
  `SyntheticGenerator` and `SyntheticBlockGenerator` outputs.
- `test_integration_02_timestamp_features`: Checks the timestamp injection functionality in both
  single and block-based generators.
- `test_real_01_block_generator`: Tests the `RealBlockGenerator`, including dynamic chunking,
  date injection, and drift scheduling.
- `test_real_02_all_methods`: Iterates through all synthesis methods available in `RealGenerator`
  to ensure they run without errors.
- `test_real_03_drift_injection_suite`: A comprehensive test of all drift types supported by `DriftInjector`.
- `test_synthetic_01_generator_features`: Tests various River generator types and drift simulations
  in `SyntheticGenerator`.
- `test_synthetic_02_block_generator`: Tests the `SyntheticBlockGenerator`, including simulating
  concept drift by varying parameters between blocks.

To Run:
    python -m data_generators.run_all_tests
"""

BASE_OUTPUT_DIR = "test_outputs"

def run_test(test_func):
    """Decorator to create a unique test directory and handle exceptions."""
    def wrapper():
        test_name = test_func.__name__
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(BASE_OUTPUT_DIR, test_name, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        print(f"--- Running test: {test_name} ---")
        print(f"Output directory: {output_dir}")
        try:
            test_func(output_dir)
            print(f"--- SUCCESS: {test_name} ---")
        except Exception as e:
            print(f"--- FAILURE: {test_name} failed with error: {e} ---")
    return wrapper

@run_test
def test_integration_01_drift_injection(output_dir):
    """Integration tests for combining synthetic generators with the DriftInjector."""
    rng = np.random.default_rng(42)
    synthetic_generator = SyntheticGenerator(random_state=42)
    block_generator = SyntheticBlockGenerator()
    factory = GeneratorFactory()

    # Test 1: SyntheticGenerator with gradual feature drift
    test_output_dir_1 = os.path.join(output_dir, "synthetic_feature_drift")
    os.makedirs(test_output_dir_1, exist_ok=True)
    base_gen = factory.create_generator(GeneratorType.AGRAWAL, GeneratorConfig(random_state=42))
    clean_df = synthetic_generator.generate(
        generator_instance=base_gen,
        output_path=test_output_dir_1,
        filename="clean_data.csv",
        n_samples=500,
        save_dataset=False
    )
    drift_injector = DriftInjector(
        original_df=clean_df,
        output_dir=test_output_dir_1,
        generator_name="feature_drift_test"
    )
    drifted_df = drift_injector.inject_feature_drift_gradual(
        df=clean_df,
        feature_cols=["salary"],
        drift_type="shift",
        drift_magnitude=1.5,
        start_index=250,
        width=200
    )
    assert isinstance(drifted_df, pd.DataFrame)

    # Test 2: SyntheticBlockGenerator with virtual drift (missing values)
    test_output_dir_2 = os.path.join(output_dir, "block_virtual_drift")
    os.makedirs(test_output_dir_2, exist_ok=True)
    clean_df_2 = block_generator.generate_blocks_simple(
        output_path=test_output_dir_2,
        filename="clean_blocks.csv",
        n_blocks=4,
        total_samples=400,
        methods='sea',
        random_state=42
    )
    drift_injector_2 = DriftInjector(
        original_df=clean_df_2,
        output_dir=test_output_dir_2,
        generator_name="virtual_drift_test",
        block_column="block"
    )
    drifted_df_2 = drift_injector_2.inject_missing_values_drift(
        df=clean_df_2,
        feature_cols=["0"],
        missing_fraction=0.9,
        block_index=3
    )
    assert isinstance(drifted_df_2, pd.DataFrame)

    # Test 3: SyntheticBlockGenerator with label shift
    test_output_dir_3 = os.path.join(output_dir, "block_label_shift")
    os.makedirs(test_output_dir_3, exist_ok=True)
    method_params = [
        {'classification_function': 0}, {'classification_function': 0}, {'classification_function': 0}
    ]
    clean_df_3 = block_generator.generate_blocks_simple(
        output_path=test_output_dir_3,
        filename="clean_blocks_for_label_shift.csv",
        n_blocks=3,
        total_samples=600,
        methods='agrawal',
        method_params=method_params,
        random_state=42
    )
    drift_injector_3 = DriftInjector(
        original_df=clean_df_3,
        output_dir=test_output_dir_3,
        generator_name="label_shift_test",
        target_column="target",
        block_column="block"
    )
    drifted_df_3 = drift_injector_3.inject_label_shift(
        df=clean_df_3,
        target_col="target",
        target_distribution={0: 0.9, 1: 0.1},
        block_index=2
    )
    assert isinstance(drifted_df_3, pd.DataFrame)

@run_test
def test_integration_02_timestamp_features(output_dir):
    """Test suite for timestamp generation functionalities."""
    synthetic_generator = SyntheticGenerator(random_state=42)
    block_generator = SyntheticBlockGenerator()
    factory = GeneratorFactory()

    # Test 1: SyntheticGenerator with timestamp
    test_output_dir_1 = os.path.join(output_dir, "synthetic_with_timestamp")
    os.makedirs(test_output_dir_1, exist_ok=True)
    base_gen = factory.create_generator(GeneratorType.SEA, GeneratorConfig(random_state=42))
    output_path = synthetic_generator.generate(
        generator_instance=base_gen,
        output_path=test_output_dir_1,
        filename="ts_synthetic.csv",
        n_samples=100,
        date_start="2023-01-01 09:00:00",
        date_every=10,
        date_step={"minutes": 15}
    )
    assert os.path.exists(output_path)

    # Test 2: SyntheticBlockGenerator with timestamp
    test_output_dir_2 = os.path.join(output_dir, "block_with_timestamp")
    os.makedirs(test_output_dir_2, exist_ok=True)
    output_path_2 = block_generator.generate_blocks_simple(
        output_path=test_output_dir_2,
        filename="ts_blocks.csv",
        n_blocks=4,
        total_samples=400,
        methods='sea',
        random_state=42,
        date_start="2023-10-01",
        date_step={"days": 7}
    )
    assert os.path.exists(output_path_2)

    # Test 3: Drift and timestamp together
    test_output_dir_3 = os.path.join(output_dir, "drift_and_timestamp")
    os.makedirs(test_output_dir_3, exist_ok=True)
    base_gen_3 = factory.create_generator(GeneratorType.AGRAWAL, GeneratorConfig(classification_function=0, random_state=42))
    drift_gen_3 = factory.create_generator(GeneratorType.AGRAWAL, GeneratorConfig(classification_function=1, random_state=42))
    output_path_3 = synthetic_generator.generate(
        generator_instance=base_gen_3,
        drift_generator=drift_gen_3,
        output_path=test_output_dir_3,
        filename="drift_with_ts.csv",
        n_samples=300,
        drift_type="abrupt",
        position_of_drift=150,
        date_start="2023-01-01",
        date_every=1
    )
    assert os.path.exists(output_path_3)

@run_test
def test_real_01_block_generator(output_dir):
    """Test suite for RealBlockGenerator using a sample numeric dataset."""
    try:
        source_data = pd.read_csv("temp_ds_42/temp_data.csv")
        source_df = source_data.sample(n=100, random_state=42).reset_index(drop=True)
    except FileNotFoundError:
        print("Skipping test_real_01_block_generator: temp_ds_42/temp_data.csv not found.")
        return

    # Test 1: Basic block generation by size
    test_output_dir_1 = os.path.join(output_dir, "generate_basic_blocks_by_size")
    os.makedirs(test_output_dir_1, exist_ok=True)
    block_generator = RealBlockGenerator(
        original_data=source_df,
        chunk_size=20,
        target_column='target',
        random_state=42
    )
    synthetic_dataset = block_generator.generate_block_dataset(
        output_dir=test_output_dir_1
    )
    assert isinstance(synthetic_dataset, pd.DataFrame)

    # Test 2: Date injection per block
    test_output_dir_2 = os.path.join(output_dir, "date_injection_per_block")
    os.makedirs(test_output_dir_2, exist_ok=True)
    block_generator_2 = RealBlockGenerator(
        original_data=source_df,
        chunk_size=20,
        target_column='target',
        random_state=42
    )
    synthetic_dataset_2 = block_generator_2.generate_block_dataset(
        output_dir=test_output_dir_2,
        date_start="2023-01-01",
        date_step={"days": 1}
    )
    assert isinstance(synthetic_dataset_2, pd.DataFrame)

    # Test 3: Feature drift schedule across blocks
    test_output_dir_3 = os.path.join(output_dir, "generate_with_feature_drift_schedule")
    os.makedirs(test_output_dir_3, exist_ok=True)
    drift_schedule = [
        {
            "mode": "gradual",
            "drift_type": "shift",
            "feature_cols": ["bmi"],
            "drift_magnitude": 0.8,
            "block_start": 2,
            "n_blocks": 2,
        }
    ]
    block_generator_3 = RealBlockGenerator(
        original_data=source_df,
        chunk_size=20,
        target_column='target',
        random_state=42
    )
    synthetic_dataset_3 = block_generator_3.generate_block_dataset(
        output_dir=test_output_dir_3,
        drift_schedule=drift_schedule
    )
    assert isinstance(synthetic_dataset_3, pd.DataFrame)

@run_test
def test_real_02_all_methods(output_dir):
    """Test all available synthesis methods from RealGenerator."""
    try:
        diabetes_data = pd.read_csv("temp_ds_42/temp_data.csv").sample(n=100, random_state=42)
    except FileNotFoundError:
        print("Skipping test_real_02_all_methods: temp_ds_42/temp_data.csv not found.")
        return

    def create_synthetic_dataset(n_samples=100):
        np.random.seed(42)
        data = {
            "age": np.random.randint(18, 65, n_samples),
            "city": np.random.choice(["New York", "London", "Paris", "Tokyo"], n_samples),
            "income": np.random.normal(50000, 15000, n_samples).round(2),
            "illness": np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        }
        return pd.DataFrame(data)

    synthetic_data = create_synthetic_dataset()

    def run_sub_test(method, data, data_name, **kwargs):
        init_keys = ['target_column', 'balance_target', 'model_params']
        synth_keys = ['custom_distributions', 'date_start', 'date_every', 'date_step', 'date_col']
        init_kwargs = {k: v for k, v in kwargs.items() if k in init_keys}
        synth_kwargs = {k: v for k, v in kwargs.items() if k in synth_keys}
        test_name_parts = [data_name, method]
        for k, v in sorted(kwargs.items()):
            if k == 'custom_distributions':
                test_name_parts.append('custom_distributions')
            else:
                test_name_parts.append(str(k))
        test_name = "_".join(test_name_parts)
        sub_test_output_dir = os.path.join(output_dir, test_name)
        os.makedirs(sub_test_output_dir, exist_ok=True)
        print(f"--- Running sub-test: {test_name} ---")
        try:
            generator = RealGenerator(
                original_data=data,
                method=method,
                auto_report=True,
                random_state=42,
                **init_kwargs
            )
            synth_data = generator.synthesize(
                n_samples=50,
                output_dir=sub_test_output_dir,
                **synth_kwargs
            )
            assert synth_data is not None
            assert not synth_data.empty
        except Exception as e:
            print(f"FAILURE: {test_name} failed with error: {e}")

    methods = ["cart", "rf", "lgbm", "gmm", "ctgan", "tvae", "copula", "datasynth", "resample"]
    for method in methods:
        if method in ["cart", "rf", "lgbm", "gmm"]:
            run_sub_test(method, diabetes_data, "diabetes", target_column="target")
        elif method != 'gmm':
            run_sub_test(method, synthetic_data, "synthetic", target_column="illness")

@run_test
def test_real_03_drift_injection_suite(output_dir):
    """A comprehensive test to generate a dataset and inject all supported types of drift."""
    N_SAMPLES = 400
    RANDOM_STATE = 42

    def setup_datasets():
        np.random.seed(RANDOM_STATE)
        data = {
            "age": np.random.randint(20, 70, N_SAMPLES),
            "city": np.random.choice(["New York", "London", "Paris", "Tokyo"], N_SAMPLES, p=[0.4, 0.3, 0.2, 0.1]),
            "income": np.random.normal(50000, 15000, N_SAMPLES).round(2),
            "illness": np.random.choice([0, 1], N_SAMPLES, p=[0.8, 0.2])
        }
        data['block'] = np.arange(N_SAMPLES) // 40
        synthetic_df = pd.DataFrame(data)
        base_numeric_df = pd.DataFrame(np.random.rand(N_SAMPLES, 4), columns=['num1', 'num2', 'num3', 'num4'])
        base_numeric_df['block'] = np.arange(N_SAMPLES) // 40
        return synthetic_df, base_numeric_df

    synthetic_df, numeric_df = setup_datasets()

    injector_synth = DriftInjector(
        original_df=synthetic_df, 
        output_dir=os.path.join(output_dir, "base_synth"), 
        generator_name="synthetic_base",
        target_column='illness',
        block_column='block',
        random_state=RANDOM_STATE
    )
    injector_numeric = DriftInjector(
        original_df=numeric_df,
        output_dir=os.path.join(output_dir, "base_numeric"),
        generator_name="numeric_base",
        block_column='block',
        random_state=RANDOM_STATE
    )

    # Feature Drift Tests
    test_feature_drift_dir = os.path.join(output_dir, "feature_drift")
    os.makedirs(test_feature_drift_dir, exist_ok=True)
    injector_synth.inject_feature_drift_gradual(
        df=synthetic_df.copy(), feature_cols=['age', 'income'], drift_type='scale',
        drift_magnitude=0.8, center=200, width=100, profile='sigmoid'
    )

@run_test
def test_synthetic_01_generator_features(output_dir):
    """Test suite for SyntheticGenerator, covering various generator types and drift."""
    generator = SyntheticGenerator(random_state=42)
    factory = GeneratorFactory()

    # Test 1: Various generator types
    test_output_dir_1 = os.path.join(output_dir, "test_various_generator_types")
    os.makedirs(test_output_dir_1, exist_ok=True)
    generator_types = [GeneratorType.AGRAWAL, GeneratorType.SEA, GeneratorType.HYPERPLANE, GeneratorType.STAGGER]
    for gen_type in generator_types:
        sub_dir = os.path.join(test_output_dir_1, gen_type.name)
        generator_instance = factory.create_generator(gen_type, GeneratorConfig(random_state=42))
        output_path = generator.generate(
            generator_instance=generator_instance,
            output_path=sub_dir,
            filename=f"data_{gen_type.name}.csv",
            n_samples=100
        )
        assert os.path.exists(output_path)

    # Test 2: Abrupt drift simulation
    test_output_dir_2 = os.path.join(output_dir, "test_abrupt_drift")
    base_gen = factory.create_generator(GeneratorType.AGRAWAL, GeneratorConfig(classification_function=0, random_state=42))
    drift_gen = factory.create_generator(GeneratorType.AGRAWAL, GeneratorConfig(classification_function=1, random_state=42))
    output_path_2 = generator.generate(
        generator_instance=base_gen,
        drift_generator=drift_gen,
        output_path=test_output_dir_2,
        filename="abrupt_drift.csv",
        n_samples=200,
        drift_type="abrupt",
        position_of_drift=100
    )
    assert os.path.exists(output_path_2)

@run_test
def test_synthetic_02_block_generator(output_dir):
    """Test suite for SyntheticBlockGenerator."""
    block_generator = SyntheticBlockGenerator()

    # Test 1: Simple block generation
    test_output_dir_1 = os.path.join(output_dir, "simple_block_generation")
    output_path = block_generator.generate_blocks_simple(
        output_path=test_output_dir_1,
        filename="simple_blocks.csv",
        n_blocks=3,
        total_samples=300,
        methods='agrawal',
        method_params={},
        random_state=42
    )
    assert os.path.exists(output_path)

    # Test 2: Drift by changing parameters between blocks
    test_output_dir_2 = os.path.join(output_dir, "drift_by_changing_block_parameters")
    method_params = [
        {'classification_function': 0},  # Block 1
        {'classification_function': 1},  # Block 2 (Drift)
        {'classification_function': 1}   # Block 3 (Stays drifted)
    ]
    output_path_2 = block_generator.generate_blocks_simple(
        output_path=test_output_dir_2,
        filename="drift_blocks.csv",
        n_blocks=3,
        total_samples=300,
        methods='agrawal',
        method_params=method_params,
        random_state=42
    )
    assert os.path.exists(output_path_2)

def main():
    """Run all tests defined in this script."""
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    test_integration_01_drift_injection()
    test_integration_02_timestamp_features()
    test_real_01_block_generator()
    test_real_02_all_methods()
    test_real_03_drift_injection_suite()
    test_synthetic_01_generator_features()
    test_synthetic_02_block_generator()

if __name__ == "__main__":
    main()