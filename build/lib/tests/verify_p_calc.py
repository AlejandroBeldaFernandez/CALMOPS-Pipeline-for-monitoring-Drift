import sys
import os
import pandas as pd
import numpy as np
import logging
import math
from sklearn.tree import DecisionTreeClassifier

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the module to test
from calmops.IPIP.modules.default_train_retrain import default_retrain, IpipModel


def calculate_expected_p_b(n_expired_train, prop_minor_frac):
    """Helper to calculate expected p and b based on R logic."""
    np_val = round(n_expired_train * prop_minor_frac)

    # Calculate p
    if n_expired_train > 1 and np_val > 0:
        denom_p = math.log(1 - 1 / n_expired_train) * np_val
        if denom_p != 0:
            p = math.ceil(math.log(0.01) / denom_p)
        else:
            p = 5
    else:
        p = 5

    # Calculate b
    if np_val > 1:
        denom_b = math.log(1 - 1 / np_val) * np_val
        if denom_b != 0:
            b = math.ceil(math.log(0.01) / denom_b)
        else:
            b = 10
    else:
        b = 10

    return p, b, np_val


def run_test_case(
    case_name, n_expired_total, n_discharge_total, prop_minor_frac, val_size=0.2
):
    print(f"\n=== Running Test Case: {case_name} ===")
    logger = logging.getLogger(f"test_{case_name}")
    logger.setLevel(logging.INFO)

    # 1. Setup Data
    # We need to consider that default_train splits data.
    # We can't control the exact split perfectly due to randomness, but we can check if results are reasonable
    # or try to force a specific split count by using a large enough dataset or checking the actual split inside the function (hard to do without modifying code).
    # Instead, we will calculate the expected p based on the *likely* train set size (stratified split).

    n_expired_train_expected = int(round(n_expired_total * (1 - val_size)))

    # Construct data
    data = {
        "feature1": np.random.rand(n_expired_total + n_discharge_total),
        "target": [1] * n_expired_total + [0] * n_discharge_total,
        "block": ["2"] * (n_expired_total + n_discharge_total),
    }
    df = pd.DataFrame(data)
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    X = df[["feature1", "block"]]
    y = df["target"]

    # Expected values
    expected_p, expected_b, np_val = calculate_expected_p_b(
        n_expired_train_expected, prop_minor_frac
    )

    print(
        f"Inputs: n_expired_total={n_expired_total}, val_size={val_size}, prop_minor_frac={prop_minor_frac}"
    )
    print(f"Expected Train Size (approx): n_expired_train={n_expired_train_expected}")
    print(
        f"Calculated Params: np_val={np_val}, Expected p={expected_p}, Expected b={expected_b}"
    )

    # 2. Real Model
    model_instance = DecisionTreeClassifier(max_depth=1, random_state=42)
    prev_model = IpipModel(ensembles=[])

    # 3. Call default_retrain
    ipip_config = {
        "prop_minor_frac": prop_minor_frac,
        "majority_prop": 0.55,
        "val_size": val_size,
    }

    try:
        final_model, _, _, results = default_retrain(
            X=X,
            y=y,
            last_processed_file="dummy.csv",
            model_path=prev_model,
            random_state=42,
            logger=logger,
            output_dir="tmp_test_output",
            block_col="block",
            ipip_config=ipip_config,
            model_instance=model_instance,
        )

        actual_p = results.get("p")
        # b is not always returned in the top level results dict in default_retrain,
        # but let's check if we can verify it or if we just verify p as requested primarily.
        # The results dict from default_retrain has 'p' and 'num_ensembles'.
        # 'b' is used internally to limit ensemble size.
        # However, default_train returns 'b' in its results, and default_retrain saves 'retrain_results.json' which might have it?
        # Looking at code: default_retrain returns `results` which has `p` but NOT explicitly `b` in the dictionary keys at the end.
        # Wait, I added `p` to results in my previous edit? No, I only modified the calculation logic.
        # Let's check the return dictionary in default_retrain.
        # It returns: "p": p, "num_ensembles": ...
        # It does NOT return 'b'. So we can only verify 'p' easily from the return value.

        print(f"Actual p: {actual_p}")

        if actual_p == expected_p:
            print("✅ SUCCESS: p matches.")
            return True
        else:
            # Allow for small deviation if split wasn't exactly as expected due to rounding/randomness
            # Recalculate p for n_expired_train +/- 1
            p_minus, _, _ = calculate_expected_p_b(
                n_expired_train_expected - 1, prop_minor_frac
            )
            p_plus, _, _ = calculate_expected_p_b(
                n_expired_train_expected + 1, prop_minor_frac
            )

            if actual_p in [p_minus, p_plus]:
                print(
                    f"⚠️ ACCEPTABLE: p={actual_p} matches expected p for +/- 1 sample deviation (expected {expected_p})."
                )
                return True
            else:
                print(f"❌ FAILURE: p mismatch. Expected {expected_p}, got {actual_p}")
                return False

    except Exception as e:
        print(f"❌ ERROR: {e}")
        # import traceback
        # traceback.print_exc()
        return False


def test_p_calculation_scenarios():
    logging.basicConfig(level=logging.ERROR)  # Reduce noise

    scenarios = [
        # Case 1: Standard (Small)
        {"name": "Standard Small", "n_expired": 20, "n_discharge": 80, "prop": 0.75},
        # Case 2: Larger dataset
        {"name": "Larger Data", "n_expired": 100, "n_discharge": 400, "prop": 0.75},
        # Case 3: Different prop_minor_frac
        {"name": "High Prop", "n_expired": 50, "n_discharge": 200, "prop": 0.90},
        # Case 4: Low Prop
        {"name": "Low Prop", "n_expired": 50, "n_discharge": 200, "prop": 0.50},
        # Case 5: Very small (Edge case -> p=5)
        {"name": "Edge Small", "n_expired": 5, "n_discharge": 20, "prop": 0.75},
    ]

    failures = 0
    for sc in scenarios:
        success = run_test_case(
            sc["name"], sc["n_expired"], sc["n_discharge"], sc["prop"]
        )
        if not success:
            failures += 1

    if failures == 0:
        print("\n🎉 ALL TEST CASES PASSED!")
        sys.exit(0)
    else:
        print(f"\n💀 {failures} TEST CASES FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    test_p_calculation_scenarios()
