import os
import shutil
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from calmops.IPIP.pipeline_ipip import run_pipeline
from calmops.IPIP.ipip_model import IpipModel
from sklearn.ensemble import RandomForestClassifier

# Setup paths
TEST_DIR = Path("/tmp/test_ipip_fidelity")
DATA_DIR = TEST_DIR / "data"
PIPELINE_NAME = "test_ipip_fidelity"
PIPELINES_ROOT = TEST_DIR / "pipelines"

# Mock get_pipelines_root to return our test dir
import calmops.utils

calmops.utils.get_pipelines_root = lambda: PIPELINES_ROOT


def setup_module(module):
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    DATA_DIR.mkdir(parents=True)
    (PIPELINES_ROOT / "pipelines" / PIPELINE_NAME).mkdir(parents=True)


def teardown_module(module):
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)


def create_dummy_data():
    # Create a dummy dataset with 3 blocks
    # Block 1: Train
    # Block 2: Train/Eval
    # Block 3: Eval

    data = []
    for block in [1, 2, 3]:
        for _ in range(50):
            # Simple pattern: x1 > 0.5 -> class 1
            x1 = np.random.rand()
            x2 = np.random.rand()
            y = 1 if x1 > 0.5 else 0
            # Add some noise to make probabilities interesting
            if np.random.rand() < 0.1:
                y = 1 - y
            data.append({"x1": x1, "x2": x2, "y": y, "block": block})

    df = pd.DataFrame(data)
    # Save as CSV
    df.to_csv(DATA_DIR / "data.csv", index=False)
    return df


def create_preprocess_file():
    content = """
import pandas as pd
def data_preprocessing(df):
    X = df[['x1', 'x2', 'block']]
    y = df['y']
    return X, y
"""
    path = TEST_DIR / "preprocess.py"
    with open(path, "w") as f:
        f.write(content)
    return str(path)


def test_ipip_metrics_fidelity():
    create_dummy_data()
    preprocess_file = create_preprocess_file()

    # Run pipeline
    run_pipeline(
        pipeline_name=PIPELINE_NAME,
        data_dir=str(DATA_DIR),
        preprocess_file=preprocess_file,
        model_instance=RandomForestClassifier(n_estimators=10),
        random_state=42,
        block_col="block",
        ipip_config={"prop_majoritaria": 0.55},
        target_file="data.csv",
    )

    # Check outputs
    metrics_dir = PIPELINES_ROOT / "pipelines" / PIPELINE_NAME / "metrics"

    # 1. Check predictions.csv
    preds_path = metrics_dir / "predictions.csv"
    assert preds_path.exists(), "predictions.csv should exist"

    preds_df = pd.read_csv(preds_path)
    print("Predictions columns:", preds_df.columns)

    assert "y_pred_proba" in preds_df.columns, (
        "y_pred_proba should be in predictions.csv"
    )
    assert "y_true_numeric" in preds_df.columns, (
        "y_true_numeric should be in predictions.csv"
    )

    # Check if probabilities are valid (0-1)
    assert preds_df["y_pred_proba"].between(0, 1).all(), (
        "Probabilities should be between 0 and 1"
    )

    # 2. Check eval_results.json for ROC AUC
    eval_path = metrics_dir / "eval_results.json"
    assert eval_path.exists(), "eval_results.json should exist"

    import json

    with open(eval_path, "r") as f:
        results = json.load(f)

    assert "roc_auc" in results["metrics"], "Global ROC AUC should be in metrics"
    print("Global ROC AUC:", results["metrics"]["roc_auc"])

    # Check per-block ROC AUC
    blocks_metrics = results["blocks"]["per_block_metrics_full"]
    for block, metrics in blocks_metrics.items():
        assert "roc_auc" in metrics, f"ROC AUC should be in metrics for block {block}"
        print(f"Block {block} ROC AUC:", metrics["roc_auc"])


if __name__ == "__main__":
    test_ipip_metrics_fidelity()
