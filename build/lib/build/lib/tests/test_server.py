import unittest
import json
import os
import shutil
import tempfile
import joblib
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

# Ensure the project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from calmops.server import app, LOADED_MODELS


class TestServer(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory to act as the project root
        self.test_dir = tempfile.mkdtemp()
        self.pipeline_name = "test_pipeline"
        self.pipeline_dir = os.path.join(self.test_dir, "pipelines", self.pipeline_name)

        # We don't need models dir if we inject into cache, but we need config/logs
        os.makedirs(os.path.join(self.pipeline_dir, "config"), exist_ok=True)
        os.makedirs(os.path.join(self.pipeline_dir, "logs"), exist_ok=True)

        # Create a dummy model and inject into cache
        self.model = MagicMock()
        self.model.predict.return_value = np.array([1, 0])
        LOADED_MODELS[self.pipeline_name] = self.model

        # Create a dummy preprocessor script
        self.preprocess_path = os.path.join(self.pipeline_dir, "preprocess.py")
        with open(self.preprocess_path, "w") as f:
            f.write("def data_preprocessing(df):\n    return df, None")

        # Create a dummy config file
        self.config_path = os.path.join(
            self.pipeline_dir, "config", "runner_config.json"
        )
        with open(self.config_path, "w") as f:
            json.dump({"preprocess_file": self.preprocess_path}, f)

        self.app = app.test_client()
        self.app.testing = True

        # Patch get_project_root globally for all tests
        self.patcher = patch("calmops.server.get_project_root")
        self.mock_get_root = self.patcher.start()
        self.mock_get_root.return_value = Path(self.test_dir)

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.test_dir)
        # Clean up cache
        if self.pipeline_name in LOADED_MODELS:
            del LOADED_MODELS[self.pipeline_name]

    def test_predict_success(self):
        data = [{"col1": 1, "col2": 2}, {"col1": 3, "col2": 4}]
        response = self.app.post(
            f"/predict/{self.pipeline_name}",
            data=json.dumps(data),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("predictions", response.json)
        self.assertEqual(response.json["predictions"], [1, 0])

    def test_predict_pipeline_not_found(self):
        response = self.app.post(
            "/predict/non_existent_pipeline",
            data=json.dumps([{"a": 1}]),
            content_type="application/json",
        )

        # Should fail because model/config won't be found
        self.assertEqual(response.status_code, 500)
        self.assertIn("error", response.json)

    def test_predict_no_json(self):
        response = self.app.post(f"/predict/{self.pipeline_name}", data="not json")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["error"], "Request must be JSON")

    def test_predict_empty_data(self):
        response = self.app.post(
            f"/predict/{self.pipeline_name}",
            data=json.dumps([]),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
