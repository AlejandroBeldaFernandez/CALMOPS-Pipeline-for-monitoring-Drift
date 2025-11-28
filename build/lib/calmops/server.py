import os
import sys
import json
import logging
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from pathlib import Path
import importlib.util

# Add the project root to the Python path
# We will use a more robust way to find the project root if possible, 
# but for now we ensure the current directory's parent is in path to find calmops
current_dir = Path(__file__).resolve().parent
if str(current_dir.parent) not in sys.path:
    sys.path.insert(0, str(current_dir.parent))

from calmops.utils import get_project_root
from calmops.logger.logger import PipelineLogger

app = Flask(__name__)

# Global caches
LOADED_MODELS = {}
LOADED_PREPROCESSORS = {}
LOGGERS = {}

def _get_pipeline_base_dir(pipeline_name: str) -> Path:
    """Constructs the base directory for a given pipeline."""
    return get_project_root() / "pipelines" / pipeline_name

def _get_logger(pipeline_name: str):
    """Retrieves or creates a logger for a specific pipeline."""
    if pipeline_name in LOGGERS:
        return LOGGERS[pipeline_name]
    
    logs_dir = _get_pipeline_base_dir(pipeline_name) / "logs"
    # Ensure logs directory exists, though PipelineLogger might handle it
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    logger = PipelineLogger(f"calmops.server.{pipeline_name}", log_dir=logs_dir).get_logger()
    LOGGERS[pipeline_name] = logger
    return logger

def _load_model(pipeline_name: str):
    """Loads a trained model for a given pipeline with caching."""
    if pipeline_name in LOADED_MODELS:
        return LOADED_MODELS[pipeline_name]

    model_path = _get_pipeline_base_dir(pipeline_name) / "models" / f"{pipeline_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found for pipeline '{pipeline_name}' at {model_path}")

    model = joblib.load(model_path)
    LOADED_MODELS[pipeline_name] = model
    return model

def _load_preprocess_func(pipeline_name: str):
    """Loads the preprocessing function for a given pipeline with caching."""
    if pipeline_name in LOADED_PREPROCESSORS:
        return LOADED_PREPROCESSORS[pipeline_name]

    config_path = _get_pipeline_base_dir(pipeline_name) / "config" / "runner_config.json"
    if not config_path.exists():
        # Fallback to old config location if needed, or just fail
        config_path = _get_pipeline_base_dir(pipeline_name) / "config" / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration not found for pipeline '{pipeline_name}'")

    with open(config_path, "r") as f:
        config = json.load(f)

    preprocess_file = config.get("preprocess_file")
    if not preprocess_file or not Path(preprocess_file).exists():
        raise FileNotFoundError(f"Preprocessing file not specified or not found for pipeline '{pipeline_name}'")

    spec = importlib.util.spec_from_file_location("custom_preprocess_module", preprocess_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "data_preprocessing"):
        raise AttributeError(f"{preprocess_file} must define data_preprocessing(df)")
    
    func = getattr(mod, "data_preprocessing")
    LOADED_PREPROCESSORS[pipeline_name] = func
    return func

@app.route('/predict/<pipeline_name>', methods=['POST'])
def predict(pipeline_name):
    # 1. Logger Management
    try:
        logger = _get_logger(pipeline_name)
    except Exception as e:
        # Fallback logger if pipeline specific fails (e.g. directory permissions)
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("calmops.server.fallback")
        logger.error(f"Failed to initialize pipeline logger: {e}")

    logger.info(f"Prediction request received for pipeline: {pipeline_name}")

    try:
        model = _load_model(pipeline_name)
        # 3. Preprocessing Caching
        preprocess_func = _load_preprocess_func(pipeline_name)
    except (FileNotFoundError, AttributeError, Exception) as e:
        logger.error(f"Error loading resources for pipeline '{pipeline_name}': {e}")
        return jsonify({"error": str(e)}), 500

    if not request.is_json:
        logger.warning("Request must be JSON")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    if not data:
        logger.warning("No data provided in JSON request")
        return jsonify({"error": "No data provided"}), 400

    try:
        # 4. Validation (Basic)
        if isinstance(data, dict):
            data = [data] # Handle single record
        
        if not isinstance(data, list):
             return jsonify({"error": "Input data must be a JSON object or a list of objects"}), 400

        input_df = pd.DataFrame(data)
        
        if input_df.empty:
             return jsonify({"error": "Input DataFrame is empty"}), 400

        # Preprocess the input data
        X_processed, _ = preprocess_func(input_df) 
        
        # Make prediction
        predictions = model.predict(X_processed).tolist()
        
        logger.info(f"Prediction successful for pipeline '{pipeline_name}'.")
        return jsonify({"predictions": predictions}), 200

    except Exception as e:
        logger.error(f"Prediction failed for pipeline '{pipeline_name}': {e}", exc_info=True)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    # 5. Project Root Handling is implicitly improved by using get_project_root() 
    # and relative imports at the top.
    
    # Example for local testing
    test_pipeline_name = os.environ.get("FLASK_PIPELINE_NAME", "my_pipeline_watchdog")
    
    # Initialize a main logger for the server startup
    logging.basicConfig(level=logging.INFO)
    main_logger = logging.getLogger("calmops.server.main")
    main_logger.info(f"Starting Flask server. Test pipeline: {test_pipeline_name}")

    app.run(debug=True, host='0.0.0.0', port=5000)
