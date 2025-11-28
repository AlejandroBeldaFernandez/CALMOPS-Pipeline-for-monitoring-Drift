# CALMOPS – A Pipeline for Monitoring Drift

**CalmOps** is a comprehensive, Python-based pipeline for monitoring *data drift* and *model drift* in machine learning systems. It is designed to be a robust and production-ready MLOps tool.

The core of the project is a pipeline that can be triggered with new data, which then performs drift analysis and model evaluation. The results are visualized in a web dashboard based on Streamlit.

## Features

-   **Drift Detection:** Univariate and multivariate data drift and model drift detectors using the `frouros` library.
-   **Model Comparison:** Implements a Champion/Challenger model promotion strategy to ensure production stability.
-   **Flexible Re-training:** Supports various re-training modes:
    -   `full`: Full re-training with new data.
    -   `incremental`: Incremental model training.
    -   `window`: Training with a sliding window of data.
    -   `stacking`: Model stacking.
    -   `replay mix`: Combination of old and new data.
    -   `recalibration`: Model recalibration.
-   **Multiple Pipeline Types:** Supports different data processing strategies:
    -   **Stream:** Processes data file by file as it arrives.
    -   **Block:** Processes data in discrete blocks or chunks.
    -   **IPIP:** A specialized pipeline with an adaptive model that continuously retrains.
-   **Interactive Dashboard:** Visualization of results in a real-time Streamlit dashboard.
-   **File-based Triggering:** Uses a `watchdog`-based file system monitor to automatically trigger pipelines when new data arrives.
-   **Production Deployment:** Supports production deployment with PM2 and Docker for persistence and scalability.
-   **Resilience:** Implements a Circuit Breaker pattern to prevent the system from being overwhelmed by repeated failures.
-   **Prediction-Only Mode:** Allows running a pre-trained model for inference without a target variable, drift detection, or re-training.
-   **Comprehensive Logging:** Provides detailed logs for each pipeline run.
-   **CLI for Management:** A command-line interface to list, delete, update, and relaunch pipelines.

## Architecture

The project is structured into several key components that work together to create a full MLOps cycle.

-   **`calmops/monitor`:** This is the entry point for running pipelines. It monitors the file system for new data and triggers the appropriate pipeline orchestrator.
-   **`calmops/pipeline` (and `pipeline_block`, `IPIP`):** These directories contain the core logic for orchestrating the ML process: data loading, preprocessing, drift detection, training/re-training, and evaluation.
-   **`calmops/pipelines`:** This is the root directory where all created pipeline instances are stored, each with its own configuration, models, logs, and metrics.
-   **`calmops/Detector`:** Contains the logic for drift detection, leveraging the `frouros` library.
-   **`calmops/web_interface`:** Contains the Streamlit dashboard for visualizing pipeline metrics, model performance, and drift results.
-   **`calmops/server.py`**: A Flask-based prediction server to expose trained models via a REST API.
-   **`calmops/cli.py`**: A command-line interface for managing the lifecycle of the pipelines.

For a more detailed explanation of the architecture, please refer to the **[MLOps Architecture Documentation](calmops/MLOPS_README.md)**.

## Data Generation

The project includes a powerful set of data generators for creating synthetic and realistic datasets for testing and development. These tools can simulate various types of data and concept drift.

-   **`Synthetic`**: Generates synthetic data based on well-known generators from the `river` library (e.g., SEA, Agrawal, Hyperplane).
-   **`Real`**: Synthesizes new data that mimics the statistical properties of a real-world dataset.
-   **`Clinic`**: A specialized generator for creating synthetic clinical data with multi-omics and longitudinal drift.
-   **`DriftInjection`**: A tool to inject various types of drift (abrupt, gradual, etc.) into an existing dataset.

The generation and preprocessing scripts are located in the `scripts/` directory:
-   `scripts/generate_scenario_*.py`: Scripts to generate different drift scenarios.
-   `scripts/preprocessing_*.py`: Preprocessing scripts for the generated data.

For a complete guide on how to use the generators, see the **[Data Generators Documentation](calmops/data_generators/README.md)**.

## Main Technologies

-   **Python:** The main language of the project.
-   **scikit-learn:** For training and evaluating machine learning models.
-   **frouros:** For drift detection.
-   **Streamlit:** For the monitoring dashboard.
-   **pandas:** For data manipulation and analysis.
-   **NumPy:** For numerical operations.
-   **watchdog:** For monitoring the file system.
-   **PM2 & Docker:** For production deployment.
-   **Flask:** For the prediction server.

## Installation

1.  Install the project dependencies using pip:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running a Pipeline

The main entry point for running a pipeline is the `calmops/cli.py` script, which allows you to launch and manage monitors. The monitors, in turn, execute the pipelines.

For example, to relaunch an existing pipeline named `my_pipeline_watchdog`:

```bash
python calmops/cli.py relaunch my_pipeline_watchdog
```

### Managing Pipelines with the CLI

The CLI provides several commands to manage your pipelines:

-   **List all pipelines:**
    ```bash
    python calmops/cli.py list
    ```

-   **Relaunch a pipeline:**
    ```bash
    python calmops/cli.py relaunch <pipeline_name>
    ```

-   **Update a pipeline's configuration:**
    ```bash
    python calmops/cli.py update <pipeline_name> --port 8502 --retrain_mode 2
    ```

-   **Delete a pipeline:**
    ```bash
    python calmops/cli.py delete <pipeline_name>
    ```

-   **Serve a model for predictions:**
    ```bash
    python calmops/cli.py serve <pipeline_name> --port 5001
    ```

### Viewing the Dashboard

Once a pipeline is running, its Streamlit dashboard will be available at `http://localhost:8501` by default. The port can be configured when the pipeline is created or updated.

## Recent Changes

### Prediction-Only Mode

A "prediction-only" mode has been added to all pipelines. This mode allows running the pipeline without a target variable, performing only predictions with a pre-trained model.

To activate this mode, the `prediction_only=True` argument must be passed to the `run_pipeline` function. In this mode, the pipeline will not perform drift detection or model training.

### IPIP Pipeline Changes

The IPIP pipeline has been updated to reflect its adaptive nature:

-   **No Performance Thresholds:** Performance thresholds (`thresholds_perf`) have been removed. The IPIP model adapts automatically.
-   **No Drift Detection:** Drift detection (`check_drift`) has been removed. The pipeline now always re-trains with new data.
-   **Updated Dashboard:** The IPIP dashboard has been updated to remove the "Drift" tab.

### Log Correction

An issue that prevented the generation of log files in the IPIP pipeline has been fixed. Now, logs are generated correctly for all pipelines.

## Production Deployment

The project supports production deployment using PM2 and Docker. You can configure the persistence mode when launching a monitor:

-   **PM2:** `persistence="pm2"`
-   **Docker:** `persistence="docker"`

When a pipeline is launched with one of these modes, the system will automatically generate the necessary configuration files (`ecosystem.config.js` for PM2, `Dockerfile` and `docker-compose.yml` for Docker) and deploy the pipeline as a persistent service.

## Development Conventions

-   **Modular Structure:** The project is organized into modules with specific responsibilities.
-   **Configuration:** The pipeline is configured through Python scripts and configuration files.
-   **Customization:** The pipeline is designed to be customizable. You can provide your own custom functions for training, re-training, and fallback strategies.
-   **Logging:** The project uses the `logging` module for complete logging. Each pipeline has its own log file.
-   **Error Handling:** The project includes robust error handling, including a Circuit Breaker pattern to prevent repeated failures.