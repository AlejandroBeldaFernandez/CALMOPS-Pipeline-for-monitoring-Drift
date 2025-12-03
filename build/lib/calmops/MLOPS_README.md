# CalmOps MLOps Architecture

This document provides an overview of the MLOps architecture of the CalmOps project. The system is designed to be modular and flexible, allowing for the management of machine learning pipelines, model monitoring, and drift detection.

## Key Components

The CalmOps architecture is divided into the following key components:

1.  **Data Generators (`calmops/data_generators/`)**:
    -   A set of tools for creating synthetic and realistic data.
    -   Supports the simulation of various scenarios, including data and concept drift.
    -   For more details, refer to the `README.md` in `calmops/data_generators/`.

2.  **Pipelines (`calmops/pipelines/`)**:
    -   The heart of the system, where ML pipelines are defined and stored.
    -   Each subdirectory in `calmops/pipelines/` represents an individual pipeline.
    -   A pipeline typically contains:
        -   `config/`: Configuration files (`config.json`, `runner_config.json`).
        -   `models/`: Trained models (usually `.pkl` files).
        -   `logs/`: Pipeline execution logs.
        -   `metrics/`: Performance metrics, drift detection results, etc.

3.  **Monitors (`calmops/monitor/`)**:
    -   Monitors are responsible for executing the pipelines.
    -   There are different types of monitors for different pipeline types and execution modes:
        -   `monitor.py`: For streaming pipelines.
        -   `monitor_schedule.py`: For scheduled pipelines.
        -   `monitor_block.py`: For pipelines that process data in blocks.
        -   `monitor_schedule_block.py`: For scheduled block pipelines.
        -   `monitor_ipip.py` and `monitor_schedule_ipip.py`: For the specific IPIP pipeline.
    -   They handle tasks such as drift detection, model retraining, and evaluation.

4.  **CLI (`calmops/cli.py`)**:
    -   The command-line interface for managing pipelines.
    -   Allows users to `list`, `delete`, `update`, and `relaunch` pipelines.
    -   It can also be used to start the prediction server (`serve`).

5.  **Prediction Server (`calmops/server.py`)**:
    -   A Flask application that exposes trained models via a REST API.
    -   Provides a `/predict/<pipeline_name>` endpoint for real-time predictions.
    -   Manages the loading of models and associated preprocessing functions.

6.  **Web Interface (`calmops/web_interface/`)**:
    -   Contains dashboards for visualizing monitoring results.
    -   Allows users to observe model performance, drift detection results, and other relevant metrics.

## Workflow

1.  **Pipeline Creation**: A pipeline is created by defining its configuration and scripts (preprocessing, training, etc.).
2.  **Launch**: The `cli.py` is used to launch a pipeline. This starts the corresponding monitor, which begins processing data.
3.  **Monitoring**: The monitor runs continuously or on a schedule, processing new data, evaluating model performance, and detecting drift.
4.  **Retraining**: If drift or a performance drop is detected, the system can (depending on the configuration) retrain the model with new data.
5.  **Visualization**: Monitoring results can be viewed in the web interface dashboards.
6.  **Prediction**: The trained model can be served through the prediction server for use in applications.