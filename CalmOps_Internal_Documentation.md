
# CalmOps Internal Documentation

This document provides a detailed explanation of the **CalmOps** MLOps framework. It covers the architecture, directory structure, key modules, functions, and configuration usage.

## 1. Project Overview

**CalmOps** is an MLOps framework designed for continuous model monitoring, drift detection, and adaptive maintenance. It supports multiple pipeline types (IPIP, Block, Flow) and deployment modes (Foreground, PM2, Docker).

### Key Features
- **Drift Detection**: Integrated with `frouros` for detecting concept and data drift.
- **Adaptive Training**: Automated retraining triggered by drift or performance degradation.
- **Resilient Operations**: Circuit breaker patterns to prevent repeated failures.
- **Synthetic Data**: Advanced data generation capabilities (`RealGenerator`) using FCS (Fully Conditional Specification).
- **Flexible Deployment**: Supports local execution, PM2 process management, and Docker containerization.

---

## 2. Directory Structure

The project root is `/home/alex/calmops`. The Python package is located in `calmops/`.

| Directory | Description |
|-----------|-------------|
| `calmops/cli.py` | **Main Entry Point**. Command Line Interface for managing pipelines. |
| `calmops/IPIP/` | **IPIP Pipeline**. Logic for the Incremental Learning pipeline. |
| `calmops/Detector/` | **Drift Detection**. Wrappers around `frouros` detectors. |
| `calmops/pipeline/` | **Flow Pipeline**. General-purpose pipeline logic (Flow) with circuit breakers. |
| `calmops/pipeline_block/`| **Block Pipeline**. Block-based processing logic. |
| `calmops/monitor/` | **Monitoring & Scheduling**. Orchestration using APScheduler. |
| `calmops/data_generators/` | **Data Synthesis**. Tools to generate synthetic datasets (`Real`, `Clinic`, `Synthetic`). |
| `calmops/web_interface/` | **Dashboard**. Streamlit-based UI for monitoring pipeline status. |
| `pipelines/` | **Configuration Storage**. Stores runtime configs (`runner_config.json`) for deployed pipelines. |

---

## 3. Module & Function Breakdown

### 3.1. Command Line Interface (`calmops/cli.py`)
This is the primary way users interact with CalmOps.

- **`main()`**: Entry point. Parses arguments and routes commands.
- **Commands**:
  - `list`: Lists active pipelines.
  - `relaunch <name>`: Restarts a pipeline using its stored config.
  - `update <name>`: Updates parameters (e.g., schedule, thresholds) of a live pipeline.
  - `delete <name>`: Removes a pipeline.
  - `tutorials`: Manages tutorial files.

### 3.2. IPIP Pipeline (`calmops/IPIP/`)
The **IPIP** (Incremental Learning) pipeline processes data in blocks and adapts dynamic parameters (`p`, `b`) based on class balance.

- **File**: `pipeline_ipip.py`
  - **`run_pipeline_ipip(...)`**: Core logic.
    - Loads data blocks.
    - Calculates dynamic parameters `p` (minority prop) and `b` (bagging size).
    - Checks for drift.
    - Performs transductive pruning during retraining.
  - **`_upsert_control_entry(...)`**: Updates the control file to track processed blocks.

### 3.3. Drift Detector (`calmops/Detector/`)
Handles statistical tests to detect distribution changes.

- **File**: `drift_detector.py`
  - **Class `DriftDetector`**:
    - **`kolmogorov_smirnov_test(X_ref, X_new)`**: Univariate KS test.
    - **`population_stability_index_test(X_ref, X_new)`**: PSI metric (Population Stability Index).
    - **`mann_whitney_test`**: Non-parametric test for location shift.

### 3.4. Flow Pipeline (`calmops/pipeline/`)
A flexible pipeline orchestrator that supports custom logic and circuit breakers.

- **File**: `pipeline_flow.py`
  - **Function `run_pipeline(...)`**: Executes the generic pipeline flow (Predict -> Drift? -> Retrain).
  - **Circuit Breaker Logic**:
    - **`_load_health()` / `_save_health()`**: Persists failure counts to `health.json`.
    - **`_should_pause(health)`**: Checks if retraining is currently paused due to too many failures.
    - **`_update_on_result(...)`**: Updates failure counts and triggers backoff if threshold `max_failures` is reached.

### 3.5. Data Generators (`calmops/data_generators/`)
- **RealGenerator (`Real/RealGenerator.py`)**:
  - **`synthesize(...)`**: Main entry point. Supports multiple methods (`cart`, `rf`, `lgbm`, `sdv`, `resample`).
  - **`_synthesize_fcs_generic(...)`**: (Refactored) Generic helper for Fully Conditional Specification synthesis. Iterates through columns, training regressors/classifiers to predict missing values based on others.

### 3.6. Monitor & Schedule (`calmops/monitor/`)
Handles the periodic execution of pipelines.

- **File**: `monitor_schedule.py`
  - **`start_monitor_schedule(...)`**: Starts the scheduler (APScheduler).
  - **Persistence**:
    - **`_launch_with_pm2(...)`**: Uses PM2 to daemonize the process.
    - **`_launch_with_docker(...)`**: Uses Docker Compose to isolate the pipeline.
  - **`_write_runner_config(...)`**: Saves the arguments to `pipelines/<name>/config/runner_config.json` so they can be reloaded.

---

## 4. Configuration (`runner_config.json`)
Each pipeline has a configuration file stored in `pipelines/<name>/config/runner_config.json`. This JSON file allows the CLI to relaunch or update the pipeline.

**Example Structure:**
```json
{
  "pipeline_name": "pipeline_ipip_sms",
  "monitor_type": "monitor_schedule_ipip",
  "data_dir": "/opt/data",
  "model_spec": {
    "module": "sklearn.ensemble",
    "class": "RandomForestClassifier",
    "params": {"n_estimators": 100}
  },
  "schedule": {
    "type": "interval",
    "params": {"minutes": 10}
  },
  "thresholds_drift": {"p_value": 0.05},
  "persistence": "pm2"
}
```

## 5. Global Variables & Environment
- **`TF_CPP_MIN_LOG_LEVEL = "2"`**: Suppresses TensorFlow info/warning logs project-wide.
- **Logging**: Configured via `calmops.logger`. Logs are typically stored in `logs/` directory.

---

## 6. How to Use
1.  **Start a Pipeline**: Typically done via a Python script calling `start_monitor_schedule_ipip` (or similar).
2.  **Manage**: Use `calmops list` to see running pipelines.
3.  **Monitor**: Access the Streamlit dashboard (default port 8501) to view drift graphs and metrics.
