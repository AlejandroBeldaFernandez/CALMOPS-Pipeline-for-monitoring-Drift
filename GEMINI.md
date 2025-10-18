# GEMINI.md - CalmOps Project

## Project Overview

This project, "CalmOps," is a comprehensive Python-based pipeline for monitoring data drift and model drift in machine learning systems. It is designed to be a robust and production-ready MLOps tool.

The core of the project is a pipeline that can be triggered by new data, which then performs drift analysis and model evaluation. The results are visualized in a Streamlit-based web dashboard.

**Main Technologies:**

*   **Python:** The core language of the project.
*   **scikit-learn:** Used for machine learning model training and evaluation.
*   **frouros:** A library for drift detection in machine learning systems.
*   **Streamlit:** Used to create the interactive web dashboard for monitoring.
*   **pandas:** Used for data manipulation and analysis.
*   **NumPy:** Used for numerical operations.
*   **watchdog:** Used for monitoring the file system for new data.
*   **PM2 & Docker:** Used for production deployment and persistence.

**Architecture:**

The project is structured into several key components:

*   **`monitor`:** This component is responsible for monitoring the file system for new data files. It uses the `watchdog` library to detect new files and then triggers the main pipeline. It also handles persistence using PM2 and Docker.
*   **`pipeline`:** This is the core of the project. It orchestrates the entire process of data loading, preprocessing, drift detection, model training/retraining, and evaluation. It implements a Champion/Challenger model promotion strategy and a circuit breaker pattern for reliability.
*   **`Detector`:** This component contains the logic for drift detection. It uses various statistical tests and distance metrics from the `frouros` library to detect both data drift and model drift.
*   **`web_interface`:** This component contains the Streamlit dashboard. It visualizes the results of the drift analysis and model performance, providing an interactive way to monitor the system.
*   **`data_generators`:** This component contains scripts for generating synthetic datasets, which can be used for testing and development.
*   **`config`:** This directory likely contains configuration files for the pipeline, such as thresholds for drift detection and model performance.

## Building and Running

**1. Install Dependencies:**

The project's dependencies are listed in the `requirements.txt` file. You can install them using pip:

```bash
pip install -r requirements.txt
```

**2. Running the Pipeline:**

The main entry point for running the pipeline is the `monitor/monitor.py` script. You can run it directly from the command line:

```bash
python monitor/monitor.py
```

This will start the file system monitor and the Streamlit dashboard. When a new data file is added to the specified data directory, the pipeline will be triggered automatically.

**3. Viewing the Dashboard:**

The Streamlit dashboard will be available at `http://localhost:8501` by default. The port can be configured in the `monitor/monitor.py` script.

**4. Production Deployment:**

The project supports production deployment using PM2 and Docker. You can configure the persistence mode in the `monitor/monitor.py` script.

*   **PM2:** `persistence="pm2"`
*   **Docker:** `persistence="docker"`

## Development Conventions

*   **Modular Structure:** The project is organized into modules with specific responsibilities (e.g., `Detector`, `pipeline`, `monitor`).
*   **Configuration:** The pipeline is configured through Python scripts and configuration files.
*   **Customization:** The pipeline is designed to be customizable. You can provide your own custom functions for training, retraining, and fallback strategies.
*   **Logging:** The project uses the `logging` module for comprehensive logging. Each pipeline has its own log file.
*   **Error Handling:** The project includes robust error handling, including a circuit breaker pattern to prevent repeated failures.

## Agent Rules

- No puedes usar emoji
- El agente debe hablar en Español
- El codigo generado siempre debe incluir comentarios en ingles
- Los prints e informacion mostrada debe ser en ingles
