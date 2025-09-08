# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CALMOPS is a comprehensive MLOps pipeline for monitoring data drift and model drift with automatic retraining capabilities. The system uses Frouros for drift detection and supports multiple retraining strategies.

## Key Architecture Components

### Core Pipeline (`pipeline_block/`)
- `pipeline_block.py`: Main orchestration engine for block-wise data processing
- `modules/data_loader.py`: Handles incremental data loading with snapshot control
- `modules/check_drift.py`: Statistical and performance-based drift detection using Frouros
- `modules/evaluador.py`: Model evaluation with threshold-based approval system
- `modules/default_train_retrain.py`: Training and retraining strategies (full, incremental, windowed, stacking)

### Data Generation (`data_generators/`)
- Uses River library (migrated from scikit-multiflow) for synthetic data generation
- Supports multiple generator types: Agrawal, SEA, Hyperplane, Sine, Stagger
- Factory pattern implementation for generator creation

### Monitoring (`monitor/`)
- File system watchers using APScheduler
- Automatic pipeline triggering on new data
- Block-wise and IPIP monitoring modes

### Web Interface (`web_interface/`)
- Streamlit-based dashboards
- Real-time visualization of drift metrics, model performance, and predictions
- Anti-delta sanitization for secure data display

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Environment activation (if using virtual environment)
source /home/alex/env-test/bin/activate
```

### Testing
```bash
# Test River migration (synthetic data generation)
python test_migration.py

# Test with custom environment
/home/alex/env-test/bin/python test_migration.py
```

### Running the Pipeline
```bash
# Basic pipeline execution (requires configuration setup)
python -m pipeline_block.pipeline_block

# Web dashboard
streamlit run web_interface/dashboard_block.py
```

### Data Generation
```bash
# Generate synthetic datasets
python -m data_generators.Synthetic.SyntheticGenerator
```

## Configuration Structure

### Pipeline Configuration
- Located in `pipelines/{pipeline_name}/config/config.json`
- Required fields: `pipeline_name`, `data_dir`, `preprocess_file`
- Pipeline outputs stored in `pipelines/{pipeline_name}/`

### Directory Structure per Pipeline
```
pipelines/{pipeline_name}/
├── config/config.json
├── modelos/{pipeline_name}.pkl
├── control/              # Snapshot control files
├── logs/                 # Pipeline execution logs  
├── metrics/              # Drift detection and evaluation results
└── candidates/           # Model training artifacts
```

## Key Dependencies

- **frouros**: Drift detection algorithms
- **river**: Streaming ML (replacement for scikit-multiflow)
- **streamlit**: Web dashboard framework  
- **scikit-learn**: ML model training
- **plotly**: Interactive visualizations
- **APScheduler**: Task scheduling
- **watchdog**: File system monitoring

## Block-wise Processing

The system is designed around "block-wise" processing where:
- Data is partitioned by a `block_col` (e.g., time periods, batches)
- Drift detection compares blocks to identify degraded segments
- Retraining targets only blocks with detected drift
- Evaluation performed on latest blocks (`eval_blocks`)

## Circuit Breaker Pattern

Built-in fault tolerance with `health.json` tracking:
- Consecutive failure counting
- Automatic pipeline pausing after threshold breaches
- Configurable backoff periods

## Testing Strategy

- Use `test_migration.py` for River integration validation
- Synthetic data generation testing via GeneratorFactory
- Pipeline execution testing requires data in `/home/alex/datos/`