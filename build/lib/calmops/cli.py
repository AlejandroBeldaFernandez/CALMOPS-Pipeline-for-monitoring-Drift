#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CalmOps CLI - Pipeline Management Tool

This script provides a command-line interface for managing CalmOps pipelines.
"""

import os
import sys
import json
import shutil
import logging
import inspect
import argparse
import importlib
from pathlib import Path

from calmops.utils import get_pipelines_root
from calmops.monitor.utils import _get_logger

# Ensure the project root is on the Python path
ROOT_DIR = get_pipelines_root() / "pipelines"


def _build_model(spec: dict):
    """Dynamically builds a model instance from its specification."""
    if not spec or not spec.get("module") or not spec.get("class"):
        print(
            "Warning: Model specification is incomplete. Cannot build model instance."
        )
        return None
    try:
        mod = importlib.import_module(spec["module"])
        cls = getattr(mod, spec["class"])
        params = spec.get("params", {})
        return cls(**params)
    except Exception as e:
        print(f"Warning: Could not build model instance from spec {spec}. Error: {e}")
        # Attempt to instantiate without params as a fallback
        try:
            return cls()
        except Exception as final_e:
            print(
                f"Fatal: Could not instantiate model class {spec.get('class')}. Error: {final_e}"
            )
            return None


def _filter_args(func, args_dict: dict) -> dict:
    """Filters a dictionary to include only keys that are parameters of a function."""
    sig = inspect.signature(func)
    param_names = set(sig.parameters.keys())

    # If the function accepts **kwargs, no filtering is needed.
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return args_dict

    return {k: v for k, v in args_dict.items() if k in param_names}


def main():
    """Main entry point for the CalmOps CLI."""
    try:
        from calmops.monitor.monitor import configure_root_logging

        configure_root_logging()
    except ImportError:
        # If dependencies are missing, we might be in a lightweight install
        logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="CalmOps Pipeline Management Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # List command
    parser_list = subparsers.add_parser("list", help="List all available pipelines.")

    # Delete command
    parser_delete = subparsers.add_parser("delete", help="Delete a pipeline.")
    parser_delete.add_argument(
        "pipeline_name", help="The name of the pipeline to delete."
    )

    # Relaunch command
    parser_relaunch = subparsers.add_parser("relaunch", help="Relaunch a pipeline.")
    parser_relaunch.add_argument(
        "pipeline_name", help="The name of the pipeline to relaunch."
    )

    # Update command (NEW)
    parser_update = subparsers.add_parser(
        "update", help="Update parameters of an existing pipeline."
    )
    parser_update.add_argument(
        "pipeline_name", help="The name of the pipeline to update."
    )
    parser_update.add_argument(
        "--preprocess_file", help="New path to the preprocessing script."
    )
    parser_update.add_argument(
        "--custom_train_file", help="New path to the custom training script."
    )
    parser_update.add_argument(
        "--custom_retrain_file", help="New path to the custom retraining script."
    )
    parser_update.add_argument(
        "--schedule",
        help='New schedule in JSON format (e.g., \'{"type": "interval", "params": {"minutes": 10}}\'',
    )
    parser_update.add_argument("--retrain_mode", type=int, help="New retrain mode.")
    parser_update.add_argument("--fallback_mode", type=int, help="New fallback mode.")
    parser_update.add_argument(
        "--thresholds_drift",
        help="New drift thresholds in JSON format (e.g., '{\"balanced_accuracy\": 0.8}'",
    )
    parser_update.add_argument(
        "--thresholds_perf",
        help="New performance thresholds in JSON format (e.g., '{\"accuracy\": 0.9}'",
    )
    parser_update.add_argument("--data_dir", help="New path to the data directory.")
    parser_update.add_argument("--random_state", type=int, help="New random state.")
    parser_update.add_argument(
        "--param_grid",
        help="New parameter grid in JSON format (e.g., '{\"n_estimators\": [50, 100]}'",
    )
    parser_update.add_argument("--cv", type=int, help="New cross-validation value.")
    parser_update.add_argument("--port", type=int, help="New port for the dashboard.")
    parser_update.add_argument("--window_size", type=int, help="New window size.")
    parser_update.add_argument("--delimiter", help="New delimiter for data files.")

    # Serve command
    parser_serve = subparsers.add_parser(
        "serve", help="Serve a pipeline's model for prediction."
    )
    parser_serve.add_argument(
        "pipeline_name", help="The name of the pipeline to serve."
    )
    parser_serve.add_argument(
        "--port", type=int, default=5000, help="Port to serve the model on."
    )

    # Tutorials command
    parser_tutorials = subparsers.add_parser(
        "tutorials", help="Manage tutorials and documentation."
    )
    tutorials_subparsers = parser_tutorials.add_subparsers(
        dest="tutorial_command", required=True
    )

    # Tutorials list
    tutorials_subparsers.add_parser("list", help="List available tutorials.")

    # Tutorials copy
    parser_tutorials_copy = tutorials_subparsers.add_parser(
        "copy", help="Copy tutorial files to a directory."
    )
    parser_tutorials_copy.add_argument(
        "module", help="Name of the module (e.g., 'Clinic', 'Privacy')."
    )
    parser_tutorials_copy.add_argument(
        "--dest",
        default=".",
        help="Destination directory (default: current directory).",
    )

    args = parser.parse_args()

    if args.command == "list":
        list_pipelines()
    elif args.command == "delete":
        delete_pipeline(args.pipeline_name)
    elif args.command == "relaunch":
        relaunch_pipeline(args.pipeline_name)
    elif args.command == "update":
        update_pipeline_config(
            pipeline_name=args.pipeline_name,
            preprocess_file=args.preprocess_file,
            custom_train_file=args.custom_train_file,
            custom_retrain_file=args.custom_retrain_file,
            schedule=args.schedule,
            retrain_mode=args.retrain_mode,
            fallback_mode=args.fallback_mode,
            thresholds_drift=args.thresholds_drift,
            thresholds_perf=args.thresholds_perf,
            data_dir=args.data_dir,
            random_state=args.random_state,
            param_grid=args.param_grid,
            cv=args.cv,
            port=args.port,
            window_size=args.window_size,
        )
    elif args.command == "serve":
        from calmops.server import app

        # Set the pipeline name for the Flask app
        os.environ["FLASK_PIPELINE_NAME"] = args.pipeline_name
        app.run(host="0.0.0.0", port=args.port)
    elif args.command == "tutorials":
        if args.tutorial_command == "list":
            list_tutorials()
        elif args.tutorial_command == "copy":
            copy_tutorial(args.module, args.dest)


def _get_available_tutorials():
    """Scans the package for available tutorials."""
    import calmops

    package_dir = Path(calmops.__file__).parent

    tutorials = []

    # Check data_generators
    generators_dir = package_dir / "data_generators"
    if generators_dir.exists():
        for item in generators_dir.iterdir():
            if item.is_dir() and (item / "tutorial.py").exists():
                tutorials.append(f"data_generators/{item.name}")

    # Check privacy
    privacy_dir = package_dir / "privacy"
    if privacy_dir.exists() and (privacy_dir / "tutorial.py").exists():
        tutorials.append("privacy")

    return sorted(tutorials)


def list_tutorials():
    """Lists available tutorials by scanning the package."""
    tutorials = _get_available_tutorials()
    print("Available Tutorials:")
    for t in tutorials:
        print(f"  - {t}")


def copy_tutorial(module_name: str, dest_dir: str):
    """Copies tutorial files for a specific module."""
    if module_name.lower() == "all":
        tutorials = _get_available_tutorials()
        print(f"Copying {len(tutorials)} tutorials to '{dest_dir}'...")
        for t in tutorials:
            copy_tutorial(t, dest_dir)
        return

    import calmops

    package_dir = Path(calmops.__file__).parent

    # Handle nested paths like data_generators/Clinic
    parts = module_name.split("/")
    source_path = package_dir
    for part in parts:
        source_path = source_path / part

    if not source_path.exists() or not source_path.is_dir():
        # Try finding it directly if user just typed 'Clinic'
        found = False
        if len(parts) == 1:
            # Check data_generators
            potential_path = package_dir / "data_generators" / module_name
            if potential_path.exists():
                source_path = potential_path
                found = True
            elif (package_dir / module_name).exists():
                source_path = package_dir / module_name
                found = True

        if not found:
            print(f"Error: Module '{module_name}' not found.")
            return

    dest_path = Path(dest_dir) / source_path.name
    dest_path.mkdir(parents=True, exist_ok=True)

    files_to_copy = ["tutorial.py", "README.md", "API_REFERENCE.md"]
    copied_count = 0

    for filename in files_to_copy:
        src_file = source_path / filename
        if src_file.exists():
            shutil.copy2(src_file, dest_path / filename)
            print(f"Copied {filename} to {dest_path}")
            copied_count += 1

    if copied_count == 0:
        print(f"Warning: No tutorial files found in {source_path}")
    else:
        print(f"Successfully copied {module_name} files to {dest_path}")


def relaunch_pipeline(pipeline_name: str):
    """Relaunches a pipeline by finding its unique runner configuration."""
    pipeline_dir = ROOT_DIR / pipeline_name
    if not pipeline_dir.is_dir():
        print(f"Error: Pipeline '{pipeline_name}' not found.")
        sys.exit(1)

    config_dir = pipeline_dir / "config"
    if not config_dir.is_dir():
        print(
            f"Error: Configuration directory not found for pipeline '{pipeline_name}'."
        )
        sys.exit(1)

    # Find the runner config file dynamically
    possible_configs = [f.name for f in config_dir.glob("runner_*.json")]

    if not possible_configs:
        # Fallback for older versions that might just have runner_config.json
        if (config_dir / "runner_config.json").exists():
            possible_configs = ["runner_config.json"]
        else:
            print(
                f"Error: No 'runner_*.json' file found for pipeline '{pipeline_name}'."
            )
            sys.exit(1)

    if len(possible_configs) > 1:
        print(
            f"Error: Multiple runner configuration files found for '{pipeline_name}'. Please resolve the ambiguity:"
        )
        for cfg in possible_configs:
            print(f"  - {cfg}")
        sys.exit(1)

    config_path = config_dir / possible_configs[0]
    print(f"Using configuration file: {possible_configs[0]}")

    print(f"Relaunching pipeline '{pipeline_name}' from configuration file...")
    with config_path.open("r") as f:
        config = json.load(f)

    monitor_type = config.get("monitor_type")
    if not monitor_type:
        print("Error: 'monitor_type' not found in configuration. Cannot relaunch.")
        sys.exit(1)

    # Reconstruct the model instance from spec
    model_instance = _build_model(config.get("model_spec"))
    if model_instance is None:
        print("Fatal: Could not rebuild model instance. Aborting.")
        sys.exit(1)

    # Prepare a dictionary with all possible arguments from the config
    monitor_args = config.copy()
    monitor_args["model_instance"] = model_instance

    # Map monitor_type to the correct function
    # Lazy imports to avoid heavy dependencies when not needed
    from calmops.monitor.monitor import start_monitor
    from calmops.monitor.monitor_schedule import start_monitor_schedule
    from calmops.monitor.monitor_block import start_monitor_block
    from calmops.monitor.monitor_schedule_block import start_monitor_schedule_block
    from calmops.monitor.monitor_ipip import start_monitor_ipip
    from calmops.monitor.monitor_schedule_ipip import start_monitor_schedule_ipip

    # Set TensorFlow environment variables and logging
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.get_logger().setLevel("ERROR")

    monitor_functions = {
        "monitor": start_monitor,
        "monitor_schedule": start_monitor_schedule,
        "monitor_block": start_monitor_block,
        "monitor_schedule_block": start_monitor_schedule_block,
        "monitor_ipip": start_monitor_ipip,
        "monitor_schedule_ipip": start_monitor_schedule_ipip,
    }

    start_function = monitor_functions.get(monitor_type)

    if start_function:
        # Filter the arguments to match the function's signature
        filtered_args = _filter_args(start_function, monitor_args)

        # Call the function with only the valid arguments
        start_function(**filtered_args)
    else:
        print(f"Error: Unknown monitor_type '{monitor_type}'.")
        sys.exit(1)


def update_pipeline_config(
    pipeline_name: str,
    preprocess_file: str = None,
    custom_train_file: str = None,
    custom_retrain_file: str = None,
    schedule: str = None,
    retrain_mode: int = None,
    fallback_mode: int = None,
    thresholds_drift: str = None,
    thresholds_perf: str = None,
    data_dir: str = None,
    random_state: int = None,
    param_grid: str = None,
    cv: int = None,
    port: int = None,
    window_size: int = None,
):
    """Updates the configuration of an existing pipeline."""
    pipeline_dir = ROOT_DIR / pipeline_name
    if not pipeline_dir.is_dir():
        print(f"Error: Pipeline '{pipeline_name}' not found.")
        sys.exit(1)

    config_path = pipeline_dir / "config" / "runner_config.json"
    if not config_path.exists():
        print(
            f"Error: Could not find runner_config.json for pipeline '{pipeline_name}'."
        )
        sys.exit(1)

    print(f"Updating configuration for pipeline '{pipeline_name}'...")
    with config_path.open("r") as f:
        config = json.load(f)

    updated = False
    if preprocess_file is not None:
        if not Path(preprocess_file).exists():
            print(
                f"Error: New preprocess_file path '{preprocess_file}' does not exist."
            )
            sys.exit(1)
        config["preprocess_file"] = preprocess_file
        updated = True
        print(f"Updated preprocess_file to: {preprocess_file}")

    if custom_train_file is not None:
        if not Path(custom_train_file).exists():
            print(
                f"Error: New custom_train_file path '{custom_train_file}' does not exist."
            )
            sys.exit(1)
        config["custom_train_file"] = custom_train_file
        updated = True
        print(f"Updated custom_train_file to: {custom_train_file}")

    if custom_retrain_file is not None:
        if not Path(custom_retrain_file).exists():
            print(
                f"Error: New custom_retrain_file path '{custom_retrain_file}' does not exist."
            )
            sys.exit(1)
        config["custom_retrain_file"] = custom_retrain_file
        updated = True
        print(f"Updated custom_retrain_file to: {custom_retrain_file}")

    if schedule is not None:
        try:
            schedule_dict = json.loads(schedule)
            if "type" not in schedule_dict or "params" not in schedule_dict:
                print(
                    "Error: Invalid schedule format. Must be a JSON string with 'type' and 'params' keys."
                )
                sys.exit(1)
            config["schedule"] = schedule_dict
            updated = True
            print(f"Updated schedule to: {schedule}")
        except json.JSONDecodeError:
            print("Error: Invalid JSON format for schedule.")
            sys.exit(1)

    if retrain_mode is not None:
        config["retrain_mode"] = retrain_mode
        updated = True
        print(f"Updated retrain_mode to: {retrain_mode}")

    if fallback_mode is not None:
        config["fallback_mode"] = fallback_mode
        updated = True
        print(f"Updated fallback_mode to: {fallback_mode}")

    if thresholds_drift is not None:
        try:
            thresholds_drift_dict = json.loads(thresholds_drift)
            config["thresholds_drift"] = thresholds_drift_dict
            updated = True
            print(f"Updated thresholds_drift to: {thresholds_drift}")
        except json.JSONDecodeError:
            print("Error: Invalid JSON format for thresholds_drift.")
            sys.exit(1)

    if thresholds_perf is not None:
        try:
            thresholds_perf_dict = json.loads(thresholds_perf)
            config["thresholds_perf"] = thresholds_perf_dict
            updated = True
            print(f"Updated thresholds_perf to: {thresholds_perf}")
        except json.JSONDecodeError:
            print("Error: Invalid JSON format for thresholds_perf.")
            sys.exit(1)

    if data_dir is not None:
        if not Path(data_dir).is_dir():
            print(
                f"Error: New data_dir path '{data_dir}' does not exist or is not a directory."
            )
            sys.exit(1)
        config["data_dir"] = data_dir
        updated = True
        print(f"Updated data_dir to: {data_dir}")

    if random_state is not None:
        config["random_state"] = random_state
        updated = True
        print(f"Updated random_state to: {random_state}")

    if param_grid is not None:
        try:
            param_grid_dict = json.loads(param_grid)
            config["param_grid"] = param_grid_dict
            updated = True
            print(f"Updated param_grid to: {param_grid}")
        except json.JSONDecodeError:
            print("Error: Invalid JSON format for param_grid.")
            sys.exit(1)

    if cv is not None:
        config["cv"] = cv
        updated = True
        print(f"Updated cv to: {cv}")

    if port is not None:
        config["port"] = port
        updated = True
        print(f"Updated port to: {port}")

    if window_size is not None:
        config["window_size"] = window_size
        updated = True
        print(f"Updated window_size to: {window_size}")

    if updated:
        with config_path.open("w") as f:
            json.dump(config, f, indent=2)
        print(f"Configuration for pipeline '{pipeline_name}' updated successfully.")
    else:
        print("No parameters specified for update.")


def list_pipelines():
    """
    Administrative utility for pipeline discovery and inventory.

    Provides operational visibility into deployed monitoring pipelines
    for management and maintenance purposes.
    """
    log = _get_logger("calmops.monitor.utils")
    try:
        pipelines = [d.name for d in ROOT_DIR.iterdir() if d.is_dir()]
        if not pipelines:
            log.info("No monitoring pipelines currently deployed")
        else:
            log.info("Currently deployed monitoring pipelines:")
            for pipeline in pipelines:
                log.info(f"- {pipeline}")
    except Exception as e:
        log.error(f"Pipeline discovery failed: {e}")


def delete_pipeline(pipeline_name):
    """
    Permanent pipeline removal with safety confirmation.

    Implements secure deletion with user confirmation to prevent
    accidental removal of production monitoring configurations.
    """
    log = _get_logger("calmops.monitor.utils")
    pipeline_path = ROOT_DIR / pipeline_name
    if not pipeline_path.exists():
        log.error(f"Pipeline '{pipeline_name}' not found in deployment registry")
        return
    try:
        confirmation = input(
            f"Are you sure you want to delete the pipeline '{pipeline_name}'? This action is irreversible. (y/n): "
        )
        if confirmation.lower() != "y":
            log.info("Pipeline deletion cancelled by user")
            return
        shutil.rmtree(pipeline_path)
        log.info(f"Pipeline '{pipeline_name}' successfully removed from system")
    except Exception as e:
        log.error(f"Pipeline deletion failed for '{pipeline_name}': {e}")


if __name__ == "__main__":
    main()
