#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CalmOps CLI - Pipeline Management Tool

This script provides a command-line interface for managing CalmOps pipelines.
"""

import os
# Set TensorFlow environment variables and logging BEFORE any other imports


import logging
logging.basicConfig(level=logging.ERROR)

import argparse
import sys
import subprocess
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.get_logger().setLevel('ERROR')
# Ensure the project root is on the Python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from monitor.utils import list_pipelines, delete_pipeline
from monitor.monitor import configure_root_logging, start_monitor
from monitor.monitor_schedule import start_monitor_schedule
from monitor.monitor_block import start_monitor_block
from monitor.monitor_schedule_block import start_monitor_schedule_block
from monitor.monitor_ipip import start_monitor_ipip
from monitor.monitor_schedule_ipip import start_monitor_schedule_ipip

def main():
    """Main entry point for the CalmOps CLI."""
    configure_root_logging()

    parser = argparse.ArgumentParser(description="CalmOps Pipeline Management Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # List command
    parser_list = subparsers.add_parser("list", help="List all available pipelines.")

    # Delete command
    parser_delete = subparsers.add_parser("delete", help="Delete a pipeline.")
    parser_delete.add_argument("pipeline_name", help="The name of the pipeline to delete.")

    # Relaunch command
    parser_relaunch = subparsers.add_parser("relaunch", help="Relaunch a pipeline.")
    parser_relaunch.add_argument("pipeline_name", help="The name of the pipeline to relaunch.")

    # Update command (NEW)
    parser_update = subparsers.add_parser("update", help="Update parameters of an existing pipeline.")
    parser_update.add_argument("pipeline_name", help="The name of the pipeline to update.")
    parser_update.add_argument("--preprocess_file", help="New path to the preprocessing script.")
    parser_update.add_argument("--custom_train_file", help="New path to the custom training script.")
    parser_update.add_argument("--custom_retrain_file", help="New path to the custom retraining script.")
    parser_update.add_argument("--schedule", help='New schedule in JSON format (e.g., \'{"type": "interval", "params": {"minutes": 10}}\')')
    parser_update.add_argument("--retrain_mode", type=int, help="New retrain mode.")
    parser_update.add_argument("--fallback_mode", type=int, help="New fallback mode.")
    parser_update.add_argument("--thresholds_drift", help='New drift thresholds in JSON format (e.g., \'{"balanced_accuracy": 0.8}\')')
    parser_update.add_argument("--thresholds_perf", help='New performance thresholds in JSON format (e.g., \'{"accuracy": 0.9}\')')
    parser_update.add_argument("--data_dir", help="New path to the data directory.")
    parser_update.add_argument("--random_state", type=int, help="New random state.")
    parser_update.add_argument("--param_grid", help='New parameter grid in JSON format (e.g., \'{"n_estimators": [50, 100]}\')')
    parser_update.add_argument("--cv", type=int, help="New cross-validation value.")
    parser_update.add_argument("--port", type=int, help="New port for the dashboard.")
    parser_update.add_argument("--window_size", type=int, help="New window size.")
    parser_update.add_argument("--delimiter", help="New delimiter for data files.")

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

def relaunch_pipeline(pipeline_name: str):
    """Relaunches a pipeline."""
    pipeline_dir = os.path.join(ROOT_DIR, "pipelines", pipeline_name)
    if not os.path.isdir(pipeline_dir):
        print(f"Error: Pipeline '{pipeline_name}' not found.")
        sys.exit(1)

    config_path = os.path.join(pipeline_dir, "config", "runner_config.json")
    if not os.path.exists(config_path):
        print(f"Error: Could not find runner_config.json for pipeline '{pipeline_name}'.")
        sys.exit(1)

    print(f"Relaunching pipeline '{pipeline_name}' from config file...")
    with open(config_path, "r") as f:
        config = json.load(f)

    # We need to reconstruct the model instance

    model_instance = None
    

    monitor_type = config.get("monitor_type")
    if not monitor_type:
        print("Error: 'monitor_type' not found in config. Cannot relaunch.")
        sys.exit(1)

    # Prepare arguments for start_monitor, ensuring only expected parameters are passed
    monitor_args = {
        "pipeline_name": config["pipeline_name"],
        "data_dir": config["data_dir"],
        "preprocess_file": config["preprocess_file"],
        "thresholds_drift": config["thresholds_drift"],
        "thresholds_perf": config["thresholds_perf"],
        "model_instance": model_instance,
        "retrain_mode": config["retrain_mode"],
        "fallback_mode": config.get("fallback_mode"),
        "random_state": config["random_state"],
        "param_grid": config.get("param_grid"),
        "cv": config.get("cv"),
        "custom_train_file": config.get("custom_train_file"),
        "custom_retrain_file": config.get("custom_retrain_file"),
        "custom_fallback_file": config.get("custom_fallback_file"),
        "delimiter": config.get("delimiter"),
        "target_file": config.get("target_file"),
        "window_size": config.get("window_size"),
        "port": config.get("port"),
        "persistence": config.get("persistence", "none"), # Default to "none" if not specified
    }

    if monitor_type == "monitor":
        start_monitor(**monitor_args)
    elif monitor_type == "monitor_schedule":
        monitor_args["schedule"] = config.get("schedule")
        monitor_args["early_start"] = config.get("early_start")
        start_monitor_schedule(**monitor_args)
    elif monitor_type == "monitor_block":
        monitor_args["block_col"] = config.get("block_col")
        monitor_args["blocks_eval"] = config.get("blocks_eval")
        
        # Remove args not expected by start_monitor_block
        args_for_block = monitor_args.copy()
        args_for_block.pop("target_file", None)
        
        start_monitor_block(**args_for_block)
    elif monitor_type == "monitor_schedule_block":
        monitor_args["schedule"] = config.get("schedule")
        monitor_args["early_start"] = config.get("early_start")
        monitor_args["block_col"] = config.get("block_col")
        start_monitor_schedule_block(**monitor_args)
    elif monitor_type == "monitor_ipip":
        monitor_args["block_col"] = config.get("block_col")
        monitor_args["ipip_config"] = config.get("ipip_config")
        start_monitor_ipip(**monitor_args)
    elif monitor_type == "monitor_schedule_ipip":
        monitor_args["schedule"] = config.get("schedule")
        monitor_args["early_start"] = config.get("early_start")
        monitor_args["block_col"] = config.get("block_col")
        monitor_args["ipip_config"] = config.get("ipip_config")
        start_monitor_schedule_ipip(**monitor_args)
    else:
        print(f"Error: Unknown monitor_type '{monitor_type}'.")
        sys.exit(1)

def update_pipeline_config(pipeline_name: str, preprocess_file: str = None, custom_train_file: str = None, custom_retrain_file: str = None, schedule: str = None, retrain_mode: int = None, fallback_mode: int = None, thresholds_drift: str = None, thresholds_perf: str = None, data_dir: str = None, random_state: int = None, param_grid: str = None, cv: int = None, port: int = None, window_size: int = None):
    """
    Updates the configuration of an existing pipeline.
    """
    pipeline_dir = os.path.join(ROOT_DIR, "pipelines", pipeline_name)
    if not os.path.isdir(pipeline_dir):
        print(f"Error: Pipeline '{pipeline_name}' not found.")
        sys.exit(1)

    config_path = os.path.join(pipeline_dir, "config", "runner_config.json")
    if not os.path.exists(config_path):
        print(f"Error: Could not find runner_config.json for pipeline '{pipeline_name}'.")
        sys.exit(1)

    print(f"Updating configuration for pipeline '{pipeline_name}'...")
    with open(config_path, "r") as f:
        config = json.load(f)

    updated = False
    if preprocess_file is not None:
        if not os.path.exists(preprocess_file):
            print(f"Error: New preprocess_file path '{preprocess_file}' does not exist.")
            sys.exit(1)
        config["preprocess_file"] = preprocess_file
        updated = True
        print(f"Updated preprocess_file to: {preprocess_file}")

    if custom_train_file is not None:
        if not os.path.exists(custom_train_file):
            print(f"Error: New custom_train_file path '{custom_train_file}' does not exist.")
            sys.exit(1)
        config["custom_train_file"] = custom_train_file
        updated = True
        print(f"Updated custom_train_file to: {custom_train_file}")

    if custom_retrain_file is not None:
        if not os.path.exists(custom_retrain_file):
            print(f"Error: New custom_retrain_file path '{custom_retrain_file}' does not exist.")
            sys.exit(1)
        config["custom_retrain_file"] = custom_retrain_file
        updated = True
        print(f"Updated custom_retrain_file to: {custom_retrain_file}")

    if schedule is not None:
        try:
            schedule_dict = json.loads(schedule)
            if "type" not in schedule_dict or "params" not in schedule_dict:
                print("Error: Invalid schedule format. Must be a JSON string with 'type' and 'params' keys.")
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
        if not os.path.isdir(data_dir):
            print(f"Error: New data_dir path '{data_dir}' does not exist or is not a directory.")
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
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Configuration for pipeline '{pipeline_name}' updated successfully.")
    else:
        print("No parameters specified for update.")

if __name__ == "__main__":
    main()
