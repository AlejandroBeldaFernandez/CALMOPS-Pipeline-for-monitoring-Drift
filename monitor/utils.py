# monitor/utils.py
# -*- coding: utf-8 -*-
"""
Pipeline Management Utilities
"""

import os
import shutil
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.get_logger().setLevel('ERROR')
def _get_logger(name: str | None = None) -> logging.Logger:
    """
    Factory function for creating contextual loggers with consistent formatting.
    """
    if name is None:
        name = __name__
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # If there's already a custom handler, don't propagate to avoid double output
    if logger.handlers:
        logger.propagate = False
    return logger

def list_pipelines(base_dir="pipelines"):
    """
    Administrative utility for pipeline discovery and inventory.
    
    Provides operational visibility into deployed monitoring pipelines
    for management and maintenance purposes.
    """
    log = _get_logger("calmops.monitor.utils")
    try:
        pipelines = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not pipelines:
            log.info("No monitoring pipelines currently deployed")
        else:
            log.info("Currently deployed monitoring pipelines:")
            for pipeline in pipelines:
                log.info(f"- {pipeline}")
    except Exception as e:
        log.error(f"Pipeline discovery failed: {e}")

def delete_pipeline(pipeline_name, base_dir="pipelines"):
    """
    Permanent pipeline removal with safety confirmation.
    
    Implements secure deletion with user confirmation to prevent
    accidental removal of production monitoring configurations.
    """
    log = _get_logger("calmops.monitor.utils")
    pipeline_path = os.path.join(base_dir, pipeline_name)
    if not os.path.exists(pipeline_path):
        log.error(f"Pipeline '{pipeline_name}' not found in deployment registry")
        return
    try:
        confirmation = input(
            f"Are you sure you want to delete the pipeline '{pipeline_name}'? This action is irreversible. (y/n): "
        )
        if confirmation.lower() != 'y':
            log.info("Pipeline deletion cancelled by user")
            return
        shutil.rmtree(pipeline_path)
        log.info(f"Pipeline '{pipeline_name}' successfully removed from system")
    except Exception as e:
        log.error(f"Pipeline deletion failed for '{pipeline_name}': {e}")
