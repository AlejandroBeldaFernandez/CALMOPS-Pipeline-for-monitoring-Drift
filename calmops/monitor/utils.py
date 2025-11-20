# monitor/utils.py
# -*- coding: utf-8 -*-
"""
Pipeline Management Utilities
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.get_logger().setLevel('ERROR')

import logging

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


