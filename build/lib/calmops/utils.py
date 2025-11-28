import os
from pathlib import Path

def get_project_root() -> Path:
    """
    Returns the project root, which is the parent directory of the 'calmops' package.
    """
    # This file is in /calmops/calmops/utils.py, so one parent up is the project root.
    return Path(__file__).parent
