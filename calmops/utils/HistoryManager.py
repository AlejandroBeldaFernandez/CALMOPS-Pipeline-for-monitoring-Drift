import json
import os
from typing import List, Dict
from datetime import datetime


class HistoryManager:
    """
    Manages the saving and loading of historical execution results for pipelines.
    """

    @staticmethod
    def load_history(history_file: str) -> List[Dict]:
        """
        Loads the history list from a JSON file.

        Args:
            history_file (str): Path to the history JSON file.

        Returns:
            List[Dict]: List of history records. Returns empty list if file doesn't exist or is invalid.
        """
        if not os.path.exists(history_file):
            return []

        try:
            with open(history_file, "r") as f:
                history = json.load(f)
                if isinstance(history, list):
                    return history
                else:
                    return []
        except (json.JSONDecodeError, IOError):
            return []

    @staticmethod
    def append_history_record(history_file: str, record: Dict, max_history: int = 5):
        """
        Appends a record to the history file, maintaining a maximum size.

        Args:
            history_file (str): Path to the history JSON file.
            record (Dict): The record to append.
            max_history (int): Maximum number of records to keep.
        """
        # Load existing history
        history = HistoryManager.load_history(history_file)

        # Add timestamp if not present
        if "timestamp" not in record:
            record["timestamp"] = datetime.now().isoformat()

        # Append new record
        history.append(record)

        # Trim to max_history (keep latest)
        if len(history) > max_history:
            history = history[-max_history:]

        # Save back to file
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(history_file)), exist_ok=True)

        try:
            with open(history_file, "w") as f:
                json.dump(history, f, indent=4)
        except IOError as e:
            print(f"Error saving history to {history_file}: {e}")
