"""Canvas utility functions for UI state persistence."""
import json
import os
import logging
from ..config import CANVASES_FILE


def load_canvases():
    """Load canvas configurations from file."""
    if os.path.exists(CANVASES_FILE):
        try:
            with open(CANVASES_FILE, "r") as f:
                content = f.read().strip()
                # If file is empty or only whitespace, return empty dict
                if not content:
                    return {}
                return json.loads(content)
        except (json.JSONDecodeError, ValueError):
            # If JSON is invalid, return empty dict and log warning
            logging.warning(f"Invalid JSON in {CANVASES_FILE}, returning empty dict")
            return {}
    return {}


def save_canvases(data):
    """Save canvas configurations to file."""
    with open(CANVASES_FILE, "w") as f:
        json.dump(data, f, indent=4)
