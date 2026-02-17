"""Topic preset utility functions for export topic persistence."""
import json
import os
import logging
from ..config import TOPIC_PRESETS_FILE


def load_topic_presets():
    """Load topic presets from file."""
    if os.path.exists(TOPIC_PRESETS_FILE):
        try:
            with open(TOPIC_PRESETS_FILE, "r") as f:
                content = f.read().strip()
                if not content:
                    return {}
                return json.loads(content)
        except (json.JSONDecodeError, ValueError):
            logging.warning(f"Invalid JSON in {TOPIC_PRESETS_FILE}, returning empty dict")
            return {}
    return {}


def save_topic_presets(data):
    """Save topic presets to file."""
    os.makedirs(os.path.dirname(TOPIC_PRESETS_FILE), exist_ok=True)
    with open(TOPIC_PRESETS_FILE, "w") as f:
        json.dump(data, f, indent=4)
