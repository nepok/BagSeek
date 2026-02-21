"""Configuration routes."""
import os
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from flask import Blueprint, jsonify, request
from ..config import EMBEDDINGS, ROSBAGS, FILE_PATH_CACHE_TTL_SECONDS, VALID_ROSBAGS_INDEX
from ..state import get_selected_rosbag, SELECTED_MODEL, set_aligned_data, set_selected_rosbag, SEARCHED_ROSBAGS, _matching_rosbag_cache, _file_path_cache_lock, _lookup_table_cache, _positional_lookup_cache
from ..utils.rosbag import extract_rosbag_name_from_path

config_bp = Blueprint('config', __name__)


def _is_safe_path(path: str) -> bool:
    """Validate path to prevent directory traversal attacks."""
    if not path:
        return False
    return '..' not in path and not path.startswith('/')


@config_bp.route('/api/get-models', methods=['GET'])
def get_models():
    """
    List available CLIP models (based on embeddings directory).
    """
    try:
        models = []
        for model in os.listdir(EMBEDDINGS):
            if not model in ['.DS_Store', 'README.md', 'completion.json']:
                models.append(model)
        return jsonify({"models": models}), 200
    except Exception as e:
        # Handle any errors that occur (e.g., directory not found, permission issues)
        return jsonify({"error": str(e)}), 500


@config_bp.route('/api/set-file-paths', methods=['POST'])
def post_file_paths():
    """
    Set the current Rosbag file. Aligned data is loaded lazily when needed.
    """
    try:
        data = request.get_json()  # Get the JSON payload
        path_value = data.get('path')  # The path value from the JSON

        # Security: Validate path to prevent directory traversal
        if not path_value or not _is_safe_path(path_value):
            return jsonify({"error": "Invalid path"}), 400

        # Update state
        set_selected_rosbag(path_value)

        # Clear cached aligned data so it reloads for the new rosbag
        set_aligned_data(None)

        SEARCHED_ROSBAGS.clear()
        SEARCHED_ROSBAGS.append(path_value)  # Reset searched rosbags to the selected one
        logging.warning(SEARCHED_ROSBAGS)

        # Pre-warm aligned data in background so the first set-reference-timestamp call
        # is fast (otherwise it would trigger a slow NAS read on-demand).
        def _prewarm_aligned_data():
            from ..state import get_aligned_data
            get_aligned_data()

        threading.Thread(target=_prewarm_aligned_data, daemon=True).start()

        return jsonify({"message": f"File path updated successfully to {path_value}."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _scan_rosbag_paths() -> list[str]:
    """
    Scan ROSBAGS directory and return list of valid rosbag paths.
    Filters to only include rosbags that have embeddings in at least one model.
    """
    rosbag_paths: list[str] = []
    base_path = ROSBAGS

    if not base_path.exists():
        return []

    # Recursively walk through all directories
    for root, dirs, files in os.walk(str(base_path), topdown=True):
        # Skip EXCLUDED and EXPORTED directories
        if "EXCLUDED" in root or "EXPORTED" in root:
            dirs[:] = []  # Don't recurse into excluded directories
            continue

        # Remove excluded subdirectories from dirs list to skip them
        dirs[:] = [d for d in dirs if "EXCLUDED" not in d and "EXPORTED" not in d]

        # If this directory has no subdirectories (or only excluded ones), it's a leaf
        if not dirs:
            root_path = Path(root)
            # Skip the base directory itself
            if root_path != base_path:
                relative_path = root_path.relative_to(base_path)
                rosbag_paths.append(str(relative_path))

    rosbag_paths.sort()

    # Filter rosbags to only include those that have embeddings in at least one model
    if EMBEDDINGS and EMBEDDINGS.exists():
        rosbags_base_path = ROSBAGS
        filtered_paths = []

        for relative_path in rosbag_paths:
            # Check if rosbag exists in ROSBAGS
            rosbag_full_path = rosbags_base_path / relative_path
            rosbag_exists = rosbag_full_path.exists() and rosbag_full_path.is_dir()

            if not rosbag_exists:
                continue

            # Check if this rosbag exists in EMBEDDINGS for any model
            found_in_embeddings = False
            for model_dir in EMBEDDINGS.iterdir():
                if not model_dir.is_dir():
                    continue

                embeddings_rosbag_path = model_dir / relative_path
                if embeddings_rosbag_path.exists() and embeddings_rosbag_path.is_dir():
                    found_in_embeddings = True
                    break

            if found_in_embeddings:
                filtered_paths.append(relative_path)

        rosbag_paths = filtered_paths

    return rosbag_paths


def _read_index_file() -> list[str] | None:
    """
    Read valid rosbag paths from index file.
    Returns None if file doesn't exist or is invalid.
    """
    try:
        if not VALID_ROSBAGS_INDEX.exists():
            return None

        with open(VALID_ROSBAGS_INDEX, 'r') as f:
            data = json.load(f)

        if not isinstance(data, dict) or 'paths' not in data:
            return None

        return data['paths']
    except (json.JSONDecodeError, IOError):
        return None


def _write_index_file(paths: list[str]) -> None:
    """Write valid rosbag paths to index file."""
    data = {
        "paths": paths,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    with open(VALID_ROSBAGS_INDEX, 'w') as f:
        json.dump(data, f, indent=2)


@config_bp.route('/api/get-file-paths', methods=['GET'])
def get_file_paths():
    """
    Return all available Rosbag file paths.
    First tries to read from index file, falls back to directory scan.
    """
    from time import time

    try:
        now = time()
        with _file_path_cache_lock:
            cached_paths = list(_matching_rosbag_cache["paths"])
            cached_timestamp = _matching_rosbag_cache["timestamp"]

        cache_age = now - cached_timestamp

        # Return from memory cache if fresh
        if cached_paths and cache_age < FILE_PATH_CACHE_TTL_SECONDS:
            return jsonify({"paths": cached_paths}), 200

        # Try to read from index file (fast path)
        index_paths = _read_index_file()
        if index_paths is not None:
            with _file_path_cache_lock:
                _matching_rosbag_cache["paths"] = list(index_paths)
                _matching_rosbag_cache["timestamp"] = now
            return jsonify({"paths": index_paths}), 200

        # Fallback: full directory scan (slow path)
        rosbag_paths = _scan_rosbag_paths()

        with _file_path_cache_lock:
            _matching_rosbag_cache["paths"] = list(rosbag_paths)
            _matching_rosbag_cache["timestamp"] = now

        return jsonify({"paths": rosbag_paths}), 200
    except Exception as e:
        logging.error(f"Error in get_file_paths: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@config_bp.route('/api/refresh-file-paths', methods=['POST'])
def refresh_file_paths():
    """
    Force refresh of rosbag file paths.
    Performs full directory scan, updates index file, and returns new paths.
    """
    from time import time

    try:
        # Perform full directory scan
        rosbag_paths = _scan_rosbag_paths()

        # Write to index file
        _write_index_file(rosbag_paths)

        # Update memory cache
        now = time()
        with _file_path_cache_lock:
            _matching_rosbag_cache["paths"] = list(rosbag_paths)
            _matching_rosbag_cache["timestamp"] = now

        return jsonify({"paths": rosbag_paths}), 200
    except Exception as e:
        logging.error(f"Error in refresh_file_paths: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@config_bp.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """
    Clear all backend caches (lookup tables, file paths, positional lookup).
    Useful for forcing fresh data reload after code changes.
    """
    try:
        # Clear lookup table cache
        _lookup_table_cache.clear()
        
        # Clear file path cache
        with _file_path_cache_lock:
            _matching_rosbag_cache["paths"] = []
            _matching_rosbag_cache["timestamp"] = 0.0
        
        # Clear positional lookup cache
        _positional_lookup_cache["data"] = None
        _positional_lookup_cache["mtime"] = None
        
        # Clear aligned data to force reload
        set_aligned_data(None)
        
        return jsonify({"message": "All caches cleared successfully"}), 200
    except Exception as e:
        logging.error(f"Error clearing cache: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@config_bp.route('/api/get-selected-rosbag', methods=['GET'])
def get_selected_rosbag_route():
    """
    Return the currently selected Rosbag file name, or null if none selected.
    """
    try:
        selected_rosbag = get_selected_rosbag()
        if selected_rosbag is None:
            return jsonify({"selectedRosbag": None}), 200

        selectedRosbag = extract_rosbag_name_from_path(str(selected_rosbag))
        return jsonify({"selectedRosbag": selectedRosbag}), 200
    except Exception as e:
        # Handle any errors that occur (e.g., directory not found, permission issues)
        return jsonify({"error": str(e)}), 500
