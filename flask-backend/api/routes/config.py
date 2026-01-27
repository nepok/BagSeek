"""Configuration routes."""
import os
import logging
from pathlib import Path
from flask import Blueprint, jsonify, request
from ..config import EMBEDDINGS, ROSBAGS, FILE_PATH_CACHE_TTL_SECONDS
from ..state import SELECTED_ROSBAG, SELECTED_MODEL, set_aligned_data, set_selected_rosbag, SEARCHED_ROSBAGS, _matching_rosbag_cache, _file_path_cache_lock
from ..utils.rosbag import extract_rosbag_name_from_path, load_lookup_tables_for_rosbag

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
    Set the current Rosbag file and update alignment data accordingly.
    """
    try:
        data = request.get_json()  # Get the JSON payload
        path_value = data.get('path')  # The path value from the JSON

        # Security: Validate path to prevent directory traversal
        if not path_value or not _is_safe_path(path_value):
            return jsonify({"error": "Invalid path"}), 400

        # Update state
        set_selected_rosbag(path_value)

        rosbag_name = extract_rosbag_name_from_path(str(path_value))
        aligned_data = load_lookup_tables_for_rosbag(rosbag_name)
        set_aligned_data(aligned_data)

        SEARCHED_ROSBAGS.clear()
        SEARCHED_ROSBAGS.append(path_value)  # Reset searched rosbags to the selected one
        logging.warning(SEARCHED_ROSBAGS)
        return jsonify({"message": f"File path updated successfully to {path_value}."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@config_bp.route('/api/get-file-paths', methods=['GET'])
def get_file_paths():
    """
    Return all available Rosbag file paths (excluding those in EXCLUDED).
    Recursively scans ROSBAGS directory and collects all leaf directories.
    """
    from time import time
    
    try:
        now = time()
        with _file_path_cache_lock:
            cached_paths = list(_matching_rosbag_cache["paths"])
            cached_timestamp = _matching_rosbag_cache["timestamp"]

        cache_age = now - cached_timestamp
        
        if (
            cached_paths
            and cache_age < FILE_PATH_CACHE_TTL_SECONDS
        ):
            return jsonify({"paths": cached_paths}), 200

        rosbag_paths: list[str] = []
        base_path = ROSBAGS
        
        if not base_path.exists():
            return jsonify({"paths": []}), 200
        
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
            
            # Filter: check if rosbag exists in ROSBAGS AND in EMBEDDINGS (any model)
            original_count = len(rosbag_paths)
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

        with _file_path_cache_lock:
            _matching_rosbag_cache["paths"] = list(rosbag_paths)
            _matching_rosbag_cache["timestamp"] = now

        return jsonify({"paths": rosbag_paths}), 200
    except Exception as e:
        logging.error(f"Error in get_file_paths: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@config_bp.route('/api/get-selected-rosbag', methods=['GET'])
def get_selected_rosbag():
    """
    Return the currently selected Rosbag file name.
    """
    try:
        selectedRosbag = extract_rosbag_name_from_path(str(SELECTED_ROSBAG))
        return jsonify({"selectedRosbag": selectedRosbag}), 200
    except Exception as e:
        # Handle any errors that occur (e.g., directory not found, permission issues)
        return jsonify({"error": str(e)}), 500
