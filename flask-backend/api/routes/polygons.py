"""Polygon routes."""
import json
from flask import Blueprint, jsonify
from ..config import POLYGONS_DIR

polygons_bp = Blueprint('polygons', __name__)


@polygons_bp.route('/api/polygons/list', methods=['GET'])
def get_polygon_files():
    """
    Return list of available polygon JSON files.
    """
    try:
        if not POLYGONS_DIR.exists():
            return jsonify({"error": "Polygons directory not found"}), 404
        
        json_files = sorted([f.name for f in POLYGONS_DIR.glob("*.json")])
        return jsonify({"files": json_files}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@polygons_bp.route('/api/polygons/<filename>', methods=['GET'])
def get_polygon_file(filename: str):
    """
    Return polygon JSON file content.
    Validates filename to prevent path traversal.
    """
    try:
        # Validate filename to prevent path traversal
        if not filename.endswith('.json') or '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({"error": "Invalid filename"}), 400
        
        file_path = POLYGONS_DIR / filename
        if not file_path.exists():
            return jsonify({"error": f"File '{filename}' not found"}), 404
        
        # Ensure file is within POLYGONS_DIR (additional safety check)
        # Using try/except with relative_to() for Python 3.8 compatibility
        try:
            file_path.resolve().relative_to(POLYGONS_DIR.resolve())
        except ValueError:
            return jsonify({"error": "Invalid file path"}), 400
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return jsonify(data), 200
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON file"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
