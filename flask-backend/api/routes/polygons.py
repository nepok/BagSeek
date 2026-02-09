"""Polygon routes."""
import json
import re
from flask import Blueprint, jsonify, request
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


@polygons_bp.route('/api/polygons', methods=['POST'])
def save_polygon_file():
    """
    Save polygon data as a GeoJSON file.
    Expects JSON body with 'name' and 'geojson' fields.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        name = data.get('name')
        geojson = data.get('geojson')

        if not name:
            return jsonify({"error": "Name is required"}), 400
        if not geojson:
            return jsonify({"error": "GeoJSON data is required"}), 400

        # Sanitize filename: allow only alphanumeric, spaces, hyphens, underscores
        sanitized_name = re.sub(r'[^a-zA-Z0-9\s\-_]', '', name).strip()
        if not sanitized_name:
            return jsonify({"error": "Invalid name"}), 400

        filename = f"{sanitized_name}.json"

        # Validate the filename doesn't contain path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({"error": "Invalid filename"}), 400

        # Ensure polygons directory exists
        if not POLYGONS_DIR.exists():
            POLYGONS_DIR.mkdir(parents=True, exist_ok=True)

        file_path = POLYGONS_DIR / filename

        # Ensure file is within POLYGONS_DIR
        try:
            file_path.resolve().relative_to(POLYGONS_DIR.resolve())
        except ValueError:
            return jsonify({"error": "Invalid file path"}), 400

        # Validate GeoJSON structure
        if not isinstance(geojson, dict) or geojson.get('type') != 'FeatureCollection':
            return jsonify({"error": "Invalid GeoJSON: must be a FeatureCollection"}), 400

        # Write the file
        with open(file_path, 'w') as f:
            json.dump(geojson, f, indent=2)

        return jsonify({"success": True, "filename": filename}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@polygons_bp.route('/api/polygons/<filename>', methods=['DELETE'])
def delete_polygon_file(filename: str):
    """
    Delete a polygon JSON file.
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
        try:
            file_path.resolve().relative_to(POLYGONS_DIR.resolve())
        except ValueError:
            return jsonify({"error": "Invalid file path"}), 400

        file_path.unlink()

        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
