"""Static file serving routes."""
from flask import Blueprint, send_from_directory, jsonify
from ..config import ADJACENT_SIMILARITIES, IMAGE_TOPIC_PREVIEWS

static_bp = Blueprint('static', __name__)


def _is_safe_path(path: str) -> bool:
    """Validate path to prevent directory traversal attacks."""
    return '..' not in path and not path.startswith('/')


@static_bp.route('/adjacency-image/<path:adjacency_image_path>')
def serve_adjacency_image(adjacency_image_path):
    """
    Serve an extracted image file from the backend image directory.
    """
    if not _is_safe_path(adjacency_image_path):
        return jsonify({'error': 'Invalid path'}), 400
    response = send_from_directory(ADJACENT_SIMILARITIES, adjacency_image_path)
    response.headers["Cache-Control"] = "public, max-age=86400"  # Cache for 1 day
    return response


@static_bp.route('/image-topic-preview/<path:image_topic_preview_path>')
def serve_image_topic_preview(image_topic_preview_path):
    """
    Serve an extracted image file from the backend image directory.
    """
    if not _is_safe_path(image_topic_preview_path):
        return jsonify({'error': 'Invalid path'}), 400
    response = send_from_directory(IMAGE_TOPIC_PREVIEWS, image_topic_preview_path)
    response.headers["Cache-Control"] = "public, max-age=86400"  # Cache for 1 day
    return response
