"""Flask application factory."""
import os
import logging
from flask import Flask
from flask_cors import CORS
from .routes import (
    auth_bp,
    health_bp,
    config_bp,
    positions_bp,
    polygons_bp,
    topics_bp,
    content_bp,
    search_bp,
    export_bp,
    canvases_bp,
    topic_presets_bp,
    static_bp,
)
from .routes.auth import require_auth_before_request


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Configure logging to show INFO level messages
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    # Configure CORS with allowed origins from environment variable
    # Default to localhost for development; set CORS_ORIGINS in production
    # Example: CORS_ORIGINS=https://example.com,https://www.example.com
    cors_origins = os.getenv('CORS_ORIGINS')
    CORS(app, origins=cors_origins, supports_credentials=True)

    # Global authentication check — runs before every request
    app.before_request(require_auth_before_request)

    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(config_bp)
    app.register_blueprint(positions_bp)
    app.register_blueprint(polygons_bp)
    app.register_blueprint(topics_bp)
    app.register_blueprint(content_bp)
    app.register_blueprint(search_bp)
    app.register_blueprint(export_bp)
    app.register_blueprint(canvases_bp)
    app.register_blueprint(topic_presets_bp)
    app.register_blueprint(static_bp)

    return app
