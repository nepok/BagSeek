"""Route blueprints for Flask API."""
from .auth import auth_bp
from .health import health_bp
from .config import config_bp
from .positions import positions_bp
from .polygons import polygons_bp
from .topics import topics_bp
from .content import content_bp
from .search import search_bp
from .export import export_bp
from .canvases import canvases_bp
from .topic_presets import topic_presets_bp
from .static import static_bp

__all__ = [
    'auth_bp',
    'health_bp',
    'config_bp',
    'positions_bp',
    'polygons_bp',
    'topics_bp',
    'content_bp',
    'search_bp',
    'export_bp',
    'canvases_bp',
    'topic_presets_bp',
    'static_bp',
]
