"""Health check route."""
from flask import Blueprint

health_bp = Blueprint('health', __name__)


@health_bp.route('/health')
def health():
    return {'status': 'healthy'}, 200
