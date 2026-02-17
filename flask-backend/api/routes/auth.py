"""Authentication routes and middleware."""
import logging
from datetime import datetime, timezone, timedelta

import jwt
from flask import Blueprint, request, jsonify, make_response

from ..config import APP_PASSWORD, JWT_SECRET, JWT_EXPIRY_SECONDS

auth_bp = Blueprint('auth', __name__)

# Paths that do NOT require authentication
AUTH_EXEMPT_PREFIXES = ('/api/login', '/api/logout', '/api/auth-check', '/health')


def _decode_token(token: str) -> dict | None:
    """Decode and validate a JWT token. Returns payload or None."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


def require_auth_before_request():
    """Flask before_request hook enforcing auth on all routes except exempted ones."""
    if not APP_PASSWORD:
        return None

    if request.path.startswith(AUTH_EXEMPT_PREFIXES):
        return None

    token = request.cookies.get('auth_token')
    if not token:
        return jsonify({'error': 'Authentication required'}), 401

    payload = _decode_token(token)
    if payload is None:
        response = make_response(jsonify({'error': 'Token expired or invalid'}), 401)
        response.delete_cookie('auth_token', path='/')
        return response

    return None


@auth_bp.route('/api/login', methods=['POST'])
def login():
    """Authenticate with the shared password. Sets httpOnly JWT cookie."""
    if not APP_PASSWORD:
        return jsonify({'message': 'Authentication disabled'}), 200

    data = request.get_json(silent=True)
    if not data or 'password' not in data:
        return jsonify({'error': 'Password is required'}), 400

    if data['password'] != APP_PASSWORD:
        return jsonify({'error': 'Invalid password'}), 401

    now = datetime.now(timezone.utc)
    payload = {
        'iat': now,
        'exp': now + timedelta(seconds=JWT_EXPIRY_SECONDS),
        'sub': 'user',
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")

    response = make_response(jsonify({'message': 'Login successful'}), 200)
    response.set_cookie(
        'auth_token',
        value=token,
        httponly=True,
        secure=False,
        samesite='Lax',
        max_age=JWT_EXPIRY_SECONDS,
        path='/',
    )
    return response


@auth_bp.route('/api/logout', methods=['POST'])
def logout():
    """Clear the authentication cookie."""
    response = make_response(jsonify({'message': 'Logged out'}), 200)
    response.delete_cookie('auth_token', path='/')
    return response


@auth_bp.route('/api/auth-check', methods=['GET'])
def auth_check():
    """Check if the current session is authenticated. Never returns 401."""
    if not APP_PASSWORD:
        return jsonify({'authenticated': True, 'authDisabled': True}), 200

    token = request.cookies.get('auth_token')
    if not token:
        return jsonify({'authenticated': False}), 200

    payload = _decode_token(token)
    if payload is None:
        response = make_response(jsonify({'authenticated': False}), 200)
        response.delete_cookie('auth_token', path='/')
        return response

    return jsonify({'authenticated': True}), 200
