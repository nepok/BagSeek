"""Flask API entry point.

This file serves as the entry point for running the Flask application.
The actual application is created by the create_app() factory in __init__.py.
"""
from . import create_app

# Create the Flask application
app = create_app()

# Start Flask server (debug mode enabled for development)
# Note: debug=True is for local development only. In production, use a WSGI server and set debug=False.
if __name__ == '__main__':
    app.run(debug=True)
