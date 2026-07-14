"""Flask application factory for the web interface."""

import os

from flask import Flask


def create_app(test_config=None):
    """Create and configure the Flask application."""
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("FLASK_SECRET_KEY", "development-only-change-me"),
        DEBUG=os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"},
    )

    if test_config is not None:
        app.config.update(test_config)

    from .routes import web

    app.register_blueprint(web)

    return app
