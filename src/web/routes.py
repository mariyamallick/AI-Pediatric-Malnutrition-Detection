"""Routes for the Pediatric Malnutrition Detection web interface."""

from flask import Blueprint, current_app, render_template, request

from .forms import validate_measurements


web = Blueprint("web", __name__)


@web.get("/")
def index():
    """Render the screening form."""
    return render_template("index.html", values={}, errors={})


@web.post("/assess")
def assess():
    """Validate submitted measurements and display screening results."""
    features, errors = validate_measurements(request.form)
    values = request.form.to_dict()

    if errors:
        return render_template("index.html", values=values, errors=errors), 400

    try:
        from src.pipeline.prediction_pipeline import assess_child

        result = assess_child(features)
    except Exception:
        current_app.logger.exception("Nutrition screening failed")
        return render_template("error.html"), 500

    return render_template("results.html", result=result, features=features)
