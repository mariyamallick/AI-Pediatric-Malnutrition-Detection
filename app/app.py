import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, render_template, request
from src.pipeline.prediction_pipeline import assess_child

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    sample_child = {

        "age_months": int(request.form["age_months"]),

        "sex": int(request.form["sex"]),

        "weight_kg": float(request.form["weight_kg"]),

        "height_cm": float(request.form["height_cm"]),

        "muac_cm": float(request.form["muac_cm"]),

        "waz": float(request.form["waz"]),

        "haz": float(request.form["haz"]),

        "whz": float(request.form["whz"])

    }

    result = assess_child(sample_child)

    return render_template(
        "results.html",
        result=result,
        features=sample_child
    )


if __name__ == "__main__":
    app.run(debug=True)