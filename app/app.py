import sys
from pathlib import Path
from datetime import datetime

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, render_template, request, send_from_directory
from src.pipeline.prediction_pipeline import assess_child
from src.reports.report_generator import generate_pdf_report

print("RUNNING APP FROM:", __file__)

app = Flask(__name__)

def validate_input(child):

    errors = []

    if not (0 <= child["age_months"] <= 60):
        errors.append("Age must be between 0 and 60 months.")

    if not (1 <= child["weight_kg"] <= 30):
        errors.append("Weight must be between 1 and 30 kg.")

    if not (30 <= child["height_cm"] <= 130):
        errors.append("Height must be between 30 and 130 cm.")

    if child["muac_cm"] is not None:
        if not (5 <= child["muac_cm"] <= 30):
            errors.append("MUAC must be between 5 and 30 cm.")
    
    return errors

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
        "muac_cm": float(request.form["muac_cm"]) if request.form.get("muac_cm") else None,
        "wealth_index": int(request.form["wealth_index"]),
        "mother_education": int(request.form["mother_education"]),
        "currently_breastfeeding": int(request.form["currently_breastfeeding"])
    }

    errors = validate_input(sample_child)
    if errors:
        return render_template("index.html", errors=errors, child=sample_child)

    result = assess_child(sample_child)

    pdf_path = generate_pdf_report(sample_child, result)

    filename = Path(pdf_path).name

    return render_template(
        "results.html",
        features=sample_child,
        result=result,
        today=datetime.now(),
        pdf_filename=filename
    )

@app.route("/download/<filename>")
def download_report(filename):

    reports_folder = PROJECT_ROOT / "generated_reports"

    return send_from_directory(
        reports_folder,
        filename,
        as_attachment=True
    )
if __name__ == "__main__":
    app.run(debug=True)