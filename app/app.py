import sys
from pathlib import Path
from datetime import datetime

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, render_template, request, send_from_directory
from src.pipeline.prediction_pipeline import assess_child
from src.reports.report_generator import generate_pdf_report

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
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
        print("=" * 50)
        print("Predict route called")
        print("=" * 50)

        result = assess_child(sample_child)
        print("Assessment completed")
       
        pdf_path = generate_pdf_report(sample_child, result)
        filename = Path(pdf_path).name
        print("PDF created at:", pdf_path)
        
        return render_template(
            "result.html",
            features=sample_child,
            result=result,
            today=datetime.now(),
            pdf_filename=filename
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template(
            "error.html",
            error=str(e)
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