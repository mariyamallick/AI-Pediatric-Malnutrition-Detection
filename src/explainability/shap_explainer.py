import shap
import pandas as pd
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"


def explain_prediction(features, model_name="underweight"):

    model = joblib.load(MODEL_DIR / f"{model_name}_model.pkl")

    X = pd.DataFrame([features])

    explainer = shap.Explainer(model)

    shap_values = explainer(X)

    if isinstance(shap_values, list):
        values = shap_values[0].values[0]
    else:
        values = shap_values.values[0]

    explanation = []

    for feature, value in zip(X.columns, values):
        explanation.append({
            "feature": feature,
            "impact": round(float(value), 3)
        })

    explanation = sorted(
        explanation,
        key=lambda x: abs(x["impact"]),
        reverse=True
    )

    return explanation