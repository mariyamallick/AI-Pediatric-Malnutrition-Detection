import joblib
from pathlib import Path

MODEL_DIR = Path("../../models")

underweight_model = joblib.load(
    MODEL_DIR / "underweight_status_model.pkl"
)

stunting_model = joblib.load(
    MODEL_DIR / "stunting_status_model.pkl"
)

wasting_model = joblib.load(
    MODEL_DIR / "wasting_status_model.pkl"
)
import pandas as pd

def predict_malnutrition(features):

    data = pd.DataFrame([features])

    underweight = underweight_model.predict(data)[0]

    stunting = stunting_model.predict(data)[0]

    wasting = wasting_model.predict(data)[0]

    return {
        "underweight": int(underweight),
        "stunting": int(stunting),
        "wasting": int(wasting)
    }
from recommendations.nutrition_recommendation import generate_recommendation
def assess_child(features):

    predictions = predict_malnutrition(features)

    result = generate_recommendation(
        age_months=features["age_months"],
        muac=features["muac_cm"],
        waz=features["waz"],
        haz=features["haz"],
        whz=features["whz"],
        underweight=predictions["underweight"],
        stunting=predictions["stunting"],
        wasting=predictions["wasting"]
    )

    return result