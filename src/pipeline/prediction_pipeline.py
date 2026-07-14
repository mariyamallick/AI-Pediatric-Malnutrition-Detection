import logging

import joblib
import pandas as pd
from pathlib import Path

from src.recommendations.nutrition_recommendation import generate_recommendation

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Models folder
MODEL_DIR = PROJECT_ROOT / "models"
FEATURE_COLUMNS = [
    "age_months", "sex", "weight_kg", "height_cm", "muac_cm",
    "waz", "haz", "whz", "bmi",
]

# Load trained models
logger = logging.getLogger(__name__)
underweight_model = joblib.load(MODEL_DIR / "underweight_status_model.pkl")
stunting_model = joblib.load(MODEL_DIR / "stunting_status_model.pkl")
wasting_model = joblib.load(MODEL_DIR / "wasting_status_model.pkl")

logger.info("All malnutrition prediction models loaded successfully.")


def predict_malnutrition(features):

    data = pd.DataFrame([features], columns=FEATURE_COLUMNS)

    underweight = underweight_model.predict(data)[0]
    stunting = stunting_model.predict(data)[0]
    wasting = wasting_model.predict(data)[0]

    return {
        "underweight": underweight,
        "stunting": stunting,
        "wasting": wasting
    }


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
