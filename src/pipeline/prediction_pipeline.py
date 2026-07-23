import logging
from pyexpat import features

import joblib

import pandas as pd

from pathlib import Path

from src.growth.growth_calculator import calculate_z_scores

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


    growth = calculate_z_scores(
    age_months=features["age_months"],
    sex=features["sex"],
    weight_kg=features["weight_kg"],
    height_cm=features["height_cm"]
    )

    features["waz"] = growth["waz"]
    features["haz"] = growth["haz"]
    features["whz"] = growth["whz"]
    features["bmi"] = growth["bmi"]

    result = {"WHO Growth": growth}

    data = pd.DataFrame([features], columns=FEATURE_COLUMNS)

    underweight = underweight_model.predict(data)[0]
    underweight_conf = max(underweight_model.predict_proba(data)[0])
    stunting = stunting_model.predict(data)[0]
    stunting_conf = max(stunting_model.predict_proba(data)[0])
    wasting = wasting_model.predict(data)[0]
    wasting_conf = max(wasting_model.predict_proba(data)[0])

    result.update({
        "underweight": underweight,
        "stunting": stunting,
        "wasting": wasting,
        "confidence": {
            "underweight": round(underweight_conf * 100, 2),
            "stunting": round(stunting_conf * 100, 2),
            "wasting": round(wasting_conf * 100, 2)
        }
    })

    return result



def calculate_risk(prediction):



    score = 0



    if prediction["Underweight"]:

        score += 1



    if prediction["Stunting"]:

        score += 1



    if prediction["Wasting"]:

        score += 1



    if score == 0:

        return "🟢 Low Risk"



    elif score == 1:

        return "🟡 Moderate Risk"



    else:

        return "🔴 High Risk"





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

    positive_predictions = sum([

        predictions["underweight"],

        predictions["stunting"],

        predictions["wasting"]

    ])



    if positive_predictions == 0:

        risk = "Low"

    elif positive_predictions == 1:

        risk = "Moderate"

    else:

        risk = "High"


    result["Confidence"]=predictions["confidence"]

    

    result["Overall Risk"] = calculate_risk(

    result["Prediction"]

    )

    print(result)      

    return result