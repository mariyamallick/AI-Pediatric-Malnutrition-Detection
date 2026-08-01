import logging

import joblib

import pandas as pd

print("Loaded prediction_pipeline.py")

from pathlib import Path

from src.growth.growth_calculator import calculate_z_scores

from src.recommendations.nutrition_recommendation import generate_recommendation

from src.recommendations.clinical_summary import generate_clinical_summary

from src.recommendations.food_recommendation import generate_food_recommendation

from src.database.database import save_assessment

# Project root directory

PROJECT_ROOT = Path(__file__).resolve().parents[2]



# Models folder

MODEL_DIR = PROJECT_ROOT / "models"

FEATURE_COLUMNS = [
    "age_months",
    "sex",
    "weight_kg",
    "height_cm",
    "wealth_index",
    "mother_education",
    "currently_breastfeeding",
    "waz",
    "haz",
    "whz"
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

    def classify_z_score(z):
        if z < -3:
            return "Severe"
        elif z < -2:
            return "Moderate"
        else:
            return "Normal"

    growth["waz_status"] = classify_z_score(growth["waz"])
    growth["haz_status"] = classify_z_score(growth["haz"])
    growth["whz_status"] = classify_z_score(growth["whz"])

    

    model_input = pd.DataFrame([{
        "age_months": features["age_months"],
        "sex": features["sex"],
        "weight_kg": features["weight_kg"],
        "height_cm": features["height_cm"],
        "wealth_index": features["wealth_index"],
        "mother_education": features["mother_education"],
        "currently_breastfeeding": features["currently_breastfeeding"],
        "waz": features["waz"],
        "haz": features["haz"],
        "whz": features["whz"]
    }])

    data = model_input



    underweight = underweight_model.predict(data)[0]
    underweight_conf = max(underweight_model.predict_proba(data)[0])
    stunting = stunting_model.predict(data)[0]
    stunting_conf = max(stunting_model.predict_proba(data)[0])
    wasting = wasting_model.predict(data)[0]
    wasting_conf = max(wasting_model.predict_proba(data)[0])

    return {
        "underweight": underweight,
        "stunting": stunting,
        "wasting": wasting,
        "confidence": {
            "underweight": round(underweight_conf * 100, 2),
            "stunting": round(stunting_conf * 100, 2),
            "wasting": round(wasting_conf * 100, 2)
    },
    "WHO Growth": growth
    }



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

    print("Inside assess_child()")

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

    result["WHO Growth"] = predictions["WHO Growth"]

    result["Food Recommendations"] = generate_food_recommendation(
        predictions["underweight"],
        predictions["stunting"],
        predictions["wasting"],
        features["age_months"]
    )



    positive_predictions = sum([

        predictions["underweight"],

        predictions["stunting"],

        predictions["wasting"]

    ])



    result["Confidence"]=predictions["confidence"]

    

    result["Overall Risk"] = calculate_risk(

    result["Prediction"]

    )

    print(result)      

    result["Clinical Summary"] = generate_clinical_summary(result)

    save_assessment(features, result)

    return result

 