"""
Nutrition Recommendation Engine
AI for Pediatric Malnutrition Detection

This module generates nutrition recommendations based on:
- Age
- MUAC
- WAZ
- HAZ
- WHZ
- Underweight prediction
- Stunting prediction
- Wasting prediction
"""

def generate_recommendation(
    age_months,
    muac,
    waz,
    haz,
    whz,
    underweight,
    stunting,
    wasting
):
    result = {
        "Prediction": {},
        "Assessment": {},
        "Recommendations": [],
        "Alerts": []
    }

    # -----------------------------
    # Prediction Results
    # -----------------------------
    result["Prediction"] = {
        "Underweight": bool(underweight),
        "Stunting": bool(stunting),
        "Wasting": bool(wasting)
    }

    # -----------------------------
    # Severity Assessment
    # -----------------------------
    if muac < 11.5:
        severity = "Severe Acute Malnutrition (SAM)"
    elif muac < 12.5:
        severity = "Moderate Acute Malnutrition (MAM)"
    else:
        severity = "Normal"

    result["Assessment"]["MUAC Status"] = severity

    if waz < -3:
        result["Assessment"]["Weight-for-Age"] = "Severely Underweight"
    elif waz < -2:
        result["Assessment"]["Weight-for-Age"] = "Underweight"
    else:
        result["Assessment"]["Weight-for-Age"] = "Normal"

    if haz < -3:
        result["Assessment"]["Height-for-Age"] = "Severely Stunted"
    elif haz < -2:
        result["Assessment"]["Height-for-Age"] = "Stunted"
    else:
        result["Assessment"]["Height-for-Age"] = "Normal"

    if whz < -3:
        result["Assessment"]["Weight-for-Height"] = "Severely Wasted"
    elif whz < -2:
        result["Assessment"]["Weight-for-Height"] = "Wasted"
    else:
        result["Assessment"]["Weight-for-Height"] = "Normal"

    # -----------------------------
    # Age Group
    # -----------------------------
    if age_months < 6:
        age_group = "Infant (0–6 months)"
    elif age_months < 24:
        age_group = "Infant/Toddler (6–24 months)"
    else:
        age_group = "Young Child (2–5 years)"

    result["Assessment"]["Age Group"] = age_group

    recommendations = []

    # -----------------------------
    # Underweight
    # -----------------------------
    if underweight:
        recommendations.extend([
            "Increase calorie intake using nutritious, age-appropriate foods.",
            "Include protein-rich foods such as eggs, pulses, dairy products, fish, or lean meat.",
            "Monitor weight every month."
        ])

    # -----------------------------
    # Stunting
    # -----------------------------
    if stunting:
        recommendations.extend([
            "Provide foods rich in protein, calcium, zinc and iron.",
            "Encourage dietary diversity including fruits and vegetables.",
            "Monitor height and growth regularly."
        ])

    # -----------------------------
    # Wasting
    # -----------------------------
    if wasting:
        recommendations.extend([
            "Provide energy-dense foods in small frequent meals.",
            "Ensure adequate hydration.",
            "Consult a pediatrician or nutrition specialist."
        ])

    # -----------------------------
    # MUAC Recommendation
    # -----------------------------
    if muac < 11.5:
        result["Alerts"].append(
            "URGENT: Child may have Severe Acute Malnutrition (SAM). Immediate medical assessment is recommended."
        )

    elif muac < 12.5:
        result["Alerts"].append(
            "Child may have Moderate Acute Malnutrition (MAM). Close follow-up is recommended."
        )

    # -----------------------------
    # Age Specific Advice
    # -----------------------------
    if age_months < 6:
        recommendations.append(
            "Exclusive breastfeeding is recommended unless advised otherwise by a healthcare professional."
        )

    elif age_months < 24:
        recommendations.append(
            "Continue breastfeeding along with appropriate complementary feeding."
        )

    else:
        recommendations.append(
            "Provide three balanced meals and healthy snacks each day."
        )

    # -----------------------------
    # General Advice
    # -----------------------------
    recommendations.extend([
        "Maintain proper hygiene and safe drinking water.",
        "Ensure routine immunization.",
        "Schedule regular growth monitoring visits."
    ])

    # Remove duplicate recommendations
    recommendations = list(dict.fromkeys(recommendations))

    result["Recommendations"] = recommendations

    return result