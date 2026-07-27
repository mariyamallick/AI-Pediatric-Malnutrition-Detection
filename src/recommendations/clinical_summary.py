def generate_clinical_summary(result):

    prediction = result["Prediction"]
    assessment = result["Assessment"]
    growth = result["WHO Growth"]

    summary = []

    # -------------------------
    # Overall Prediction
    # -------------------------

    if (
        not prediction["Underweight"]
        and not prediction["Stunting"]
        and not prediction["Wasting"]
    ):
        summary.append(
            "The child shows no evidence of underweight, stunting, or wasting based on the AI assessment."
        )

    else:

        conditions = []

        if prediction["Underweight"]:
            conditions.append("underweight")

        if prediction["Stunting"]:
            conditions.append("stunting")

        if prediction["Wasting"]:
            conditions.append("wasting")

        summary.append(
            "The AI assessment indicates " +
            ", ".join(conditions[:-1]) +
            (" and " + conditions[-1] if len(conditions) > 1 else conditions[0]) +
            "."
        )

    # -------------------------
    # WHO Growth
    # -------------------------

    summary.append(
        f"WHO growth indicators show WAZ ({growth['waz_status']}), "
        f"HAZ ({growth['haz_status']}), "
        f"and WHZ ({growth['whz_status']})."
    )

    # -------------------------
    # MUAC
    # -------------------------

    summary.append(
        f"MUAC assessment indicates {assessment['MUAC Status']}."
    )

    # -------------------------
    # Overall Risk
    # -------------------------

    summary.append(
        f"The overall nutritional risk is classified as {result['Overall Risk']}."
    )

    # -------------------------
    # Recommendation
    # -------------------------

    if result["Overall Risk"] == "🔴 High Risk":
        summary.append(
            "Immediate nutritional intervention and pediatric consultation are recommended."
        )

    elif result["Overall Risk"] == "🟡 Moderate Risk":
        summary.append(
            "Regular growth monitoring and nutritional counselling are recommended."
        )

    else:
        summary.append(
            "Continue age-appropriate nutrition and routine growth monitoring."
        )

    return " ".join(summary)