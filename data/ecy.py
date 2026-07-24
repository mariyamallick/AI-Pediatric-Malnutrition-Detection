def confidence_level(score):

    if score >= 95:
        return "Very High"

    elif score >= 85:
        return "High"

    elif score >= 70:
        return "Moderate"

    else:
        return "Low"