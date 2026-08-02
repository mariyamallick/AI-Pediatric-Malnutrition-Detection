def generate_feature_importance(model, feature_names):

    importance = model.feature_importances_

    result = []

    for feature, value in zip(feature_names, importance):

        result.append({

            "feature": feature,

            "importance": round(float(value), 4)

        })

    result.sort(

        key=lambda x: x["importance"],

        reverse=True

    )

    return result