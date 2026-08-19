"""
Step 5A - Calibration Analysis
Pediatric Malnutrition Detection

Evaluates calibration of the selected final models:
- Underweight
- Stunting
- Wasting

Generates:
1. Calibration curves
2. Calibration CSV results

Author: Mariya Mallick
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "data/raw/dhs_children_combined.csv"

MODEL_DIR = "models"
OUTPUT_DIR = "evaluation/calibration"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 5A - CALIBRATION ANALYSIS")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded:")
print(f"Samples: {len(df)}")


# ============================================================
# FEATURES
# ============================================================

features = [
    "age_months",
    "sex",
    "weight_kg",
    "height_cm",
    "wealth_index",
    "mother_education",
    "currently_breastfeeding"
]


# Encode sex
df["sex"] = df["sex"].map({
    "M": 1,
    "F": 0
})


# Encode breastfeeding
df["currently_breastfeeding"] = (
    df["currently_breastfeeding"]
    .fillna(False)
    .astype(bool)
    .astype(int)
)


# ============================================================
# LABEL ENCODING
# ============================================================

label_map = {
    "normal": 0,
    "moderate": 1,
    "severe": 2
}


targets = {
    "underweight": "underweight_status",
    "stunting": "stunting_status",
    "wasting": "wasting_status"
}


for target_column in targets.values():
    df[target_column] = df[target_column].map(label_map)


# ============================================================
# CALIBRATION FUNCTION
# ============================================================

def calculate_multiclass_calibration(
    y_true,
    probabilities,
    disease
):
    """
    Calculates one-vs-rest calibration curves
    for a three-class problem.

    Classes:
        0 = Normal
        1 = Moderate
        2 = Severe
    """

    results = []

    class_names = {
        0: "Normal",
        1: "Moderate",
        2: "Severe"
    }

    plt.figure(figsize=(8, 7))

    # Perfect calibration line
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    for class_id, class_name in class_names.items():

        # One-vs-rest true labels
        y_binary = (y_true == class_id).astype(int)

        # Probability for this class
        class_probability = probabilities[:, class_id]

        # Calibration curve
        fraction_positive, mean_predicted = calibration_curve(
            y_binary,
            class_probability,
            n_bins=10,
            strategy="quantile"
        )

        # Brier score
        brier = brier_score_loss(
            y_binary,
            class_probability
        )

        print(
            f"{class_name:10s} "
            f"Brier Score: {brier:.6f}"
        )

        # Store curve values
        for predicted, observed in zip(
            mean_predicted,
            fraction_positive
        ):
            results.append({
                "Disease": disease,
                "Class": class_name,
                "Predicted Probability": predicted,
                "Observed Frequency": observed,
                "Brier Score": brier
            })

        # Plot
        plt.plot(
            mean_predicted,
            fraction_positive,
            marker="o",
            label=class_name
        )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")

    plt.title(
        f"Calibration Curve - {disease.capitalize()}"
    )

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    output_path = (
        f"{OUTPUT_DIR}/"
        f"{disease}_calibration_curve.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")

    return results


# ============================================================
# MAIN ANALYSIS
# ============================================================

all_results = []


for disease, target_column in targets.items():

    print("\n" + "=" * 70)
    print(f"CALIBRATION - {disease.upper()}")
    print("=" * 70)

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    data = df[
        features + [target_column]
    ].dropna()

    X = data[features]

    y = data[target_column].astype(int)

    # --------------------------------------------------------
    # Same train/test split used in previous evaluation
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------------
    # Load selected final model
    # --------------------------------------------------------

    model_path = (
        f"{MODEL_DIR}/"
        f"{disease}_final_model.pkl"
    )

    if not os.path.exists(model_path):

        print(
            f"WARNING: Model not found:"
            f" {model_path}"
        )

        print(
            "Trying selected model naming convention..."
        )

        # Fallback for your existing naming
        alternative_paths = [
            f"{MODEL_DIR}/{disease}_status_model.pkl",
            f"{MODEL_DIR}/{disease}_model.pkl"
        ]

        model_path = None

        for path in alternative_paths:

            if os.path.exists(path):
                model_path = path
                break

    if model_path is None:

        print(
            f"ERROR: No model found for {disease}."
        )

        continue

    print(f"Loading model: {model_path}")

    model = joblib.load(model_path)

    # --------------------------------------------------------
    # Probability predictions
    # --------------------------------------------------------

    if not hasattr(model, "predict_proba"):

        print(
            f"ERROR: {disease} model does not "
            "support predict_proba()."
        )

        continue

    probabilities = model.predict_proba(X_test)

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    results = calculate_multiclass_calibration(
        y_test.to_numpy(),
        probabilities,
        disease
    )

    all_results.extend(results)


# ============================================================
# SAVE RESULTS
# ============================================================

if all_results:

    results_df = pd.DataFrame(all_results)

    output_csv = (
        f"{OUTPUT_DIR}/"
        "calibration_results.csv"
    )

    results_df.to_csv(
        output_csv,
        index=False
    )

    print("\n" + "=" * 70)
    print("CALIBRATION ANALYSIS COMPLETE")
    print("=" * 70)

    print(f"\nSaved: {output_csv}")

    print("\nResults preview:")
    print(results_df.head(15))

else:

    print("\nNo calibration results were generated.")