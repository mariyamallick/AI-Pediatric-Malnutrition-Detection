"""
Step 5B - Calibration Summary

Calculates:
- Overall multiclass Brier score
- Class-wise Brier scores
- Class-wise Expected Calibration Error (ECE)
- Mean ECE across classes

Uses the already-trained final models.
No model retraining is performed.

Author: Mariya Mallick
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "data/raw/dhs_children_combined.csv"
MODEL_DIR = "models"
OUTPUT_DIR = "evaluation/calibration"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "age_months",
    "sex",
    "weight_kg",
    "height_cm",
    "wealth_index",
    "mother_education",
    "currently_breastfeeding"
]


TARGETS = {
    "underweight": "underweight_status",
    "stunting": "stunting_status",
    "wasting": "wasting_status"
}


LABEL_MAP = {
    "normal": 0,
    "moderate": 1,
    "severe": 2
}


CLASS_NAMES = {
    0: "Normal",
    1: "Moderate",
    2: "Severe"
}


# ============================================================
# ECE FUNCTION
# ============================================================

def calculate_ece(y_true, probabilities, n_bins=10):
    """
    Calculate class-wise Expected Calibration Error.

    ECE is calculated using equal-width probability bins.

    For each class:

        ECE = sum(
            bin_weight * |observed_frequency - predicted_probability|
        )

    Lower ECE = better calibration.
    """

    results = []

    for class_id, class_name in CLASS_NAMES.items():

        y_binary = (
            np.asarray(y_true) == class_id
        ).astype(int)

        class_probability = probabilities[:, class_id]

        bin_edges = np.linspace(
            0.0,
            1.0,
            n_bins + 1
        )

        ece = 0.0
        total_samples = len(y_binary)

        for i in range(n_bins):

            lower = bin_edges[i]
            upper = bin_edges[i + 1]

            if i == n_bins - 1:

                mask = (
                    (class_probability >= lower)
                    &
                    (class_probability <= upper)
                )

            else:

                mask = (
                    (class_probability >= lower)
                    &
                    (class_probability < upper)
                )

            if not np.any(mask):
                continue

            predicted_mean = np.mean(
                class_probability[mask]
            )

            observed_frequency = np.mean(
                y_binary[mask]
            )

            bin_weight = (
                np.sum(mask) / total_samples
            )

            ece += (
                bin_weight
                * abs(
                    observed_frequency
                    - predicted_mean
                )
            )

        # Class-wise Brier score
        brier = brier_score_loss(
            y_binary,
            class_probability
        )

        results.append({
            "Class": class_name,
            "Brier Score": brier,
            "ECE": ece
        })

    return results


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 5B - CALIBRATION SUMMARY")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(
    f"\nDataset samples: {len(df)}"
)


# ============================================================
# PREPROCESSING
# ============================================================

df["sex"] = df["sex"].map({
    "M": 1,
    "F": 0
})

df["currently_breastfeeding"] = (
    df["currently_breastfeeding"]
    .fillna(False)
    .astype(bool)
    .astype(int)
)


for column in TARGETS.values():

    df[column] = (
        df[column]
        .map(LABEL_MAP)
    )


# ============================================================
# RESULTS
# ============================================================

summary_results = []


# ============================================================
# PROCESS EACH DISEASE
# ============================================================

for disease, target_column in TARGETS.items():

    print("\n" + "=" * 70)
    print(f"CALIBRATION SUMMARY - {disease.upper()}")
    print("=" * 70)

    data = df[
        FEATURES + [target_column]
    ].dropna()

    X = data[FEATURES]

    y = data[target_column].astype(int)

    # Same held-out split used throughout project
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------------
    # Find existing final model
    # --------------------------------------------------------

    possible_models = [

        f"{MODEL_DIR}/{disease}_final_model.pkl",

        f"{MODEL_DIR}/{disease}_status_model.pkl",

        f"{MODEL_DIR}/{disease}_model.pkl",

        f"{MODEL_DIR}/{disease}_best_model.pkl"
    ]

    model_path = None

    for path in possible_models:

        if os.path.exists(path):

            model_path = path
            break

    if model_path is None:

        print(
            f"ERROR: Final model not found for {disease}"
        )

        continue

    print(
        f"Model: {model_path}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = joblib.load(model_path)

    if not hasattr(model, "predict_proba"):

        print(
            "ERROR: Model does not provide "
            "predict_proba()."
        )

        continue

    # --------------------------------------------------------
    # Probability predictions
    # --------------------------------------------------------

    probabilities = model.predict_proba(X_test)

    # --------------------------------------------------------
    # Calculate ECE
    # --------------------------------------------------------

    class_results = calculate_ece(
        y_test.to_numpy(),
        probabilities,
        n_bins=10
    )

    # --------------------------------------------------------
    # Overall multiclass Brier score
    # --------------------------------------------------------

    y_one_hot = np.zeros_like(
        probabilities
    )

    y_one_hot[
        np.arange(len(y_test)),
        y_test.to_numpy()
    ] = 1

    multiclass_brier = np.mean(
        np.sum(
            (y_one_hot - probabilities) ** 2,
            axis=1
        )
    )

    mean_ece = np.mean([
        row["ECE"]
        for row in class_results
    ])

    mean_class_brier = np.mean([
        row["Brier Score"]
        for row in class_results
    ])

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"\nOverall Multiclass Brier Score: "
        f"{multiclass_brier:.6f}"
    )

    print(
        f"Mean Class-wise Brier Score: "
        f"{mean_class_brier:.6f}"
    )

    print(
        f"Mean ECE: "
        f"{mean_ece:.6f}"
    )

    print("\nClass-wise results:")

    for row in class_results:

        print(
            f"{row['Class']:10s} | "
            f"Brier: {row['Brier Score']:.6f} | "
            f"ECE: {row['ECE']:.6f}"
        )

    # --------------------------------------------------------
    # Store disease-level summary
    # --------------------------------------------------------

    summary_results.append({

        "Disease": disease,

        "Multiclass Brier Score":
            multiclass_brier,

        "Mean Class-wise Brier Score":
            mean_class_brier,

        "Normal Brier":
            class_results[0]["Brier Score"],

        "Moderate Brier":
            class_results[1]["Brier Score"],

        "Severe Brier":
            class_results[2]["Brier Score"],

        "Normal ECE":
            class_results[0]["ECE"],

        "Moderate ECE":
            class_results[1]["ECE"],

        "Severe ECE":
            class_results[2]["ECE"],

        "Mean ECE":
            mean_ece
    })


# ============================================================
# SAVE FINAL SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary_results
)

output_path = (
    f"{OUTPUT_DIR}/"
    "final_calibration_summary.csv"
)

summary_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# DISPLAY FINAL TABLE
# ============================================================

print("\n" + "=" * 70)
print("FINAL CALIBRATION SUMMARY")
print("=" * 70)

print(
    summary_df.to_string(
        index=False
    )
)

print(
    f"\nSaved: {output_path}"
)

print(
    "\nStep 5B completed successfully."
)