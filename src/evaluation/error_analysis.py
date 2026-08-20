"""
Step 6 - Detailed Error Analysis

Analyzes errors made by the selected final model for:
    - Underweight
    - Stunting
    - Wasting

The analysis includes:
    1. Overall error counts
    2. Confusion matrices
    3. Class-wise error counts
    4. Misclassified samples
    5. Errors near clinical z-score boundaries
    6. Error rates by age group
    7. Error rates by sex
    8. Summary CSV files
    9. Error-analysis figures

IMPORTANT:
This script does NOT retrain all 8 models.
It loads the already-selected final models.

Author: Mariya Mallick
"""

import os
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "data/raw/dhs_children_combined.csv"

MODEL_DIR = "models"

OUTPUT_DIR = "evaluation/error_analysis"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


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


# ============================================================
# TARGETS
# ============================================================

TARGETS = {
    "underweight": "underweight_status",
    "stunting": "stunting_status",
    "wasting": "wasting_status"
}


# ============================================================
# Z-SCORE VARIABLES
# ============================================================

Z_SCORE_COLUMNS = {
    "underweight": "waz",
    "stunting": "haz",
    "wasting": "whz"
}


# ============================================================
# LABELS
# ============================================================

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
# MODEL NAME NORMALIZATION
# ============================================================

def normalize_model_name(name):

    name = str(name).lower()

    replacements = {
        "logistic regression": "logistic_regression",
        "decision tree": "decision_tree",
        "random forest": "random_forest",
        "gradient boosting": "gradient_boosting",
        "gaussian naive bayes": "gaussian_naive_bayes",
        "naive bayes": "gaussian_naive_bayes",
        "xgboost": "xgboost",
        "knn": "knn",
        "svm": "svm"
    }

    for key, value in replacements.items():

        if key in name:
            return value

    return name.replace(" ", "_")


# ============================================================
# FIND FINAL MODEL
# ============================================================

def find_final_model(disease):

    """
    Attempts to locate the already-selected final model.

    First checks common final-model filenames.

    Then checks FINAL_MODEL_SELECTION_SUMMARY.csv
    if available.
    """

    candidates = [

        f"{MODEL_DIR}/{disease}_final_model.pkl",

        f"{MODEL_DIR}/{disease}_best_model.pkl",

        f"{MODEL_DIR}/{disease}_status_model.pkl",

        f"{MODEL_DIR}/{disease}_model.pkl"
    ]

    for path in candidates:

        if os.path.exists(path):

            print(
                f"Using model: {path}"
            )

            return path


    # --------------------------------------------------------
    # Search final model selection summary
    # --------------------------------------------------------

    summary_path = (
        "evaluation/"
        "FINAL_MODEL_SELECTION_SUMMARY.csv"
    )

    if os.path.exists(summary_path):

        summary = pd.read_csv(
            summary_path
        )

        disease_columns = [
            c for c in summary.columns
            if "disease" in c.lower()
        ]

        model_columns = [
            c for c in summary.columns
            if "model" in c.lower()
        ]

        if disease_columns and model_columns:

            disease_col = disease_columns[0]
            model_col = model_columns[0]

            matches = summary[
                summary[disease_col]
                .astype(str)
                .str.lower()
                == disease.lower()
            ]

            if len(matches) > 0:

                model_name = matches.iloc[0][model_col]

                normalized = normalize_model_name(
                    model_name
                )

                possible = [

                    f"{MODEL_DIR}/{disease}_{normalized}.pkl",

                    f"{MODEL_DIR}/{disease}_{normalized}_model.pkl",

                    f"{MODEL_DIR}/{normalized}_{disease}.pkl",

                    f"{MODEL_DIR}/{normalized}.pkl"
                ]

                for path in possible:

                    if os.path.exists(path):

                        print(
                            f"Using selected model: {path}"
                        )

                        return path


    # --------------------------------------------------------
    # Search all model files
    # --------------------------------------------------------

    all_models = glob.glob(
        f"{MODEL_DIR}/*.pkl"
    )

    disease_models = [
        path
        for path in all_models
        if disease.lower()
        in os.path.basename(path).lower()
    ]

    if len(disease_models) == 1:

        print(
            f"Using available model: "
            f"{disease_models[0]}"
        )

        return disease_models[0]


    print(
        f"\nERROR: Could not identify final model "
        f"for {disease}."
    )

    print("Available model files:")

    for path in all_models:
        print("   ", path)

    return None


# ============================================================
# CREATE AGE GROUP
# ============================================================

def age_group(age):

    if age < 12:
        return "<12 months"

    elif age < 60:
        return "12-59 months"

    else:
        return "60+ months"


# ============================================================
# CLINICAL BOUNDARY ANALYSIS
# ============================================================

def boundary_category(z):

    """
    Identifies observations close to the
    -2 and -3 z-score boundaries.

    Within ±0.25 SD of either threshold
    is considered a boundary case.
    """

    if pd.isna(z):
        return "Unknown"

    distance_minus_2 = abs(z + 2)
    distance_minus_3 = abs(z + 3)

    closest_distance = min(
        distance_minus_2,
        distance_minus_3
    )

    if closest_distance <= 0.25:

        if distance_minus_2 <= distance_minus_3:
            return "Near -2 SD boundary"

        return "Near -3 SD boundary"

    return "Away from boundary"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 6 - DETAILED ERROR ANALYSIS")
print("=" * 70)

df = pd.read_csv(
    DATA_PATH
)

print(
    f"\nDataset size: {df.shape}"
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


# ============================================================
# MAP TARGETS
# ============================================================

for disease, target in TARGETS.items():

    df[target] = (
        df[target]
        .astype(str)
        .str.lower()
        .map(LABEL_MAP)
    )


# ============================================================
# STORAGE
# ============================================================

overall_results = []

class_errors = []

boundary_results = []

age_results = []

sex_results = []


# ============================================================
# PROCESS EACH DISEASE
# ============================================================

for disease, target in TARGETS.items():

    print("\n")
    print("=" * 70)
    print(
        f"ERROR ANALYSIS - "
        f"{disease.upper()}"
    )
    print("=" * 70)


    # --------------------------------------------------------
    # Find model
    # --------------------------------------------------------

    model_path = find_final_model(
        disease
    )

    if model_path is None:

        print(
            f"Skipping {disease}."
        )

        continue


    model = joblib.load(
        model_path
    )


    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    required_columns = (
        FEATURES
        + [
            target,
            Z_SCORE_COLUMNS[disease]
        ]
    )

    data = df[
        required_columns
    ].dropna()


    X = data[
        FEATURES
    ]

    y = data[
        target
    ].astype(int)


    # --------------------------------------------------------
    # Same train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # Align original test rows
    # --------------------------------------------------------

    test_data = data.loc[
        X_test.index
    ].copy()


    test_data[
        "true_label"
    ] = y_test.values


    test_data[
        "predicted_label"
    ] = predictions


    test_data[
        "true_class"
    ] = test_data[
        "true_label"
    ].map(CLASS_NAMES)


    test_data[
        "predicted_class"
    ] = test_data[
        "predicted_label"
    ].map(CLASS_NAMES)


    # --------------------------------------------------------
    # Error flag
    # --------------------------------------------------------

    test_data[
        "is_error"
    ] = (
        test_data["true_label"]
        != test_data["predicted_label"]
    )


    # ========================================================
    # OVERALL ERROR
    # ========================================================

    total = len(test_data)

    errors = int(
        test_data["is_error"].sum()
    )

    correct = total - errors

    accuracy = (
        correct / total
    )

    error_rate = (
        errors / total
    )


    print(
        f"\nTotal test samples : {total}"
    )

    print(
        f"Correct predictions: {correct}"
    )

    print(
        f"Errors             : {errors}"
    )

    print(
        f"Accuracy           : {accuracy:.4f}"
    )

    print(
        f"Error rate         : {error_rate:.4f}"
    )


    overall_results.append({

        "Disease": disease,

        "Test Samples": total,

        "Correct": correct,

        "Errors": errors,

        "Accuracy": accuracy,

        "Error Rate": error_rate
    })


    # ========================================================
    # MISCLASSIFIED SAMPLES
    # ========================================================

    misclassified = test_data[
        test_data["is_error"]
    ].copy()


    misclassified[
        "z_score"
    ] = misclassified[
        Z_SCORE_COLUMNS[disease]
    ]


    misclassified[
        "Boundary Category"
    ] = misclassified[
        "z_score"
    ].apply(
        boundary_category
    )


    # --------------------------------------------------------
    # Save misclassified cases
    # --------------------------------------------------------

    misclassified_path = (

        f"{OUTPUT_DIR}/"
        f"{disease}_misclassified_cases.csv"
    )


    misclassified.to_csv(
        misclassified_path,
        index=False
    )


    print(
        f"\nSaved misclassified cases:"
        f"\n{misclassified_path}"
    )


    # ========================================================
    # CLASS-WISE ERRORS
    # ========================================================

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1, 2]
    )


    cm_df = pd.DataFrame(

        cm,

        index=[
            "True Normal",
            "True Moderate",
            "True Severe"
        ],

        columns=[
            "Pred Normal",
            "Pred Moderate",
            "Pred Severe"
        ]
    )


    cm_path = (

        f"{OUTPUT_DIR}/"
        f"{disease}_error_confusion_matrix.csv"
    )


    cm_df.to_csv(
        cm_path
    )


    # --------------------------------------------------------
    # Plot confusion matrix
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        f"Error Analysis - {disease.title()}"
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "True Class"
    )

    plt.xticks(
        [0, 1, 2],
        ["Normal", "Moderate", "Severe"]
    )

    plt.yticks(
        [0, 1, 2],
        ["Normal", "Moderate", "Severe"]
    )


    for i in range(3):

        for j in range(3):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )


    plt.colorbar()

    plt.tight_layout()


    plt.savefig(
        f"{OUTPUT_DIR}/"
        f"{disease}_error_confusion_matrix.png",
        dpi=300
    )

    plt.close()


    # ========================================================
    # CLASS ERROR COUNTS
    # ========================================================

    for class_id in [0, 1, 2]:

        true_class_mask = (
            test_data["true_label"]
            == class_id
        )

        class_total = int(
            true_class_mask.sum()
        )

        class_errors_count = int(
            (
                true_class_mask
                & test_data["is_error"]
            ).sum()     
        )

        class_errors_rate = (   
            
            class_errors_count / class_total

            if class_total > 0

            else 0
        )

        class_errors.append({   
            
            "Disease": disease,

            "Class": CLASS_NAMES[class_id],

            "Samples": class_total,

            "Errors": class_errors_count,

            "Error Rate": class_errors_rate
        })

    # ========================================================
    # BOUNDARY ANALYSIS
    # ========================================================

    test_data[
        "z_score"
    ] = test_data[
        Z_SCORE_COLUMNS[disease]
    ]


    test_data[
        "Boundary Category"
    ] = test_data[
        "z_score"
    ].apply(
        boundary_category
    )


    for category, group in test_data.groupby(
        "Boundary Category"
    ):

        total_group = len(group)

        errors_group = int(
            group["is_error"].sum()
        )

        error_rate_group = (

            errors_group / total_group

            if total_group > 0

            else 0
        )


        boundary_results.append({

            "Disease": disease,

            "Boundary Category":
                category,

            "Samples":
                total_group,

            "Errors":
                errors_group,

            "Error Rate":
                error_rate_group
        })


    # ========================================================
    # AGE GROUP ERROR ANALYSIS
    # ========================================================

    test_data[
        "Age Group"
    ] = test_data[
        "age_months"
    ].apply(
        age_group
    )


    for group_name, group in test_data.groupby(
        "Age Group"
    ):

        total_group = len(group)

        errors_group = int(
            group["is_error"].sum()
        )

        age_results.append({

            "Disease": disease,

            "Age Group":
                group_name,

            "Samples":
                total_group,

            "Errors":
                errors_group,

            "Error Rate":
                errors_group / total_group
        })


    # ========================================================
    # SEX ERROR ANALYSIS
    # ========================================================

    for sex_value, group in test_data.groupby(
        "sex"
    ):

        total_group = len(group)

        errors_group = int(
            group["is_error"].sum()
        )

        sex_name = (
            "Male"
            if sex_value == 1
            else "Female"
        )


        sex_results.append({

            "Disease": disease,

            "Sex":
                sex_name,

            "Samples":
                total_group,

            "Errors":
                errors_group,

            "Error Rate":
                errors_group / total_group
        })


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(

        y_test,

        predictions,

        labels=[0, 1, 2],

        target_names=[
            "Normal",
            "Moderate",
            "Severe"
        ],

        output_dict=True,

        zero_division=0
    )


    report_df = pd.DataFrame(
        report
    ).transpose()


    report_df.to_csv(

        f"{OUTPUT_DIR}/"
        f"{disease}_error_classification_report.csv"
    )


# ============================================================
# SAVE ALL SUMMARY FILES
# ============================================================

pd.DataFrame(
    overall_results
).to_csv(

    f"{OUTPUT_DIR}/"
    "overall_error_summary.csv",

    index=False
)


pd.DataFrame(
    class_errors
).to_csv(

    f"{OUTPUT_DIR}/"
    "class_error_summary.csv",

    index=False
)


pd.DataFrame(
    boundary_results
).to_csv(

    f"{OUTPUT_DIR}/"
    "boundary_error_summary.csv",

    index=False
)


pd.DataFrame(
    age_results
).to_csv(

    f"{OUTPUT_DIR}/"
    "age_error_summary.csv",

    index=False
)


pd.DataFrame(
    sex_results
).to_csv(

    f"{OUTPUT_DIR}/"
    "sex_error_summary.csv",

    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("STEP 6 COMPLETED")
print("=" * 70)

print("\nOverall Error Summary:")

print(
    pd.DataFrame(
        overall_results
    ).to_string(
        index=False
    )
)


print(
    "\nFiles saved in:"
)

print(
    OUTPUT_DIR
)