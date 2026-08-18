"""
Step 4A - Sex Subgroup / Bias Analysis
Pediatric Malnutrition Detection

Evaluates the selected final model separately for:
    - Male children
    - Female children

Outcomes:
    - Underweight
    - Stunting
    - Wasting

The analysis reports:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - Macro F1
    - Balanced Accuracy
    - Class-wise performance
    - Confusion matrices

Author: Mariya Mallick
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "data/raw/dhs_children_combined.csv"

MODEL_PATHS = {
    "underweight": "models/underweight_status_model.pkl",
    "stunting": "models/stunting_status_model.pkl",
    "wasting": "models/wasting_status_model.pkl"
}

OUTPUT_DIR = "evaluation/sex_subgroup"

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


# ============================================================
# LABEL ENCODING
# ============================================================

LABEL_MAP = {
    "normal": 0,
    "moderate": 1,
    "severe": 2
}


LABEL_NAMES = [
    "Normal",
    "Moderate",
    "Severe"
]


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("STEP 4A - SEX SUBGROUP / BIAS ANALYSIS")
print("=" * 70)

print("\nLoading DHS dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Total records loaded: {len(df):,}")


# ============================================================
# PREPROCESSING
# ============================================================

df = df[
    [
        "age_months",
        "sex",
        "weight_kg",
        "height_cm",
        "wealth_index",
        "mother_education",
        "currently_breastfeeding",
        "underweight_status",
        "stunting_status",
        "wasting_status"
    ]
].copy()


# Encode sex
df["sex"] = df["sex"].map({
    "M": 1,
    "F": 0
})


# Encode target variables
for target in [
    "underweight_status",
    "stunting_status",
    "wasting_status"
]:
    df[target] = df[target].map(LABEL_MAP)


# Convert breastfeeding to binary
df["currently_breastfeeding"] = (
    df["currently_breastfeeding"]
    .fillna(False)
    .astype(bool)
    .astype(int)
)


# Remove rows with missing required values
required_columns = FEATURES + [
    "underweight_status",
    "stunting_status",
    "wasting_status"
]

df = df.dropna(subset=required_columns)


print(f"Records after preprocessing: {len(df):,}")


# ============================================================
# SEX DISTRIBUTION
# ============================================================

print("\nSex distribution:")

print(
    df["sex"]
    .map({
        0: "Female",
        1: "Male"
    })
    .value_counts()
)


# ============================================================
# LOAD FINAL MODELS
# ============================================================

print("\nLoading selected final models...")

models = {}

for disease, model_path in MODEL_PATHS.items():

    if not os.path.exists(model_path):

        print(
            f"WARNING: Model not found: {model_path}"
        )

        continue

    models[disease] = joblib.load(model_path)

    print(
        f"Loaded {disease} model: {model_path}"
    )


# ============================================================
# SUBGROUP ANALYSIS FUNCTION
# ============================================================

def evaluate_subgroup(
    model,
    X,
    y,
    subgroup_name,
    disease
):

    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        average="weighted",
        zero_division=0
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0
    )

    balanced_accuracy = balanced_accuracy_score(
        y,
        predictions
    )

    print("\n" + "-" * 70)

    print(
        f"{disease.upper()} - {subgroup_name}"
    )

    print("-" * 70)

    print(
        f"Samples             : {len(y):,}"
    )

    print(
        f"Accuracy            : {accuracy:.4f}"
    )

    print(
        f"Precision           : {precision:.4f}"
    )

    print(
        f"Recall              : {recall:.4f}"
    )

    print(
        f"F1 Score            : {f1:.4f}"
    )

    print(
        f"Macro F1            : {macro_f1:.4f}"
    )

    print(
        f"Balanced Accuracy   : {balanced_accuracy:.4f}"
    )


    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y,
        predictions,
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0
    )


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y,
        predictions,
        labels=[0, 1, 2]
    )

    figure_name = (
        f"{disease}_{subgroup_name.lower()}"
        .replace(" ", "_")
        + "_confusion_matrix.png"
    )

    figure_path = os.path.join(
        OUTPUT_DIR,
        figure_name
    )


    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=LABEL_NAMES
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format="d"
    )

    ax.set_title(
        f"{disease.title()} - {subgroup_name}"
    )

    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    return {
        "Disease": disease,
        "Sex": subgroup_name,
        "Samples": len(y),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Macro F1": macro_f1,
        "Balanced Accuracy": balanced_accuracy,

        "Normal Precision":
            report["Normal"]["precision"],

        "Normal Recall":
            report["Normal"]["recall"],

        "Normal F1":
            report["Normal"]["f1-score"],

        "Moderate Precision":
            report["Moderate"]["precision"],

        "Moderate Recall":
            report["Moderate"]["recall"],

        "Moderate F1":
            report["Moderate"]["f1-score"],

        "Severe Precision":
            report["Severe"]["precision"],

        "Severe Recall":
            report["Severe"]["recall"],

        "Severe F1":
            report["Severe"]["f1-score"]
    }


# ============================================================
# RUN SEX SUBGROUP ANALYSIS
# ============================================================

all_results = []


for disease, model in models.items():

    target = f"{disease}_status"

    X = df[FEATURES].copy()

    y = df[target]


    # --------------------------------------------------------
    # FEMALE
    # --------------------------------------------------------

    female_mask = df["sex"] == 0

    X_female = X.loc[
        female_mask
    ]

    y_female = y.loc[
        female_mask
    ]


    female_result = evaluate_subgroup(
        model,
        X_female,
        y_female,
        "Female",
        disease
    )

    all_results.append(
        female_result
    )


    # --------------------------------------------------------
    # MALE
    # --------------------------------------------------------

    male_mask = df["sex"] == 1

    X_male = X.loc[
        male_mask
    ]

    y_male = y.loc[
        male_mask
    ]


    male_result = evaluate_subgroup(
        model,
        X_male,
        y_male,
        "Male",
        disease
    )

    all_results.append(
        male_result
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)


results_path = os.path.join(
    OUTPUT_DIR,
    "sex_subgroup_results.csv"
)


results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# CALCULATE SEX PERFORMANCE GAP
# ============================================================

gap_rows = []


for disease in results_df["Disease"].unique():

    disease_results = results_df[
        results_df["Disease"] == disease
    ]

    female = disease_results[
        disease_results["Sex"] == "Female"
    ].iloc[0]

    male = disease_results[
        disease_results["Sex"] == "Male"
    ].iloc[0]


    gap_rows.append({

        "Disease": disease,

        "Accuracy Gap":
            abs(
                female["Accuracy"]
                -
                male["Accuracy"]
            ),

        "Precision Gap":
            abs(
                female["Precision"]
                -
                male["Precision"]
            ),

        "Recall Gap":
            abs(
                female["Recall"]
                -
                male["Recall"]
            ),

        "F1 Gap":
            abs(
                female["F1 Score"]
                -
                male["F1 Score"]
            ),

        "Macro F1 Gap":
            abs(
                female["Macro F1"]
                -
                male["Macro F1"]
            ),

        "Balanced Accuracy Gap":
            abs(
                female["Balanced Accuracy"]
                -
                male["Balanced Accuracy"]
            )
    })


gap_df = pd.DataFrame(
    gap_rows
)


gap_path = os.path.join(
    OUTPUT_DIR,
    "sex_performance_gaps.csv"
)


gap_df.to_csv(
    gap_path,
    index=False
)


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SEX SUBGROUP ANALYSIS COMPLETE")
print("=" * 70)

print("\nMain Results:")

print(
    results_df[
        [
            "Disease",
            "Sex",
            "Samples",
            "Accuracy",
            "Macro F1",
            "Balanced Accuracy"
        ]
    ].to_string(index=False)
)


print("\nPerformance Gaps:")

print(
    gap_df.to_string(index=False)
)


print("\nFiles generated:")

print(
    f"1. {results_path}"
)

print(
    f"2. {gap_path}"
)

print(
    f"3. Sex-specific confusion matrices in {OUTPUT_DIR}/"
)


print("\nStep 4A completed successfully.")