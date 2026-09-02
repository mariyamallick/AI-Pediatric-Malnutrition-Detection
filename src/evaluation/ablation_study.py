"""
Step 7 - Feature Ablation Study

Purpose:
Evaluate how much each predictive feature contributes to the
selected final model for:
    - Underweight
    - Stunting
    - Wasting

The selected model is retrained after removing one feature at a time.

Author: Mariya Mallick
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "data/raw/dhs_children_combined.csv"

MODEL_DIR = "models"

OUTPUT_DIR = "evaluation/ablation"

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
# TARGET ENCODING
# ============================================================

LABEL_MAP = {
    "normal": 0,
    "moderate": 1,
    "severe": 2
}


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_PATHS = {

    "underweight":
        "models/underweight_best_model.pkl",

    "stunting":
        "models/stunting_best_model.pkl",

    "wasting":
        "models/wasting_best_model.pkl"
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 7 - FEATURE ABLATION STUDY")
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

# ---------- SEX ----------

df["sex"] = (
    df["sex"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "m": 1,
        "male": 1,
        "1": 1,

        "f": 0,
        "female": 0,
        "2": 0
    })
)


# ---------- BREASTFEEDING ----------

df["currently_breastfeeding"] = (
    df["currently_breastfeeding"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "yes": 1,
        "y": 1,
        "true": 1,
        "1": 1,

        "no": 0,
        "n": 0,
        "false": 0,
        "0": 0
    })
)


# ---------- ENSURE NUMERIC FEATURES ----------

numeric_features = [
    "age_months",
    "weight_kg",
    "height_cm",
    "wealth_index",
    "mother_education"
]

for column in numeric_features:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ---------- CHECK DATA ----------

print("\nFeature data types:")

print(
    df[FEATURES].dtypes
)

print("\nMissing values after preprocessing:")

print(
    df[FEATURES].isnull().sum()
)

# ============================================================
# RESULTS
# ============================================================

all_results = []


# ============================================================
# PROCESS EACH DISEASE
# ============================================================

for disease, target in TARGETS.items():

    print("\n")
    print("=" * 70)
    print(
        f"ABLATION ANALYSIS - {disease.upper()}"
    )
    print("=" * 70)


    model_path = MODEL_PATHS[disease]


    if not os.path.exists(model_path):

        print(
            f"ERROR: Model not found:"
            f" {model_path}"
        )

        continue


    # --------------------------------------------------------
    # Load selected model
    # --------------------------------------------------------

    original_model = joblib.load(
        model_path
    )


    print(
        f"Selected model: {model_path}"
    )


    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    required = FEATURES + [target]

    data = df[
        required
    ].dropna()


    # Encode target

    y = (
        data[target]
        .astype(str)
        .str.lower()
        .map(LABEL_MAP)
    )


    valid = y.notna()

    data = data.loc[valid]

    y = y.loc[valid].astype(int)


    # ========================================================
    # FIXED TRAIN/TEST SPLIT
    # ========================================================

    train_idx, test_idx = train_test_split(

        np.arange(len(data)),

        test_size=0.20,

        random_state=42,

        stratify=y
    )


    # ========================================================
    # FULL FEATURE SET
    # ========================================================

    feature_sets = {

        "All features":
            FEATURES.copy()
    }


    # ========================================================
    # ONE FEATURE REMOVED AT A TIME
    # ========================================================

    for feature in FEATURES:

        reduced_features = [

            f for f in FEATURES

            if f != feature
        ]

        feature_sets[
            f"Without {feature}"
        ] = reduced_features


    # ========================================================
    # RUN ABLATIONS
    # ========================================================

    for variant, selected_features in feature_sets.items():

        print("\n")
        print("-" * 60)

        print(
            f"Variant: {variant}"
        )

        print(
            f"Features used: "
            f"{len(selected_features)}"
        )


        X = data[
            selected_features
        ]


        X_train = X.iloc[
            train_idx
        ]

        X_test = X.iloc[
            test_idx
        ]

        y_train = y.iloc[
            train_idx
        ]

        y_test = y.iloc[
            test_idx
        ]


        # ----------------------------------------------------
        # Clone selected model
        # ----------------------------------------------------

        try:

            model = clone(
                original_model
            )

        except Exception:

            model = original_model


        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train
        )


        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        predictions = model.predict(
            X_test
        )


        # ----------------------------------------------------
        # Basic metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            predictions
        )


        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )


        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )


        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )


        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )


        balanced_acc = balanced_accuracy_score(
            y_test,
            predictions
        )


        # ----------------------------------------------------
        # ROC-AUC
        # ----------------------------------------------------

        roc_auc = np.nan


        if hasattr(
            model,
            "predict_proba"
        ):

            try:

                probabilities = (
                    model.predict_proba(
                        X_test
                    )
                )


                roc_auc = roc_auc_score(

                    y_test,

                    probabilities,

                    multi_class="ovr",

                    average="macro"
                )

            except Exception:

                roc_auc = np.nan


        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        all_results.append({

            "Disease":
                disease,

            "Variant":
                variant,

            "Features Used":
                len(selected_features),

            "Removed Feature":
                "None"
                if variant == "All features"
                else variant.replace(
                    "Without ",
                    ""
                ),

            "Accuracy":
                accuracy,

            "Precision":
                precision,

            "Recall":
                recall,

            "F1 Score":
                f1,

            "Macro F1":
                macro_f1,

            "Balanced Accuracy":
                balanced_acc,

            "ROC AUC":
                roc_auc
        })


        print(
            f"Accuracy          : {accuracy:.4f}"
        )

        print(
            f"Precision         : {precision:.4f}"
        )

        print(
            f"Recall            : {recall:.4f}"
        )

        print(
            f"F1 Score          : {f1:.4f}"
        )

        print(
            f"Macro F1          : {macro_f1:.4f}"
        )

        print(
            f"Balanced Accuracy : {balanced_acc:.4f}"
        )

        if not np.isnan(roc_auc):

            print(
                f"ROC-AUC           : "
                f"{roc_auc:.4f}"
            )

        else:

            print(
                "ROC-AUC           : N/A"
            )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)


output_path = (
    f"{OUTPUT_DIR}/"
    "feature_ablation_results.csv"
)


results_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# CALCULATE PERFORMANCE CHANGE
# ============================================================

comparison_results = []


for disease in TARGETS.keys():

    disease_results = results_df[
        results_df["Disease"]
        == disease
    ].copy()


    if disease_results.empty:

        continue


    baseline = disease_results[
        disease_results["Variant"]
        == "All features"
    ]


    if baseline.empty:

        continue


    baseline_row = baseline.iloc[0]


    for _, row in disease_results.iterrows():

        comparison_results.append({

            "Disease":
                disease,

            "Variant":
                row["Variant"],

            "Removed Feature":
                row["Removed Feature"],

            "Baseline Accuracy":
                baseline_row["Accuracy"],

            "Ablated Accuracy":
                row["Accuracy"],

            "Accuracy Change":
                row["Accuracy"]
                - baseline_row["Accuracy"],

            "Baseline Macro F1":
                baseline_row["Macro F1"],

            "Ablated Macro F1":
                row["Macro F1"],

            "Macro F1 Change":
                row["Macro F1"]
                - baseline_row["Macro F1"],

            "Baseline Balanced Accuracy":
                baseline_row[
                    "Balanced Accuracy"
                ],

            "Ablated Balanced Accuracy":
                row[
                    "Balanced Accuracy"
                ],

            "Balanced Accuracy Change":
                row[
                    "Balanced Accuracy"
                ]
                -
                baseline_row[
                    "Balanced Accuracy"
                ],

            "Baseline ROC AUC":
                baseline_row["ROC AUC"],

            "Ablated ROC AUC":
                row["ROC AUC"],

            "ROC AUC Change":
                row["ROC AUC"]
                -
                baseline_row["ROC AUC"]
        })


comparison_df = pd.DataFrame(
    comparison_results
)


comparison_path = (
    f"{OUTPUT_DIR}/"
    "feature_ablation_comparison.csv"
)


comparison_df.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("STEP 7 COMPLETED")
print("=" * 70)


print(
    f"\nSaved:"
    f"\n{output_path}"
    f"\n{comparison_path}"
)


print("\nAblation Results:")

print(
    results_df.to_string(
        index=False
    )
)