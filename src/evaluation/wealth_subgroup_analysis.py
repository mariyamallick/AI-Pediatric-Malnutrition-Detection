"""
Socioeconomic Subgroup / Bias Analysis
AI Pediatric Malnutrition Detection

Step 4C:
Evaluates final selected models across socioeconomic
groups based on DHS wealth_index tertiles.

Groups:
1. Low Wealth
2. Middle Wealth
3. High Wealth

No model retraining is performed.
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
    balanced_accuracy_score
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

OUTPUT_DIR = "evaluation/wealth_subgroup"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SOCIOECONOMIC SUBGROUP / BIAS ANALYSIS")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset size: {df.shape}")


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
# LABEL ENCODING
# ============================================================

label_map = {
    "normal": 0,
    "moderate": 1,
    "severe": 2
}

for target in [
    "underweight_status",
    "stunting_status",
    "wasting_status"
]:

    df[target] = df[target].map(label_map)


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

required_columns = [
    "age_months",
    "sex",
    "weight_kg",
    "height_cm",
    "wealth_index",
    "mother_education"
]

df = df.dropna(
    subset=required_columns
).copy()


# ============================================================
# CREATE WEALTH GROUPS
# ============================================================

print("\nCreating wealth groups...")

# Create tertiles from the actual dataset distribution.
# duplicates="drop" prevents failure if identical values
# produce duplicate quantile boundaries.

df["wealth_group"] = pd.qcut(
    df["wealth_index"],
    q=3,
    labels=[
        "Low Wealth",
        "Middle Wealth",
        "High Wealth"
    ],
    duplicates="drop"
)


print("\nWealth Group Distribution:")

print(
    df["wealth_group"]
    .value_counts()
    .sort_index()
)


print("\nWealth Group Boundaries:")

print(
    df.groupby(
        "wealth_group",
        observed=True
    )["wealth_index"]
    .agg(
        ["min", "max", "count"]
    )
)


# ============================================================
# LOAD FINAL MODELS
# ============================================================

models = {}

print("\nLoading final models...")

for disease, path in MODEL_PATHS.items():

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Model not found: {path}"
        )

    models[disease] = joblib.load(path)

    print(
        f"Loaded: {disease}"
    )


# ============================================================
# SUBGROUP EVALUATION
# ============================================================

results = []


wealth_groups = [
    "Low Wealth",
    "Middle Wealth",
    "High Wealth"
]


for disease, model in models.items():

    print("\n" + "=" * 70)

    print(
        f"{disease.upper()} - "
        "SOCIOECONOMIC SUBGROUP ANALYSIS"
    )

    print("=" * 70)

    target = f"{disease}_status"

    for wealth_group in wealth_groups:

        subgroup = df[
            df["wealth_group"] == wealth_group
        ].copy()

        if len(subgroup) == 0:

            print(
                f"\n{wealth_group}: No samples"
            )

            continue

        X = subgroup[features]

        y = subgroup[target]

        # Remove missing target values
        valid = y.notna()

        X = X[valid]

        y = y[valid]

        if len(y) == 0:

            continue

        predictions = model.predict(X)


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

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

        macro_f1 = f1_score(
            y,
            predictions,
            average="macro",
            zero_division=0
        )

        balanced_accuracy = (
            balanced_accuracy_score(
                y,
                predictions
            )
        )


        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print(
            f"\nWealth Group: "
            f"{wealth_group}"
        )

        print(
            f"Samples             : "
            f"{len(y)}"
        )

        print(
            f"Accuracy            : "
            f"{accuracy:.4f}"
        )

        print(
            f"Weighted Precision  : "
            f"{precision:.4f}"
        )

        print(
            f"Weighted Recall     : "
            f"{recall:.4f}"
        )

        print(
            f"Macro F1            : "
            f"{macro_f1:.4f}"
        )

        print(
            f"Balanced Accuracy   : "
            f"{balanced_accuracy:.4f}"
        )


        results.append({

            "Disease": disease,

            "Wealth Group": wealth_group,

            "Samples": len(y),

            "Accuracy": accuracy,

            "Weighted Precision": precision,

            "Weighted Recall": recall,

            "Macro F1": macro_f1,

            "Balanced Accuracy":
                balanced_accuracy

        })


# ============================================================
# SAVE MAIN RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_csv = (
    f"{OUTPUT_DIR}/wealth_subgroup_results.csv"
)

results_df.to_csv(
    results_csv,
    index=False
)


print("\n" + "=" * 70)

print(
    "SOCIOECONOMIC SUBGROUP RESULTS"
)

print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

print(
    f"\nSaved: {results_csv}"
)


# ============================================================
# PERFORMANCE GAP ANALYSIS
# ============================================================

gap_results = []


for disease in results_df["Disease"].unique():

    disease_data = results_df[
        results_df["Disease"] == disease
    ]


    for metric in [
        "Accuracy",
        "Macro F1",
        "Balanced Accuracy"
    ]:

        maximum = (
            disease_data[metric].max()
        )

        minimum = (
            disease_data[metric].min()
        )

        gap = maximum - minimum


        max_group = disease_data.loc[
            disease_data[metric].idxmax(),
            "Wealth Group"
        ]

        min_group = disease_data.loc[
            disease_data[metric].idxmin(),
            "Wealth Group"
        ]


        gap_results.append({

            "Disease": disease,

            "Metric": metric,

            "Maximum": maximum,

            "Minimum": minimum,

            "Wealth Group Gap": gap,

            "Best Performing Group":
                max_group,

            "Lowest Performing Group":
                min_group

        })


gap_df = pd.DataFrame(
    gap_results
)


gap_csv = (
    f"{OUTPUT_DIR}/wealth_subgroup_gaps.csv"
)


gap_df.to_csv(
    gap_csv,
    index=False
)


print(
    f"Saved: {gap_csv}"
)


# ============================================================
# PLOT 1 — MACRO F1
# ============================================================

pivot_f1 = results_df.pivot(
    index="Wealth Group",
    columns="Disease",
    values="Macro F1"
)

ax = pivot_f1.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title(
    "Macro F1 Across Socioeconomic Groups"
)

plt.xlabel(
    "Socioeconomic Group"
)

plt.ylabel(
    "Macro F1"
)

plt.ylim(
    0,
    1
)

plt.xticks(
    rotation=0
)

plt.legend(
    title="Outcome"
)

plt.tight_layout()


f1_path = (
    f"{OUTPUT_DIR}/wealth_macro_f1.png"
)


plt.savefig(
    f1_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PLOT 2 — BALANCED ACCURACY
# ============================================================

pivot_balanced = results_df.pivot(
    index="Wealth Group",
    columns="Disease",
    values="Balanced Accuracy"
)

ax = pivot_balanced.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title(
    "Balanced Accuracy Across Socioeconomic Groups"
)

plt.xlabel(
    "Socioeconomic Group"
)

plt.ylabel(
    "Balanced Accuracy"
)

plt.ylim(
    0,
    1
)

plt.xticks(
    rotation=0
)

plt.legend(
    title="Outcome"
)

plt.tight_layout()


balanced_path = (
    f"{OUTPUT_DIR}/wealth_balanced_accuracy.png"
)


plt.savefig(
    balanced_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)

print(
    "STEP 4C COMPLETED"
)

print("=" * 70)

print("\nGenerated files:")

print(
    f"1. {results_csv}"
)

print(
    f"2. {gap_csv}"
)

print(
    f"3. {f1_path}"
)

print(
    f"4. {balanced_path}"
)

print(
    "\nSocioeconomic subgroup analysis "
    "completed successfully."
)