"""
Age Subgroup / Bias Analysis
AI Pediatric Malnutrition Detection

Step 4B:
Evaluates final selected models across child age groups.

Age groups:
1. <12 months
2. 12-59 months
3. >=60 months
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

OUTPUT_DIR = "evaluation/age_subgroup"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("AGE SUBGROUP / BIAS ANALYSIS")
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


# Remove rows with missing required values
df = df.dropna(
    subset=[
        "age_months",
        "sex",
        "weight_kg",
        "height_cm",
        "wealth_index",
        "mother_education"
    ]
).copy()


# ============================================================
# CREATE AGE GROUPS
# ============================================================

def assign_age_group(age):

    if age < 12:
        return "<12 months"

    elif age < 60:
        return "12-59 months"

    else:
        return ">=60 months"


df["age_group"] = df["age_months"].apply(assign_age_group)


print("\nAge Group Distribution:")
print(df["age_group"].value_counts().sort_index())


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
# LOAD MODELS
# ============================================================

models = {}

print("\nLoading final models...")

for disease, path in MODEL_PATHS.items():

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found: {path}"
        )

    models[disease] = joblib.load(path)

    print(f"Loaded: {disease}")


# ============================================================
# AGE SUBGROUP EVALUATION
# ============================================================

results = []


for disease, model in models.items():

    print("\n" + "=" * 70)
    print(f"{disease.upper()} - AGE SUBGROUP ANALYSIS")
    print("=" * 70)

    target = f"{disease}_status"

    for age_group in [
        "<12 months",
        "12-59 months",
        ">=60 months"
    ]:

        subgroup = df[
            df["age_group"] == age_group
        ].copy()

        if len(subgroup) == 0:

            print(
                f"\n{age_group}: No samples"
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

        balanced_accuracy = balanced_accuracy_score(
            y,
            predictions
        )

        print(
            f"\nAge Group: {age_group}"
        )

        print(
            f"Samples             : {len(y)}"
        )

        print(
            f"Accuracy            : {accuracy:.4f}"
        )

        print(
            f"Weighted Precision  : {precision:.4f}"
        )

        print(
            f"Weighted Recall     : {recall:.4f}"
        )

        print(
            f"Macro F1            : {macro_f1:.4f}"
        )

        print(
            f"Balanced Accuracy   : {balanced_accuracy:.4f}"
        )

        results.append({

            "Disease": disease,

            "Age Group": age_group,

            "Samples": len(y),

            "Accuracy": accuracy,

            "Weighted Precision": precision,

            "Weighted Recall": recall,

            "Macro F1": macro_f1,

            "Balanced Accuracy": balanced_accuracy

        })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

output_csv = (
    f"{OUTPUT_DIR}/age_subgroup_results.csv"
)

results_df.to_csv(
    output_csv,
    index=False
)

print("\n" + "=" * 70)
print("AGE SUBGROUP RESULTS")
print("=" * 70)

print(results_df.to_string(index=False))

print(
    f"\nSaved: {output_csv}"
)


# ============================================================
# CALCULATE AGE-RELATED PERFORMANCE GAPS
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

        maximum = disease_data[metric].max()
        minimum = disease_data[metric].min()

        gap = maximum - minimum

        gap_results.append({

            "Disease": disease,

            "Metric": metric,

            "Maximum": maximum,

            "Minimum": minimum,

            "Age Group Gap": gap

        })


gap_df = pd.DataFrame(gap_results)

gap_csv = (
    f"{OUTPUT_DIR}/age_subgroup_gaps.csv"
)

gap_df.to_csv(
    gap_csv,
    index=False
)

print(
    f"\nSaved: {gap_csv}"
)


# ============================================================
# PLOT 1 — MACRO F1
# ============================================================

pivot_f1 = results_df.pivot(
    index="Age Group",
    columns="Disease",
    values="Macro F1"
)

ax = pivot_f1.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title(
    "Macro F1 Across Child Age Groups"
)

plt.xlabel("Age Group")

plt.ylabel("Macro F1")

plt.ylim(0, 1)

plt.xticks(
    rotation=0
)

plt.legend(
    title="Outcome"
)

plt.tight_layout()

f1_path = (
    f"{OUTPUT_DIR}/age_macro_f1.png"
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
    index="Age Group",
    columns="Disease",
    values="Balanced Accuracy"
)

ax = pivot_balanced.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title(
    "Balanced Accuracy Across Child Age Groups"
)

plt.xlabel("Age Group")

plt.ylabel("Balanced Accuracy")

plt.ylim(0, 1)

plt.xticks(
    rotation=0
)

plt.legend(
    title="Outcome"
)

plt.tight_layout()

balanced_path = (
    f"{OUTPUT_DIR}/age_balanced_accuracy.png"
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
print("STEP 4B COMPLETED")
print("=" * 70)

print("\nGenerated files:")

print(
    f"1. {output_csv}"
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

print("\nAge subgroup analysis completed successfully.")