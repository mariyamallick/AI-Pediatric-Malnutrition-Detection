"""
Permutation Feature Importance Analysis
Pediatric Malnutrition Detection

Author: Mariya Mallick
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "data/raw/dhs_children_combined.csv"

OUTPUT_DIR = "evaluation/permutation_importance"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading DHS dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# SELECT FEATURES
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

targets = [
    "underweight_status",
    "stunting_status",
    "wasting_status"
]

for target in targets:
    df[target] = df[target].map(label_map)


# ============================================================
# RUN PERMUTATION IMPORTANCE
# ============================================================

for target in targets:

    print("\n" + "=" * 70)
    print(f"PERMUTATION IMPORTANCE: {target.upper()}")
    print("=" * 70)

    # --------------------------------------------------------
    # Remove missing values
    # --------------------------------------------------------

    analysis_df = df[features + [target]].dropna()

    X = analysis_df[features]
    y = analysis_df[target]

    print(f"Samples used: {len(X)}")

    # --------------------------------------------------------
    # Train-test split
    # Same split used in your previous evaluation
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    print("Training Random Forest...")

    model.fit(X_train, y_train)

    print(
        f"Test accuracy: "
        f"{model.score(X_test, y_test):.4f}"
    )

    # --------------------------------------------------------
    # PERMUTATION IMPORTANCE
    # --------------------------------------------------------

    print("Calculating permutation importance...")

    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="f1_macro",
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    # --------------------------------------------------------
    # CREATE RESULTS TABLE
    # --------------------------------------------------------

    importance_df = pd.DataFrame({

        "Feature": X_test.columns,

        "Permutation Importance Mean":
            result.importances_mean,

        "Permutation Importance Std":
            result.importances_std
    })

    importance_df = importance_df.sort_values(
        by="Permutation Importance Mean",
        ascending=False
    )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    csv_path = os.path.join(
        OUTPUT_DIR,
        f"{target}_permutation_importance.csv"
    )

    importance_df.to_csv(
        csv_path,
        index=False
    )

    print(f"\nSaved: {csv_path}")

    print("\nFeature Ranking:")

    print(importance_df.to_string(index=False))

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.barh(
        importance_df["Feature"],
        importance_df["Permutation Importance Mean"],
        xerr=importance_df["Permutation Importance Std"]
    )

    plt.xlabel(
        "Decrease in Macro F1 after permutation"
    )

    plt.ylabel("Feature")

    plt.title(
        f"Permutation Feature Importance - "
        f"{target.replace('_', ' ').title()}"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    png_path = os.path.join(
        OUTPUT_DIR,
        f"{target}_permutation_importance.png"
    )

    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {png_path}")


print("\n" + "=" * 70)
print("PERMUTATION IMPORTANCE ANALYSIS COMPLETED")
print("=" * 70)