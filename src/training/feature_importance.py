"""
Feature Importance Analysis
Pediatric Malnutrition Detection

Author: Mariya Mallick
"""

import pandas as pd
import joblib
import matplotlib

# Prevent Tkinter GUI errors
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "raw" / "dhs_children_combined.csv"
MODEL_DIR = BASE_DIR / "models"
EVAL_DIR = BASE_DIR / "evaluation"


EVAL_DIR.mkdir(exist_ok=True)


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
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset loaded: {len(df):,} rows")


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

targets = {
    "underweight": "underweight_status",
    "stunting": "stunting_status",
    "wasting": "wasting_status"
}

for target_column in targets.values():

    df[target_column] = df[target_column].map(label_map)


# ============================================================
# FEATURE IMPORTANCE FUNCTION
# ============================================================

def analyze_feature_importance(model_file, target_column, disease):

    print("\n" + "=" * 70)
    print(f"FEATURE IMPORTANCE - {disease.upper()}")
    print("=" * 70)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model_path = MODEL_DIR / model_file

    print(f"Loading model: {model_path.name}")

    model = joblib.load(model_path)

    # --------------------------------------------------------
    # Prepare X
    # --------------------------------------------------------

    X = df[FEATURES].copy()

    y = df[target_column]

    # Remove rows with missing target
    valid_rows = y.notna()

    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    if not hasattr(model, "feature_importances_"):

        print(
            f"ERROR: {model.__class__.__name__} "
            "does not provide feature_importances_."
        )

        return

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": importances
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    importance_df["Percentage"] = (
        importance_df["Importance"] * 100
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nFeature Importance Ranking:\n")

    for _, row in importance_df.iterrows():

        print(
            f"{row['Feature']:30s} "
            f"{row['Percentage']:.2f}%"
        )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_path = (
        EVAL_DIR /
        f"{disease}_feature_importance.csv"
    )

    importance_df.to_csv(
        csv_path,
        index=False
    )

    print(f"\nSaved: {csv_path}")

    # --------------------------------------------------------
    # Create plot
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title(
        f"Feature Importance - {disease.capitalize()}"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plot_path = (
        EVAL_DIR /
        f"{disease}_feature_importance.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved plot: {plot_path}")

    return importance_df


# ============================================================
# RUN ANALYSIS
# ============================================================

# Underweight
analyze_feature_importance(
    "underweight_status_model.pkl",
    "underweight_status",
    "underweight"
)


# Stunting
analyze_feature_importance(
    "stunting_status_model.pkl",
    "stunting_status",
    "stunting"
)


# Wasting
analyze_feature_importance(
    "wasting_status_model.pkl",
    "wasting_status",
    "wasting"
)


print("\n" + "=" * 70)
print("FEATURE IMPORTANCE ANALYSIS COMPLETED")
print("=" * 70)

print("\nGenerated files:")
print("evaluation/underweight_feature_importance.csv")
print("evaluation/stunting_feature_importance.csv")
print("evaluation/wasting_feature_importance.csv")

print("\nGenerated plots:")
print("evaluation/underweight_feature_importance.png")
print("evaluation/stunting_feature_importance.png")
print("evaluation/wasting_feature_importance.png")