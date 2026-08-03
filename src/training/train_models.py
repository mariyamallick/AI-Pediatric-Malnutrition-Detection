"""
Training pipeline for Pediatric Malnutrition Detection models.

Dataset:
    DHS Child Dataset

Models:
    - Underweight
    - Stunting
    - Wasting

Author: Mariya Mallick
"""
import pandas as pd
import joblib
import pathlib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

def train_model(X, y, model_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted")
    recall = recall_score(y_test, predictions, average="weighted")
    f1 = f1_score(y_test, predictions, average="weighted")

    print(f"\n{'='*60}")
    print(model_name)
    print(f"{'='*60}")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report")
    print(classification_report(y_test, predictions))


    results = pd.DataFrame([{
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }])

    results.to_csv(
        f"evaluation/{model_name}_metrics.csv",
        index=False
    )


    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(
        f"evaluation/{model_name}_classification_report.csv"
    )


    cm = confusion_matrix(y_test, predictions)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot(cmap="Blues")

    plt.title(model_name)

    plt.savefig(
        f"evaluation/{model_name}_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    joblib.dump(
        model,
        f"models/{model_name}.pkl"
    )

    print(f"{model_name} saved successfully.")


def compare_models(X, y, disease):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {

        "Logistic Regression": LogisticRegression(max_iter=1000),

        "Decision Tree": DecisionTreeClassifier(random_state=42),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )

    }

    results = []

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        results.append({

            "Model": name,

            "Accuracy": accuracy_score(y_test, predictions),

            "Precision": precision_score(
                y_test,
                predictions,
                average="weighted"
            ),

            "Recall": recall_score(
                y_test,
                predictions,
                average="weighted"
            ),

            "F1 Score": f1_score(
                y_test,
                predictions,
                average="weighted"
            )

        })

    results = pd.DataFrame(results)

    print(results)

    results.to_csv(
        f"evaluation/{disease}_model_comparison.csv",
        index=False
    )

def cross_validation(X, y, disease):
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy"
    )

    print("\n" + "="*60)
    print("5-Fold Cross Validation")
    print("="*60)

    print("Fold Accuracies:")
    print(scores)

    print(f"\nAverage Accuracy : {scores.mean():.4f}")
    print(f"Standard Deviation: {scores.std():.4f}")

    cv_results = pd.DataFrame({
        "Fold": [1,2,3,4,5],
        "Accuracy": scores
    })

    cv_results.to_csv(
        f"evaluation/{disease}_cross_validation.csv",
        index=False
    )

df = pd.read_csv("data/raw/dhs_children_combined.csv")

print(df.head())

df = df[
    [
        "age_months",
        "sex",
        "weight_kg",
        "height_cm",
        "wealth_index",
        "mother_education",
        "currently_breastfeeding",
        "haz",
        "waz",
        "whz",
        "underweight_status",
        "stunting_status",
        "wasting_status"
    ]
]

print(df.head())

features = [
    "age_months",
    "sex",
    "weight_kg",
    "height_cm",
    "wealth_index",
    "mother_education",
    "currently_breastfeeding"
]

df["sex"] = df["sex"].map({
    "M":1,
    "F":0
})

print(df["sex"].value_counts())

print("\nUnderweight Labels:")
print(df["underweight_status"].unique())

print("\nStunting Labels:")
print(df["stunting_status"].unique())

print("\nWasting Labels:")
print(df["wasting_status"].unique())

label_map = {
    "normal": 0,
    "moderate": 1,
    "severe": 2
}

df["underweight_status"] = df["underweight_status"].map(label_map)
df["stunting_status"] = df["stunting_status"].map(label_map)
df["wasting_status"] = df["wasting_status"].map(label_map)

print(df.head())

X = df[
    [
        "age_months",
        "sex",
        "weight_kg",
        "height_cm",
        "wealth_index",
        "mother_education",
        "currently_breastfeeding",
        "waz",
        "haz",
        "whz"
    ]
]
X = X.copy()

X["currently_breastfeeding"] = (
    X["currently_breastfeeding"]
    .fillna(False)
    .astype(bool)
    .astype(int)
)

y_underweight = df["underweight_status"]
y_stunting = df["stunting_status"]
y_wasting = df["wasting_status"]

train_model(
    X,
    df["underweight_status"],
    "underweight_status_model"
)

train_model(
    X,
    df["stunting_status"],
    "stunting_status_model"
)

train_model(
    X,
    df["wasting_status"],
    "wasting_status_model"
)

print("\nComparing Models for Underweight Prediction...\n")
compare_models(X, df["underweight_status"], "underweight")



print("\n5-Fold Cross Validation - Underweight\n")
cross_validation(X, df["underweight_status"], "underweight")


print("\nComparing Models for Stunting Prediction...\n")
compare_models(X, df["stunting_status"], "stunting")

print("\n5-Fold Cross Validation - Stunting\n")
cross_validation(X, df["stunting_status"], "stunting")



print("\nComparing Models for Wasting Prediction...\n")
compare_models(X, df["wasting_status"], "wasting")


print("\n5-Fold Cross Validation - Wasting\n")
cross_validation(X, df["wasting_status"], "wasting")