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

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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

    print(f"\n{'='*50}")
    print(model_name)
    print(f"{'='*50}")

    print("Accuracy:", accuracy_score(y_test, predictions))

    print(classification_report(y_test, predictions))

    joblib.dump(
        model,
        f"models/{model_name}.pkl"
    )

    print(f"{model_name} saved successfully.")

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
        "currently_breastfeeding"
    ]
]
X["currently_breastfeeding"] = X["currently_breastfeeding"].astype(int)

y_underweight = df["underweight_status"]
y_stunting = df["stunting_status"]
y_wasting = df["wasting_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_underweight,
    test_size=0.2,
    random_state=42
)


underweight_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

underweight_model.fit(X_train, y_train)
predictions = underweight_model.predict(X_test)


accuracy = accuracy_score(y_test, predictions)

print("Underweight Accuracy:", accuracy)

y = df["underweight_status"]

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