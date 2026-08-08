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

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    cross_validate,
    StratifiedKFold
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
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

    precision = precision_score(
        y_test,
        predictions,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
    )

    probabilities = model.predict_proba(X_test)

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
        multi_class="ovr",
        average="macro"  
    )


    print(f"\n{'='*60}")
    print(model_name)
    print(f"{'='*60}")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print(f"Macro F1 : {macro_f1:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")


    print("\nClassification Report")
    print(classification_report(y_test, predictions))


    results = pd.DataFrame([{
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Macro F1": macro_f1,
        "Balanced Accuracy": balanced_accuracy,
        "ROC AUC": roc_auc
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

        "Logistic Regression": LogisticRegression(
            max_iter=10000,
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=15,
            min_samples_leaf=20,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )

    }

    results = []

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro"
        )

        balanced_accuracy = balanced_accuracy_score(
            y_test,
            predictions
        )

        roc_auc = roc_auc_score(
            y_test,
            probabilities,
            multi_class="ovr",
            average="macro"
        )

        if name == "Decision Tree":
            print("Decision Tree Depth:", model.get_depth())
            print("Decision Tree Leaves:", model.get_n_leaves())

        results.append({

            "Model": name,

            "Accuracy": accuracy_score(
                y_test, 
                predictions
            ),

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
            ),

            "Macro F1": macro_f1,

            "Balanced Accuracy": balanced_accuracy,
            "ROC AUC": roc_auc
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

def compare_models_cross_validation(X, y, disease):

    print("\n" + "=" * 70)
    print(f"5-FOLD MODEL CROSS-VALIDATION - {disease.upper()}")
    print("=" * 70)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=5000,
            solver="lbfgs"
        ),

        "Decision Tree":DecisionTreeClassifier(
            max_depth=15,
            min_samples_leaf=20,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    }

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted"
    }

    results = []

    for name, model in models.items():

        print(f"\nTesting: {name}")

        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        accuracy_mean = scores["test_accuracy"].mean()
        accuracy_std = scores["test_accuracy"].std()

        precision_mean = scores["test_precision"].mean()
        precision_std = scores["test_precision"].std()

        recall_mean = scores["test_recall"].mean()
        recall_std = scores["test_recall"].std()

        f1_mean = scores["test_f1"].mean()
        f1_std = scores["test_f1"].std()

        print(
            f"Accuracy : {accuracy_mean:.4f} ± {accuracy_std:.4f}"
        )

        print(
            f"Precision: {precision_mean:.4f} ± {precision_std:.4f}"
        )

        print(
            f"Recall   : {recall_mean:.4f} ± {recall_std:.4f}"
        )

        print(
            f"F1 Score : {f1_mean:.4f} ± {f1_std:.4f}"
        )

        results.append({
            "Disease": disease,
            "Model": name,

            "Accuracy Mean": accuracy_mean,
            "Accuracy Std": accuracy_std,

            "Precision Mean": precision_mean,
            "Precision Std": precision_std,

            "Recall Mean": recall_mean,
            "Recall Std": recall_std,

            "F1 Mean": f1_mean,
            "F1 Std": f1_std
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="F1 Mean",
        ascending=False
    )

    results_df.to_csv(
        f"evaluation/{disease}_full_cross_validation.csv",
        index=False
    )

    print("\nFinal Ranking:")
    print(
        results_df[
            [
                "Model",
                "Accuracy Mean",
                "Accuracy Std",
                "F1 Mean",
                "F1 Std"
            ]
        ]
    )

    print(
        f"\nSaved: evaluation/"
        f"{disease}_full_cross_validation.csv"
    )

    return results_df


def inspect_decision_tree(X, y, disease):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    tree = DecisionTreeClassifier(
        max_depth=15,
        min_samples_leaf=20,
        random_state=42
    )

    tree.fit(X_train, y_train)

    print("\n" + "=" * 60)
    print(f"DECISION TREE COMPLEXITY - {disease.upper()}")
    print("=" * 60)

    print("Tree Depth:", tree.get_depth())
    print("Number of Leaves:", tree.get_n_leaves())

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

# Base features used for leakage-controlled evaluation
base_features = [
    "age_months",
    "sex",
    "weight_kg",
    "height_cm",
    "wealth_index",
    "mother_education",
    "currently_breastfeeding"
]

X = df[base_features].copy()

X["currently_breastfeeding"] = (
    X["currently_breastfeeding"]
    .fillna(False)
    .astype(bool)
    .astype(int)
)

# Leakage-controlled feature sets
# Remove the z-score that was directly used to create each target.

X_underweight = X.copy()

X_stunting = X.copy()

X_wasting = X.copy()

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
    X_underweight,
    df["underweight_status"],
    "underweight_status_model"
)

train_model(
    X_stunting,
    df["stunting_status"],
    "stunting_status_model"
)

train_model(
    X_wasting,
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

print("\n\nRunning Full Cross-Validation Model Comparison...\n")


underweight_cv = compare_models_cross_validation(
    X,
    df["underweight_status"],
    "underweight"
)


stunting_cv = compare_models_cross_validation(
    X,
    df["stunting_status"],
    "stunting"
)


wasting_cv = compare_models_cross_validation(
    X,
    df["wasting_status"],
    "wasting"
)