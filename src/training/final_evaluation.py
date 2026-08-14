"""
Final Evaluation Pipeline
=========================

Project:
AI-Pediatric-Malnutrition-Detection

Dataset:
DHS Child Dataset

Targets:
1. Underweight
2. Stunting
3. Wasting

Models:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. SVM
5. KNN
6. Gaussian Naive Bayes
7. Gradient Boosting
8. XGBoost

Evaluation:
- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1
- Macro F1
- Balanced Accuracy
- Multiclass ROC-AUC (OVR, macro)
- Training time
- Inference time
- Confusion Matrix
- Classification Report
- 5-Fold CV for selected model only

Author: Mariya Mallick
"""

# ============================================================
# 1. IMPORTS
# ============================================================

import os
import time
import warnings
import pathlib

# Prevent GUI/Tkinter errors during plotting
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

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

warnings.filterwarnings("ignore")


# ============================================================
# 2. PATHS
# ============================================================

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "dhs_children_combined.csv"

EVALUATION_DIR = PROJECT_ROOT / "evaluation"
MODELS_DIR = PROJECT_ROOT / "models"

EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. MODEL DEFINITIONS
# ============================================================

def get_models():

    models = {

        "Logistic Regression": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    random_state=42
                )
            )
        ]),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=15,
            min_samples_leaf=20,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),

        # probability=False intentionally.
        # This avoids the probability=True warning
        # and makes SVM considerably faster.
        "SVM": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                SVC(
                    kernel="linear",
                    probability=False,
                    random_state=42
                )
            )
        ]),

        "KNN": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                KNeighborsClassifier(
                    n_neighbors=5,
                    n_jobs=-1
                )
            )
        ]),

        "Gaussian Naive Bayes": GaussianNB(),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ),

    }

    # XGBoost is optional.
    # If installed, include it.
    try:

        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1
        )

    except ImportError:

        print("\nWARNING:")
        print("XGBoost is not installed.")
        print("Install it with:")
        print("pip install xgboost")
        print()

    return models


# ============================================================
# 4. ROC-AUC FUNCTION
# ============================================================

def calculate_roc_auc(model, X_test, y_test):

    """
    Calculates multiclass ROC-AUC.

    Uses:
    - predict_proba() when available
    - decision_function() for models such as SVM
    """

    try:

        if hasattr(model, "predict_proba"):

            scores = model.predict_proba(X_test)

        elif hasattr(model, "decision_function"):

            scores = model.decision_function(X_test)

        else:

            return float("nan")

        auc = roc_auc_score(
            y_test,
            scores,
            multi_class="ovr",
            average="macro"
        )

        return auc

    except Exception:

        return float("nan")


# ============================================================
# 5. CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    y_test,
    predictions,
    model_name,
    disease
):

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1, 2]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Normal",
            "Moderate",
            "Severe"
        ]
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    display.plot(
        ax=ax,
        cmap="Blues",
        colorbar=True
    )

    ax.set_title(
        f"{model_name} - {disease.title()}",
        fontsize=14
    )

    plt.tight_layout()

    safe_model_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    filename = (
        f"{disease}_"
        f"{safe_model_name}_"
        f"confusion_matrix.png"
    )

    filepath = EVALUATION_DIR / filename

    plt.savefig(
        filepath,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close("all")

    return filepath


# ============================================================
# 6. CLASSIFICATION REPORT
# ============================================================

def save_classification_report(
    y_test,
    predictions,
    disease,
    model_name
):

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

    report_df = pd.DataFrame(report).transpose()

    safe_model_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    filepath = (
        EVALUATION_DIR /
        f"{disease}_{safe_model_name}_classification_report.csv"
    )

    report_df.to_csv(filepath)

    return filepath


# ============================================================
# 7. EVALUATE ONE MODEL
# ============================================================

def evaluate_model(
    model,
    model_name,
    X_train,
    X_test,
    y_train,
    y_test,
    disease
):

    print("\n" + "-" * 60)
    print(f"Training: {model_name}")
    print("-" * 60)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    start_train = time.perf_counter()

    model.fit(
        X_train,
        y_train
    )

    training_time = (
        time.perf_counter() -
        start_train
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    start_inference = time.perf_counter()

    predictions = model.predict(
        X_test
    )

    inference_time = (
        time.perf_counter() -
        start_inference
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
    )

    roc_auc = calculate_roc_auc(
        model,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"Accuracy           : {accuracy:.4f}"
    )

    print(
        f"Precision          : {precision:.4f}"
    )

    print(
        f"Recall             : {recall:.4f}"
    )

    print(
        f"F1 Score           : {f1:.4f}"
    )

    print(
        f"Macro F1           : {macro_f1:.4f}"
    )

    print(
        f"Balanced Accuracy  : {balanced_accuracy:.4f}"
    )

    if pd.isna(roc_auc):

        print(
            "ROC-AUC            : N/A"
        )

    else:

        print(
            f"ROC-AUC            : {roc_auc:.4f}"
        )

    print(
        f"Training Time      : {training_time:.2f} sec"
    )

    print(
        f"Inference Time     : {inference_time:.4f} sec"
    )

    # --------------------------------------------------------
    # Decision Tree information
    # --------------------------------------------------------

    if isinstance(
        model,
        DecisionTreeClassifier
    ):

        print(
            f"Tree Depth         : {model.get_depth()}"
        )

        print(
            f"Tree Leaves        : {model.get_n_leaves()}"
        )

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    save_confusion_matrix(
        y_test,
        predictions,
        model_name,
        disease
    )

    # --------------------------------------------------------
    # Save classification report
    # --------------------------------------------------------

    save_classification_report(
        y_test,
        predictions,
        disease,
        model_name
    )

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    result = {

        "Disease": disease,

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "Macro F1": macro_f1,

        "Balanced Accuracy": balanced_accuracy,

        "ROC AUC": roc_auc,

        "Training Time (sec)": training_time,

        "Inference Time (sec)": inference_time
    }

    return result, model


# ============================================================
# 8. MODEL COMPARISON
# ============================================================

def compare_models(
    X,
    y,
    disease
):

    print("\n")
    print("=" * 80)
    print(
        f"8-MODEL COMPARISON - {disease.upper()}"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = get_models()

    results = []

    trained_models = {}

    # --------------------------------------------------------
    # Train each model ONCE
    # --------------------------------------------------------

    for name, model in models.items():

        try:

            result, trained_model = evaluate_model(
                model,
                name,
                X_train,
                X_test,
                y_train,
                y_test,
                disease
            )

            results.append(result)

            trained_models[name] = trained_model

        except Exception as error:

            print(
                f"\nERROR while training {name}:"
            )

            print(error)

            print(
                "Skipping this model..."
            )

    # --------------------------------------------------------
    # Create results table
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # Sort primarily by Macro F1,
    # then Balanced Accuracy,
    # then ROC-AUC.

    results_df = results_df.sort_values(
        by=[
            "Macro F1",
            "Balanced Accuracy",
            "ROC AUC"
        ],
        ascending=False,
        na_position="last"
    )

    # --------------------------------------------------------
    # Print final ranking
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print(
        f"FINAL MODEL RANKING - {disease.upper()}"
    )
    print("=" * 80)

    print(
        results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save complete results
    # --------------------------------------------------------

    filepath = (
        EVALUATION_DIR /
        f"{disease}_final_model_comparison.csv"
    )

    results_df.to_csv(
        filepath,
        index=False
    )

    print(
        f"\nSaved: {filepath}"
    )

    return (
        results_df,
        trained_models,
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# 9. SELECT BEST MODEL
# ============================================================

def select_best_model(results_df):

    """
    Model selection priority:

    1. Macro F1
    2. Balanced Accuracy
    3. ROC-AUC
    4. Weighted F1
    """

    valid = results_df.copy()

    valid = valid.sort_values(
        by=[
            "Macro F1",
            "Balanced Accuracy",
            "ROC AUC",
            "F1 Score"
        ],
        ascending=False,
        na_position="last"
    )

    return valid.iloc[0]["Model"]


# ============================================================
# 10. SAVE BEST MODEL
# ============================================================

def save_best_model(
    trained_models,
    best_model_name,
    disease
):

    model = trained_models[
        best_model_name
    ]

    filepath = (
        MODELS_DIR /
        f"{disease}_best_model.pkl"
    )

    joblib.dump(
        model,
        filepath
    )

    print(
        f"Best model saved: {filepath}"
    )

    return model


# ============================================================
# 11. FINAL 5-FOLD CV
# ============================================================

def final_cross_validation(
    X,
    y,
    disease,
    best_model_name
):

    print("\n")
    print("=" * 80)
    print(
        f"FINAL 5-FOLD CROSS-VALIDATION - "
        f"{disease.upper()}"
    )
    print(
        f"Selected Model: {best_model_name}"
    )
    print("=" * 80)

    models = get_models()

    if best_model_name not in models:

        print(
            "Selected model definition not found."
        )

        return None

    model = models[
        best_model_name
    ]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {

        "accuracy": "accuracy",

        "precision": "precision_weighted",

        "recall": "recall_weighted",

        "f1": "f1_weighted",

        "macro_f1": "f1_macro",

        "balanced_accuracy":
            "balanced_accuracy"
    }

    # --------------------------------------------------------
    # Run CV
    # --------------------------------------------------------

    start = time.perf_counter()

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )

    elapsed = (
        time.perf_counter() -
        start
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"\nCross-validation time: "
        f"{elapsed:.2f} seconds"
    )

    print(
        f"\nAccuracy          : "
        f"{scores['test_accuracy'].mean():.4f} "
        f"± {scores['test_accuracy'].std():.4f}"
    )

    print(
        f"Precision         : "
        f"{scores['test_precision'].mean():.4f} "
        f"± {scores['test_precision'].std():.4f}"
    )

    print(
        f"Recall            : "
        f"{scores['test_recall'].mean():.4f} "
        f"± {scores['test_recall'].std():.4f}"
    )

    print(
        f"Weighted F1       : "
        f"{scores['test_f1'].mean():.4f} "
        f"± {scores['test_f1'].std():.4f}"
    )

    print(
        f"Macro F1          : "
        f"{scores['test_macro_f1'].mean():.4f} "
        f"± {scores['test_macro_f1'].std():.4f}"
    )

    print(
        f"Balanced Accuracy : "
        f"{scores['test_balanced_accuracy'].mean():.4f} "
        f"± {scores['test_balanced_accuracy'].std():.4f}"
    )

    # --------------------------------------------------------
    # Save fold results
    # --------------------------------------------------------

    cv_df = pd.DataFrame({

        "Fold": [1, 2, 3, 4, 5],

        "Accuracy":
            scores["test_accuracy"],

        "Precision":
            scores["test_precision"],

        "Recall":
            scores["test_recall"],

        "Weighted F1":
            scores["test_f1"],

        "Macro F1":
            scores["test_macro_f1"],

        "Balanced Accuracy":
            scores["test_balanced_accuracy"]
    })

    filepath = (
        EVALUATION_DIR /
        f"{disease}_best_model_5fold_cv.csv"
    )

    cv_df.to_csv(
        filepath,
        index=False
    )

    print(
        f"\nSaved: {filepath}"
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = {

        "Disease": disease,

        "Selected Model":
            best_model_name,

        "Accuracy Mean":
            scores["test_accuracy"].mean(),

        "Accuracy Std":
            scores["test_accuracy"].std(),

        "Precision Mean":
            scores["test_precision"].mean(),

        "Precision Std":
            scores["test_precision"].std(),

        "Recall Mean":
            scores["test_recall"].mean(),

        "Recall Std":
            scores["test_recall"].std(),

        "Weighted F1 Mean":
            scores["test_f1"].mean(),

        "Weighted F1 Std":
            scores["test_f1"].std(),

        "Macro F1 Mean":
            scores["test_macro_f1"].mean(),

        "Macro F1 Std":
            scores["test_macro_f1"].std(),

        "Balanced Accuracy Mean":
            scores["test_balanced_accuracy"].mean(),

        "Balanced Accuracy Std":
            scores["test_balanced_accuracy"].std()
    }

    summary_df = pd.DataFrame(
        [summary]
    )

    summary_path = (
        EVALUATION_DIR /
        f"{disease}_best_model_cv_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print(
        f"Saved: {summary_path}"
    )

    return summary_df


# ============================================================
# 12. DATA PREPARATION
# ============================================================

def load_and_prepare_data():

    print("\n")
    print("=" * 80)
    print("LOADING DHS DATASET")
    print("=" * 80)

    print(
        f"Dataset: {DATA_PATH}"
    )

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Total rows loaded: {len(df):,}"
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [

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

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + str(missing)
        )

    # --------------------------------------------------------
    # Select columns
    # --------------------------------------------------------

    df = df[
        required_columns
    ].copy()

    # --------------------------------------------------------
    # Encode sex
    # --------------------------------------------------------

    df["sex"] = df[
        "sex"
    ].map({
        "M": 1,
        "F": 0
    })

    # --------------------------------------------------------
    # Encode breastfeeding
    # --------------------------------------------------------

    df[
        "currently_breastfeeding"
    ] = (

        df[
            "currently_breastfeeding"
        ]

        .fillna(False)

        .astype(bool)

        .astype(int)
    )

    # --------------------------------------------------------
    # Encode labels
    # --------------------------------------------------------

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

        df[target] = df[
            target
        ].map(label_map)

    # --------------------------------------------------------
    # Remove rows with missing
    # values in model variables
    # --------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=[
            "age_months",
            "sex",
            "weight_kg",
            "height_cm",
            "wealth_index",
            "mother_education",
            "underweight_status",
            "stunting_status",
            "wasting_status"
        ]
    )

    after = len(df)

    print(
        f"Rows removed due to missing values: "
        f"{before - after:,}"
    )

    print(
        f"Final usable rows: {after:,}"
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    features = [

        "age_months",

        "sex",

        "weight_kg",

        "height_cm",

        "wealth_index",

        "mother_education",

        "currently_breastfeeding"
    ]

    X = df[
        features
    ].copy()

    # --------------------------------------------------------
    # Targets
    # --------------------------------------------------------

    targets = {

        "underweight": df[
            "underweight_status"
        ],

        "stunting": df[
            "stunting_status"
        ],

        "wasting": df[
            "wasting_status"
        ]
    }

    print(
        f"\nNumber of features: "
        f"{len(features)}"
    )

    print(
        f"Features: {features}"
    )

    # --------------------------------------------------------
    # Class distributions
    # --------------------------------------------------------

    print("\nClass distributions:")

    for disease, y in targets.items():

        print(
            f"\n{disease.title()}:"
        )

        print(
            y.value_counts()
            .sort_index()
            .to_string()
        )

    return X, targets


# ============================================================
# 13. MAIN
# ============================================================

def main():

    total_start = time.perf_counter()

    print("\n")
    print("=" * 80)
    print("FINAL PEDIATRIC MALNUTRITION MODEL EVALUATION")
    print("=" * 80)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X, targets = load_and_prepare_data()

    final_results = []

    # --------------------------------------------------------
    # Evaluate each disease
    # --------------------------------------------------------

    for disease, y in targets.items():

        (
            results_df,
            trained_models,
            X_train,
            X_test,
            y_train,
            y_test
        ) = compare_models(
            X,
            y,
            disease
        )

        # ----------------------------------------------------
        # Select best model
        # ----------------------------------------------------

        best_model_name = select_best_model(
            results_df
        )

        print("\n")
        print("=" * 80)
        print(
            f"SELECTED MODEL - "
            f"{disease.upper()}"
        )
        print("=" * 80)

        print(
            f"Best Model: {best_model_name}"
        )

        best_row = results_df[
            results_df["Model"]
            == best_model_name
        ].iloc[0]

        print(
            f"Macro F1: "
            f"{best_row['Macro F1']:.4f}"
        )

        print(
            f"Balanced Accuracy: "
            f"{best_row['Balanced Accuracy']:.4f}"
        )

        print(
            f"ROC-AUC: "
            f"{best_row['ROC AUC']:.4f}"
            if not pd.isna(
                best_row["ROC AUC"]
            )
            else
            "ROC-AUC: N/A"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        save_best_model(
            trained_models,
            best_model_name,
            disease
        )

        # ----------------------------------------------------
        # Final 5-fold CV
        # ----------------------------------------------------

        cv_summary = final_cross_validation(
            X,
            y,
            disease,
            best_model_name
        )

        # ----------------------------------------------------
        # Store final result
        # ----------------------------------------------------

        final_results.append({

            "Disease":
                disease,

            "Selected Model":
                best_model_name,

            "Test Accuracy":
                best_row["Accuracy"],

            "Test Macro F1":
                best_row["Macro F1"],

            "Test Balanced Accuracy":
                best_row["Balanced Accuracy"],

            "Test ROC AUC":
                best_row["ROC AUC"],

            "5-Fold Accuracy":
                (
                    cv_summary.iloc[0]
                    ["Accuracy Mean"]
                    if cv_summary is not None
                    else None
                ),

            "5-Fold Macro F1":
                (
                    cv_summary.iloc[0]
                    ["Macro F1 Mean"]
                    if cv_summary is not None
                    else None
                ),

            "5-Fold Balanced Accuracy":
                (
                    cv_summary.iloc[0]
                    ["Balanced Accuracy Mean"]
                    if cv_summary is not None
                    else None
                )
        })

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    final_df = pd.DataFrame(
        final_results
    )

    summary_path = (
        EVALUATION_DIR /
        "FINAL_MODEL_SELECTION_SUMMARY.csv"
    )

    final_df.to_csv(
        summary_path,
        index=False
    )

    print("\n")
    print("=" * 80)
    print("FINAL MODEL SELECTION SUMMARY")
    print("=" * 80)

    print(
        final_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {summary_path}"
    )

    # --------------------------------------------------------
    # Total runtime
    # --------------------------------------------------------

    total_time = (
        time.perf_counter() -
        total_start
    )

    print("\n")
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    print(
        f"Total runtime: "
        f"{total_time / 60:.2f} minutes"
    )

    print(
        "\nAll final evaluation files are in:"
    )

    print(
        EVALUATION_DIR
    )

    print(
        "\nBest models are in:"
    )

    print(
        MODELS_DIR
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()