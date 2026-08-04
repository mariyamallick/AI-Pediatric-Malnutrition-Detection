# 🩺 AI-Powered Pediatric Malnutrition Detection System

An Explainable Artificial Intelligence (XAI) based web application for early identification of pediatric malnutrition using Machine Learning, WHO Growth Standards, and clinical decision support.

---

## 📌 Overview

Pediatric malnutrition remains a major global health challenge, particularly in low- and middle-income countries. Early identification of undernutrition can significantly improve clinical outcomes and support timely nutritional interventions.

This project combines Machine Learning with WHO Growth Standard calculations to predict:

- Underweight
- Stunting
- Wasting

The system also provides:

- WHO Growth Indicators
- BMI Calculation
- Personalized Nutrition Recommendations
- Food Recommendations
- Clinical Summary
- SHAP Explainability
- AI-generated Prediction Summary
- Assessment History
- PDF Report Generation

---

## 🚀 Features

### ✅ Machine Learning Prediction

Random Forest models predict:

- Underweight
- Stunting
- Wasting

using demographic and anthropometric information.

---

### 📊 WHO Growth Assessment

The application automatically calculates:

- Weight-for-Age Z Score (WAZ)
- Height-for-Age Z Score (HAZ)
- Weight-for-Height Z Score (WHZ)
- Body Mass Index (BMI)

according to WHO Child Growth Standards.

---

### 🧠 Explainable AI (SHAP)

The system provides transparent predictions using SHAP (SHapley Additive Explanations).

For every prediction it shows:

- Most influential features
- Feature impact
- Increased/Reduced Risk
- AI-generated explanation

---

### 🥗 Nutrition Recommendation Engine

Provides nutritional guidance based on:

- Predicted malnutrition status
- WHO Growth Indicators
- Child age

---

### 🍎 Food Recommendation System

Suggests age-appropriate food recommendations for:

- Underweight children
- Stunted children
- Wasted children

---

### 🏥 Clinical Summary

Automatically generates a concise clinical interpretation for healthcare workers or caregivers.

---

### 📄 PDF Report Generation

Creates downloadable assessment reports including:

- Patient Information
- Prediction Results
- WHO Growth Assessment
- Nutrition Advice
- Food Recommendations
- Clinical Summary

---

### 📜 Assessment History

Stores previous assessments using SQLite.

History page includes:

- Previous assessments
- BMI
- Overall Risk
- Prediction history
- Summary statistics

---

## 🛠 Technology Stack

### Backend

- Python
- Flask

### Machine Learning

- Scikit-learn
- Random Forest
- SHAP

### Data Processing

- Pandas
- NumPy

### Database

- SQLite

### Reports

- ReportLab

### Frontend

- HTML
- CSS
- Bootstrap
- Jinja2

---

## 📂 Project Structure

```
AI-Pediatric-Malnutrition-Detection/

│

├── app/
│   └── app.py

│

├── data/
│   └── raw/
│       └── dhs_children_combined.csv

│

├── evaluation/
│   ├── confusion matrices
│   ├── classification reports
│   ├── metrics
│   ├── model comparison
│   └── cross validation

│

├── generated_reports/

│

├── models/
│   ├── underweight_status_model.pkl
│   ├── stunting_status_model.pkl
│   └── wasting_status_model.pkl

│

├── src/
│   ├── database/
│   ├── explainability/
│   ├── growth/
│   ├── pipeline/
│   ├── recommendations/
│   ├── reports/
│   └── training/

│

├── templates/

│

├── database.db

│

└── README.md
```

---

## 📊 Machine Learning Models

The project trains three independent Random Forest classifiers.

| Model | Target |
|--------|----------|
| Model 1 | Underweight |
| Model 2 | Stunting |
| Model 3 | Wasting |

---

## 📈 Model Evaluation

The training pipeline automatically performs:

- Train-Test Split (80:20)
- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix
- Model Comparison
- 5-Fold Cross Validation

Models compared:

- Logistic Regression
- Decision Tree
- Random Forest

Evaluation reports are automatically saved inside the **evaluation/** folder.

---

## 🩺 WHO Growth Indicators

The system computes:

- Weight-for-Age Z-score (WAZ)
- Height-for-Age Z-score (HAZ)
- Weight-for-Height Z-score (WHZ)
- BMI

These indicators are displayed to assist clinical interpretation alongside AI predictions.

---

## 🔍 Explainable AI

SHAP is used to improve transparency by identifying which patient features contributed most to the model's prediction.

This helps users understand *why* the AI reached its conclusion.

---

## 💾 Database

SQLite stores assessment history including:

- Date
- Child Information
- BMI
- Overall Risk
- Underweight Prediction
- Stunting Prediction
- Wasting Prediction

---

## 📄 Generated Reports

Each assessment can be exported as a PDF containing:

- Child Details
- Prediction Results
- WHO Growth Assessment
- Nutrition Recommendations
- Food Recommendations
- Clinical Summary

---

## ▶️ Running the Project

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train models

```bash
python src/training/train_models.py
```

### Run the application

```bash
python app/app.py
```

Open:

```
http://127.0.0.1:5000
```

---

## 🎯 Future Enhancements

- Deep Learning Models
- XGBoost Integration
- Multi-country Validation
- Mobile Application
- Growth Trend Visualization
- Cloud Deployment
- Real-time Clinical Dashboard
- Bias and Fairness Analysis

---

## 👩‍💻 Author

**Mariya Mallick**

B.Tech Computer Science & Engineering

Jamia Hamdard University

---

## 📜 License

This project is intended for educational and research purposes only and is not a substitute for professional medical diagnosis.