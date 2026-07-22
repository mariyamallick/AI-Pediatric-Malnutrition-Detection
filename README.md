# AI for Pediatric Malnutrition Detection

An AI-powered healthcare system that detects pediatric malnutrition and provides personalized nutritional recommendations.

## Overview

Pediatric malnutrition remains a major global health challenge, particularly in low-resource settings where early diagnosis can be difficult.

This project aims to develop an AI-powered system capable of detecting malnutrition in children using anthropometric measurements, clinical indicators, and machine learning techniques. The system will also provide personalized nutritional recommendations based on the predicted deficiencies to assist healthcare professionals and caregivers.

The project is intended as a clinical decision-support tool and does not replace professional medical diagnosis.

## Objectives

    - Detect pediatric malnutrition using AI.
- Predict the severity of malnutrition.
- Identify probable nutrient deficiencies.
- Recommend personalized nutritional guidance.
- Generate reports for healthcare professionals.

## Features

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Machine Learning prediction
- Nutrient deficiency estimation
- Personalized diet recommendations
- Explainable AI
- PDF report generation
- Web application

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Flask/FastAPI
- HTML
- CSS
- JavaScript
- Git
- GitHub

## Status

🚧 Currently under development.

## Documentation

Project research and planning documents are available in the `docs/` folder.git 

## Web Application

The Flask application provides an AI-assisted screening form for children aged 0–60 months. It validates measurements, calculates BMI, runs the three trained models, and displays the recommendation engine's assessment.

### Run locally

Activate the project's virtual environment and install dependencies if needed:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then start the application from the project root:

```powershell
python app.py
```

Open `http://127.0.0.1:5000` in a browser. For a non-development deployment, set a strong `FLASK_SECRET_KEY` environment variable before starting the app.

## Project Documentation

- Research Notes: `docs/research.md`
- Dataset Planning: `docs/dataset.md`

## Dataset

This project uses the DHS (Demographic and Health Surveys) child dataset as the primary source for model training.

The dataset contains over 213,000 pediatric health records with anthropometric measurements, nutritional indicators, and dietary information.

Synthetic data is used only for application testing and demonstration.
