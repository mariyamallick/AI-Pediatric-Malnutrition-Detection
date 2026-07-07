# System Architecture

## AI-Assisted Pediatric Malnutrition Detection and Personalized Nutritional Recommendation System

## Overview

The system is designed as an end-to-end clinical decision support tool for children aged 0–59 months. It combines anthropometric measurements, machine learning, WHO growth standards, and evidence-based nutritional recommendations.

---

## System Workflow

Child Information
        │
        ▼
Data Validation
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Machine Learning Model
        │
        ▼
Malnutrition Classification
        │
        ▼
Explainable AI
        │
        ▼
Nutrition Recommendation Engine
        │
        ▼
PDF Report
        │
        ▼
Web Application

---

## Input Features

- Age (months)
- Sex
- Weight (kg)
- Height (cm)
- MUAC (cm)
- WAZ
- HAZ
- WHZ

Future Features

- Hemoglobin
- Appetite
- Fever
- Diarrhea
- Breastfeeding Status
- Immunization Status
- Maternal Education

---

## Machine Learning Output

The AI model predicts:

- Normal
- Underweight
- Stunted
- Wasted
- Severe Acute Malnutrition (SAM)
- Moderate Acute Malnutrition (MAM)

---

## Recommendation Engine

The recommendation module provides:

- Nutritional assessment
- Suggested food groups
- Protein recommendations
- Iron-rich foods
- Vitamin A sources
- Zinc-rich foods
- Follow-up advice
- Referral recommendations (when required)

---

## Technologies

Backend:
- Python
- Flask/FastAPI

Machine Learning:
- Scikit-learn
- XGBoost

Visualization:
- Matplotlib
- Plotly

Frontend:
- HTML
- CSS
- JavaScript

Deployment:
- Docker
- GitHub