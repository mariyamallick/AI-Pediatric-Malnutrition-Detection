# Data Source Registry

## Project

AI-Assisted Pediatric Malnutrition Detection and Personalized Nutritional Recommendation System

---

## Dataset 1: Synthetic Pediatric Dataset

**Status:** Available

**Source:** Self-generated synthetic dataset

**Purpose:**
- Initial machine learning model development
- Feature engineering
- Exploratory Data Analysis
- Testing preprocessing pipeline

**Expected Features**
- Age (months)
- Sex
- Weight
- Height
- MUAC
- WAZ
- HAZ
- WHZ
- Underweight Status
- Stunting Status
- Wasting Status

---

## Dataset 2: DHS Child Records

**Status:** Access Requested

**Source:** Demographic and Health Surveys (DHS)

**Purpose**
- External validation
- Improve model robustness
- Compare synthetic and real-world data

Expected Features:
- Child age
- Weight
- Height
- Sex
- Household information
- Maternal education
- Health indicators

---

## Dataset 3: WHO Child Growth Standards

**Status:** To Download

**Source:** World Health Organization (WHO)

Purpose:
- Growth validation
- Z-score reference
- Clinical comparison

---

## Dataset 4: Nutrition Knowledge Base

**Status:** To Create

Purpose:
Provide evidence-based nutritional recommendations based on AI predictions.

Example Fields:

- Malnutrition Category
- Recommended Foods
- Calories
- Protein Sources
- Iron-rich Foods
- Vitamin A Sources
- Zinc Sources
- Medical Referral Required

---

## Data Pipeline

Synthetic Dataset
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Machine Learning Model
        ↓
Prediction
        ↓
Recommendation Engine