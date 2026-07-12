# Dataset Planning

## Purpose

The AI model requires high-quality pediatric health data to identify malnutrition and estimate possible nutrient deficiencies.

The dataset should contain anthropometric measurements, demographic information, and clinical indicators that can be used to predict nutritional status.

## Proposed Input Features

The following features are planned for the AI model:

| Feature | Description |
|---------|-------------|
| Age | Child's age (months or years) |
| Gender | Male/Female |
| Weight | Weight in kilograms |
| Height | Height in centimeters |
| BMI | Body Mass Index |
| MUAC | Mid Upper Arm Circumference |
| Hemoglobin | Blood hemoglobin level (if available) |
| Appetite | Normal / Reduced |
| Fever | Yes / No |
| Diarrhea | Yes / No |
| Recent Illness | Yes / No |

## Target Variable

The AI model will classify children into one of the following categories:

- Normal
- Mild Malnutrition
- Moderate Malnutrition
- Severe Malnutrition

## Planned Data Sources

Potential datasets include:

- WHO Child Growth Standards
- UNICEF Nutrition Data
- Kaggle Pediatric Nutrition Datasets
- Demographic and Health Surveys (DHS)
- National Family Health Survey (NFHS), India

## Data Preprocessing Plan

The following preprocessing steps are planned:

- Handle missing values
- Remove duplicate records
- Detect and treat outliers
- Encode categorical variables
- Normalize numerical features
- Split data into training and testing sets


## Future Dataset Enhancements

Future versions of the project may include:

- Clinical symptoms
- Laboratory test results
- Dietary history
- Geographic information
- Socioeconomic indicators

# Synthetic Dataset Documentation

## Overview

The synthetic pediatric dataset was created during the initial phase of this project to support machine learning model development before incorporating real-world clinical data.

## Purpose

The dataset is used for:

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature engineering
- Initial machine learning model training
- Pipeline testing

## Dataset Characteristics

Number of Records:
5000

Features:

- Age (months)
- Sex
- Weight (kg)
- Height (cm)
- MUAC (cm)
- Weight-for-Age Z-score (WAZ)
- Height-for-Age Z-score (HAZ)
- Weight-for-Height Z-score (WHZ)
- Underweight Status
- Stunting Status
- Wasting Status
- SAM MUAC Flag
- MAM MUAC Flag

## Data Source

The dataset was synthetically generated with AI assistance to simulate realistic pediatric anthropometric measurements based on WHO child growth concepts. It is intended solely for model development and testing and does not contain real patient information.

## Project Usage

This dataset is used for the initial development of the machine learning pipeline.

The final model will be trained and validated using real-world DHS child health records.