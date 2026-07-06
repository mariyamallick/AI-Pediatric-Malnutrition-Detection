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