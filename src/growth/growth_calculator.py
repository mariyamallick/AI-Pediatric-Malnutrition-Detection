"""
WHO Growth Calculator

Calculates:
- WAZ (Weight-for-Age)
- HAZ / LAZ (Height/Length-for-Age)
- WHZ (Weight-for-Height)
- BMI

Author: Mariya Mallick
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"


def load_table(filename):
    df = pd.read_excel(DATA_DIR / filename)

    # Normalize column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
    )

    return df


# -------------------------------
# Load WHO Tables
# -------------------------------

boys_wfa = load_table("boys_wfa.xlsx")
girls_wfa = load_table("girls_wfa.xlsx")
boys_lfa = load_table("boys_lfa.xlsx")
girls_lfa = load_table("girls_lfa.xlsx")
boys_hfa = load_table("boys_hfa.xlsx")
girls_hfa = load_table("girls_hfa.xlsx")
boys_wfh = load_table("boys_wfh.xlsx")
girls_wfh = load_table("girls_wfh.xlsx")

# -------------------------------
# WHO LMS Formula
# -------------------------------

def calculate_lms(value, L, M, S):

    if L == 0:
        return np.log(value / M) / S

    return (((value / M) ** L) - 1) / (L * S)


# -------------------------------
# Find nearest row
# -------------------------------

def nearest_row(table, column, value):

    idx = (table[column] - value).abs().idxmin()

    return table.loc[idx]


# -------------------------------
# Main Function
# -------------------------------

def calculate_z_scores(
    age_months,
    sex,
    weight_kg,
    height_cm
):

    sex = int(sex)

    # -------------------------
    # Weight-for-Age
    # -------------------------

    wfa = boys_wfa if sex == 1 else girls_wfa

    wfa_row = nearest_row(
        wfa,
        "Month",
        age_months
    )

    # -------------------------
    # Length / Height for Age
    # -------------------------

    if age_months < 24:

        hfa_table = boys_lfa if sex == 1 else girls_lfa

    # Use child's measured length/height
        hfa_row = nearest_row(
            hfa_table,
            "Length",
            height_cm
        )

    else:

        hfa_table = boys_hfa if sex == 1 else girls_hfa

        hfa_row = nearest_row(
            hfa_table,
            "Month",
            age_months
        )
    # -------------------------
    # Weight-for-Height
    # -------------------------

    wfh_table = boys_wfh if sex == 1 else girls_wfh

    first_column = "Height"

    wfh_row = nearest_row(
        wfh_table,
        first_column,
        height_cm
    )

    # -------------------------
    # Calculate Z Scores
    # -------------------------

    waz = calculate_lms(
        weight_kg,
        float(wfa_row["L"]),
        float(wfa_row["M"]),
        float(wfa_row["S"])
    )

    haz = calculate_lms(
        height_cm,
        float(hfa_row["L"]),
        float(hfa_row["M"]),
        float(hfa_row["S"])
    )

    whz = calculate_lms(
        weight_kg,
        float(wfh_row["L"]),
        float(wfh_row["M"]),
        float(wfh_row["S"])
    )

    bmi = weight_kg / ((height_cm / 100) ** 2)

    return {

        "waz": round(waz, 2),

        "haz": round(haz, 2),

        "whz": round(whz, 2),

        "bmi": round(bmi, 2)

    }


# -------------------------------
# Testing
# -------------------------------

if __name__ == "__main__":

    result = calculate_z_scores(

        age_months=18,

        sex=1,

        weight_kg=10.4,

        height_cm=79

    )

    print(result)