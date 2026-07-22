"""
WHO Growth Calculator

Calculates:
- Weight-for-Age Z Score (WAZ)
- Height-for-Age Z Score (HAZ)
- Weight-for-Height Z Score (WHZ)

Author: Mariya Mallick
"""

from pathlib import Path
import pandas as pd
import numpy as np

# -------------------------------------------------------
# WHO Growth Tables
# -------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"

boys_wfa = pd.read_excel(DATA_DIR / "boys_weight_for_age.xlsx")
girls_wfa = pd.read_excel(DATA_DIR / "girls_weight_for_age.xlsx")

boys_hfa = pd.read_excel(DATA_DIR / "boys_height_for_age.xlsx")
girls_hfa = pd.read_excel(DATA_DIR / "girls_height_for_age.xlsx")

boys_wfh = pd.read_excel(DATA_DIR / "boys_weight_for_height.xlsx")
girls_wfh = pd.read_excel(DATA_DIR / "girls_weight_for_height.xlsx")


# -------------------------------------------------------
# LMS Formula
# -------------------------------------------------------

def lms_zscore(value, L, M, S):
    """
    WHO LMS Z-score formula.
    """

    if L == 0:
        return np.log(value / M) / S

    return (((value / M) ** L) - 1) / (L * S)


# -------------------------------------------------------
# Lookup Helpers
# -------------------------------------------------------

def get_wfa_row(age_months, sex):

    table = boys_wfa if sex == 1 else girls_wfa

    row = table.iloc[(table["Month"] - age_months).abs().argsort()[:1]]

    return row.iloc[0]


def get_hfa_row(age_months, sex):

    table = boys_hfa if sex == 1 else girls_hfa

    row = table.iloc[(table["Month"] - age_months).abs().argsort()[:1]]

    return row.iloc[0]


def get_wfh_row(height_cm, sex):

    table = boys_wfh if sex == 1 else girls_wfh

    height_column = table.columns[0]

    row = table.iloc[(table[height_column] - height_cm).abs().argsort()[:1]]

    return row.iloc[0]


# -------------------------------------------------------
# Main Calculator
# -------------------------------------------------------

def calculate_z_scores(
    age_months,
    sex,
    weight_kg,
    height_cm
):
    """
    Parameters
    ----------
    sex
        Male = 1
        Female = 0
    """

    wfa = get_wfa_row(age_months, sex)
    hfa = get_hfa_row(age_months, sex)
    wfh = get_wfh_row(height_cm, sex)

    waz = lms_zscore(
        weight_kg,
        wfa["L"],
        wfa["M"],
        wfa["S"]
    )

    haz = lms_zscore(
        height_cm,
        hfa["L"],
        hfa["M"],
        hfa["S"]
    )

    whz = lms_zscore(
        weight_kg,
        wfh["L"],
        wfh["M"],
        wfh["S"]
    )

    return {
        "waz": round(float(waz), 2),
        "haz": round(float(haz), 2),
        "whz": round(float(whz), 2)
    }


# -------------------------------------------------------
# Testing
# -------------------------------------------------------

if __name__ == "__main__":

    result = calculate_z_scores(
        age_months=18,
        sex=1,
        weight_kg=10.4,
        height_cm=79
    )

    print(result)