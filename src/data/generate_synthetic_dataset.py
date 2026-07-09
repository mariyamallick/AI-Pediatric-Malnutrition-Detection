"""
generate_synthetic_dataset.py

Generates a synthetic pediatric anthropometric dataset for children aged
0-59 months, for use in developing and testing the malnutrition-detection
ML pipeline.

IMPORTANT LIMITATIONS (read before using results in any report/paper):
-----------------------------------------------------------------------
1. This script uses SIMPLIFIED mean/SD approximations of child growth by
   age-in-months, NOT the official WHO LMS (Box-Cox power-exponential)
   reference tables. Real WHO z-scores are computed with the LMS method,
   which handles skewness in the growth distribution more accurately,
   especially at the tails (severe malnutrition).
2. This data is SYNTHETIC. It is useful for:
     - building and unit-testing the preprocessing/training/inference
       pipeline end-to-end
     - prototyping the recommendation engine
   It is NOT sufficient, on its own, to support clinical or research
   claims. Before finalizing results for a paper, validate the pipeline
   against a real dataset (e.g. DHS survey data) or replace the
   reference tables below with the official WHO LMS tables.
3. Correlations between weight, height, and MUAC are approximated to be
   directionally realistic (e.g. wasted children have low weight-for-height
   AND low MUAC), but are not derived from real epidemiological data.

Usage:
    python generate_synthetic_dataset.py --n 5000 --seed 42 \
        --output ../../data/raw/synthetic_children.csv

Author: <you> + AI engineering partner
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Approximate WHO reference medians (M) and coefficients of variation (CV)
#    for weight-for-age and height-for-age, by sex, sampled at a few anchor
#    ages (in months) and linearly interpolated in between.
#
#    Values are rounded approximations of the WHO Child Growth Standards
#    (2006) median tables, adequate for generating a realistic-shaped
#    synthetic population. Replace with official LMS tables for production
#    accuracy (see docs/dataset.md for source links).
# ---------------------------------------------------------------------------

AGE_ANCHORS_MONTHS = [0, 6, 12, 24, 36, 48, 59]

# median weight (kg) at each anchor age
WEIGHT_MEDIAN_BOYS = [3.3, 7.9, 9.6, 12.2, 14.3, 16.3, 18.0]
WEIGHT_MEDIAN_GIRLS = [3.2, 7.3, 8.9, 11.5, 13.9, 16.1, 17.9]

# median length/height (cm) at each anchor age
HEIGHT_MEDIAN_BOYS = [49.9, 67.6, 75.7, 87.1, 96.1, 103.3, 109.2]
HEIGHT_MEDIAN_GIRLS = [49.1, 65.7, 74.0, 85.7, 95.1, 102.7, 108.4]

# Approximate coefficient of variation (SD as fraction of median).
# Growth measurements roughly maintain a fairly stable CV across ages.
WEIGHT_CV = 0.14
HEIGHT_CV = 0.045

# MUAC (cm) reference: fairly stable after 6 months in well-nourished
# children; we model it as correlated with WHZ rather than age.
MUAC_MEDIAN_CM = 14.5
MUAC_SD_CM = 1.1


@dataclass
class GeneratorConfig:
    n_samples: int = 5000
    seed: int = 42
    malnutrition_prevalence: float = 0.28  # fraction NOT in "normal" band
    severe_fraction_of_malnourished: float = 0.25
    output_path: str = "synthetic_children.csv"


def _interp_median(age_months: np.ndarray, anchors: list, medians: list) -> np.ndarray:
    """Linearly interpolate a median growth value at arbitrary ages."""
    return np.interp(age_months, anchors, medians)


def _assign_health_status(rng: np.random.Generator, n: int, cfg: GeneratorConfig) -> np.ndarray:
    """
    Assign each synthetic child a latent health status that will drive
    a downward shift in weight/height/MUAC. This is what lets the
    generated dataset contain a realistic mix of normal, moderately
    malnourished, and severely malnourished children instead of only
    "normal" cases.

    Returns an array of z-score shifts (0 = normal, negative = malnourished).
    """
    status = rng.random(n)
    shift = np.zeros(n)

    malnourished_mask = status < cfg.malnutrition_prevalence
    severe_mask = malnourished_mask & (
        rng.random(n) < cfg.severe_fraction_of_malnourished
    )
    moderate_mask = malnourished_mask & ~severe_mask

    # Shifts expressed directly in z-score units, with noise so the
    # boundary between categories isn't a hard cliff.
    shift[moderate_mask] = rng.normal(loc=-2.4, scale=0.3, size=moderate_mask.sum())
    shift[severe_mask] = rng.normal(loc=-3.6, scale=0.4, size=severe_mask.sum())

    return shift


def generate_dataset(cfg: GeneratorConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_samples

    age_months = rng.integers(0, 60, size=n)
    sex = rng.choice(["M", "F"], size=n)

    weight_median = np.where(
        sex == "M",
        _interp_median(age_months, AGE_ANCHORS_MONTHS, WEIGHT_MEDIAN_BOYS),
        _interp_median(age_months, AGE_ANCHORS_MONTHS, WEIGHT_MEDIAN_GIRLS),
    )
    height_median = np.where(
        sex == "M",
        _interp_median(age_months, AGE_ANCHORS_MONTHS, HEIGHT_MEDIAN_BOYS),
        _interp_median(age_months, AGE_ANCHORS_MONTHS, HEIGHT_MEDIAN_GIRLS),
    )

    weight_sd = weight_median * WEIGHT_CV
    height_sd = height_median * HEIGHT_CV

    # Latent health status shifts a child's weight and height jointly
    # (wasting/stunting are correlated with overall nutritional status),
    # plus independent measurement noise.
    health_shift = _assign_health_status(rng, n, cfg)

    weight_kg = weight_median + health_shift * weight_sd + rng.normal(0, weight_sd * 0.4, n)
    height_cm = height_median + health_shift * 0.7 * height_sd + rng.normal(0, height_sd * 0.5, n)

    weight_kg = np.clip(weight_kg, 1.5, None)
    height_cm = np.clip(height_cm, 35, None)

    # MUAC correlated with the same health shift (captures acute wasting).
    muac_cm = MUAC_MEDIAN_CM + health_shift * MUAC_SD_CM * 0.8 + rng.normal(0, 0.5, n)
    muac_cm = np.clip(muac_cm, 8.0, 20.0)

    # --- Derived z-scores (approximate, SD-based rather than true LMS) ---
    waz = (weight_kg - weight_median) / weight_sd
    haz = (height_cm - height_median) / height_sd

    # Weight-for-height needs an expected weight at the child's *actual*
    # height rather than their age, so we approximate by re-deriving an
    # expected-weight-at-height ratio from the age-based median trend.
    expected_weight_for_height = weight_median * (height_cm / height_median)
    whz = (weight_kg - expected_weight_for_height) / weight_sd

    df = pd.DataFrame(
        {
            "age_months": age_months,
            "sex": sex,
            "weight_kg": weight_kg.round(2),
            "height_cm": height_cm.round(1),
            "muac_cm": muac_cm.round(1),
            "waz": waz.round(2),
            "haz": haz.round(2),
            "whz": whz.round(2),
        }
    )

    df["underweight_status"] = pd.cut(
        df["waz"], bins=[-np.inf, -3, -2, np.inf],
        labels=["severe", "moderate", "normal"],
    )
    df["stunting_status"] = pd.cut(
        df["haz"], bins=[-np.inf, -3, -2, np.inf],
        labels=["severe", "moderate", "normal"],
    )
    df["wasting_status"] = pd.cut(
        df["whz"], bins=[-np.inf, -3, -2, np.inf],
        labels=["severe", "moderate", "normal"],
    )
    df["sam_muac_flag"] = df["muac_cm"] < 11.5  # severe acute malnutrition cutoff
    df["mam_muac_flag"] = (df["muac_cm"] >= 11.5) & (df["muac_cm"] < 12.5)

    return df


def parse_args() -> GeneratorConfig:
    parser = argparse.ArgumentParser(description="Generate synthetic pediatric growth dataset.")
    parser.add_argument("--n", type=int, default=5000, help="Number of synthetic children.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--prevalence", type=float, default=0.28,
        help="Fraction of children who are NOT nutritionally normal.",
    )
    parser.add_argument(
        "--output", type=str, default="synthetic_children.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()
    return GeneratorConfig(
        n_samples=args.n,
        seed=args.seed,
        malnutrition_prevalence=args.prevalence,
        output_path=args.output,
    )


def main() -> None:
    cfg = parse_args()
    df = generate_dataset(cfg)
    df.to_csv(cfg.output_path, index=False)
    print(f"Generated {len(df)} synthetic records -> {cfg.output_path}")
    print("\nClass distribution (wasting_status):")
    print(df["wasting_status"].value_counts(normalize=True).round(3))
    print(
        "\nReminder: this is SYNTHETIC data using approximate reference "
        "tables. See module docstring for limitations before using in "
        "any report or paper."
    )


if __name__ == "__main__":
    main()
