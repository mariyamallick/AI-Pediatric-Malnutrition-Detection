"""
load_dhs_data.py

Loads and standardizes real DHS "Children's Recode" (KR) datasets from
multiple countries into a single, clean dataframe matching the same
schema produced by generate_synthetic_dataset.py, so the rest of the
pipeline (preprocessing/training/evaluation) works unchanged on either
source.

Supported countries in this version (Standard DHS, South/Southeast Asia):
    - Bangladesh  (BDKR81FL.DTA, BDHS 2022)
    - Cambodia    (KHKR82FL.DTA, CDHS 2021-22)
    - Nepal       (NPKR82FL.DTA, NDHS 2022)
    - Pakistan    (PKKR71FL.DTA, PDHS 2017-18)
    - India       (IAKR7EFL.DTA, NFHS-5 2019-21) -- add path once available

Key DHS variable reference (standard DHS-7/8 recode, confirmed against
uploaded files):
    hw1   - child's age in months
    hw2   - weight in kg x10 (one decimal implied)
    hw3   - height in cm x10 (one decimal implied)
    hw70  - height-for-age z-score x100 (WHO)
    hw71  - weight-for-age z-score x100 (WHO)
    hw72  - weight-for-height z-score x100 (WHO)
    b4    - sex of child (1=male, 2=female)
    v190  - wealth index quintile (1=poorest ... 5=richest)
    v106  - mother's highest education level
    v404  - currently breastfeeding (0/1)
    v414* - IYCF: gave child [food] in last 24 hours (0=no, 1=yes, 8=dk, 9=missing)

DHS special/missing codes handled here:
    - hw2/hw3 raw values >= 9990 -> not measured / flagged -> treated as NaN
    - hw70/hw71/hw72 raw values with abs() >= 9990 (e.g. 9998 "flagged as
      biologically implausible") -> treated as NaN
    - v414* codes 8 (don't know) and 9 (missing) are conservatively mapped
      to 0 (not consumed). This is a simplifying assumption -- document it
      in your methodology section, since it can slightly underestimate
      dietary diversity if "don't know" responses were actually "yes".

MUAC note: standard KR files for these surveys do NOT include mid-upper
arm circumference (MUAC) as a recode variable (confirmed absent in the
uploaded files). SAM/MAM flags based on MUAC are therefore not available
from this source; z-score-based severity (WHZ/HAZ/WAZ) is used instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# IYCF food-group mapping, following the WHO/UNICEF Minimum Dietary Diversity
# for Children (MDD-C) 7-group definition. Each entry lists the raw DHS
# variables that count as "consumed" for that group if ANY of them = 1.
# ---------------------------------------------------------------------------
FOOD_GROUP_SOURCE_VARS = {
    "grains_roots_tubers": ["v414e", "v414f"],
    "legumes_nuts": ["v414o", "v414c"],
    "dairy": ["v411", "v411a", "v414p", "v414v", "v413a"],
    "flesh_foods": ["v414h", "v414m", "v414n", "v414b"],
    "eggs": ["v414g"],
    "vitamin_a_fruits_veg": ["v414i", "v414k"],
    "other_fruits_veg": ["v414a", "v414j", "v414l"],
}
MIN_DIETARY_DIVERSITY_THRESHOLD = 4

# Core variables required for a child record to be usable at all.
REQUIRED_VARS = ["hw1", "b4", "hw70", "hw71", "hw72"]


@dataclass
class CountryFile:
    country: str
    path: str


@dataclass
class DHSLoaderConfig:
    files: list = field(default_factory=list)  # list[CountryFile]
    min_age_months: int = 0
    max_age_months: int = 59
    output_path: str = "data/processed/dhs_children_combined.csv"


def _clean_special_codes(series: pd.Series, threshold: float) -> pd.Series:
    """Replace DHS 'flagged/not measured' sentinel codes with NaN."""
    return series.where(series.abs() < threshold, np.nan)


def _binary_from_dhs_code(series: pd.Series) -> pd.Series:
    """
    Convert a raw DHS 0/1/8/9 IYCF variable into a clean 0/1 flag.
    8 (don't know) and 9 (missing) are conservatively treated as 0 (not
    consumed) -- see module docstring for the rationale/limitation.
    """
    return series.isin([1]).astype(int)


def _compute_food_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive the 7 MDD-C food-group flags and dietary diversity score.

    Important: DHS only asks the IYCF 24-hour recall questions (v414*) for
    a subset of children (in practice, overwhelmingly those under ~24
    months -- this matches the official WHO/UNICEF IYCF indicator scope of
    0-23 months). For children who were never asked, all v414* fields are
    NaN. We must NOT silently treat that as "consumed nothing" (score=0),
    since that would fabricate a false signal for older children. Instead
    we add an explicit `iycf_applicable` flag so downstream code can filter
    or handle these two cases (truly low diversity vs. not asked) separately.
    """
    out = pd.DataFrame(index=df.index)

    # A child was "asked" if at least one source variable has a non-null
    # response; if all are null, the module was skipped for this child.
    all_source_vars = sorted({v for vs in FOOD_GROUP_SOURCE_VARS.values() for v in vs})
    available_source_vars = [v for v in all_source_vars if v in df.columns]
    out["iycf_applicable"] = df[available_source_vars].notna().any(axis=1)

    for group, source_vars in FOOD_GROUP_SOURCE_VARS.items():
        available = [v for v in source_vars if v in df.columns]
        if not available:
            out[group] = np.nan
            continue
        flags = pd.DataFrame({v: _binary_from_dhs_code(df[v]) for v in available})
        group_flag = (flags.sum(axis=1) > 0).astype(float)
        # Preserve NaN for children who were never asked, instead of
        # collapsing them to 0.
        out[group] = group_flag.where(out["iycf_applicable"], np.nan)

    known_groups = [g for g in FOOD_GROUP_SOURCE_VARS if out[g].notna().any()]
    out["dietary_diversity_score"] = out[known_groups].sum(axis=1, min_count=1)
    mdd_met = (out["dietary_diversity_score"] >= MIN_DIETARY_DIVERSITY_THRESHOLD).astype("boolean")
    mdd_met[~out["iycf_applicable"]] = pd.NA
    out["min_dietary_diversity_met"] = mdd_met

    return out


def _classify_zscore(z: pd.Series) -> pd.Series:
    """Apply standard WHO severity cutoffs to a z-score series."""
    return pd.cut(
        z, bins=[-np.inf, -3, -2, np.inf], labels=["severe", "moderate", "normal"]
    )


def load_one_country(country: str, path: str) -> pd.DataFrame:
    """Load and standardize a single country's DHS Children's Recode file."""
    raw = pd.read_stata(path, convert_categoricals=False)

    missing_required = [v for v in REQUIRED_VARS if v not in raw.columns]
    if missing_required:
        raise ValueError(
            f"{country}: missing required variables {missing_required}. "
            "This file may not be a standard KR recode."
        )

    df = pd.DataFrame(index=raw.index)
    df["country"] = country
    df["age_months"] = raw["hw1"]
    df["sex"] = raw["b4"].map({1: "M", 2: "F"})

    if "hw2" in raw.columns:
        df["weight_kg"] = _clean_special_codes(raw["hw2"], threshold=9990) / 10.0
    if "hw3" in raw.columns:
        df["height_cm"] = _clean_special_codes(raw["hw3"], threshold=9990) / 10.0

    df["haz"] = _clean_special_codes(raw["hw70"], threshold=9990) / 100.0
    df["waz"] = _clean_special_codes(raw["hw71"], threshold=9990) / 100.0
    df["whz"] = _clean_special_codes(raw["hw72"], threshold=9990) / 100.0

    df["stunting_status"] = _classify_zscore(df["haz"])
    df["underweight_status"] = _classify_zscore(df["waz"])
    df["wasting_status"] = _classify_zscore(df["whz"])

    if "v190" in raw.columns:
        df["wealth_index"] = raw["v190"]
    if "v106" in raw.columns:
        df["mother_education"] = raw["v106"]
    if "v404" in raw.columns:
        df["currently_breastfeeding"] = raw["v404"].astype("boolean")

    food_df = _compute_food_groups(raw)
    df = pd.concat([df, food_df], axis=1)

    return df


def load_all_countries(cfg: DHSLoaderConfig) -> pd.DataFrame:
    frames = []
    for cf in cfg.files:
        print(f"Loading {cf.country} from {cf.path} ...")
        df = load_one_country(cf.country, cf.path)
        before = len(df)
        df = df[
            (df["age_months"] >= cfg.min_age_months)
            & (df["age_months"] <= cfg.max_age_months)
            & df["haz"].notna()
            & df["waz"].notna()
            & df["whz"].notna()
        ].copy()
        print(f"  {before} records -> {len(df)} after age/quality filtering")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined


def main() -> None:
    cfg = DHSLoaderConfig(
        files=[
            CountryFile("India", "/home/claude/IAKR7EFL.DTA"),
            CountryFile("Bangladesh", "/mnt/user-data/uploads/BDKR81FL.DTA"),
            CountryFile("Cambodia", "/mnt/user-data/uploads/KHKR82FL.DTA"),
            CountryFile("Nepal", "/mnt/user-data/uploads/NPKR82FL.DTA"),
            CountryFile("Pakistan", "/mnt/user-data/uploads/PKKR71FL.DTA"),
        ],
        output_path="dhs_children_combined.csv",
    )
    combined = load_all_countries(cfg)
    combined.to_csv(cfg.output_path, index=False)

    print(f"\nTotal combined records: {len(combined)}")
    print("\nRecords per country:")
    print(combined["country"].value_counts())
    print("\nWasting status distribution (overall):")
    print(combined["wasting_status"].value_counts(normalize=True).round(3))
    print("\nWasting status distribution by country:")
    print(
        combined.groupby("country", observed=True)["wasting_status"]
        .value_counts(normalize=True)
        .round(3)
    )
    print("\nIYCF module applicability (fraction of children who WERE asked "
          "the 24h dietary recall questions), by country:")
    print(combined.groupby("country", observed=True)["iycf_applicable"].mean().round(3))
    print(f"\nSaved combined dataset -> {cfg.output_path}")


if __name__ == "__main__":
    main()
