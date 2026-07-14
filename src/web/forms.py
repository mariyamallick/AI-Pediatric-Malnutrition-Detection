"""Request parsing and validation for nutrition screening measurements."""

from dataclasses import dataclass

from werkzeug.datastructures import MultiDict


@dataclass(frozen=True)
class FieldSpec:
    minimum: float
    maximum: float
    label: str
    integer: bool = False


FIELD_SPECS = {
    "age_months": FieldSpec(0, 60, "Age in months", integer=True),
    "weight_kg": FieldSpec(0.5, 40, "Weight"),
    "height_cm": FieldSpec(30, 130, "Height / length"),
    "muac_cm": FieldSpec(5, 30, "MUAC"),
    "waz": FieldSpec(-6, 6, "Weight-for-age z-score"),
    "haz": FieldSpec(-6, 6, "Height-for-age z-score"),
    "whz": FieldSpec(-6, 6, "Weight-for-height z-score"),
}


def validate_measurements(form: MultiDict):
    """Return normalized features and per-field validation errors."""
    features, errors = {}, {}

    for name, spec in FIELD_SPECS.items():
        raw_value = form.get(name, "").strip()
        if not raw_value:
            errors[name] = f"{spec.label} is required."
            continue
        try:
            value = int(raw_value) if spec.integer else float(raw_value)
        except ValueError:
            errors[name] = f"{spec.label} must be a number."
            continue
        if not spec.minimum <= value <= spec.maximum:
            errors[name] = f"{spec.label} must be between {spec.minimum:g} and {spec.maximum:g}."
            continue
        features[name] = value

    sex = form.get("sex", "")
    if sex not in {"0", "1"}:
        errors["sex"] = "Select the child's sex."
    else:
        features["sex"] = int(sex)

    if not errors:
        features["bmi"] = round(
            features["weight_kg"] / (features["height_cm"] / 100) ** 2, 2
        )

    return features, errors
