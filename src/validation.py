"""
Clinical range validation for sidebar inputs (P2-5).

Returns a list of (severity, message) warnings for values that are
biologically unusual or combinations that warrant a flag, without
blocking the user from proceeding.
"""

# (field, low, high, label) — soft "unusual but possible" bounds
RANGES = {
    "trestbps": (90, 180, "Resting blood pressure"),
    "chol":     (120, 400, "Cholesterol"),
    "thalch":   (60, 202, "Maximum heart rate"),
    "oldpeak":  (0.0, 4.0, "ST depression (oldpeak)"),
}


def validate_inputs(inputs: dict) -> list[tuple[str, str]]:
    warnings: list[tuple[str, str]] = []

    for field, (lo, hi, label) in RANGES.items():
        val = inputs.get(field)
        if val is None:
            continue
        if val < lo or val > hi:
            warnings.append((
                "warning",
                f"{label} of {val} is outside the typical clinical range "
                f"({lo}\u2013{hi}). Please double-check this value.",
            ))

    # Cross-field sanity checks
    age = inputs.get("age")
    thalch = inputs.get("thalch")
    if age is not None and thalch is not None:
        # Rough max predicted HR ~ 220 - age; flag if observed max HR exceeds it notably
        predicted_max = 220 - age
        if thalch > predicted_max + 15:
            warnings.append((
                "info",
                f"Max heart rate ({thalch} bpm) is notably above the "
                f"age-predicted maximum (~{predicted_max} bpm for age {age}).",
            ))

    chol = inputs.get("chol")
    if chol is not None and chol < 130 and age is not None and age > 40:
        warnings.append((
            "info",
            f"Cholesterol of {chol} mg/dL is unusually low for a patient "
            f"of age {age}. Verify lab units (mg/dL vs mmol/L).",
        ))

    return warnings
