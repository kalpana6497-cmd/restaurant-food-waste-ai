"""Input validation and prediction using the saved pipelines.

The saved pipelines already include feature engineering + preprocessing, so this module
only assembles a DataFrame with the exact raw input columns recorded in the metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

FALLBACK_COLUMNS = [
    "date", "meals_served", "kitchen_staff", "temperature_C", "humidity_percent",
    "day_of_week", "special_event", "past_waste_kg", "staff_experience", "waste_category",
]


@dataclass
class PredictionResult:
    waste_kg: float
    threshold_label: str
    classifier_label: str
    class_probabilities: Dict[str, float]
    inputs: dict
    input_frame: pd.DataFrame = field(repr=False, default=None)

    @property
    def models_agree(self) -> bool:
        return self.threshold_label == self.classifier_label


def input_columns(meta: dict | None) -> List[str]:
    if meta and meta.get("input_columns"):
        return list(meta["input_columns"])
    return FALLBACK_COLUMNS


def validate_inputs(inputs: dict, ref: dict | None = None) -> Tuple[List[str], List[str]]:
    """Return (errors, warnings). Errors block prediction; warnings do not."""
    errors, warnings = [], []
    if inputs["meals_served"] <= 0:
        errors.append("Meals served must be greater than 0.")
    if inputs["kitchen_staff"] <= 0:
        errors.append("Kitchen staff must be at least 1.")
    if inputs["past_waste_kg"] < 0:
        errors.append("Past waste cannot be negative.")
    if not 0 <= inputs["humidity_percent"] <= 100:
        errors.append("Humidity must be between 0 and 100 %.")
    if not 0 <= inputs["day_of_week"] <= 6:
        errors.append("Day of week must be between 0 and 6.")
    if inputs["special_event"] not in (0, 1):
        errors.append("Special event must be Yes or No.")
    if not -30 <= inputs["temperature_C"] <= 60:
        warnings.append("Temperature is outside the usual -30 to 60 °C range - please double-check.")
    if ref and ref.get("year") and abs(inputs["date"].year - ref["year"]) > 3:
        warnings.append(
            f"The model's training data is centred on {int(ref['year'])}; a date this far away "
            "may give less reliable predictions."
        )
    return errors, warnings


def build_input_frame(inputs: dict, columns: List[str]) -> pd.DataFrame:
    missing = [c for c in columns if c not in inputs]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")
    row = {c: inputs[c] for c in columns}
    row["date"] = pd.Timestamp(row["date"])
    return pd.DataFrame([row], columns=columns)


def label_from_thresholds(waste_kg: float, thresholds: dict) -> str:
    low, high = thresholds["low_threshold"], thresholds["high_threshold"]
    labels = thresholds.get("labels", ["Low Waste", "Medium Waste", "High Waste"])
    if waste_kg <= low:
        return labels[0]
    if waste_kg <= high:
        return labels[1]
    return labels[2]


def run_prediction(reg, clf, thresholds: dict, inputs: dict, columns: List[str]) -> PredictionResult:
    frame = build_input_frame(inputs, columns)
    waste_kg = float(reg.predict(frame)[0])
    clf_label = str(clf.predict(frame)[0])
    probs = {str(c): float(p) for c, p in zip(clf.classes_, clf.predict_proba(frame)[0])}
    return PredictionResult(
        waste_kg=waste_kg,
        threshold_label=label_from_thresholds(waste_kg, thresholds),
        classifier_label=clf_label,
        class_probabilities=probs,
        inputs=dict(inputs),
        input_frame=frame,
    )


def sweep_meals(reg, base_inputs: dict, columns: List[str], meals_values) -> pd.DataFrame:
    """Model-simulated what-if: vary only meals_served with the pipeline's own prediction."""
    rows = []
    for m in meals_values:
        r = dict(base_inputs)
        r["meals_served"] = float(m)
        rows.append(build_input_frame(r, columns).iloc[0])
    frame = pd.DataFrame(rows, columns=columns)
    preds = reg.predict(frame)
    return pd.DataFrame({"meals_served": np.asarray(meals_values, dtype=float), "predicted_waste_kg": preds})
