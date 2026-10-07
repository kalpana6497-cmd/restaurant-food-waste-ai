"""Rule-based decision-support recommendations driven by the real prediction and inputs."""
from __future__ import annotations

from typing import List

LEVEL_ACTIONS = {
    "High Waste": [
        "Consider reducing preparation quantities for the day.",
        "Prepare in smaller batches instead of one large batch.",
        "Monitor actual demand before preparing additional food.",
        "Review historical waste records for similar days.",
        "Adjust inventory and ordering plans to avoid over-stocking.",
    ],
    "Medium Waste": [
        "Monitor preparation quantities during service.",
        "Compare expected demand with previous patterns.",
        "Adjust batch sizes if demand is lower than planned.",
    ],
    "Low Waste": [
        "Maintain the current preparation strategy.",
        "Continue monitoring waste levels to keep them low.",
    ],
}

DISCLAIMER = ("These are decision-support suggestions based on a model prediction. "
              "They are not guaranteed outcomes - use them alongside your own kitchen experience.")


def level_recommendations(level: str) -> List[str]:
    return LEVEL_ACTIONS.get(level, [])


def context_recommendations(inputs: dict, thresholds: dict, ref: dict) -> List[str]:
    """Recommendations tied to the specific input conditions."""
    recs: List[str] = []

    if inputs["special_event"] == 1:
        recs.append("Special events can change demand patterns. Consider monitoring actual demand "
                    "closely before preparing additional batches.")

    # Past waste and the target are both in kg, so the high-waste threshold is a fair yardstick.
    if inputs["past_waste_kg"] >= thresholds["high_threshold"]:
        recs.append("Historical waste is relatively high. Review previous preparation quantities "
                    "before increasing today's production.")

    mps_ref = ref.get("meals_per_staff")
    if mps_ref and inputs["kitchen_staff"] > 0:
        mps = inputs["meals_served"] / inputs["kitchen_staff"]
        if mps > 1.5 * mps_ref:
            recs.append(f"Workload is high ({mps:.0f} meals per staff member vs a training median of "
                        f"{mps_ref:.0f}). Busy kitchens can over-prepare - consider clear batch plans.")

    if inputs["temperature_C"] >= 30:
        recs.append("It is a hot day. Perishable items spoil faster, so favour smaller, "
                    "more frequent preparation.")

    if inputs["day_of_week"] >= 5:
        recs.append("Weekend demand can differ from weekdays. Compare with previous weekend patterns.")

    if inputs["staff_experience"] == "beginner":
        recs.append("With a less experienced team, standard portioning guides and a "
                    "shared prep sheet can help keep preparation aligned with demand.")

    if inputs["waste_category"] in ("dairy", "vegetables"):
        recs.append(f"{inputs['waste_category'].capitalize()} are typically perishable - "
                    "track shelf life and use first-in-first-out stock rotation.")
    return recs


def build_recommendations(inputs: dict, level: str, thresholds: dict, ref: dict) -> dict:
    return {
        "level": level,
        "actions": level_recommendations(level),
        "context": context_recommendations(inputs, thresholds, ref),
        "disclaimer": DISCLAIMER,
    }
