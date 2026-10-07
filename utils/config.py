"""Central configuration: file locations and UI constants."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"

REGRESSION_MODEL_FILE = "food_waste_best_regression_model.pkl"
CLASSIFICATION_MODEL_FILE = "food_waste_best_classification_model.pkl"
THRESHOLDS_FILE = "food_waste_thresholds.json"
META_FILE = "food_waste_model_meta.json"
SUBMISSION_FILE = "final_submission.csv"
# Optional: add this file to display real evaluation metrics (see README).
METRICS_FILE = "food_waste_metrics.json"

APP_TITLE = "Restaurant Food Wastage Prediction & Management System"
APP_SUBTITLE = "AI-powered prediction and decision support for reducing restaurant food waste."

LEVEL_COLORS = {"Low Waste": "#2E9E6B", "Medium Waste": "#E0A100", "High Waste": "#D64545"}
LEVEL_SHORT = {"Low Waste": "LOW", "Medium Waste": "MEDIUM", "High Waste": "HIGH"}
LEVEL_ICONS = {"Low Waste": "🟢", "Medium Waste": "🟡", "High Waste": "🔴"}

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

FEATURE_LABELS = {
    "date": "Date",
    "meals_served": "Meals served",
    "kitchen_staff": "Kitchen staff",
    "temperature_C": "Temperature (°C)",
    "humidity_percent": "Humidity (%)",
    "day_of_week": "Day of week",
    "special_event": "Special event",
    "past_waste_kg": "Past waste (kg)",
    "staff_experience": "Staff experience",
    "waste_category": "Food category",
}
