# 🍽️ Restaurant Food Wastage Prediction & Management System

AI-powered prediction and decision support for reducing restaurant food waste.
A Streamlit frontend built around the **actual trained models** (no dummy data, no retraining).

## Problem statement
Restaurants can experience food waste because of changing customer demand, over-preparation,
inaccurate demand estimation and inefficient planning. This app predicts expected waste from daily
operating conditions and turns the prediction into planning suggestions.

## ML tasks
| Task | Model | Output |
|---|---|---|
| Regression | Random Forest Regressor | `food_waste_kg` |
| Classification | Logistic Regression | Low / Medium / High Waste |

Thresholds (`food_waste_thresholds.json`): Low ≤ 35.64 kg, High > 46.65 kg.

## Features
Home dashboard · Prediction form + result card · Waste analysis · Model performance & insights ·
Recommendations (level-based and context-aware) · About page · Friendly error handling.

## How prediction works
1. The user enters the 10 raw columns: `date, meals_served, kitchen_staff, temperature_C,
   humidity_percent, day_of_week, special_event, past_waste_kg, staff_experience, waste_category`.
2. They are passed as a one-row DataFrame to the saved pipelines. **Feature engineering
   (`food_waste_features.prepare`) and preprocessing are inside the pipelines**, so the app does not
   duplicate them.
3. The regression pipeline returns kg; the classification pipeline returns the level and probabilities.
   The saved thresholds are also applied to the kg value and shown alongside.
4. Recommendations are generated from the level and the input conditions.

## Project architecture
```
restaurant-food-waste/
├── app.py
├── food_waste_features.py      # required to unpickle the pipelines (do not rename/move)
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/config.toml
├── models/
│   ├── food_waste_best_regression_model.pkl
│   ├── food_waste_best_classification_model.pkl
│   ├── food_waste_thresholds.json
│   ├── food_waste_model_meta.json
│   └── final_submission.csv    # model predictions on a test set (NOT actual waste)
└── utils/
    ├── config.py  model_loader.py  predictor.py  recommendations.py  charts.py
```

## Installation & run
Python 3.10 – 3.13 recommended.
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
`scikit-learn==1.6.1` is pinned because the `.pkl` files were saved with that version.

## Performance metrics
The files supplied with the models contain **no evaluation metrics** and the training notebook was not
included, so the app does not display RMSE / MAE / R² / accuracy / F1 by default (nothing is invented).
To show your real results, add `models/food_waste_metrics.json`:
```json
{
  "regression": {"best_model": "Random Forest Regressor", "rmse": 0.0, "mae": 0.0, "r2": 0.0},
  "classification": {"best_model": "Logistic Regression", "accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0},
  "regression_comparison": [{"Model": "...", "RMSE": 0.0, "MAE": 0.0, "R²": 0.0}],
  "classification_comparison": [{"Model": "...", "Accuracy": 0.0, "F1": 0.0}]
}
```
(Replace the zeros with the values from your notebook.) Feature importances shown in the app are read
directly from the trained Random Forest.

## Data provenance
`final_submission.csv` holds predictions (ID + predicted kg), not historical actuals. The Waste Analysis
page labels it as such and never mixes it with user predictions.

## Future improvements
Add evaluation metrics file, prediction intervals, batch (CSV) prediction, SHAP explanations,
model monitoring with real logged outcomes.
