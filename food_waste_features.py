import numpy as np
import pandas as pd

EXP_MAP = {"beginner": 0, "intermediate": 1, "expert": 2}
CAT_COLS = ["waste_category", "staff_experience"]
NUM_COLS = [
    "meals_served", "kitchen_staff", "temperature_C", "humidity_percent", "day_of_week",
    "special_event", "past_waste_kg", "year", "month", "day", "day_of_year", "is_weekend",
    "month_sin", "month_cos", "experience_ord", "log_meals", "meals_per_staff",
    "past_waste_per_meal", "past_waste_per_staff", "temp_x_humidity", "hot_day", "meals_x_event"
]

def _clean_text(s):
    s = s.astype(object).map(lambda v: np.nan if pd.isna(v) else str(v).strip().lower())
    return s.replace({"nan": np.nan, "": np.nan, "none": np.nan, "null": np.nan, "na": np.nan})

def clean_raw(df):
    df = df.copy()
    for c in CAT_COLS:
        if c in df:
            df[c] = _clean_text(df[c])
        else:
            df[c] = np.nan

    df["date"] = pd.to_datetime(df["date"], errors="coerce") if "date" in df else pd.NaT

    numeric_cols = ["meals_served", "kitchen_staff", "temperature_C", "humidity_percent",
                    "day_of_week", "special_event", "past_waste_kg"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan

    df.loc[df["meals_served"] <= 0, "meals_served"] = np.nan
    df.loc[df["kitchen_staff"] <= 0, "kitchen_staff"] = np.nan
    df.loc[~df["humidity_percent"].between(0, 100), "humidity_percent"] = np.nan
    df.loc[df["past_waste_kg"] < 0, "past_waste_kg"] = np.nan
    df.loc[~df["day_of_week"].between(0, 6), "day_of_week"] = np.nan
    return df

def _div(a, b):
    return a / b.where(b > 0, np.nan)

def add_features(df):
    d = pd.DataFrame(index=df.index)
    for c in ["meals_served", "kitchen_staff", "temperature_C", "humidity_percent",
              "day_of_week", "special_event", "past_waste_kg"]:
        d[c] = df[c]

    dt = df["date"]
    d["year"] = dt.dt.year
    d["month"] = dt.dt.month
    d["day"] = dt.dt.day
    d["day_of_year"] = dt.dt.dayofyear
    d["is_weekend"] = (df["day_of_week"] >= 5).astype(float).where(df["day_of_week"].notna(), np.nan)
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)
    d["experience_ord"] = df["staff_experience"].map(EXP_MAP)
    d["log_meals"] = np.log1p(df["meals_served"])
    d["meals_per_staff"] = _div(df["meals_served"], df["kitchen_staff"])
    d["past_waste_per_meal"] = _div(df["past_waste_kg"], df["meals_served"])
    d["past_waste_per_staff"] = _div(df["past_waste_kg"], df["kitchen_staff"])
    d["temp_x_humidity"] = df["temperature_C"] * df["humidity_percent"] / 100.0
    d["hot_day"] = (df["temperature_C"] >= 30).astype(float).where(df["temperature_C"].notna(), np.nan)
    d["meals_x_event"] = df["meals_served"] * df["special_event"]

    for c in CAT_COLS:
        d[c] = df[c]
    return d[NUM_COLS + CAT_COLS]

def prepare(df):
    return add_features(clean_raw(df))
