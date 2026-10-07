"""Cached loading of the trained pipelines and supporting files.

Only the expected files inside ``models/`` are ever loaded. Every loader returns a
``Loaded`` object carrying either the object or a friendly error message, so the app
never crashes because a file is missing or incompatible.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Optional

import joblib
import pandas as pd
import streamlit as st

from utils import config

# The pickled pipelines reference ``food_waste_features.prepare`` (the feature-engineering
# step that is *inside* the pipeline). That module lives in the project root and must be
# importable at unpickling time.
if str(config.ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(config.ROOT_DIR))

import food_waste_features  # noqa: E402,F401


@dataclass
class Loaded:
    obj: Any = None
    error: Optional[str] = None
    detail: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.obj is not None


def _load_pipeline(filename: str, kind: str) -> Loaded:
    path = config.MODELS_DIR / filename
    if not path.exists():
        return Loaded(error=f"{kind} model file not found: models/{filename}")
    try:
        model = joblib.load(path)
        if not hasattr(model, "predict"):
            return Loaded(error=f"models/{filename} is not a valid scikit-learn pipeline.")
        return Loaded(obj=model)
    except Exception as exc:  # noqa: BLE001
        import traceback

        return Loaded(
            error=(f"Could not load the {kind.lower()} model. This is usually a scikit-learn "
                   "version mismatch - install the versions in requirements.txt."),
            detail=f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
        )


def _load_json(filename: str, label: str) -> Loaded:
    path = config.MODELS_DIR / filename
    if not path.exists():
        return Loaded(error=f"{label} file not found: models/{filename}")
    try:
        return Loaded(obj=json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:  # noqa: BLE001
        return Loaded(error=f"Could not read {label} file.", detail=str(exc))


@st.cache_resource(show_spinner=False)
def load_regression_model() -> Loaded:
    return _load_pipeline(config.REGRESSION_MODEL_FILE, "Regression")


@st.cache_resource(show_spinner=False)
def load_classification_model() -> Loaded:
    return _load_pipeline(config.CLASSIFICATION_MODEL_FILE, "Classification")


@st.cache_resource(show_spinner=False)
def load_thresholds() -> Loaded:
    res = _load_json(config.THRESHOLDS_FILE, "Threshold")
    if res.ok and not {"low_threshold", "high_threshold"} <= set(res.obj):
        return Loaded(error="Threshold file is missing low_threshold / high_threshold.")
    return res


@st.cache_resource(show_spinner=False)
def load_metadata() -> Loaded:
    return _load_json(config.META_FILE, "Model metadata")


@st.cache_resource(show_spinner=False)
def load_metrics() -> Loaded:
    """Optional evaluation metrics file (not required to run the app)."""
    return _load_json(config.METRICS_FILE, "Metrics")


@st.cache_data(show_spinner=False)
def load_submission() -> Loaded:
    path = config.MODELS_DIR / config.SUBMISSION_FILE
    if not path.exists():
        return Loaded(error=f"Prediction file not found: models/{config.SUBMISSION_FILE}")
    try:
        df = pd.read_csv(path)
        if "food_waste_kg" not in df.columns:
            return Loaded(error="final_submission.csv has no 'food_waste_kg' column.")
        return Loaded(obj=df)
    except Exception as exc:  # noqa: BLE001
        return Loaded(error="Could not read final_submission.csv.", detail=str(exc))


def training_reference(reg_pipeline) -> dict:
    """Training-set medians stored inside the fitted pipeline's numeric imputer.

    These are real statistics learned during training (``SimpleImputer(strategy='median')``)
    and are used for sensible form defaults and context-aware recommendations.
    """
    try:
        pre = reg_pipeline.named_steps["preprocess"]
        num_cols = list(pre.transformers_[0][2])
        stats = pre.named_transformers_["num"].named_steps["imputer"].statistics_
        ref = {c: float(v) for c, v in zip(num_cols, stats)}
        cat_imp = pre.named_transformers_["cat"].named_steps["imputer"]
        cat_cols = list(pre.transformers_[1][2])
        ref.update({c: str(v) for c, v in zip(cat_cols, cat_imp.statistics_)})
        return ref
    except Exception:  # noqa: BLE001
        return {}


def categorical_options(reg_pipeline) -> dict:
    """Allowed categories per categorical column, read from the fitted OneHotEncoder."""
    try:
        pre = reg_pipeline.named_steps["preprocess"]
        cat_cols = list(pre.transformers_[1][2])
        enc = pre.named_transformers_["cat"].named_steps["onehot"]
        return {c: [str(x) for x in cats] for c, cats in zip(cat_cols, enc.categories_)}
    except Exception:  # noqa: BLE001
        return {}


def feature_importances(reg_pipeline) -> Optional[pd.DataFrame]:
    """Impurity-based importances of the trained Random Forest (real model attribute)."""
    try:
        pre = reg_pipeline.named_steps["preprocess"]
        model = reg_pipeline.named_steps["model"]
        names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
        df = pd.DataFrame({"feature": names, "importance": model.feature_importances_})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
    except Exception:  # noqa: BLE001
        return None
