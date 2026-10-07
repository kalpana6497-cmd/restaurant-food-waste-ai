"""Restaurant Food Wastage Prediction & Management System - Streamlit frontend.

All predictions come from the saved scikit-learn pipelines in ``models/``.
No training happens here.
"""
from __future__ import annotations

import traceback
from datetime import date

import pandas as pd
import streamlit as st

from utils import config
from utils.charts import (category_figure, distribution_figure, importance_figure,
                          probability_figure, sweep_figure, waste_band_figure)
from utils.model_loader import (categorical_options, feature_importances, load_classification_model,
                                load_metadata, load_metrics, load_regression_model,
                                load_submission, load_thresholds, training_reference)
from utils.predictor import (input_columns, label_from_thresholds, run_prediction, sweep_meals,
                             validate_inputs)
from utils.recommendations import build_recommendations

st.set_page_config(page_title="Food Waste Prediction", page_icon="🍽️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
.hero h1 {font-size: 2.1rem; margin-bottom: .2rem; color:#1F4E3D;}
.hero p {font-size: 1.1rem; color:#52606D; margin-top:0;}
.card {border:1px solid #E1E8E4; border-radius:14px; padding:1.1rem 1.2rem; background:#F8FBF9; height:100%;}
.card h4 {margin:0 0 .3rem 0; color:#1F4E3D; font-size:.95rem; letter-spacing:.06em;}
.card p {margin:0; color:#52606D; font-size:.95rem;}
.step {text-align:center; border-radius:12px; padding:.8rem .4rem; background:#EAF4EE; color:#1F4E3D; font-weight:600;}
.arrow {text-align:center; font-size:1.5rem; color:#7B8794; line-height:2.6rem;}
.result {border-radius:18px; padding:1.6rem; text-align:center; border:2px solid; background:#fff;}
.result .lab {letter-spacing:.12em; font-size:.85rem; color:#52606D; font-weight:600;}
.result .val {font-size:3.2rem; font-weight:800; line-height:1.1; color:#1F2933;}
.badge {display:inline-block; padding:.25rem .9rem; border-radius:999px; color:#fff; font-weight:700; letter-spacing:.08em;}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------- loading
reg_res = load_regression_model()
clf_res = load_classification_model()
thr_res = load_thresholds()
meta_res = load_metadata()
metrics_res = load_metrics()
sub_res = load_submission()

meta = meta_res.obj if meta_res.ok else None
COLUMNS = input_columns(meta)
REF = training_reference(reg_res.obj) if reg_res.ok else {}
OPTIONS = categorical_options(reg_res.obj) if reg_res.ok else {}
core_ready = reg_res.ok and clf_res.ok and thr_res.ok


def show_load_problems(include_optional: bool = False) -> None:
    items = [("Regression model", reg_res), ("Classification model", clf_res), ("Thresholds", thr_res)]
    if include_optional:
        items.append(("Metadata", meta_res))
    for name, res in items:
        if not res.ok:
            st.error(f"**{name}:** {res.error}")
            if res.detail:
                with st.expander(f"Technical details - {name}"):
                    st.code(res.detail)


def level_color(level: str) -> str:
    return config.LEVEL_COLORS.get(level, "#52606D")


def card(title: str, body: str) -> None:
    st.markdown(f'<div class="card"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------- sidebar
PAGES = ["🏠 Home", "🔮 Food Waste Prediction", "📊 Waste Analysis", "🧪 Model Performance",
         "💡 Recommendations", "ℹ️ About Project"]
st.sidebar.markdown("## 🍽️ Food Waste AI")
page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.markdown("**System status**")
st.sidebar.markdown(f"{'✅' if reg_res.ok else '❌'} Regression model")
st.sidebar.markdown(f"{'✅' if clf_res.ok else '❌'} Classification model")
st.sidebar.markdown(f"{'✅' if thr_res.ok else '❌'} Thresholds")
if meta:
    st.sidebar.caption(f"Regression: {meta.get('regression_model', '-')}  \n"
                       f"Classification: {meta.get('classification_model', '-')}")


# ----------------------------------------------------------------------------- pages
def page_home() -> None:
    st.markdown(f'<div class="hero"><h1>{config.APP_TITLE}</h1><p>{config.APP_SUBTITLE}</p></div>',
                unsafe_allow_html=True)
    st.markdown("#### The problem")
    st.write("Restaurants can experience food waste because of changing customer demand, "
             "over-preparation, inaccurate demand estimation and inefficient planning. "
             "This system predicts expected waste from daily operating conditions so kitchens "
             "can plan preparation more deliberately.")

    st.markdown("#### How it works")
    steps = ["Restaurant Inputs", "ML Prediction", "Expected Food Waste", "Waste Classification",
             "Recommendations"]
    cols = st.columns([3, 1, 3, 1, 3, 1, 3, 1, 3])
    for i, s in enumerate(steps):
        cols[i * 2].markdown(f'<div class="step">{s}</div>', unsafe_allow_html=True)
        if i < len(steps) - 1:
            cols[i * 2 + 1].markdown('<div class="arrow">→</div>', unsafe_allow_html=True)

    st.markdown("#### Capabilities")
    c = st.columns(4)
    with c[0]:
        card("REGRESSION", "Predict food waste in kilograms.")
    with c[1]:
        card("CLASSIFICATION", "Identify Low / Medium / High waste.")
    with c[2]:
        card("PATTERN ANALYSIS", "Understand waste-related patterns.")
    with c[3]:
        card("RECOMMENDATIONS", "Provide decision-support suggestions.")

    st.write("")
    if core_ready:
        st.success("Trained models are loaded and ready. Open **Food Waste Prediction** in the sidebar to start.")
    else:
        show_load_problems()


def collect_form_defaults() -> dict:
    d = {"meals": 300, "staff": 10, "temp": 22.0, "hum": 60.0, "past": 25.0, "date": date.today()}
    if REF:
        d["meals"] = int(round(REF.get("meals_served", d["meals"])))
        d["staff"] = int(round(REF.get("kitchen_staff", d["staff"])))
        d["temp"] = round(REF.get("temperature_C", d["temp"]), 1)
        d["hum"] = round(REF.get("humidity_percent", d["hum"]), 1)
        d["past"] = round(REF.get("past_waste_kg", d["past"]), 1)
        try:
            d["date"] = date(int(REF["year"]), int(REF["month"]), int(REF["day"]))
        except Exception:  # noqa: BLE001
            pass
    return d


def page_prediction() -> None:
    st.title("🔮 Food Waste Prediction")
    st.caption("Enter the restaurant's conditions. The saved regression and classification pipelines "
               "produce the result - nothing is hardcoded.")
    if not core_ready:
        show_load_problems()
        return

    d = collect_form_defaults()
    exp_opts = OPTIONS.get("staff_experience", ["beginner", "intermediate", "expert"])
    cat_opts = OPTIONS.get("waste_category", ["dairy", "grains", "meat", "vegetables"])

    with st.form("prediction_form"):
        st.markdown("##### 🏪 Restaurant Information")
        a, b, c = st.columns(3)
        in_date = a.date_input("Date", value=d["date"], min_value=date(2000, 1, 1), max_value=date(2100, 12, 31))
        staff = b.number_input("Kitchen staff", min_value=1, value=d["staff"], step=1,
                               help=f"Training median: {REF.get('kitchen_staff', '-'):g}" if REF else None)
        exp = c.selectbox("Staff experience", exp_opts,
                          index=exp_opts.index(REF["staff_experience"]) if REF.get("staff_experience") in exp_opts else 0,
                          format_func=str.capitalize)
        a, b = st.columns(2)
        cat = a.selectbox("Food category", cat_opts, format_func=str.capitalize)

        st.markdown("##### 🍽️ Demand Information")
        a, b, c = st.columns(3)
        meals = a.number_input("Meals served", min_value=1, value=d["meals"], step=10,
                               help=f"Training median: {REF.get('meals_served', '-'):g}" if REF else None)
        day_choice = b.selectbox("Day of week", ["Auto (from date)"] + config.DAY_NAMES,
                                 help="Monday = 0 … Sunday = 6 (the model treats 5 and 6 as weekend).")
        event = c.radio("Special event", ["No", "Yes"], horizontal=True)

        st.markdown("##### 🌡️ Environmental Conditions")
        a, b = st.columns(2)
        temp = a.number_input("Temperature (°C)", min_value=-30.0, max_value=60.0, value=float(d["temp"]), step=0.5)
        hum = b.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=float(d["hum"]), step=1.0)

        st.markdown("##### 🗂️ Historical Waste Information")
        past = st.number_input("Past waste (kg)", min_value=0.0, value=float(d["past"]), step=1.0,
                               help=f"Training median: {REF.get('past_waste_kg', 0):.1f} kg" if REF else None)
        submitted = st.form_submit_button("Predict Food Waste", type="primary", width="stretch")

    if submitted:
        dow = in_date.weekday() if day_choice.startswith("Auto") else config.DAY_NAMES.index(day_choice)
        inputs = {
            "date": in_date, "meals_served": float(meals), "kitchen_staff": float(staff),
            "temperature_C": float(temp), "humidity_percent": float(hum), "day_of_week": dow,
            "special_event": 1 if event == "Yes" else 0, "past_waste_kg": float(past),
            "staff_experience": exp, "waste_category": cat,
        }
        errors, warns = validate_inputs(inputs, REF)
        if errors:
            for e in errors:
                st.error(e)
        else:
            for w in warns:
                st.warning(w)
            try:
                with st.spinner("Running the trained models..."):
                    result = run_prediction(reg_res.obj, clf_res.obj, thr_res.obj, inputs, COLUMNS)
                    recs = build_recommendations(inputs, result.classifier_label, thr_res.obj, REF)
                st.session_state["last"] = {"result": result, "recs": recs}
            except Exception as exc:  # noqa: BLE001
                st.error("The prediction could not be completed. Please check the inputs or the model files.")
                with st.expander("Technical details"):
                    st.code(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")

    last = st.session_state.get("last")
    if last:
        render_result(last["result"], last["recs"])


def render_result(result, recs) -> None:
    st.divider()
    level = result.classifier_label
    color = level_color(level)
    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            f'<div class="result" style="border-color:{color}">'
            f'<div class="lab">EXPECTED FOOD WASTE</div><div class="val">{result.waste_kg:.2f} kg</div>'
            f'<div style="margin-top:.7rem"><span class="badge" style="background:{color}">'
            f'{config.LEVEL_SHORT.get(level, level)}</span></div>'
            f'<div class="lab" style="margin-top:.4rem">WASTE LEVEL (classification model)</div></div>',
            unsafe_allow_html=True)
    with right:
        st.markdown("**Your inputs**")
        i = result.inputs
        m1, m2 = st.columns(2)
        m1.metric("Meals served", f"{i['meals_served']:.0f}")
        m2.metric("Past waste", f"{i['past_waste_kg']:g} kg")
        m3, m4 = st.columns(2)
        m3.metric("Special event", "Yes" if i["special_event"] else "No")
        m4.metric("Kitchen staff", f"{i['kitchen_staff']:.0f}")

    st.plotly_chart(waste_band_figure(result.waste_kg, thr_res.obj), width="stretch")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Classifier confidence** (Logistic Regression probabilities)")
        st.plotly_chart(probability_figure(result.class_probabilities), width="stretch")
    with t2:
        st.markdown("**Threshold-based level**")
        st.info(f"Applying the saved thresholds (Low ≤ {thr_res.obj['low_threshold']:.2f} kg, "
                f"High > {thr_res.obj['high_threshold']:.2f} kg) to the regression output gives: "
                f"**{result.threshold_label}**.")
        if not result.models_agree:
            st.warning("The classifier and the threshold rule disagree for this input (usually "
                       "near a boundary). The classifier's level is used for recommendations.")
    st.caption("Predictions are estimates and carry uncertainty - treat the kg value as a guide, not an exact figure.")

    st.markdown("### 💡 Recommendations")
    render_recommendations(recs)


def render_recommendations(recs: dict) -> None:
    level = recs["level"]
    box = {"High Waste": st.error, "Medium Waste": st.warning, "Low Waste": st.success}.get(level, st.info)
    box(f"{config.LEVEL_ICONS.get(level, '')} Predicted level: **{level}**")
    a, b = st.columns(2)
    with a:
        st.markdown("**Recommended actions**")
        for r in recs["actions"]:
            st.markdown(f"- {r}")
    with b:
        st.markdown("**Based on your input conditions**")
        if recs["context"]:
            for r in recs["context"]:
                st.markdown(f"- {r}")
        else:
            st.markdown("_No special conditions detected in the inputs._")
    st.caption(recs["disclaimer"])


def page_analysis() -> None:
    st.title("📊 Waste Analysis")
    st.warning("**Data provenance:** the only dataset provided is `final_submission.csv`, which contains "
               "the **model's predictions** for a test set (ID + predicted kg). These are *not* actual "
               "historical waste values. Training data was not provided, so no historical charts are shown.")
    if not sub_res.ok:
        st.error(sub_res.error)
        return
    if not thr_res.ok:
        st.error(thr_res.error)
        return
    df, thr = sub_res.obj, thr_res.obj
    last = st.session_state.get("last")
    user_kg = last["result"].waste_kg if last else None

    s = df["food_waste_kg"]
    c = st.columns(4)
    c[0].metric("Predicted records", f"{len(df):,}")
    c[1].metric("Mean predicted", f"{s.mean():.1f} kg")
    c[2].metric("Median predicted", f"{s.median():.1f} kg")
    c[3].metric("Range", f"{s.min():.1f} – {s.max():.1f} kg")

    st.markdown("#### 🔷 Predicted test data")
    st.plotly_chart(distribution_figure(df, thr, user_kg), width="stretch")
    if user_kg is not None:
        st.caption("Red line = your latest prediction from the Prediction page (user prediction, shown separately).")

    labels = s.map(lambda v: label_from_thresholds(v, thr))
    counts = labels.value_counts().reindex(thr.get("labels", ["Low Waste", "Medium Waste", "High Waste"])).fillna(0).astype(int)
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(category_figure(counts), width="stretch")
    with c2:
        st.markdown("**Level shares (predicted)**")
        st.dataframe((counts / counts.sum() * 100).round(1).rename("% of records").to_frame(),
                     width="stretch")
        st.caption("Levels are derived by applying the saved thresholds to predicted kg.")

    st.markdown("#### 🔶 Model-simulated what-if: meals served")
    st.caption("This calls the trained regression pipeline repeatedly, changing only *meals served*. "
               "It shows how the model responds - it is not historical data and not a causal claim.")
    if reg_res.ok:
        base = (last["result"].inputs if last else None)
        if base is None:
            st.info("Run a prediction first - the what-if curve uses your latest inputs as the baseline.")
        else:
            m0 = base["meals_served"]
            vals = [max(1.0, m0 * f) for f in [x / 20 for x in range(5, 41)]]
            try:
                sweep = sweep_meals(reg_res.obj, base, COLUMNS, vals)
                st.plotly_chart(sweep_figure(sweep, thr, m0), width="stretch")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not compute the what-if curve: {exc}")

    st.markdown("#### Not available")
    st.caption("Charts such as waste by day, food category, special event, meals/past waste vs waste, or "
               "monthly patterns need the model's input features for each record, which are not part of the "
               "provided files, so they are intentionally not shown.")


def render_metric_block(title: str, block: dict, keys: list) -> None:
    st.markdown(f"**{title}**" + (f" - {block['best_model']}" if block.get("best_model") else ""))
    cols = st.columns(len(keys))
    for col, (k, label) in zip(cols, keys):
        v = block.get(k)
        col.metric(label, f"{v:.4g}" if isinstance(v, (int, float)) else "n/a")


def page_performance() -> None:
    st.title("🧪 Model Performance & Insights")
    t1, t2, t3, t4 = st.tabs(["Evaluation metrics", "Model details", "Top predictive factors", "Input schema"])

    with t1:
        if metrics_res.ok:
            m = metrics_res.obj
            if "regression" in m:
                render_metric_block("Regression", m["regression"], [("rmse", "RMSE"), ("mae", "MAE"), ("r2", "R²")])
            if "classification" in m:
                render_metric_block("Classification", m["classification"],
                                    [("accuracy", "Accuracy"), ("f1", "F1-score"),
                                     ("precision", "Precision"), ("recall", "Recall")])
            for key, title in (("regression_comparison", "Regression model comparison"),
                               ("classification_comparison", "Classification model comparison")):
                if m.get(key):
                    st.markdown(f"**{title}**")
                    st.dataframe(pd.DataFrame(m[key]), width="stretch", hide_index=True)
        else:
            st.info("**Evaluation metrics are not displayed because they were not included in the files provided.** "
                    "`food_waste_model_meta.json` lists only the selected model names and input columns, and the "
                    "training notebook was not supplied. Nothing has been estimated or invented.\n\n"
                    "To show your real RMSE / MAE / R² / accuracy / F1, add `models/food_waste_metrics.json` "
                    "(format in the README) - this page will pick it up automatically.")
        st.markdown("**What the metrics mean**")
        st.markdown(
            "- **R²** - how much of the variation in food waste the model explains.\n"
            "- **MAE** - average absolute prediction error, in kilograms.\n"
            "- **RMSE** - like MAE but penalises larger errors more strongly.\n"
            "- **Accuracy / F1 / Precision / Recall** - how well the Low / Medium / High classes are identified.")

    with t2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Regression")
            if reg_res.ok:
                p = reg_res.obj.named_steps["model"].get_params()
                st.write(f"**Model:** {meta.get('regression_model') if meta else type(reg_res.obj.named_steps['model']).__name__}")
                st.write(f"n_estimators = {p.get('n_estimators')}, min_samples_leaf = {p.get('min_samples_leaf')}, "
                         f"random_state = {p.get('random_state')}")
                st.write("**Pipeline:** feature engineering → median imputation (numeric) / "
                         "most-frequent imputation + one-hot (categorical) → Random Forest")
            else:
                st.error(reg_res.error)
        with c2:
            st.markdown("##### Classification")
            if clf_res.ok:
                p = clf_res.obj.named_steps["model"].get_params()
                st.write(f"**Model:** {meta.get('classification_model') if meta else type(clf_res.obj.named_steps['model']).__name__}")
                st.write(f"max_iter = {p.get('max_iter')}, random_state = {p.get('random_state')}")
                st.write(f"**Classes:** {', '.join(map(str, clf_res.obj.classes_))}")
                st.write("**Pipeline:** feature engineering → median imputation + standard scaling (numeric) / "
                         "one-hot (categorical) → Logistic Regression")
            else:
                st.error(clf_res.error)
        if thr_res.ok:
            st.markdown("##### Waste-level thresholds (from `food_waste_thresholds.json`)")
            t = thr_res.obj
            a, b, c = st.columns(3)
            a.metric("Low Waste", f"≤ {t['low_threshold']:.2f} kg")
            b.metric("Medium Waste", f"{t['low_threshold']:.2f} – {t['high_threshold']:.2f} kg")
            c.metric("High Waste", f"> {t['high_threshold']:.2f} kg")
            if sub_res.ok:
                sh = sub_res.obj["food_waste_kg"].map(lambda v: label_from_thresholds(v, t)).value_counts(normalize=True)
                st.caption("Share of predicted test records per level: " +
                           ", ".join(f"{k} {v*100:.0f}%" for k, v in sh.items()) +
                           " - i.e. the thresholds split predictions into roughly equal-sized groups.")

    with t3:
        imp = feature_importances(reg_res.obj) if reg_res.ok else None
        if imp is None:
            st.error("Feature importances could not be read from the model.")
        else:
            st.plotly_chart(importance_figure(imp, 10), width="stretch")
            st.caption("Importances are read from the trained Random Forest (impurity-based). The model identifies "
                       "these features as important predictors; this does not imply causation. Engineered features "
                       "such as `log_meals` and `meals_per_staff` are derived from the raw inputs inside the pipeline.")
            with st.expander("Show all features"):
                st.dataframe(imp, width="stretch", hide_index=True)

    with t4:
        st.markdown("**Raw input columns the pipelines expect** (from `food_waste_model_meta.json`)")
        rows = [{"Column": c, "Label": config.FEATURE_LABELS.get(c, c)} for c in COLUMNS]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        if meta:
            st.caption(f"Target: `{meta.get('target')}`. Feature engineering and preprocessing live inside "
                       "the saved pipelines (`food_waste_features.py`), so the app passes raw inputs only.")


def page_recommendations() -> None:
    st.title("💡 Recommendations")
    last = st.session_state.get("last")
    if last:
        r = last["result"]
        st.write(f"Based on your latest prediction: **{r.waste_kg:.2f} kg** - **{r.classifier_label}**.")
        render_recommendations(last["recs"])
    else:
        st.info("Run a prediction on the **Food Waste Prediction** page to get recommendations tailored "
                "to your inputs. General guidance for each level is shown below.")
    st.markdown("### General guidance by waste level")
    from utils.recommendations import LEVEL_ACTIONS
    cols = st.columns(3)
    for col, lvl in zip(cols, ["Low Waste", "Medium Waste", "High Waste"]):
        with col:
            st.markdown(f"**{config.LEVEL_ICONS[lvl]} {lvl}**")
            for a in LEVEL_ACTIONS[lvl]:
                st.markdown(f"- {a}")


def page_about() -> None:
    st.title("ℹ️ About Project")
    st.markdown(f"**{config.APP_TITLE}** predicts daily restaurant food waste and supports planning decisions.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### ML tasks")
        st.markdown(f"- **Regression** - predict `food_waste_kg` ({meta.get('regression_model') if meta else 'Random Forest'})\n"
                    f"- **Classification** - Low / Medium / High waste ({meta.get('classification_model') if meta else 'Logistic Regression'})\n"
                    "- **Recommendations** - rule-based decision support from the prediction and inputs")
        st.markdown("##### How prediction works")
        st.markdown("1. You enter the 10 raw input fields.\n"
                    "2. They are passed as a DataFrame to the saved pipelines, which perform feature "
                    "engineering and preprocessing internally.\n"
                    "3. The regression pipeline returns kg; the classification pipeline returns the level.\n"
                    "4. Recommendations are generated from the level and your inputs.")
    with c2:
        st.markdown("##### Honesty notes")
        st.markdown("- No models are trained in this app.\n"
                    "- Only the files in `models/` are loaded.\n"
                    "- `final_submission.csv` holds predictions, not actual waste.\n"
                    "- Evaluation metrics are shown only if `models/food_waste_metrics.json` is provided.")
        st.markdown("##### Tech stack")
        st.markdown("Python · Streamlit · scikit-learn · pandas · Plotly")


{"🏠 Home": page_home, "🔮 Food Waste Prediction": page_prediction, "📊 Waste Analysis": page_analysis,
 "🧪 Model Performance": page_performance, "💡 Recommendations": page_recommendations,
 "ℹ️ About Project": page_about}[page]()
