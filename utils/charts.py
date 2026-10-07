"""Plotly figure builders. All figures are built from real model outputs / files only."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from utils.config import LEVEL_COLORS

_LAYOUT = dict(margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="white", paper_bgcolor="white",
               font=dict(family="sans-serif", size=13))


def waste_band_figure(waste_kg: float, thresholds: dict) -> go.Figure:
    low, high = thresholds["low_threshold"], thresholds["high_threshold"]
    labels = thresholds.get("labels", ["Low Waste", "Medium Waste", "High Waste"])
    axis_max = max(high * 1.5, waste_kg * 1.1)
    segs = [(0, low, labels[0]), (low, high, labels[1]), (high, axis_max, labels[2])]
    fig = go.Figure()
    for a, b, lab in segs:
        fig.add_trace(go.Bar(x=[b - a], y=[""], base=a, orientation="h", name=lab,
                             marker_color=LEVEL_COLORS.get(lab, "#999"), opacity=0.35,
                             hovertemplate=f"{lab}: {a:.1f} - {b:.1f} kg<extra></extra>"))
    fig.add_trace(go.Scatter(x=[waste_kg], y=[""], mode="markers+text", text=[f"{waste_kg:.1f} kg"],
                             textposition="top center", marker=dict(size=18, color="#1F2933",
                             symbol="diamond"), name="Prediction", hoverinfo="skip"))
    fig.update_layout(barmode="stack", height=150, showlegend=True,
                      legend=dict(orientation="h", y=-0.5), xaxis_title="Predicted food waste (kg)",
                      yaxis=dict(visible=False), **_LAYOUT)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    fig.update_xaxes(range=[0, axis_max])
    return fig


def probability_figure(probs: dict) -> go.Figure:
    order = [k for k in ("Low Waste", "Medium Waste", "High Waste") if k in probs]
    fig = go.Figure(go.Bar(x=[probs[k] * 100 for k in order], y=order, orientation="h",
                           marker_color=[LEVEL_COLORS.get(k, "#999") for k in order],
                           text=[f"{probs[k]*100:.1f}%" for k in order], textposition="auto"))
    fig.update_layout(height=180, xaxis=dict(range=[0, 100], title="Classifier probability (%)"),
                      **_LAYOUT)
    return fig


def distribution_figure(df: pd.DataFrame, thresholds: dict, user_kg: float | None = None) -> go.Figure:
    fig = go.Figure(go.Histogram(x=df["food_waste_kg"], nbinsx=40, marker_color="#5B9BD5",
                                 name="Predicted (test set)"))
    for key, name in (("low_threshold", "Low/Medium"), ("high_threshold", "Medium/High")):
        fig.add_vline(x=thresholds[key], line_dash="dash", line_color="#888",
                      annotation_text=f"{name}: {thresholds[key]:.1f}", annotation_position="top")
    if user_kg is not None:
        fig.add_vline(x=user_kg, line_color="#D64545", line_width=3,
                      annotation_text=f"Your prediction: {user_kg:.1f}", annotation_position="bottom")
    fig.update_layout(height=380, xaxis_title="Predicted food waste (kg)", yaxis_title="Records",
                      title="Predicted waste distribution (model output on test data)", **_LAYOUT)
    return fig


def category_figure(counts: pd.Series) -> go.Figure:
    fig = go.Figure(go.Bar(x=counts.index, y=counts.values,
                           marker_color=[LEVEL_COLORS.get(i, "#999") for i in counts.index],
                           text=counts.values, textposition="auto"))
    fig.update_layout(height=340, yaxis_title="Records",
                      title="Predicted waste level counts (thresholds applied to predictions)", **_LAYOUT)
    return fig


def importance_figure(imp: pd.DataFrame, top_n: int = 10) -> go.Figure:
    top = imp.head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(x=top["importance"], y=top["feature"], orientation="h",
                           marker_color="#2E7D5B", text=[f"{v:.3f}" for v in top["importance"]],
                           textposition="auto"))
    fig.update_layout(height=420, xaxis_title="Importance (Random Forest, impurity-based)",
                      title=f"Top {top_n} predictors identified by the model", **_LAYOUT)
    return fig


def sweep_figure(sweep: pd.DataFrame, thresholds: dict, current_meals: float | None = None) -> go.Figure:
    fig = go.Figure(go.Scatter(x=sweep["meals_served"], y=sweep["predicted_waste_kg"],
                               mode="lines+markers", line=dict(color="#2E7D5B"), name="Model prediction"))
    for key, name in (("low_threshold", "Low/Medium"), ("high_threshold", "Medium/High")):
        fig.add_hline(y=thresholds[key], line_dash="dash", line_color="#888",
                      annotation_text=name, annotation_position="right")
    if current_meals:
        fig.add_vline(x=current_meals, line_color="#D64545", annotation_text="Your input")
    fig.update_layout(height=380, xaxis_title="Meals served", yaxis_title="Predicted waste (kg)",
                      title="Model-simulated: predicted waste vs meals served (other inputs fixed)",
                      **_LAYOUT)
    return fig
