import numpy as np
import pandas as pd
import os
import joblib
from constants import LABEL_MAP
from model_utils import prepare_features

"""
Module for making and formatting predictions from trained models.
"""

def predict_target(model, X_new):
    return model.predict(X_new)

def format_predictions(preds, variable="T2M"):
    label = LABEL_MAP.get(variable, variable)
    return f"Predicted {label}: {preds[0]:.2f}"

def prepare_single_input(year, month, reference_df, target, model, forecast_mode=False):
    """
    Prepare a single input row matching training features for prediction.

    Uses reference_df to match feature structure.
    Uses model.named_steps['model'].feature_names_in_ to guarantee correct order.

    forecast_mode=True estimates season using historical precipitation percentiles.
    """
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    row = {
        "Month": month,
        "Month_sin": month_sin,
        "Month_cos": month_cos
    }

    # Estimate Season value
    if forecast_mode:
        if "month" not in reference_df.columns:
            reference_df["month"] = reference_df["YearMonth"].dt.month
        region = reference_df["Region"].iloc[0]
        q25, q75 = reference_df[reference_df["Region"] == region]["PRECTOTCORR"].quantile([0.25, 0.75])
        avg_precip = reference_df[reference_df["month"] == month]["PRECTOTCORR"].mean()
        if avg_precip <= q25:
            row["Season"] = "Dry"
        elif avg_precip >= q75:
            row["Season"] = "Rainy"
        else:
            row["Season"] = "Transition"
    else:
        # Infer from historical data if available
        if "month" not in reference_df.columns:
            reference_df["month"] = reference_df["YearMonth"].dt.month
        if "year" not in reference_df.columns:
            reference_df["year"] = reference_df["YearMonth"].dt.year
        matched = reference_df[(reference_df["month"] == month) & (reference_df["year"] == year)]
        row["Season"] = matched["Season"].iloc[0] if not matched.empty else "Transition"

    # Convert to DataFrame and apply one-hot encoding
    df_input = pd.DataFrame([row])
    df_input = pd.get_dummies(df_input)

    # Get expected features from the trained model
    expected_features = list(model.named_steps["model"].feature_names_in_)

    # Add any missing features as 0
    for col in expected_features:
        if col not in df_input.columns:
            df_input[col] = 0

    # Ensure correct column order
    df_input = df_input[expected_features]

    return df_input

def make_prediction(model, year, month, reference_df, target, forecast_mode=False):
    """
    Make a prediction using aligned input for either historical or forecast context.
    """
    X_new = prepare_single_input(year, month, reference_df, target, model=model, forecast_mode=forecast_mode)
    return model.predict(X_new)[0]

def get_historical_context(df, month, variable="T2M"):
    """
    Return (year, value) pairs for the selected month for trend plotting.
    """
    years = df['year'].unique()
    hist = []
    for y in sorted(years):
        match = df[(df['year'] == y) & (df['month'] == month)]
        if not match.empty:
            hist.append((y, match[variable].values[0]))
    return hist

def get_historical_average(df, month, variable="T2M"):
    return df[df['month'] == month][variable].mean()

def detect_anomaly(prediction, historical_mean, threshold=2.0):
    delta = prediction - historical_mean
    if abs(delta) >= threshold:
        return f"⚠️ Anomaly: Predicted value is {'higher' if delta > 0 else 'lower'} than historical average by {delta:.2f}"
    return "✅ Prediction is within normal range of historical average."
