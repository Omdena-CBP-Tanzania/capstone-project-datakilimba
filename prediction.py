"""
Module for making and formatting predictions from trained models.
"""
import numpy as np
import pandas as pd
from constants import LABEL_MAP

def predict_target(model, X_new):
    """
    Predict the target variable given new input features.
    Returns: prediction(s)
    """
    return model.predict(X_new)

def format_predictions(preds, variable="T2M"):
    """
    Format predictions into a friendly labeled dictionary or string.
    """
    label = LABEL_MAP.get(variable, variable)
    return f"Predicted {label}: {preds[0]:.2f}"

def prepare_single_input(year, month, reference_df):
    """
    Prepare a single row of features for prediction
    based on expected structure from reference_df.
    """
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    base = {
        'Year': year,
        'Month': month,
        'Month_sin': month_sin,
        'Month_cos': month_cos
    }

    expected_cols = [col for col in reference_df.columns if col not in ["T2M", "PRECTOTCORR", "WS2M", "RH2M", "Season", "Region", "YearMonth"]]
    for col in expected_cols:
        if col not in base:
            base[col] = 0

    return pd.DataFrame([base])

def make_prediction(model, year, month, reference_df):
    """
    Make a prediction using model and single date input.
    """
    X_new = prepare_single_input(year, month, reference_df)
    pred = predict_target(model, X_new)
    return pred[0]

def get_historical_context(df, month, variable="T2M"):
    """
    Return (year, value) for the same month over previous years.
    """
    years = df['year'].unique()
    hist = []
    for y in sorted(years):
        match = df[(df['year'] == y) & (df['month'] == month)]
        if not match.empty:
            hist.append((y, match[variable].values[0]))
    return hist

def get_historical_average(df, month, variable="T2M"):
    """
    Return the historical average value of the variable for a given month.
    """
    return df[df['month'] == month][variable].mean()

def analyze_anomaly(prediction, df, month, variable="T2M"):
    """
    Compare prediction to historical stats and return anomaly insight.
    """
    historical = df[df['month'] == month][variable]
    mean = historical.mean()
    percentile_25 = historical.quantile(0.25)
    percentile_75 = historical.quantile(0.75)

    if prediction < percentile_25:
        return f"⬇️ Predicted {LABEL_MAP.get(variable, variable)} is unusually low (below 25th percentile)."
    elif prediction > percentile_75:
        return f"⬆️ Predicted {LABEL_MAP.get(variable, variable)} is unusually high (above 75th percentile)."
    elif abs(prediction - mean) > 2:
        direction = "higher" if prediction > mean else "lower"
        return f"⚠️ Prediction is over 2°C {direction} than the long-term average of {mean:.2f}."
    else:
        return f"✅ Prediction is within typical historical range (mean: {mean:.2f})."