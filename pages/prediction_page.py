import streamlit as st
import pandas as pd
import os
from datetime import datetime
from data_utils import load_data, preprocess_data
from model_utils import load_model
from prediction import (
    make_prediction,
    format_predictions,
    get_historical_context,
    get_historical_average
)
from visualizations import plot_prediction_context
from constants import LABEL_MAP

st.set_page_config(page_title="📈 Climate Prediction", layout="centered")
st.title("📈 Predict Future Climate")

# Load and preprocess
raw_df = load_data()
df = preprocess_data(raw_df.copy())

# Sidebar selections
regions = df["Region"].unique().tolist()
variables = ["T2M", "PRECTOTCORR", "WS2M", "RH2M"]
region = st.selectbox("Select Region", regions)
target = st.selectbox("Select Climate Variable to Predict", variables)
label = LABEL_MAP.get(target, target)

# Date selection
current_year = datetime.now().year
col1, col2 = st.columns(2)
with col1:
    year = st.number_input("Prediction Year", min_value=2000, max_value=2050, value=current_year)
with col2:
    month = st.selectbox("Prediction Month", list(range(1, 13)))

# Forecast mode detection
df_region = df[df["Region"] == region]
if "year" not in df_region.columns:
    df_region["year"] = df_region["YearMonth"].dt.year
if "month" not in df_region.columns:
    df_region["month"] = df_region["YearMonth"].dt.month
forecast_mode = (year, month) not in list(zip(df_region["year"], df_region["month"]))
if forecast_mode:
    st.warning("🔮 Forecast mode: Predicting for future date not in historical data.")

# Predict
if st.button("🔮 Predict"):
    # Try to match any saved model with region/target (e.g. RandomForest, Ridge, etc.)
    model_path = None
    model_type = "Unknown"
    for fname in os.listdir("models"):
        if fname.startswith(f"{region}_{target}_") and fname.endswith(".pkl"):
            model_path = os.path.join("models", fname)
            model_type = fname.split("_")[-1].replace(".pkl", "")
            break

    if not model_path or not os.path.exists(model_path):
        st.error(f"No trained model found for {region} and {label}. Please train it first.")
    else:
        model = load_model(model_path)
        prediction = make_prediction(model, year, month, df_region, target=target, forecast_mode=forecast_mode)

        # Show prediction
        st.success(format_predictions([prediction], variable=target))

        # Show model type (fallback to embedded _model_name if available)
        model_obj = model.named_steps.get("model", None)
        embedded_name = getattr(model_obj, "_model_name", None)
        if embedded_name:
            model_type = embedded_name
        st.info(f"Model used: **{model_type}**")

        # Anomaly detection
        hist_avg = get_historical_average(df_region, month, variable=target)
        diff = prediction - hist_avg
        if abs(diff) >= 2:
            direction = "above" if diff > 0 else "below"
            st.warning(f"⚠️ Anomaly detected: Predicted value is **{abs(diff):.2f}** units {direction} the historical monthly average.")

        # Plot context
        hist_context = get_historical_context(df_region, month, variable=target)
        fig = plot_prediction_context(hist_context, year, month, prediction, variable=target)
        st.pyplot(fig)
