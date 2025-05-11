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
    get_historical_average,
    detect_anomaly
)
from visualizations import plot_prediction_context
from constants import LABEL_MAP

st.set_page_config(page_title="📈 Climate Prediction", layout="centered")
st.title("📈 Predict Future Climate")

# Load and preprocess data
raw_df = load_data()
df = preprocess_data(raw_df.copy())

# Sidebar selections
regions = df["Region"].unique().tolist()
variables = ["T2M", "PRECTOTCORR", "WS2M", "RH2M"]
region = st.selectbox("Select Region", regions)
target = st.selectbox("Select Climate Variable to Predict", variables)
label = LABEL_MAP.get(target, target)

# Year and month inputs
current_year = datetime.now().year
col1, col2 = st.columns(2)
with col1:
    year = st.number_input("Prediction Year", min_value=2000, max_value=2050, value=current_year)
with col2:
    month = st.selectbox("Prediction Month", list(range(1, 13)))

# Filter region-specific data
df_region = df[df["Region"] == region].copy()

# Ensure required columns
if "year" not in df_region.columns:
    df_region["year"] = df_region["YearMonth"].dt.year
if "month" not in df_region.columns:
    df_region["month"] = df_region["YearMonth"].dt.month

# Forecast mode check
forecast_mode = (year, month) not in list(zip(df_region["year"], df_region["month"]))
if forecast_mode:
    st.warning("🔮 Forecast mode: Predicting for a future date not in historical data.")

# Prediction
if st.button("🔮 Predict"):
    model_dir = "models"
    matched_model_file = None
    model_type = None

    # Locate model file
    for fname in os.listdir(model_dir):
        if fname.startswith(f"{region}_{target}_") and fname.endswith(".pkl"):
            matched_model_file = fname
            model_type = fname.split("_")[-1].replace(".pkl", "")
            break

    if matched_model_file:
        model_path = os.path.join(model_dir, matched_model_file)
        model = load_model(model_path)

        prediction = make_prediction(
            model, year, month, reference_df=df_region,
            target=target, forecast_mode=forecast_mode
        )

        st.success(format_predictions([prediction], variable=target))
        st.info(f"Model used: **{model_type}**")

        # Anomaly detection
        hist_avg = get_historical_average(df_region, month, variable=target)
        st.warning(detect_anomaly(prediction, hist_avg))

        # Historical context plot
        hist_context = get_historical_context(df_region, month, variable=target)
        fig = plot_prediction_context(hist_context, year, month, prediction, variable=target)
        st.pyplot(fig)
    else:
        st.error(f"No trained model found for {region} and {label}. Please train it first.")
