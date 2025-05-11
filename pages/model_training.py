import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR

from data_utils import load_data, preprocess_data
from model_utils import compare_multiple_regressors
from constants import LABEL_MAP

# Page configuration
st.set_page_config(page_title="Model Training", layout="wide")
st.title("🤖 Climate Model Training")

# Load and preprocess data
raw_df = load_data()
df = preprocess_data(raw_df.copy())

# Sidebar inputs
st.sidebar.header("🛠️ Model Configuration")
region = st.sidebar.selectbox("Select Region", df["Region"].unique())
target = st.sidebar.selectbox("Select Target Variable", ["T2M", "PRECTOTCORR", "WS2M", "RH2M"])
run_button = st.sidebar.button("Train Models")

# Show selected configuration
st.markdown(f"### Training Models to Predict **{LABEL_MAP.get(target, target)}** in **{region}**")

if run_button:
    st.info("⏳ Training in progress... Please wait.")
    
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Support Vector": SVR(kernel='rbf')
    }

    results_df = compare_multiple_regressors(df, region, target, models)
    st.success("✅ Training complete. Results below:")

    st.dataframe(results_df.style.format({"RMSE": "{:.2f}", "MAE": "{:.2f}", "R2": "{:.2f}"}))

    # Highlight best model
    best_model_row = results_df.loc[results_df['RMSE'].idxmin()]
    st.markdown("### 🏆 Best Performing Model")
    st.write(f"**Model**: {best_model_row['Model']}")
    st.write(f"**RMSE**: {best_model_row['RMSE']:.2f}, **MAE**: {best_model_row['MAE']:.2f}, **R²**: {best_model_row['R2']:.2f}")

    # Path of saved model
    model_path = f"models/{region}_{target}_{best_model_row['Model'].replace(' ', '')}.pkl"
    st.info(f"💾 Best model saved to: `{model_path}`")
else:
    st.warning("⚠️ Click the 'Train Models' button to start training.")
