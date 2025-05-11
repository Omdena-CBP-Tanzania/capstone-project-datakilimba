import streamlit as st
from data_utils import load_data, preprocess_data
from model_utils import prepare_features, split_data, make_model_pipeline, train_model, evaluate_model, save_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import os

st.set_page_config(page_title="Model Training", layout="wide")
st.title("🧪 Model Training and Comparison")

# Load and preprocess
raw_df = load_data()
df = preprocess_data(raw_df.copy())

# UI: Select Region and Variable
region = st.selectbox("Select Region", df["Region"].unique())
target = st.selectbox("Select Target Variable", ["T2M", "PRECTOTCORR", "WS2M", "RH2M"])

# Prepare features
df_region = df[df["Region"] == region]
X, y = prepare_features(df_region, target=target)
X_train, X_test, y_train, y_test = split_data(X, y)

# Define models
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "LinearRegression": LinearRegression(),
    "RidgeRegression": Ridge(),
    "SupportVector": SVR()
}

results = []
best_rmse = float("inf")
best_model = None
best_model_name = ""
best_metrics = {}

st.subheader("🔍 Comparing Models...")

for name, model in models.items():
    pipeline = make_model_pipeline(model)
    trained = train_model(pipeline, X_train, y_train)
    preds = trained.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results.append({
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })

    if rmse < best_rmse:
        best_rmse = rmse
        best_model = trained
        best_model_name = name
        best_metrics = {"RMSE": rmse, "MAE": mae, "R2": r2}

# Show results table
results_df = pd.DataFrame(results).sort_values("RMSE")
st.dataframe(results_df, use_container_width=True)

# Save best model
if best_model:
    os.makedirs("models", exist_ok=True)
    save_path = f"models/{region}_{target}_{best_model_name}.pkl"
    save_model(best_model, save_path, model_type=best_model_name)
    st.info(f"💾 Best model (**{best_model_name}**) saved to: `{save_path}`")

    st.markdown("### 🧮 Best Model Metrics")
    st.write(f"**RMSE**: {best_metrics['RMSE']:.2f}, **MAE**: {best_metrics['MAE']:.2f}, **R²**: {best_metrics['R2']:.2f}")
