import streamlit as st
import pandas as pd
from data_utils import load_data, preprocess_data
from visualizations import (
    plot_climate_trends, plot_time_series, plot_seasonal_patterns,
    plot_yearly_trends, plot_feature_distributions, plot_correlation_matrix,
    decompose_seasonality
)
from constants import LABEL_MAP

st.set_page_config(page_title="Climate Data Explorer", layout="wide")
st.title("📊 Climate Data Exploration - Select Regions")

# Load and preprocess data
raw_df = load_data()
df = preprocess_data(raw_df.copy())

# Sidebar filters
regions = df['Region'].unique().tolist()
variables = ["T2M", "PRECTOTCORR", "WS2M", "RH2M"]
region = st.sidebar.selectbox("Select Region", regions)
variable = st.sidebar.selectbox("Select Climate Variable", variables)

st.markdown(f"### Exploring **{LABEL_MAP.get(variable, variable)}** in **{region}**")
df_region = df[df['Region'] == region]

# Layout
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(plot_climate_trends(df_region, variable), use_container_width=True)
    st.pyplot(plot_time_series(df_region.rename(columns={"YearMonth": "dates"}), variable=variable))

with col2:
    st.pyplot(plot_seasonal_patterns(df_region, variable))
    st.pyplot(plot_yearly_trends(df_region, variable))

st.subheader("📌 Distribution and Correlation")
st.pyplot(plot_correlation_matrix(df_region, features=variables))
for fig in plot_feature_distributions(df_region, features=variables):
    st.pyplot(fig)

# Optional: Decompose
with st.expander("🔍 Seasonal Decomposition (Optional)"):
    fig = decompose_seasonality(df, variable=variable, region=region)
    st.pyplot(fig)

# 🔍 Optional Insights Based on Correlation
st.subheader("🧠 Automated Insight")
correlated = df_region[variables].corr()[variable].drop(variable).sort_values(ascending=False)
strongest = correlated.idxmax()
strength = correlated.max()
if strength > 0.7:
    st.info(f"**{LABEL_MAP.get(variable)}** appears to have a strong positive correlation with **{LABEL_MAP.get(strongest)}** (ρ = {strength:.2f}).")
elif strength < -0.7:
    st.info(f"**{LABEL_MAP.get(variable)}** appears to have a strong negative correlation with **{LABEL_MAP.get(strongest)}** (ρ = {strength:.2f}).")
else:
    st.info(f"No strong correlations (ρ > ±0.7) observed between **{LABEL_MAP.get(variable)}** and other variables.")
