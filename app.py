import streamlit as st

st.set_page_config(
    page_title="Climate Analysis App",
    page_icon="🌦️",
    layout="wide",
)

st.title("🌍 Climate Modeling Dashboard – Tanzania")
st.markdown("""
Welcome to the **Climate Data Modeling App** built for the Omdena Capstone.

Use the sidebar to navigate between:
- 📊 **Data Exploration** – visualize trends and patterns
- 🧠 **Model Training** – build regression and classification models
- 🔮 **Prediction Page** – make climate forecasts for future months

---
""")
st.info("Use the sidebar ➡️ to get started.")
