
# 🌍 Climate Prediction Dashboard - Tanzania

A climate analytics and forecasting tool built as part of the **Omdena CBP Tanzania Capstone Project**. This application enables users to explore climate trends and generate monthly forecasts for key variables across Tanzanian regions using machine learning models.

## 📌 Project Objectives

- Develop a user-friendly dashboard to explore historical climate trends in Tanzania.
- Build and compare multiple regression models (e.g., Random Forest, Ridge, SVR) to predict monthly values of climate variables.
- Enable real-time single-point prediction for any selected month and region.
- Detect anomalies and visualize predicted values in historical context.
- Apply learnings from the **Omdena ML/AI for Social Good** course on model lifecycle, collaboration, and storytelling.

---

## 🧪 Climate Variables Modeled

- **T2M** – Air Temperature (°C)
- **PRECTOTCORR** – Precipitation (mm)
- **WS2M** – Wind Speed at 2m (m/s)
- **RH2M** – Relative Humidity (%)

Each variable is modeled and predicted on a **monthly** basis for multiple Tanzanian regions.

---

## 🔍 Methodology

### 1. **Data Processing**
- Raw data is loaded via `data_utils.py`, then cleaned and engineered with:
  - Cyclical encoding for `month` (`sin`/`cos`)
  - One-hot encoding for `Season`
  - Derived columns: `Month_sin`, `Month_cos`, `Season`, `Year`

### 2. **Exploratory Data Analysis (EDA)**
Implemented in `data_exploration.py`, using `visualizations.py`:
- Time series and yearly trend plots
- Boxplots for monthly distribution
- Correlation matrix and feature distributions
- Automated insights for correlated variables
- Optional seasonal decomposition

### 3. **Model Training & Comparison**
Available in `model_training.py`:
- Train five regression models:
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Support Vector Regressor
  - Ridge Regression
  - Linear Regression
- Evaluate using RMSE, MAE, R²
- Save the best model for each variable-region combo (`*_best.pkl`)

### 4. **Prediction Module**
In `prediction_page.py`:
- User inputs year, month, region, and variable
- Selects best model from saved files
- Forecast or historical mode auto-detected
- Detects anomalies using a 2-unit threshold
- Visualizes prediction within historical context

---

## 📊 Model Performance Summary

Model comparison was based on **RMSE** across different regions and climate variables.

| Model               | Typical RMSE (T2M) | Notes                        |
|--------------------|--------------------|------------------------------|
| Random Forest       | ~0.85–1.2          | Best general performance     |
| Ridge Regression    | ~1.1–1.5           | Good linear baseline         |
| SVR                 | ~1.3–1.6           | Sensitive to feature scaling |
| Gradient Boosting   | ~0.9–1.3           | Occasionally outperforms RF  |
| Linear Regression   | ~1.4–1.8           | Simplest but least accurate  |

---

## 💡 Learnings from Omdena Course

As a participant of the **Omdena Community Bootcamp**, I gained hands-on skills in:

- Structuring a real-world machine learning pipeline: data → modeling → app.
- Comparing models in a reproducible way using `sklearn` and `joblib`.
- Building Streamlit dashboards with modular Python code.
- Communicating insights from EDA and model performance clearly.
- Collaborating within a best-practices ML repository (Git/GitHub).

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Make sure `models/` directory exists for saving models.

---

## 📂 Project Structure

```
capstone-project-datakilimba/
│
├── data_utils.py
├── model_utils.py
├── prediction.py
├── visualizations.py
├── pages/
│   ├── data_exploration.py
│   ├── model_training.py
│   └── prediction_page.py
├── models/                # Auto-created to store trained models
└── constants.py
```

---

## 🙌 Acknowledgments

Thanks to the **Omdena CBP Tanzania Chapter** and mentors for guidance. This project contributes toward climate-smart agriculture and environmental planning in Tanzania.
