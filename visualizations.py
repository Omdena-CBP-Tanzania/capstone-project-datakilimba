"""
Module for generating EDA and model visualizations.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from constants import LABEL_MAP

def plot_time_series(df, variable="T2M"):
    label = LABEL_MAP.get(variable, variable)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['dates'], df[variable])
    ax.set_xlabel("Date")
    ax.set_ylabel(label)
    ax.set_title(f"Monthly Average {label}")
    ax.grid(True)
    return fig

def plot_seasonal_patterns(df, variable="T2M"):
    df = df.rename(columns={"Month": "month"})  # Add this line
    label = LABEL_MAP.get(variable, variable)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='month', y=variable, data=df, ax=ax)
    ax.set_xlabel("Month")
    ax.set_ylabel(label)
    ax.set_title(f"Monthly Distribution of {label}")
    return fig

def plot_yearly_trends(df, variable="T2M"):
    df = df.rename(columns={"Year": "year"})  # Fix column name
    label = LABEL_MAP.get(variable, variable)
    year_avg = df.groupby('year')[variable].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(year_avg['year'], year_avg[variable], marker='o')
    ax.set_xlabel("Year")
    ax.set_ylabel(label)
    ax.set_title(f"Yearly Average {label}")
    ax.grid(True)
    return fig


def plot_actual_vs_predicted(y_test, y_pred, target="T2M"):
    label = LABEL_MAP.get(target, target)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    ax.set_xlabel(f"Actual {label}")
    ax.set_ylabel(f"Predicted {label}")
    ax.set_title(f"Actual VS Predicted {label}")
    return fig

def plot_prediction_context(hist_temps, pred_year, pred_month, prediction, variable="T2M"):
    years_hist, temp_hist = zip(*hist_temps)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(years_hist, temp_hist, label=f"Historical (Month {pred_month})", color='blue')
    ax.plot(years_hist, temp_hist, 'b--', alpha=0.6)
    ax.scatter([pred_year], [prediction], color='red', s=100, label='Prediction')
    z = np.polyfit(years_hist, temp_hist, 1)
    p = np.poly1d(z)
    ax.plot(range(2010, pred_year + 1), p(range(2010, pred_year + 1)), 'g-', label='Trend')
    ax.set_xlabel("Year")
    label = LABEL_MAP.get(variable, variable)
    ax.set_ylabel(f"{label} for Month {pred_month}")
    ax.set_title("Historical Context")
    ax.legend()
    ax.grid(True)
    return fig

def plot_climate_trends(df, variable="T2M"):
    label = LABEL_MAP.get(variable, variable)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df, x="YearMonth", y=variable, hue="Region", ax=ax)
    ax.set_title(f"Monthly Trend of {label} by Region")
    ax.set_xlabel("Year-Month")
    ax.set_ylabel(label)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig

def plot_feature_distributions(df, features=["T2M", "PRECTOTCORR", "WS2M", "RH2M"]):
    figs = []
    for feature in features:
        label = LABEL_MAP.get(feature, feature)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df[feature], ax=axes[0], kde=True, bins=30, color="skyblue")
        axes[0].set_title(f"Distribution of {label}")
        sns.boxplot(y=df[feature], ax=axes[1], color="lightgreen")
        axes[1].set_title(f"Boxplot of {label}")
        fig.tight_layout()
        figs.append(fig)
    return figs

def plot_correlation_matrix(df, features=["T2M", "PRECTOTCORR", "WS2M", "RH2M"]):
    friendly_labels = [LABEL_MAP.get(col, col) for col in features]
    corr = df[features].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True, ax=ax,
                xticklabels=friendly_labels, yticklabels=friendly_labels)
    ax.set_title("Correlation Matrix")
    fig.tight_layout()
    return fig

def decompose_seasonality(df, variable="T2M", region="Dodoma"):
    ts = df[df["Region"] == region].set_index("YearMonth")[variable].asfreq("MS")
    result = seasonal_decompose(ts, model="additive")
    fig = result.plot()
    label = LABEL_MAP.get(variable, variable)
    fig.suptitle(f"{region}: Seasonal Decomposition of {label}")
    fig.tight_layout()
    return fig

def plot_confusion_matrix(model, X_test, y_test, class_names=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=class_names, cmap="Blues", ax=ax
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig
