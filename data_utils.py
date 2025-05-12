# Import libraries
import pandas as pd
import numpy as np
import streamlit as st
import requests
import os

# Load the data
@st.cache_data
def load_data():
    """
    Generate or load the climate dataset.

    This function checks if 'nasa_power_monthly_from_daily.csv' exists locally.
    If not, it fetches daily climate data via the NASA POWER API and aggregates it to monthly.
    Returns a clean DataFrame ready for preprocessing.
    """

    output_file = "data/nasa_power_monthly_from_daily.csv"

    # Define regions and coordinates
    regions = {
        'Singida': {'lat': -4.817, 'lon': 34.750},
        'Manyara': {'lat': -4.315, 'lon': 36.954},
        'Dodoma': {'lat': -6.173, 'lon': 35.742},
        'Songwe': {'lat': -8.524, 'lon': 32.537},
        'Rukwa': {'lat': -8.011, 'lon': 31.446}
    }

    parameters = ["T2M", "PRECTOTCORR", "WS2M", "RH2M"]
    start_date = "2000-01-01"
    end_date = "2024-12-31"
    url_base = "https://power.larc.nasa.gov/api/temporal/daily/point"

    def get_and_aggregate_daily(region, lat, lon):
        params = {
            "start": start_date.replace("-", ""),
            "end": end_date.replace("-", ""),
            "latitude": lat,
            "longitude": lon,
            "community": "ag",
            "parameters": ",".join(parameters),
            "format": "JSON"
        }

        response = requests.get(url_base, params=params)
        if response.status_code != 200:
            st.warning(f"API error {response.status_code} for {region}")
            return pd.DataFrame()

        try:
            data = response.json()
            param_data = data['properties']['parameter']

            # Build daily DataFrame
            first_param = list(param_data.keys())[0]
            df = pd.DataFrame.from_dict(param_data[first_param], orient='index', columns=[first_param])
            for p in parameters[1:]:
                temp_df = pd.DataFrame.from_dict(param_data[p], orient='index', columns=[p])
                df = df.join(temp_df)

            df.index.name = "Date"
            df.reset_index(inplace=True)
            df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
            df["Region"] = region
            df["YearMonth"] = df["Date"].dt.to_period("M")

            # Aggregate monthly
            monthly = df.groupby(["Region", "YearMonth"]).agg({
                "T2M": "mean",
                "PRECTOTCORR": "sum",
                "WS2M": "mean",
                "RH2M": "mean"
            }).reset_index()
            monthly["YearMonth"] = monthly["YearMonth"].dt.to_timestamp()
            return monthly

        except Exception as e:
            st.error(f"Error processing {region}: {e}")
            return pd.DataFrame()

    # Check local cache
    if os.path.exists(output_file):
        df = pd.read_csv(output_file, parse_dates=["YearMonth"])
    else:
        df = pd.concat([get_and_aggregate_daily(r, c['lat'], c['lon']) for r, c in regions.items()])
        df.to_csv(output_file, index=False)
        st.success(f"Data saved to {output_file}")

    return df

def preprocess_data(df):
    """
    Clean and prepare the climate dataset for EDA and modeling.
    Includes:
    - Missing value handling
    - Time feature extraction
    - Seasonality encoding
    - Region-specific precipitation-based season classification
    """

    st.subheader("📋 Data Preprocessing Summary")

    original_len = len(df)
    df["YearMonth"] = pd.to_datetime(df["YearMonth"], errors="coerce")

    # Initial null report
    st.write("Missing values before cleaning:")
    st.write(df.isnull().sum())

    # Drop rows with invalid or missing dates
    df = df.dropna(subset=["YearMonth"])

    # Extract year and month
    df["Year"] = df["YearMonth"].dt.year
    df["Month"] = df["YearMonth"].dt.month

    # Drop rows with missing climate values
    df = df.dropna(subset=["T2M", "PRECTOTCORR", "WS2M", "RH2M"])

    # Add cyclical encoding for month
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    # Region-specific quantile-based seasonal classification
    st.write("📊 Calculating region-specific precipitation thresholds...")
    thresholds = df.groupby("Region")["PRECTOTCORR"].quantile([0.25, 0.75]).unstack()
    thresholds.columns = ["Q25", "Q75"]
    st.write(thresholds)

    def assign_season(row):
        q25 = thresholds.loc[row["Region"], "Q25"]
        q75 = thresholds.loc[row["Region"], "Q75"]
        if row["PRECTOTCORR"] <= q25:
            return "Dry"
        elif row["PRECTOTCORR"] >= q75:
            return "Rainy"
        else:
            return "Transition"

    df["Season"] = df.apply(assign_season, axis=1)

    # Summary of rows dropped
    final_len = len(df)
    dropped = original_len - final_len
    percent_dropped = (dropped / original_len) * 100
    st.write(f"🧹 Rows dropped due to missing data: {dropped} ({percent_dropped:.2f}%)")

    if percent_dropped > 10:
        st.warning("⚠️ More than 10% of the data was dropped. Consider interpolation or imputation.")
    else:
        st.success("✅ Data loss is minimal. Proceeding with row deletion is acceptable.")

    return df