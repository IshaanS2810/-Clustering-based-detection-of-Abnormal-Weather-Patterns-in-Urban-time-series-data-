import streamlit as st
import pandas as pd
from components.sidebar import render_sidebar
from utils.loader import load_data

st.set_page_config(page_title="Weather Anomaly Dashboard", layout="wide")
from components.charts import (
    get_plot_columns,
    get_temperature_plot,
    get_humidity_plot,
    get_scatter_plot,
    get_anomaly_scatter_plot,
)
from config import DEFAULT_HEAD_ROWS

st.markdown(
    """
    <style>
    .stApp > header {visibility: hidden;}
    .css-18e3th9 {padding-top: 0rem;}
    .css-1outpf7 {padding-top: 1rem;}
    .css-1d391kg {max-width: 95%; margin: 0 auto;}
    .css-1lcbmhc {max-width: 95%; margin: 0 auto;}
    .streamlit-expanderHeader {font-weight: 800; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Weather Anomaly Detection Dashboard")

# Render sidebar
selected_city, show_graphs, show_full_data, date_from, date_to, time_from, time_to = render_sidebar()

if selected_city:
    # Add file-modified time as extra cache key to avoid stale caching when parquet is reprocessed.
    from pathlib import Path
    from config import PROCESSED_DATA_PATH

    parquet_file = PROCESSED_DATA_PATH / f"{selected_city.lower()}.parquet"
    data_mtime = parquet_file.stat().st_mtime if parquet_file.exists() else 0

    # Load data directly (cached internally)
    df = load_data(selected_city, data_mtime)

    if df.empty:
        st.error("No data available for selected city.")
    else:
        # Convert date column to datetime if present
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors='coerce')

        # Apply user-selected date filter
        if date_from is not None and date_to is not None and "date" in df.columns:
            start_date = pd.to_datetime(date_from)
            end_date = pd.to_datetime(date_to)
            df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

        # Apply user-selected time filter
        if time_from is not None and time_to is not None and "date" in df.columns:
            df = df[df["date"].dt.time.between(time_from, time_to)]

        # Top meta / KPIs
        counts = df.shape[0]
        anomaly_counts = int(df["anomaly"].value_counts().get(-1, 0)) if "anomaly" in df.columns else 0
        normal_counts = int(df["anomaly"].value_counts().get(1, 0)) if "anomaly" in df.columns else counts - anomaly_counts
        avg_temp = df["temperature"].mean() if "temperature" in df.columns else None
        avg_hum = df["humidity"].mean() if "humidity" in df.columns else None

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Total records", f"{counts:,}")
        kpi2.metric("Normal points", f"{normal_counts:,}")
        kpi3.metric("Anomalies", f"{anomaly_counts:,}")
        kpi4.metric("Avg temperature", f"{avg_temp:.2f}" if avg_temp is not None else "N/A")
        kpi5.metric("Avg humidity", f"{avg_hum:.2f}" if avg_hum is not None else "N/A")

        st.markdown("---")

        # Dataset overview
        st.subheader("Dataset Overview")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Preview**")
            st.dataframe(df.head(DEFAULT_HEAD_ROWS))
            if show_full_data:
                st.dataframe(df)

        with col2:
            st.markdown("**Summary Statistics**")
            st.dataframe(df.describe())

        st.markdown("---")

        if show_graphs:
            st.header("Weather Analysis")

            # Keep plotting data small for speed
            trimmed = df.tail(2000).reset_index(drop=True)
            temp_col, hum_col = get_plot_columns(trimmed)

            if temp_col is None or hum_col is None:
                st.warning("Required temperature or humidity columns not found in dataset.")
            else:
                temp_series = tuple(trimmed[temp_col].astype('float32', errors='ignore').tolist())
                hum_series = tuple(trimmed[hum_col].astype('float32', errors='ignore').tolist())
                time_series = tuple(pd.to_datetime(trimmed["date"], errors='coerce').tolist()) if "date" in trimmed.columns else tuple()
                anomaly_series = tuple(trimmed["anomaly"].map({1: 'Normal', -1: 'Anomaly'}).fillna('Normal').tolist()) if "anomaly" in trimmed.columns else tuple()

                # line charts side-by-side
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Temperature Over Time")
                    temperature_fig = get_temperature_plot(temp_col, temp_series, selected_city, time_series)
                    if temperature_fig is not None:
                        st.plotly_chart(temperature_fig, use_container_width=True)

                with c2:
                    st.subheader("Humidity Over Time")
                    humidity_fig = get_humidity_plot(hum_col, hum_series, selected_city, time_series)
                    if humidity_fig is not None:
                        st.plotly_chart(humidity_fig, use_container_width=True)

                # scatter and anomaly side-by-side with equal width
                c3, c4 = st.columns([1, 1])
                with c3:
                    st.subheader("Temperature vs Humidity")
                    scatter_fig = get_scatter_plot(temp_col, hum_col, temp_series, hum_series, selected_city)
                    if scatter_fig is not None:
                        st.plotly_chart(scatter_fig, use_container_width=True)

                st.markdown("---")

                if "anomaly" in df.columns:
                    with c4:
                        st.subheader("Anomaly Scan")
                        anomaly_fig = get_anomaly_scatter_plot(temp_col, hum_col, temp_series, hum_series, anomaly_series, selected_city)
                        if anomaly_fig is not None:
                            st.plotly_chart(anomaly_fig, use_container_width=True)
                        st.metric("Normal count", f"{normal_counts:,}")
                        st.metric("Anomaly count", f"{anomaly_counts:,}")
                        st.caption("Anomalies marked by IsolationForest: -1 = anomaly, 1 = normal")

                        st.markdown("#### Anomaly sample rows")
                        anomaly_rows = trimmed[trimmed["anomaly"] == -1]
                        if not anomaly_rows.empty:
                            display_cols = [c for c in ["date", temp_col, hum_col, "wind_speed", "temp_change", "humidity_change", "is_anomaly"] if c in anomaly_rows.columns]
                            st.dataframe(anomaly_rows[display_cols].head(10).reset_index(drop=True))
                        else:
                            st.write("No anomaly rows in the plotted subset.")
