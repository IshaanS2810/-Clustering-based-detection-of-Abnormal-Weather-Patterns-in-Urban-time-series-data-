import streamlit as st
from utils.loader import list_datasets, load_data
from datetime import datetime, date, time
from pathlib import Path
from config import PROCESSED_DATA_PATH
import pandas as pd

def render_sidebar():
    """Render the sidebar with city selection and controls."""
    st.sidebar.title("Weather Anomaly Detection")

    # City selection
    cities = list_datasets()
    if not cities:
        st.sidebar.error("No processed datasets found in data/processed/. Run preprocess_data.py first.")
        return None, False, False, None, None, None, None

    selected_city = st.sidebar.selectbox("Select City", cities, key="city_select")

    # Determine dataset date bounds if city is selected
    min_date = date(2000, 1, 1)
    max_date = date(2100, 1, 1)

    if selected_city:
        parquet_file = PROCESSED_DATA_PATH / f"{selected_city.lower()}.parquet"
        data_mtime = parquet_file.stat().st_mtime if parquet_file.exists() else 0
        df = load_data(selected_city, data_mtime)
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            if not df["date"].isna().all():
                min_date = df["date"].min().date()
                max_date = df["date"].max().date()

    # Toggles
    show_graphs = st.sidebar.checkbox("Show Graphs", value=False, key="show_graphs")
    show_full_data = st.sidebar.checkbox("Show Full Dataset", value=False, key="show_full")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Date + Time Filters")

    # Date filter inputs (from start/end dates) - dataset bounds
    date_from_default = st.session_state.get("date_from", min_date)
    date_to_default = st.session_state.get("date_to", max_date)
    if date_from_default < min_date or date_from_default > max_date:
        date_from_default = min_date
    if date_to_default < min_date or date_to_default > max_date:
        date_to_default = max_date

    date_cols = st.sidebar.columns(2)
    with date_cols[0]:
        date_from = st.date_input(
            "Start date",
            min_value=min_date,
            max_value=max_date,
            value=date_from_default,
            key="date_from",
        )
    with date_cols[1]:
        date_to = st.date_input(
            "End date",
            min_value=min_date,
            max_value=max_date,
            value=date_to_default,
            key="date_to",
        )

    if date_from > date_to:
        st.sidebar.error("Start date cannot be after end date. Please correct the selection.")

    time_cols = st.sidebar.columns(2)
    with time_cols[0]:
        time_from = st.time_input("Start time", value=time(0, 0), key="time_from")
    with time_cols[1]:
        time_to = st.time_input("End time", value=time(23, 59), key="time_to")

    if time_from is not None and time_to is not None and time_from > time_to:
        st.sidebar.error("Start time cannot be after end time. Please correct the selection.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Detection Method")

    # Algorithm selection dropdown
    algorithm = st.sidebar.selectbox(
        "Select Detection Method",
        ["Isolation Forest", "KMeans", "Hybrid"],
        index=2,  # Default to Hybrid
        key="algorithm"
    )

    return selected_city, show_graphs, show_full_data, date_from, date_to, time_from, time_to, algorithm