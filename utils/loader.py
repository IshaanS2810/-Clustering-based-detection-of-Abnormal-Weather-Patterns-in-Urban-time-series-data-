import pandas as pd
import streamlit as st
from pathlib import Path
from config import PROCESSED_DATA_PATH

@st.cache_data(show_spinner=False)
def list_datasets():
    """List all available processed datasets."""
    cities = []
    for file in PROCESSED_DATA_PATH.glob("*.parquet"):
        city = file.stem.title()
        cities.append(city)
    return sorted(cities)

@st.cache_data(show_spinner=False)
def load_data(city: str, data_mtime: float) -> pd.DataFrame:
    """Load processed data for a city via cached parquet read with modification timestamp."""
    city_lower = city.lower()
    parquet_path = PROCESSED_DATA_PATH / f"{city_lower}.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def get_dataset_mapping() -> dict:
    """Return mapping of city names to parquet paths."""
    mapping = {}
    for file in PROCESSED_DATA_PATH.glob("*.parquet"):
        mapping[file.stem.title()] = file
    return mapping