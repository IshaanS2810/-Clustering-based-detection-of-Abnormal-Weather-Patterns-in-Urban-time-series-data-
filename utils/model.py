import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from config import ISOLATION_FOREST_PARAMS

@st.cache_resource
def get_model():
    """Get cached IsolationForest model."""
    return IsolationForest(**ISOLATION_FOREST_PARAMS)

def run_anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Run anomaly detection on the dataframe and add anomaly column."""
    if df.empty:
        return df

    features = df.select_dtypes(include=[float, int])
    if features.empty:
        return df

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    model = get_model()
    df = df.copy()
    df["anomaly"] = model.fit_predict(scaled)
    return df