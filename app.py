import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.title("Weather Anomaly Detection Dashboard")

# dataset folder (relative to this file)
DATA_DIR = Path(__file__).resolve().parent / "dataset"

# helper to list the available dataset CSVs
# (the app was previously using a single static file; now we load from a set of CSVs)
def list_dataset_files(exclude=None):
    exclude = set(exclude or [])
    if not DATA_DIR.exists():
        return []
    files = sorted(DATA_DIR.glob("*.csv"))
    return [p for p in files if p.name not in exclude]

# helper to load a dataset path
def load_csv(path: Path):
    try:
        return pd.read_csv(path)
    except Exception as e:  # includes EmptyDataError, FileNotFoundError
        st.warning(f"Unable to load {path.name}: {e}")
        return pd.DataFrame()

# choose which dataset to use
available_files = list_dataset_files(exclude={"weather.csv", "open-meteo-52.55N13.41E38m.csv"})
if not available_files:
    st.warning("No dataset CSVs found in the dataset folder. Please add dataset files.")
    base_df = pd.DataFrame()
else:
    dataset_names = ["All datasets"] + [p.name for p in available_files]
    selected = st.selectbox("Select dataset", dataset_names, index=0)
    if selected == "All datasets":
        # concatenate all available datasets into one DataFrame
        dfs = [load_csv(p) for p in available_files]
        base_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    else:
        base_df = load_csv(DATA_DIR / selected)

# simple preprocessing for uploaded files (normalize column names, types)
def preprocess_df(df):
    if df.empty:
        return df
    # lowercase and strip spaces from column names
    df = df.rename(columns=lambda c: c.strip().lower())
    # attempt to convert all columns to numeric where possible
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        except Exception:
            pass
    # fill NaNs with 0 for numeric columns
    nums = df.select_dtypes(include=[np.number]).columns
    df[nums] = df[nums].fillna(0)
    return df

# main data-loading logic
# always start with the selected dataset (from the dataset folder)
default_dataset_name = selected if available_files else None

uploaded_file = st.file_uploader("(optional) Upload a custom weather CSV", type=["csv"])
use_uploaded = False
if uploaded_file:
    try:
        uploaded_file.seek(0)
        temp = pd.read_csv(uploaded_file)
        temp = preprocess_df(temp)
        if not temp.empty:
            use_uploaded = True
            df = temp
        else:
            st.error("Uploaded file appears to be empty after preprocessing; using dataset selection instead.")
            df = base_df
    except Exception as e:
        st.error(f"Unable to parse uploaded file: {e}; using selected dataset.")
        df = base_df
else:
    df = base_df

if use_uploaded:
    st.info("Using data from uploaded file (preprocessed).")
else:
    if default_dataset_name:
        st.info(f"Using dataset: {default_dataset_name}")
    else:
        st.info("No dataset available. Upload a CSV or add dataset files to the dataset folder.")

if df.empty:
    st.warning("No data available. Please upload a valid weather CSV file or add datasets to the dataset folder.")
else:
    # use city names as index when available
    if "city" in df.columns:
        df = df.set_index("city")

    # ---------------------------------------------------------
    # Section 1 — Dataset Overview
    # ---------------------------------------------------------
    st.header("1. Dataset Overview")
    st.subheader("Preview")
    st.dataframe(df.head())
    if st.checkbox("Show raw data"):
        st.dataframe(df)

    st.subheader("Summary Statistics")
    st.write(df.describe())

    # ---------------------------------------------------------
    # Section 2 — Weather Analysis
    # ---------------------------------------------------------
    st.header("2. Weather Analysis")
    if "temperature" in df.columns:
        st.subheader("Temperature Trend")
        st.line_chart(df["temperature"])
    if "humidity" in df.columns:
        st.subheader("Humidity Trend")
        st.line_chart(df["humidity"])

    # distribution plots for numeric variables
    st.subheader("Distribution of Numeric Features")
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    if nums:
        fig_dist, axes = plt.subplots(len(nums), 1, figsize=(8, 3*len(nums)))
        if len(nums) == 1:
            axes = [axes]
        for ax, col in zip(axes, nums):
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(col)
        st.pyplot(fig_dist)

    # ---------------------------------------------------------
    # Section 3 — Feature Relationships
    # ---------------------------------------------------------
    st.header("3. Feature Relationships")
    st.subheader("Correlation Heatmap")
    corr = df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # scatter example (temperature vs humidity)
    if "temperature" in df.columns and "humidity" in df.columns:
        st.subheader("Temperature vs Humidity")
        fig_sc, ax_sc = plt.subplots(figsize=(10,6))
        ax_sc.scatter(df["temperature"], df["humidity"], alpha=0.6)
        # annotate each point with city name (index)
        for city, row in df.iterrows():
            ax_sc.text(row["temperature"], row["humidity"], str(city), fontsize=7)
        ax_sc.set_xlabel("Temperature")
        ax_sc.set_ylabel("Humidity")
        st.pyplot(fig_sc)

    # ---------------------------------------------------------
    # Section 4 — Anomaly Detection
    # ---------------------------------------------------------
    st.header("4. Anomaly Detection")
    if st.checkbox("Run anomaly detection"):
        with st.spinner("Computing anomalies..."):
            features = df.select_dtypes(include=[np.number])
            scaler = StandardScaler()
            scaled = scaler.fit_transform(features.fillna(0))
            model = IsolationForest(contamination=0.05, random_state=42)
            df["anomaly"] = model.fit_predict(scaled)

        # display using metrics to avoid repetition later
        counts = df["anomaly"].value_counts().rename(index={1:"normal", -1:"anomaly"})
        st.metric("Normal points", counts.get(1,0))
        st.metric("Anomalous points", counts.get(-1,0))

        st.subheader("Anomaly Scatter (temp vs humidity)")
        if "temperature" in df.columns and "humidity" in df.columns:
            fig2, ax2 = plt.subplots(figsize=(10,6))
            ax2.scatter(df["temperature"], df["humidity"], c=df["anomaly"], cmap="coolwarm")
            # label points with city
            for city, row in df.iterrows():
                ax2.text(row["temperature"], row["humidity"], str(city), fontsize=7)
            ax2.set_xlabel("Temperature")
            ax2.set_ylabel("Humidity")
            st.pyplot(fig2)

        st.subheader("Anomaly Timeline (temperature)")
        if "temperature" in df.columns:
            fig3, ax3 = plt.subplots(figsize=(12,5))
            # if index is city, plot will use it on x-axis automatically
            ax3.plot(df["temperature"], label="Temperature")
            anomalies = df[df["anomaly"] == -1]
            ax3.scatter(anomalies.index, anomalies["temperature"], color="red", label="Anomaly")
            ax3.legend()
            st.pyplot(fig3)

    # ---------------------------------------------------------
    # Section 5 — Summary
    # ---------------------------------------------------------
    st.header("5. Summary")
    # add a little extra spacing for readability
    st.write("\n")
    if "anomaly" in df.columns:
        counts = df["anomaly"].value_counts().rename(index={1:"normal", -1:"anomaly"})
        total = counts.sum()
        anom = counts.get(-1, 0)
        pct = anom / total * 100 if total > 0 else 0
        st.markdown(
            f"""**Total records:** {total}  
**Anomalies detected:** {anom} ({pct:.1f}% of data)"""
        )
        if anom > 0:
            st.markdown("**Sample anomalous rows:**")
            st.dataframe(df[df["anomaly"] == -1].head())
        st.markdown(
            "_These numbers are also shown above; adjust contamination or examine data quality to change them._"
        )
    else:
        st.markdown("No anomaly results yet. Run the detection toggle in Section 4 to generate a summary.")
