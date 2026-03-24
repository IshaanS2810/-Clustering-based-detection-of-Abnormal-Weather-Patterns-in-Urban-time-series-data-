import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH, ISOLATION_FOREST_PARAMS
from utils.clustering import perform_kmeans

def get_relevant_features(df: pd.DataFrame) -> list:
    """
    Select only relevant features for clustering and anomaly detection.
    Ignores irrelevant columns like timestamps, ids, etc.
    """
    relevant_columns = ["tempc", "humidity", "pressure", "windspeedkmph"]
    available_features = [col for col in relevant_columns if col in df.columns]
    return available_features

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values using forward fill, backward fill, then mean.
    """
    df = df.copy()

    # Forward fill
    df = df.ffill()

    # Backward fill
    df = df.bfill()

    # For any remaining NaN, use mean of numeric columns
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    return df

def preprocess_single_file(csv_path: Path) -> pd.DataFrame:
    """Preprocess a single CSV file."""
    df = pd.read_csv(csv_path)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    # Convert numeric columns
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    df[numeric_cols] = df[numeric_cols].astype('float32', errors='ignore')

    # Handle missing values using improved method
    df = handle_missing_values(df)

    return df

def add_anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Add anomaly detection, clustering, and hybrid anomaly columns."""
    df = df.copy()

    # Get relevant features
    features = get_relevant_features(df)
    if not features:
        df["anomaly"] = 1  # Default normal
        df["cluster"] = -1
        df["cluster_anomaly"] = 1
        df["final_anomaly"] = 1
        return df

    # Isolation Forest anomaly detection
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    model = IsolationForest(**ISOLATION_FOREST_PARAMS)
    df["anomaly"] = model.fit_predict(X_scaled)

    # K-Means clustering (includes cluster_anomaly assignment inside perform_kmeans)
    df = perform_kmeans(df, features, n_clusters=2)

    # Ensure cluster_anomaly exists (for backward compatibility)
    if "cluster_anomaly" not in df.columns:
        cluster_counts = df["cluster"].value_counts()
        threshold = max(int(len(df) * 0.03), 1)
        anomalous_clusters = cluster_counts[cluster_counts < threshold].index
        df["cluster_anomaly"] = df["cluster"].apply(lambda x: -1 if x in anomalous_clusters else 1)

    # Hybrid anomaly: Isolation Forest OR cluster anomaly
    df["final_anomaly"] = np.where((df["anomaly"] == -1) | (df["cluster_anomaly"] == -1), -1, 1)

    return df

def main():
    """Main preprocessing function."""
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

    for csv_file in RAW_DATA_PATH.glob("*.csv"):
        print(f"Processing {csv_file.name}")
        df = preprocess_single_file(csv_file)
        df = add_anomaly_detection(df)

        city_name = csv_file.stem.lower()
        output_path = PROCESSED_DATA_PATH / f"{city_name}.parquet"
        df.to_parquet(output_path, index=False)
        print(f"Saved {output_path}")

if __name__ == "__main__":
    main()