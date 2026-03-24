import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from preprocess_data import preprocess_single_file, get_relevant_features


def process_city(csv_path: Path, output_dir: Path) -> None:
    """Process one city's raw CSV and save processed parquet with anomaly results."""
    city = csv_path.stem
    print(f"Processing city: {city}...")

    # 1) Load & preprocess raw data (cleaning, missing values, feature selection)
    df = preprocess_single_file(csv_path)

    features = get_relevant_features(df)
    if not features:
        raise ValueError(f"No relevant features found in {city} after preprocessing.")

    # 2) Isolation Forest anomaly detection
    scaler_if = StandardScaler()
    X_if = scaler_if.fit_transform(df[features])
    if_model = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly"] = if_model.fit_predict(X_if)

    # 3) KMeans clustering
    scaler_km = StandardScaler()
    X_km = scaler_km.fit_transform(df[features])

    km_model = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster"] = km_model.fit_predict(X_km)
    centroids = km_model.cluster_centers_

    # 4) Distance-based KMeans anomaly
    distances = cdist(X_km, centroids, metric="euclidean")
    min_distances = distances.min(axis=1)
    threshold = np.percentile(min_distances, 95)
    df["kmeans_anomaly"] = np.where(min_distances > threshold, -1, 1)

    # 5) Hybrid anomaly logic
    df["final_anomaly"] = np.where(
        (df["anomaly"] == -1) | (df["kmeans_anomaly"] == -1), -1, 1
    )

    # 6) Save results to parquet
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{city}.parquet"
    df.to_parquet(out_file, index=False)

    # 7) Logging counts
    count_if = int((df["anomaly"] == -1).sum())
    count_km = int((df["kmeans_anomaly"] == -1).sum())
    count_final = int((df["final_anomaly"] == -1).sum())

    print(
        f"{city}: rows={len(df)}, IF={count_if}, KM={count_km}, FINAL={count_final}"
    )


def main() -> None:
    raw_folder = Path("data")
    output_folder = Path("processed_data")

    if not raw_folder.exists():
        raise FileNotFoundError("Raw data folder 'data' not found.")

    csv_files = sorted(raw_folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV files found in 'data' folder.")

    start = time.perf_counter()

    for csv_path in csv_files:
        try:
            process_city(csv_path, output_folder)
        except Exception as exc:
            print(f"Error processing {csv_path.name}: {exc}")
            continue

    elapsed = time.perf_counter() - start
    print(f"Pipeline done in {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
