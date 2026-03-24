import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def perform_kmeans(df: pd.DataFrame, features: list, n_clusters: int = 2) -> pd.DataFrame:
    """
    Perform K-Means clustering on the given features using sampling for large datasets.

    Args:
        df: Input dataframe
        features: List of feature column names to use for clustering
        n_clusters: Number of clusters to create

    Returns:
        DataFrame with added 'cluster' column
    """
    df = df.copy()

    if not features or df.empty:
        df["cluster"] = -1  # Default cluster for empty/invalid data
        return df

    # Select features and handle any remaining missing values
    X = df[features].copy()
    X = X.fillna(X.mean())  # Fallback for any missing values

    # For large datasets, sample for training to improve performance
    sample_size = min(50000, len(X))  # Sample up to 50k points for training
    if len(X) > sample_size:
        sample_indices = X.sample(sample_size, random_state=42).index
        X_sample = X.loc[sample_indices]
    else:
        X_sample = X

    # Scale features
    scaler = StandardScaler()
    scaled_sample = scaler.fit_transform(X_sample)
    scaled_data = scaler.transform(X)  # Transform all data

    # Use MiniBatchKMeans for better performance
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000, n_init=5, init='random')
    kmeans.fit(scaled_sample)  # Fit on sample
    clusters = kmeans.predict(scaled_data)  # Predict on all data

    # Add cluster column
    df["cluster"] = clusters

    # Determine cluster-level anomalies by size (small clusters are anomalies)
    cluster_counts = df["cluster"].value_counts()
    threshold = max(int(len(df) * 0.03), 1)  # 3% of dataset as small-cluster cutoff
    anomalous_clusters = cluster_counts[cluster_counts < threshold].index

    if len(anomalous_clusters) > 0:
        df["cluster_anomaly"] = df["cluster"].apply(lambda x: -1 if x in anomalous_clusters else 1)
    else:
        # fallback: farthest 3% examples as anomalies by distance to centroid
        dist = kmeans.transform(scaled_data)
        own_dist = pd.Series(dist[np.arange(len(df)), clusters], index=df.index)
        cutoff = own_dist.quantile(0.97)
        df["cluster_anomaly"] = own_dist.apply(lambda d: -1 if d >= cutoff else 1)

    # Add metadata for analysis
    df.attrs['kmeans_silhouette'] = silhouette_score(scaled_data, clusters) if len(set(clusters)) > 1 else None
    df.attrs['kmeans_centers'] = kmeans.cluster_centers_

    return df