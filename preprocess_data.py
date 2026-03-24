import pandas as pd
import os
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH, ISOLATION_FOREST_PARAMS

def preprocess_single_file(csv_path: Path) -> pd.DataFrame:
    """Preprocess a single CSV file."""
    df = pd.read_csv(csv_path)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    # Convert numeric columns
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    df[numeric_cols] = df[numeric_cols].astype('float32', errors='ignore')

    # Handle missing values
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df

def add_anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Add anomaly detection column."""
    features = df.select_dtypes(include=[float, int])
    if features.empty:
        return df

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    model = IsolationForest(**ISOLATION_FOREST_PARAMS)
    df = df.copy()
    df["anomaly"] = model.fit_predict(scaled)
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