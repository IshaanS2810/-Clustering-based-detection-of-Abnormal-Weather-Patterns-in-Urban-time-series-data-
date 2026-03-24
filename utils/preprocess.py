import pandas as pd
from pathlib import Path
from config import PROCESSED_DATA_PATH

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names: strip, lowercase."""
    df.columns = df.columns.str.strip().str.lower()
    return df

def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric columns to float32 where possible."""
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32', errors='ignore')
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in numeric columns with 0."""
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols):
        df[numeric_cols] = df[numeric_cols].fillna(0)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline."""
    if df is None or df.empty:
        return df
    df = clean_column_names(df)
    df = convert_numeric_columns(df)
    df = handle_missing_values(df)
    return df

def save_processed_data(df: pd.DataFrame, city: str):
    """Save processed dataframe as parquet."""
    if df is None:
        raise ValueError("DataFrame must not be None")
    if not isinstance(city, str) or not city.strip():
        raise ValueError("City name must be a non-empty string")
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    parquet_path = PROCESSED_DATA_PATH / f"{city.lower().strip()}.parquet"
    df.to_parquet(parquet_path, index=False)