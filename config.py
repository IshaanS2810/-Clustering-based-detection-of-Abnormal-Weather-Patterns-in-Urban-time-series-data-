import os
from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).resolve().parent

# Data paths
RAW_DATA_PATH = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed"

# Model parameters
ISOLATION_FOREST_PARAMS = {
    "contamination": "auto",
    "random_state": 42,
    "n_estimators": 100
}

# UI settings
DEFAULT_HEAD_ROWS = 100
DOWNSAMPLE_STEP = 10