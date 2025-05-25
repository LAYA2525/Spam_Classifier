"""
utils.py
General-purpose helpers for loading the SMS Spam Collection dataset
and saving / loading artefacts.
Modify, extend, or replace as you see fit.
"""
from pathlib import Path
import pandas as pd
import joblib

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "spam.csv"


def load_raw_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and normalise column names."""
    df = pd.read_csv(path, encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "message"]
    return df


def save_artifact(obj, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, file_path)


def load_artifact(file_path: Path):
    return joblib.load(file_path)
