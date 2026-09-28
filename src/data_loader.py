"""
Data Loader Module for Q-WATER
Handles loading and configuring paths for all 5 CSV datasets.
"""

import os
from typing import Dict, Optional, Tuple
import pandas as pd

DEFAULT_DATASET_PATHS = {
    "dataset1": "data/dataset1.csv",
    "dataset2": "data/dataset2.csv",
    "dataset3": "data/dataset3.csv",
    "dataset4": "data/dataset4.csv",
    "dataset5": "data/dataset5.csv",
}

DATASET_METADATA = {
    "dataset1": {
        "name": "Dataset 1: Agriculture & Crop Data",
        "description": "Crop soil conditions, irrigation frequency, growth stages, crop density, water efficiency (22 crops).",
        "expected_rows": 2200,
        "primary_key": "label",
        "fallback_file": "smart.csv"
    },
    "dataset2": {
        "name": "Dataset 2: Crop & Weather Supporting Data",
        "description": "Supporting crop and climate reference conditions (7 crops).",
        "expected_rows": 1000,
        "primary_key": "crop",
        "fallback_file": "crop.csv"
    },
    "dataset3": {
        "name": "Dataset 3: AP Rainfall Telemetry",
        "description": "Hourly rainfall telemetry across Andhra Pradesh stations including Krishna and Godavari basins (mm).",
        "expected_rows": 64510,
        "primary_key": "Station",
        "fallback_file": "rainfall.csv"
    },
    "dataset4": {
        "name": "Dataset 4: AP Reservoir Storage",
        "description": "Daily reservoir storage measurements for major projects in Krishna basin (mcm).",
        "expected_rows": 98888,
        "primary_key": "Station",
        "fallback_file": "reservoir.csv"
    },
    "dataset5": {
        "name": "Dataset 5: AP River Discharge",
        "description": "Daily river discharge measurements across Krishna and Godavari river stations (m3/sec).",
        "expected_rows": 99411,
        "primary_key": "Station",
        "fallback_file": "discharge.csv"
    },
}


def get_default_paths() -> Dict[str, str]:
    """Returns a copy of the default paths dictionary."""
    return DEFAULT_DATASET_PATHS.copy()


def resolve_filepath(path: str, fallback_filename: Optional[str] = None) -> str:
    """
    Checks if a path exists. If not, checks root directory or fallback.
    """
    if os.path.exists(path):
        return path
    
    # Check if file exists in current working directory
    basename = os.path.basename(path)
    if os.path.exists(basename):
        return basename
    
    if fallback_filename and os.path.exists(fallback_filename):
        return fallback_filename

    fallback_data_path = os.path.join("data", fallback_filename) if fallback_filename else None
    if fallback_data_path and os.path.exists(fallback_data_path):
        return fallback_data_path
    
    return path


def load_csv_dataset(
    filepath: str,
    dataset_key: Optional[str] = None,
    nrows: Optional[int] = None
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Safely load a CSV file using pandas.
    Returns (DataFrame, error_message).
    """
    fallback = DATASET_METADATA.get(dataset_key, {}).get("fallback_file") if dataset_key else None
    resolved = resolve_filepath(filepath, fallback)
    
    if not os.path.exists(resolved):
        return None, f"File not found: '{filepath}'. Please check the data configuration."

    try:
        df = pd.read_csv(resolved, nrows=nrows)
        return df, None
    except Exception as e:
        return None, f"Error reading '{resolved}': {str(e)}"


def load_all_datasets(paths: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """
    Loads all 5 configured datasets.
    Returns:
        datasets: Dict[str, pd.DataFrame] containing loaded dataframes
        errors: Dict[str, str] containing error messages for failed files
    """
    if paths is None:
        paths = get_default_paths()

    datasets = {}
    errors = {}

    for key, path in paths.items():
        df, err = load_csv_dataset(path, dataset_key=key)
        if err:
            errors[key] = err
        else:
            datasets[key] = df

    return datasets, errors
