"""
Preprocessing Pipeline for Q-WATER
Provides transparent data validation, date conversion, cleaning, missing value handling,
duplicate removal, and region filtering for Krishna and Godavari basins.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = {
    "dataset1": [
        "N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label",
        "soil_moisture", "crop_density", "growth_stage", "water_usage_efficiency",
        "irrigation_frequency"
    ],
    "dataset2": [
        "N", "P", "K", "temperature", "humidity", "ph", "rainfall", "crop"
    ],
    "dataset3": [
        "Station", "District", "River", "Basin", "Data Acquisition Time",
        "Telemetry Hourly Rainfall (mm)"
    ],
    "dataset4": [
        "Station", "District", "River", "Basin", "Data Acquisition Time",
        "Manual Daily Reservoir storage (mcm)"
    ],
    "dataset5": [
        "Station", "District", "River", "Basin", "Data Acquisition Time",
        "Manual Daily River Water Discharge (m3/sec)"
    ],
}

MEASUREMENT_COLUMNS = {
    "dataset3": "Telemetry Hourly Rainfall (mm)",
    "dataset4": "Manual Daily Reservoir storage (mcm)",
    "dataset5": "Manual Daily River Water Discharge (m3/sec)",
}


def validate_columns(df: pd.DataFrame, dataset_name: str) -> Tuple[bool, List[str]]:
    """
    Validates that required columns exist in the DataFrame.
    Returns (is_valid, missing_columns).
    """
    required = REQUIRED_COLUMNS.get(dataset_name, [])
    missing = [c for c in required if c not in df.columns]
    return (len(missing) == 0), missing


def convert_dates(
    df: pd.DataFrame,
    date_col: str = "Data Acquisition Time",
    date_format: str = "%d-%m-%Y %H:%M"
) -> Tuple[pd.DataFrame, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Converts a date column to datetime.
    Returns (updated_df, min_date, max_date).
    """
    if date_col not in df.columns:
        return df, None, None

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors="coerce")
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    return df, min_date, max_date


def remove_duplicates(df: pd.DataFrame, dataset_name: str) -> Tuple[pd.DataFrame, int]:
    """
    Identifies and removes duplicate records.
    Returns (deduped_df, duplicates_count).
    """
    dupes_count = int(df.duplicated().sum())
    if dupes_count > 0:
        cleaned_df = df.drop_duplicates().copy()
    else:
        cleaned_df = df.copy()
    return cleaned_df, dupes_count


def handle_missing_values(
    df: pd.DataFrame,
    dataset_name: str,
    action: str = "drop"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Transparently handles missing values and sensor anomalies.
    - Dataset 4 (Reservoir): 8 missing values in mcm.
    - Dataset 5 (Discharge): 1 missing value in m3/sec.
    - Dataset 3 (Rainfall): 49 negative telemetry anomaly values.
    Returns (cleaned_df, report_dict).
    """
    df = df.copy()
    initial_rows = len(df)
    total_missing = int(df.isna().sum().sum())
    missing_by_col = df.isna().sum().to_dict()
    missing_by_col = {k: int(v) for k, v in missing_by_col.items() if v > 0}

    anomalies_flagged = 0
    anomalies_desc = ""

    # Specific handling for rainfall telemetry sensor errors
    if dataset_name == "dataset3" and "Telemetry Hourly Rainfall (mm)" in df.columns:
        rf_col = "Telemetry Hourly Rainfall (mm)"
        neg_mask = df[rf_col] < 0
        anomalies_flagged = int(neg_mask.sum())
        if anomalies_flagged > 0:
            anomalies_desc = (
                f"Flagged {anomalies_flagged} negative telemetry readings (< 0 mm, sensor glitch codes). "
                f"Clipped negative values to 0.0 mm."
            )
            df.loc[neg_mask, rf_col] = 0.0

    # Specific handling for measurement missing values in Dataset 4 and 5
    val_col = MEASUREMENT_COLUMNS.get(dataset_name)
    rows_dropped = 0
    if val_col and val_col in df.columns:
        val_missing = int(df[val_col].isna().sum())
        if val_missing > 0:
            if action == "drop":
                df = df.dropna(subset=[val_col]).copy()
                rows_dropped = val_missing
                anomalies_desc += (
                    f" Removed {val_missing} row(s) with missing '{val_col}' measurements transparently."
                )
            elif action == "median":
                med_val = df[val_col].median()
                df[val_col] = df[val_col].fillna(med_val)
                anomalies_desc += (
                    f" Imputed {val_missing} row(s) of '{val_col}' using historical median ({med_val:.2f})."
                )

    report = {
        "initial_rows": initial_rows,
        "final_rows": len(df),
        "total_missing_cells": total_missing,
        "missing_by_col": missing_by_col,
        "rows_dropped": rows_dropped,
        "anomalies_flagged": anomalies_flagged,
        "notes": anomalies_desc if anomalies_desc else "No missing measurement values found."
    }

    return df, report


def clean_dataset(df: pd.DataFrame, dataset_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the full preprocessing pipeline on a single dataset:
    1. Column validation
    2. Missing value handling
    3. Duplicate removal
    4. Date conversion (if applicable)
    Returns (cleaned_df, audit_report).
    """
    is_valid, missing_cols = validate_columns(df, dataset_name)
    if not is_valid:
        raise ValueError(
            f"Dataset '{dataset_name}' is missing required columns: {missing_cols}"
        )

    # 1. Handle missing values
    cleaned_df, missing_report = handle_missing_values(df, dataset_name)

    # 2. Remove duplicates
    cleaned_df, dupes_count = remove_duplicates(cleaned_df, dataset_name)

    # 3. Convert dates if present
    date_col = "Data Acquisition Time"
    min_date, max_date = None, None
    if date_col in cleaned_df.columns:
        cleaned_df, min_date, max_date = convert_dates(cleaned_df, date_col=date_col)

    audit_report = {
        "dataset_name": dataset_name,
        "original_rows": len(df),
        "cleaned_rows": len(cleaned_df),
        "num_columns": len(cleaned_df.columns),
        "columns": list(cleaned_df.columns),
        "missing_values": missing_report["total_missing_cells"],
        "missing_details": missing_report["missing_by_col"],
        "duplicates_removed": dupes_count,
        "date_range": (
            f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
            if min_date is not None and pd.notna(min_date) else "N/A"
        ),
        "cleaning_notes": missing_report["notes"]
    }

    return cleaned_df, audit_report


def filter_by_region(
    df: pd.DataFrame,
    region: str,
    dataset_key: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Filters hydrological datasets (rainfall, reservoir, discharge) by selected region:
    'Krishna', 'Godavari', or 'Combined Krishna–Godavari'.

    CRITICAL INTEGRITY RULES:
    1. Dataset 4 (Reservoir) contains ONLY Krishna records.
       If Godavari is selected, returns an EMPTY DataFrame with an explicit notice.
       DO NOT claim or fabricate Godavari reservoir data!
    2. Datasets 3 and 5 contain verified Krishna and Godavari records filtered via Basin/River.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(), {"stations": [], "record_count": 0, "notice": "Empty input dataset"}

    region_clean = region.strip()
    basin_col = "Basin" if "Basin" in df.columns else None
    river_col = "River" if "River" in df.columns else None

    # Handle Dataset 4: Reservoir
    if dataset_key == "dataset4":
        if region_clean == "Godavari":
            empty_df = df.iloc[0:0].copy()
            meta = {
                "region": region_clean,
                "stations": [],
                "record_count": 0,
                "has_data": False,
                "notice": (
                    "CRITICAL NOTICE: Dataset 4 (AP Reservoir Storage) does NOT contain Godavari "
                    "reservoir records. Genuine reservoir data exists only for Krishna basin projects "
                    "(Srisailam, Sunkesula, Bhairavanitippa). In accordance with project integrity rules, "
                    "no fake Godavari reservoir data is generated."
                )
            }
            return empty_df, meta
        else:
            # Krishna or Combined: filter for Krishna in Basin or River
            mask = pd.Series(False, index=df.index)
            if basin_col:
                mask = mask | df[basin_col].str.contains("Krishna", case=False, na=False)
            if river_col:
                mask = mask | df[river_col].str.contains("Krishna", case=False, na=False)
            
            filtered = df[mask].copy()
            stations = sorted(filtered["Station"].dropna().unique().tolist()) if "Station" in filtered.columns else []
            meta = {
                "region": region_clean,
                "stations": stations,
                "record_count": len(filtered),
                "has_data": len(filtered) > 0,
                "notice": (
                    "Verified Krishna basin reservoir storage data loaded. "
                    + ("(Combined mode: Only Krishna reservoir data is available in Dataset 4)."
                       if "Combined" in region_clean else "")
                )
            }
            return filtered, meta

    # Handle Datasets 3 (Rainfall) and 5 (Discharge)
    mask = pd.Series(False, index=df.index)
    if region_clean == "Krishna":
        if basin_col:
            mask = mask | df[basin_col].str.contains("Krishna", case=False, na=False)
        if river_col:
            mask = mask | df[river_col].str.contains("Krishna", case=False, na=False)
    elif region_clean == "Godavari":
        if basin_col:
            mask = mask | df[basin_col].str.contains("Godavari", case=False, na=False)
        if river_col:
            mask = mask | df[river_col].str.contains("Godavari", case=False, na=False)
    else:  # Combined Krishna–Godavari
        if basin_col:
            mask = mask | df[basin_col].str.contains("Krishna|Godavari", case=False, na=False)
        if river_col:
            mask = mask | df[river_col].str.contains("Krishna|Godavari", case=False, na=False)

    filtered = df[mask].copy()
    stations = sorted(filtered["Station"].dropna().unique().tolist()) if "Station" in filtered.columns else []
    
    meta = {
        "region": region_clean,
        "stations": stations,
        "record_count": len(filtered),
        "has_data": len(filtered) > 0,
        "notice": f"Filtered {len(filtered):,} records across {len(stations)} station(s) in {region_clean} region."
    }
    return filtered, meta
