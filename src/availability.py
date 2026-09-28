"""
Water Availability Estimation Module for Q-WATER
Aggregates regional measurements from Datasets 3, 4, and 5 and computes a transparent
normalized water availability scale for optimization.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# Reference baseline parameters for normalization into common allocation scale
# These baselines reflect regional operating regimes in Andhra Pradesh command areas
BASELINE_RAINFALL_MM = 150.0       # Regional reference monthly/telemetry rainfall baseline
BASELINE_RESERVOIR_MCM = 1000.0    # Reference live storage capacity benchmark (Krishna reservoirs)
BASELINE_DISCHARGE_M3S = 100.0     # Reference canal/river conveyance discharge benchmark

# Target allocation pool scale (in Normalized Allocation Units, matching typical agricultural command zone needs)
TARGET_SCALE_MAX = 120.0


def calculate_water_availability(
    rainfall_df: Optional[pd.DataFrame],
    reservoir_df: Optional[pd.DataFrame],
    discharge_df: Optional[pd.DataFrame],
    region_name: str = "Krishna",
    weight_rain: float = 0.30,
    weight_res: float = 0.45,
    weight_dis: float = 0.25
) -> Dict[str, Any]:
    """
    Computes regional water availability from verified CSV data.
    Preserves original engineering units:
      - Rainfall: mm
      - Reservoir Storage: mcm (million cubic meters)
      - River Discharge: m3/sec
    
    Transforms each via non-dimensional indexing into a composite Normalized Availability (AU).
    """
    # 1. Rainfall (mm)
    rf_col = "Telemetry Hourly Rainfall (mm)"
    if rainfall_df is not None and len(rainfall_df) > 0 and rf_col in rainfall_df.columns:
        mean_rainfall_mm = float(rainfall_df[rf_col].mean())
        std_rainfall_mm = float(rainfall_df[rf_col].std())
        max_rainfall_mm = float(rainfall_df[rf_col].max())
        count_rf = len(rainfall_df)
    else:
        mean_rainfall_mm = 0.0
        std_rainfall_mm = 0.0
        max_rainfall_mm = 0.0
        count_rf = 0

    # 2. Reservoir Storage (mcm)
    res_col = "Manual Daily Reservoir storage (mcm)"
    is_godavari = "godavari" in region_name.lower() and "krishna" not in region_name.lower()
    
    if reservoir_df is not None and len(reservoir_df) > 0 and res_col in reservoir_df.columns:
        mean_res_mcm = float(reservoir_df[res_col].mean())
        std_res_mcm = float(reservoir_df[res_col].std())
        max_res_mcm = float(reservoir_df[res_col].max())
        count_res = len(reservoir_df)
        has_res_data = True
    else:
        mean_res_mcm = 0.0
        std_res_mcm = 0.0
        max_res_mcm = 0.0
        count_res = 0
        has_res_data = False

    # 3. River Discharge (m3/sec)
    dis_col = "Manual Daily River Water Discharge (m3/sec)"
    if discharge_df is not None and len(discharge_df) > 0 and dis_col in discharge_df.columns:
        mean_dis_m3s = float(discharge_df[dis_col].mean())
        std_dis_m3s = float(discharge_df[dis_col].std())
        max_dis_m3s = float(discharge_df[dis_col].max())
        count_dis = len(discharge_df)
    else:
        mean_dis_m3s = 0.0
        std_dis_m3s = 0.0
        max_dis_m3s = 0.0
        count_dis = 0

    # Non-dimensional indices in [0, 1]
    idx_rain = float(np.clip(mean_rainfall_mm / BASELINE_RAINFALL_MM, 0.0, 1.5))
    idx_dis = float(np.clip(mean_dis_m3s / BASELINE_DISCHARGE_M3S, 0.0, 1.5))

    # Weighting logic when reservoir data is absent (e.g. Godavari)
    if not has_res_data:
        idx_res = 0.0
        # Re-normalize weights between rain and discharge
        tot_w = weight_rain + weight_dis
        eff_w_rain = weight_rain / tot_w if tot_w > 0 else 0.5
        eff_w_res = 0.0
        eff_w_dis = weight_dis / tot_w if tot_w > 0 else 0.5
        res_note = (
            "No reservoir data in Dataset 4 for Godavari. "
            "Availability weights reallocated entirely to rainfall and river discharge."
        )
    else:
        idx_res = float(np.clip(mean_res_mcm / BASELINE_RESERVOIR_MCM, 0.0, 1.5))
        eff_w_rain = weight_rain
        eff_w_res = weight_res
        eff_w_dis = weight_dis
        res_note = "Krishna reservoir storage records active."

    # Composite availability index in [0, 1.5]
    composite_index = (
        eff_w_rain * idx_rain
        + eff_w_res * idx_res
        + eff_w_dis * idx_dis
    )

    # Scaled total availability in Normalized Allocation Units (AU)
    # Typical baseline: ~60 to ~100 AU
    total_normalized_availability = float(np.round(
        np.clip(composite_index * 80.0 + 15.0, 15.0, TARGET_SCALE_MAX), 2
    ))

    explanation = (
        f"Normalization Methodology:\n"
        f"1. Rainfall Index: I_rain = min(1.5, {mean_rainfall_mm:.1f} mm / {BASELINE_RAINFALL_MM} mm) = {idx_rain:.3f} (weight: {eff_w_rain:.2f})\n"
        f"2. Reservoir Index: I_res = min(1.5, {mean_res_mcm:.1f} mcm / {BASELINE_RESERVOIR_MCM} mcm) = {idx_res:.3f} (weight: {eff_w_res:.2f})\n"
        f"3. Discharge Index: I_dis = min(1.5, {mean_dis_m3s:.1f} m3/s / {BASELINE_DISCHARGE_M3S} m3/s) = {idx_dis:.3f} (weight: {eff_w_dis:.2f})\n"
        f"Composite Index = {composite_index:.3f}\n"
        f"Normalized Availability = (Composite * 80.0) + 15.0 = {total_normalized_availability:.1f} AU"
    )

    return {
        "region": region_name,
        "rainfall_mean_mm": round(mean_rainfall_mm, 2),
        "rainfall_std_mm": round(std_rainfall_mm, 2),
        "rainfall_records": count_rf,
        "rainfall_index": round(idx_rain, 3),
        "reservoir_mean_mcm": round(mean_res_mcm, 2),
        "reservoir_std_mcm": round(std_res_mcm, 2),
        "reservoir_records": count_res,
        "reservoir_index": round(idx_res, 3),
        "has_reservoir_data": has_res_data,
        "reservoir_note": res_note,
        "discharge_mean_m3s": round(mean_dis_m3s, 2),
        "discharge_std_m3s": round(std_dis_m3s, 2),
        "discharge_records": count_dis,
        "discharge_index": round(idx_dis, 3),
        "weights": {
            "rain": round(eff_w_rain, 2),
            "reservoir": round(eff_w_res, 2),
            "discharge": round(eff_w_dis, 2)
        },
        "composite_index": round(composite_index, 3),
        "total_normalized_availability": total_normalized_availability,
        "explanation": explanation
    }
