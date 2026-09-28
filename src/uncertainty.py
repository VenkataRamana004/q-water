"""
Uncertainty and Robustness Module for Q-WATER
Derives empirical hydrological variance from CSV records and computes robust available water
under Low, Medium, and High inflow uncertainty regimes.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


def compute_historical_variability(
    rainfall_df: Optional[pd.DataFrame],
    discharge_df: Optional[pd.DataFrame]
) -> Dict[str, float]:
    """
    Computes real empirical statistics (mean, std, CV) from historical rainfall and discharge data.
    """
    stats = {
        "rainfall_mean": 0.0,
        "rainfall_std": 0.0,
        "rainfall_cv": 0.30,  # default fallback
        "discharge_mean": 0.0,
        "discharge_std": 0.0,
        "discharge_cv": 0.40,  # default fallback
        "composite_cv": 0.35
    }

    if rainfall_df is not None and len(rainfall_df) > 0:
        rf_col = "Telemetry Hourly Rainfall (mm)"
        if rf_col in rainfall_df.columns:
            rf_clean = rainfall_df[rf_col].dropna()
            m = float(rf_clean.mean())
            s = float(rf_clean.std())
            stats["rainfall_mean"] = m
            stats["rainfall_std"] = s
            stats["rainfall_cv"] = (s / m) if m > 0.001 else 0.30

    if discharge_df is not None and len(discharge_df) > 0:
        dis_col = "Manual Daily River Water Discharge (m3/sec)"
        if dis_col in discharge_df.columns:
            dis_clean = discharge_df[dis_col].dropna()
            m = float(dis_clean.mean())
            s = float(dis_clean.std())
            stats["discharge_mean"] = m
            stats["discharge_std"] = s
            stats["discharge_cv"] = (s / m) if m > 0.001 else 0.40

    stats["composite_cv"] = float(0.5 * stats["rainfall_cv"] + 0.5 * stats["discharge_cv"])
    return stats


def apply_uncertainty_discount(
    estimated_available_water: float,
    uncertainty_level: str = "Medium",
    historical_stats: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Calculates the robust conservative available water:
        robust_available_water = estimated_available_water - uncertainty_penalty

    Transparent Multipliers:
      - Low: Mild precautionary buffer against minor diurnal/telemetry noise.
      - Medium: Standard seasonal risk buffer derived directly from historical CV.
      - High: Extreme drought / delayed monsoon buffer (stress test).
    """
    if historical_stats is None:
        historical_stats = {
            "rainfall_cv": 0.35,
            "discharge_cv": 0.45,
            "composite_cv": 0.40
        }

    comp_cv = historical_stats.get("composite_cv", 0.40)
    # Bound the effective CV multiplier to reasonable agronomic reserve bounds [0.10, 0.70]
    bounded_cv = float(np.clip(comp_cv, 0.20, 0.60))

    level = uncertainty_level.capitalize().strip()

    if level == "Low":
        # 10% - 15% discount
        discount_factor = 0.35 * bounded_cv
        risk_label = "Low (Mild Precautionary Reserve)"
        scenario_desc = "Near-normal inflows expected. Precautionary reserve retained for operational delivery losses."
    elif level == "High":
        # 35% - 50% discount
        discount_factor = 1.25 * bounded_cv
        risk_label = "High (Severe Inflow Deficit / Drought Reserve)"
        scenario_desc = "High hydrological volatility scenario. Heavy buffer withheld to protect essential base demand."
    else:  # Medium default
        # 20% - 30% discount
        discount_factor = 0.75 * bounded_cv
        risk_label = "Medium (Standard Historical Volatility Buffer)"
        scenario_desc = "Inflow variance scaled proportionally to historical seasonal coefficient of variation."

    discount_factor = float(np.clip(discount_factor, 0.05, 0.60))
    uncertainty_penalty = float(round(estimated_available_water * discount_factor, 2))
    robust_available = float(round(max(10.0, estimated_available_water - uncertainty_penalty), 2))

    reduction_pct = float(round((uncertainty_penalty / estimated_available_water) * 100.0, 1)) if estimated_available_water > 0 else 0.0

    formula_str = (
        f"robust_available_water = estimated_available_water - uncertainty_penalty\n"
        f"                       = {estimated_available_water:.1f} AU - {uncertainty_penalty:.1f} AU\n"
        f"                       = {robust_available:.1f} AU (-{reduction_pct}% buffer)"
    )

    return {
        "uncertainty_level": level,
        "risk_label": risk_label,
        "scenario_desc": scenario_desc,
        "discount_factor": round(discount_factor, 3),
        "reduction_pct": reduction_pct,
        "estimated_available_water": estimated_available_water,
        "uncertainty_penalty": uncertainty_penalty,
        "robust_available_water": robust_available,
        "historical_cv": round(comp_cv, 3),
        "formula": formula_str
    }
