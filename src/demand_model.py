"""
Prototype Water Demand Model for Q-WATER
Calculates transparent normalized irrigation water demand scores using Dataset 1.
"""

from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np

# Agronomic reference baseline demand factors by crop category (relative to standard cereal crop)
CROP_BASE_FACTORS = {
    "rice": 1.40,
    "banana": 1.35,
    "watermelon": 1.30,
    "papaya": 1.25,
    "coconut": 1.25,
    "grapes": 1.20,
    "cotton": 1.15,
    "jute": 1.15,
    "coffee": 1.10,
    "apple": 1.10,
    "orange": 1.05,
    "pomegranate": 1.00,
    "maize": 1.00,
    "muskmelon": 0.95,
    "mango": 0.95,
    "kidneybeans": 0.90,
    "pigeonpeas": 0.85,
    "blackgram": 0.80,
    "mungbean": 0.80,
    "chickpea": 0.75,
    "mothbeans": 0.70,
    "lentil": 0.70,
}

GROWTH_STAGE_WEIGHTS = {
    1: 0.75,  # Vegetative / Initial Establishment
    2: 1.15,  # Flowering / Reproductive / Peak Canopy
    3: 0.65,  # Late Season / Ripening / Senescence
}


def compute_single_demand(
    crop: str,
    soil_moisture: float,
    rainfall: float,
    temperature: float,
    humidity: float,
    crop_density: float,
    growth_stage: int,
    water_usage_efficiency: float,
    irrigation_frequency: float = 3.5,
    ph: float = 6.5,
    N: float = 50.0,
    P: float = 50.0,
    K: float = 50.0
) -> Dict[str, float]:
    """
    Transparent formulation of the Prototype Water Demand Model.
    
    DEMAND INFLUENCE PRINCIPLES:
    - Higher: Crop base need, Crop density, Growth-stage requirement (Stage 2 peak), Temperature
    - Lower (increases deficit): Soil moisture, Rainfall, Water-use efficiency
    """
    # 1. Base crop factor
    c_base = CROP_BASE_FACTORS.get(str(crop).lower(), 1.0)

    # 2. Growth stage factor (reproductive stage 2 is peak water demand)
    f_stage = GROWTH_STAGE_WEIGHTS.get(int(round(growth_stage)), 0.85)

    # 3. Crop density factor (range 5 to 20 plants/m2)
    # Higher plant density increases transpiration canopy
    f_density = 1.0 + 0.35 * ((float(crop_density) - 5.0) / 15.0)

    # 4. Temperature / Evapotranspiration stress factor (range 10 to 45 C)
    # Higher temperature drives higher atmospheric evaporative demand
    f_temp = 1.0 + 0.025 * (float(temperature) - 25.0)

    # 5. Atmospheric humidity factor (range 15 to 100%)
    # Lower humidity accelerates vapor pressure deficit
    f_humidity = 1.0 + 0.20 * ((65.0 - float(humidity)) / 50.0)

    # 6. Soil moisture deficit factor (range 10 to 30% volumetric moisture)
    # Lower soil moisture means higher irrigation water is required to restore field capacity
    # 30% = saturated/field capacity (deficit = 0.2); 10% = wilting point (deficit = 1.0)
    f_soil_deficit = max(0.1, (32.0 - float(soil_moisture)) / 22.0)

    # 7. Local precipitation offset factor (range 20 to 300 mm)
    # Higher local rainfall satisfies part of crop requirement, lowering net irrigation demand
    f_rain_offset = max(0.15, 1.0 - (float(rainfall) / 280.0))

    # 8. Water Usage Efficiency (WUE) penalty factor (range 1 to 5)
    # Lower WUE means more water is wasted in conveyance/application, increasing gross demand requirement
    f_wue = (6.0 - float(water_usage_efficiency)) / 4.0

    # 9. Irrigation frequency coefficient
    f_freq = 0.9 + 0.05 * float(irrigation_frequency)

    # Calculate raw prototype demand score
    # Baseline scale target: ~20 to ~120
    demand_score = (
        c_base
        * f_stage
        * f_density
        * f_temp
        * f_humidity
        * (0.55 * f_soil_deficit + 0.45 * f_rain_offset)
        * f_wue
        * f_freq
        * 100.0
    )

    demand_score = float(np.round(max(5.0, demand_score), 2))

    # Normalized demand mapped to a 0-100 scale
    # Reference anchor: 100 is max prototype peak demand
    normalized_demand = float(np.round(np.clip(demand_score * 0.85, 10.0, 100.0), 2))

    return {
        "crop": str(crop).capitalize(),
        "demand_score": demand_score,
        "normalized_demand": normalized_demand,
        "soil_moisture": float(round(soil_moisture, 2)),
        "rainfall": float(round(rainfall, 2)),
        "temperature": float(round(temperature, 2)),
        "humidity": float(round(humidity, 2)),
        "growth_stage": int(growth_stage),
        "crop_density": float(round(crop_density, 2)),
        "water_usage_efficiency": float(round(water_usage_efficiency, 2)),
        "model_label": "Prototype Water Demand Model (Not Scientifically Validated)"
    }


def estimate_water_demand(
    df1: pd.DataFrame,
    selected_crop: Optional[str] = None
) -> pd.DataFrame:
    """
    Estimates water demand across crops from Dataset 1.
    If selected_crop is given, returns the row for that crop or filters for it.
    If None, returns aggregated summary table across all 22 crops.
    """
    if df1 is None or len(df1) == 0:
        return pd.DataFrame()

    results = []

    if selected_crop:
        crop_sub = df1[df1["label"].str.lower() == selected_crop.lower()]
        if len(crop_sub) == 0:
            crop_sub = df1.head(1)
        mean_row = crop_sub.mean(numeric_only=True)
        res = compute_single_demand(
            crop=selected_crop,
            soil_moisture=mean_row.get("soil_moisture", 20.0),
            rainfall=mean_row.get("rainfall", 100.0),
            temperature=mean_row.get("temperature", 25.0),
            humidity=mean_row.get("humidity", 70.0),
            crop_density=mean_row.get("crop_density", 12.0),
            growth_stage=int(round(mean_row.get("growth_stage", 2))),
            water_usage_efficiency=mean_row.get("water_usage_efficiency", 3.0),
            irrigation_frequency=mean_row.get("irrigation_frequency", 3.5),
            ph=mean_row.get("ph", 6.5),
            N=mean_row.get("N", 50.0),
            P=mean_row.get("P", 50.0),
            K=mean_row.get("K", 50.0),
        )
        results.append(res)
    else:
        # Group by crop
        for crop_name, grp in df1.groupby("label"):
            m = grp.mean(numeric_only=True)
            res = compute_single_demand(
                crop=str(crop_name),
                soil_moisture=m.get("soil_moisture", 20.0),
                rainfall=m.get("rainfall", 100.0),
                temperature=m.get("temperature", 25.0),
                humidity=m.get("humidity", 70.0),
                crop_density=m.get("crop_density", 12.0),
                growth_stage=int(round(m.get("growth_stage", 2))),
                water_usage_efficiency=m.get("water_usage_efficiency", 3.0),
                irrigation_frequency=m.get("irrigation_frequency", 3.5),
                ph=m.get("ph", 6.5),
                N=m.get("N", 50.0),
                P=m.get("P", 50.0),
                K=m.get("K", 50.0),
            )
            results.append(res)

    res_df = pd.DataFrame(results)
    return res_df
