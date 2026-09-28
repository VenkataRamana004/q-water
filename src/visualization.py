"""
Visualization Module for Q-WATER
Creates publication-quality interactive Plotly figures for the Streamlit dashboard.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def plot_workflow_diagram() -> go.Figure:
    """Creates a sleek flowchart of the Q-WATER pipeline."""
    steps = [
        "1. Real Datasets\n(5 AP CSVs)",
        "2. Preprocessing\n& Validation",
        "3. Basin Filter\n(Krishna/Godavari)",
        "4. Demand Model\n(Crop / Soil)",
        "5. Availability\n(Rain+Res+Dis)",
        "6. Uncertainty\n(Historical CV)",
        "7. QUBO Matrix\n(x[i,j] Levels)",
        "8. Qiskit QAOA\n(Simulation)",
        "9. Optimal Allocation\n& Classical Benchmark"
    ]
    x_pos = list(range(len(steps)))
    y_pos = [0] * len(steps)

    fig = go.Figure()
    # Add nodes
    fig.add_trace(go.Scatter(
        x=x_pos, y=y_pos,
        mode="markers+text",
        marker=dict(
            size=48,
            color=["#1E88E5", "#00ACC1", "#43A047", "#FB8C00", "#3949AB", "#D81B60", "#8E24AA", "#5E35B1", "#00897B"],
            line=dict(color="#ffffff", width=2)
        ),
        text=[f"<b>Step {i+1}</b>" for i in range(len(steps))],
        textposition="top center",
        hovertext=steps,
        hoverinfo="text"
    ))

    # Add connecting line
    fig.add_trace(go.Scatter(
        x=x_pos, y=y_pos,
        mode="lines",
        line=dict(color="#78909C", width=3, dash="dot"),
        hoverinfo="skip"
    ))

    fig.update_layout(
        title="<b>End-to-End Quantum-Assisted Water Allocation Workflow</b>",
        xaxis=dict(
            tickmode="array",
            tickvals=x_pos,
            ticktext=steps,
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.8, 0.8]),
        height=260,
        margin=dict(l=20, r=20, t=50, b=50),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_regional_map(
    rf_df: Optional[pd.DataFrame],
    res_df: Optional[pd.DataFrame],
    dis_df: Optional[pd.DataFrame]
) -> go.Figure:
    """Plots geographical stations across Andhra Pradesh."""
    fig = go.Figure()

    def add_station_trace(df, col_val, name, color, symbol):
        if df is not None and "Latitude" in df.columns and "Longitude" in df.columns:
            sub = df.dropna(subset=["Latitude", "Longitude", "Station"]).drop_duplicates(subset=["Station"])
            # Filter reasonable AP coordinates (lat 12 to 20, lon 76 to 85)
            sub = sub[(sub["Latitude"] >= 12.0) & (sub["Latitude"] <= 20.5) &
                      (sub["Longitude"] >= 76.0) & (sub["Longitude"] <= 85.0)]
            if len(sub) > 0:
                fig.add_trace(go.Scattergeo(
                    lat=sub["Latitude"],
                    lon=sub["Longitude"],
                    text=sub["Station"] + "<br>Basin: " + sub.get("Basin", "N/A").astype(str) + "<br>River: " + sub.get("River", "N/A").astype(str),
                    mode="markers+text",
                    textposition="top right",
                    name=name,
                    marker=dict(size=9, color=color, symbol=symbol, line=dict(width=1, color="white"))
                ))

    add_station_trace(rf_df, "Telemetry Hourly Rainfall (mm)", "Rainfall Stations (Telemetry)", "#00BCD4", "circle")
    add_station_trace(res_df, "Manual Daily Reservoir storage (mcm)", "Reservoirs (Krishna Basin)", "#4CAF50", "diamond")
    add_station_trace(dis_df, "Manual Daily River Water Discharge (m3/sec)", "Discharge Gauges", "#FF9800", "square")

    fig.update_layout(
        title="<b>Geographical Station Network (Krishna & Godavari Basins)</b>",
        geo=dict(
            scope="asia",
            center=dict(lat=16.5, lon=80.5),
            projection_scale=22,
            showland=True,
            landcolor="#ECEFF1",
            subunitcolor="#90A4AE",
            countrycolor="#90A4AE",
            showlakes=True,
            lakecolor="#CFD8DC"
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        height=450,
        margin=dict(l=10, r=10, t=40, b=30)
    )
    return fig


def plot_hydro_time_series(df: pd.DataFrame, col_name: str, title: str, yaxis_title: str, color: str = "#2196F3") -> go.Figure:
    """Plots time series trend with 30-period rolling average."""
    fig = go.Figure()
    if df is None or len(df) == 0 or col_name not in df.columns:
        fig.add_annotation(text="No data available for this selection", showarrow=False, font=dict(size=14))
        return fig

    sub = df.dropna(subset=[col_name]).copy()
    date_col = "Data Acquisition Time"

    if date_col in sub.columns and pd.api.types.is_datetime64_any_dtype(sub[date_col]):
        sub = sub.sort_values(date_col)
        # Sample if too large to render swiftly
        if len(sub) > 2500:
            sample_sub = sub.iloc[::(len(sub) // 2500)].copy()
        else:
            sample_sub = sub.copy()

        x = sample_sub[date_col]
        y = sample_sub[col_name]
    else:
        sample_sub = sub.head(1000)
        x = list(range(len(sample_sub)))
        y = sample_sub[col_name]

    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        line=dict(color=color, width=1.5),
        opacity=0.75,
        name="Recorded"
    ))

    # Rolling trend
    if len(y) > 30:
        rolling = pd.Series(y.values).rolling(window=30, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=x, y=rolling,
            mode="lines",
            line=dict(color="#FF5722", width=2.5),
            name="Trend (Rolling Avg)"
        ))

    fig.update_layout(
        title=f"<b>{title}</b>",
        xaxis_title="Timeline",
        yaxis_title=yaxis_title,
        template="plotly_white",
        height=320,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_crop_demands_bar(demand_df: pd.DataFrame, selected_crop: Optional[str] = None) -> go.Figure:
    """Plots comparative water demand scores across crops from Dataset 1."""
    if demand_df is None or len(demand_df) == 0:
        return go.Figure()

    df_sorted = demand_df.sort_values("demand_score", ascending=True).copy()
    colors = [
        "#E91E63" if (selected_crop and c.lower() == selected_crop.lower()) else "#1976D2"
        for c in df_sorted["crop"]
    ]

    fig = go.Figure(go.Bar(
        x=df_sorted["demand_score"],
        y=df_sorted["crop"],
        orientation="h",
        marker=dict(color=colors),
        text=df_sorted["demand_score"].round(1),
        textposition="outside"
    ))

    fig.update_layout(
        title="<b>Prototype Water Demand Scores Across Crops (Dataset 1)</b>",
        xaxis_title="Demand Score (Prototype Model Units)",
        yaxis_title="Crop",
        height=550,
        template="plotly_white",
        margin=dict(l=100, r=40, t=50, b=40)
    )
    return fig


def plot_qubo_heatmap(matrix: np.ndarray, var_names: List[str]) -> go.Figure:
    """Visualizes the genuine QUBO matrix as a heatmap."""
    fig = px.imshow(
        matrix,
        x=var_names,
        y=var_names,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="<b>Genuine Upper-Triangular QUBO Coupling Matrix Q (K × K)</b>"
    )
    fig.update_layout(
        xaxis_title="Binary Decision Variable x[i, j]",
        yaxis_title="Binary Decision Variable x[i, j]",
        height=480,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig


def plot_qaoa_probabilities(probs: List[Tuple[str, float]], best_bitstring: str) -> go.Figure:
    """Plots measurement probabilities of top bitstrings from QAOA."""
    labels = [b for b, _ in probs]
    values = [p * 100.0 for _, p in probs]
    colors = ["#4CAF50" if b == best_bitstring else "#2196F3" for b in labels]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker=dict(color=colors),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside"
    ))

    fig.update_layout(
        title="<b>QAOA Quantum Simulator: Top Bitstring Measurement Probabilities</b>",
        xaxis_title="Quantum Bitstring (x[0,0]...x[N,L])",
        yaxis_title="Measurement Probability (%)",
        template="plotly_white",
        height=360,
        margin=dict(l=40, r=40, t=50, b=50)
    )
    return fig


def plot_allocation_comparison(allocations: List[Dict[str, Any]]) -> go.Figure:
    """Grouped bar chart comparing crop demand vs allocated water."""
    crops = [a["zone_crop"] for a in allocations]
    demands = [a["demand"] for a in allocations]
    allocated = [a["allocated"] for a in allocations]
    shortage = [a["shortage"] for a in allocations]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Estimated Demand",
        x=crops, y=demands,
        marker_color="#90A4AE",
        text=[f"{d:.1f}" for d in demands],
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        name="Allocated Water",
        x=crops, y=allocated,
        marker_color="#2E7D32",
        text=[f"{a:.1f}" for a in allocated],
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        name="Water Shortage",
        x=crops, y=shortage,
        marker_color="#C62828",
        text=[f"{s:.1f}" for s in shortage],
        textposition="auto"
    ))

    fig.update_layout(
        barmode="group",
        title="<b>Crop Water Demand vs Optimized Allocation vs Shortage</b>",
        xaxis_title="Irrigation Zone / Crop",
        yaxis_title="Water Allocation Units (AU)",
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_classical_vs_qaoa_chart(comparison: Dict[str, Any]) -> go.Figure:
    """Direct visual comparison between Classical and QAOA optimization results."""
    categories = ["Allocated Water", "Water Shortage", "Objective Value"]
    c = comparison["classical"]
    q = comparison["qaoa"]

    c_vals = [c["total_allocated"], c["total_shortage"], c["objective_value"]]
    q_vals = [q["total_allocated"], q["total_shortage"], q["objective_value"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=f"Classical ({c['method']})",
        x=categories, y=c_vals,
        marker_color="#3F51B5",
        text=[f"{v:.1f}" for v in c_vals],
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        name=f"Quantum ({q['method']})",
        x=categories, y=q_vals,
        marker_color="#00897B",
        text=[f"{v:.1f}" for v in q_vals],
        textposition="auto"
    ))

    fig.update_layout(
        barmode="group",
        title="<b>Classical Baseline vs QAOA Quantum Solution Benchmark</b>",
        yaxis_title="Metric Value",
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
