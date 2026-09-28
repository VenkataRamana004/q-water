# Q-WATER: Quantum-Assisted Robust Water Allocation for Irrigation Systems Under Uncertain Inflows

**Q-WATER** is a functional prototype application combining agricultural demand modeling, regional hydrological telemetry from Andhra Pradesh river basins (Krishna and Godavari), robust uncertainty bounds, and genuine quantum optimization using **Qiskit QAOA** (Quantum Approximate Optimization Algorithm).

---

## 🌊 Core Project Workflow

```
5 CSV DATASETS
      ↓
DATA PREPROCESSING
      ↓
KRISHNA–GODAVARI REGION FILTERING
      ↓
WATER DEMAND ESTIMATION
      ↓
WATER AVAILABILITY ESTIMATION
      ↓
UNCERTAINTY / ROBUSTNESS
      ↓
QUBO FORMULATION
      ↓
QAOA USING QISKIT
      ↓
OPTIMAL WATER ALLOCATION
      ↓
CLASSICAL vs QAOA COMPARISON
      ↓
STREAMLIT DASHBOARD
```

---

## 📁 Datasets & Integrity Rules

The application uses real CSV datasets located in `data/` (configurable in the application):

| Dataset | File | Description & Size |
| :--- | :--- | :--- |
| **Dataset 1** | `dataset1.csv` (`smart.csv`) | 2,200 rows, 23 columns. 22 crops, soil moisture, crop density, growth stages, WUE. |
| **Dataset 2** | `dataset2.csv` (`crop.csv`) | 1,000 rows, 8 columns. Supporting crop and weather records. |
| **Dataset 3** | `dataset3.csv` (`rainfall.csv`) | 64,510 rows. AP hourly rainfall telemetry (mm). Filtered via River/Basin. |
| **Dataset 4** | `dataset4.csv` (`reservoir.csv`) | 98,888 rows. AP reservoir storage (mcm). Verified Krishna records only. |
| **Dataset 5** | `dataset5.csv` (`discharge.csv`) | 99,411 rows. AP river discharge (m³/sec). Verified Krishna & Godavari records. |

### ⚠️ Critical Data Fidelity Rules
1. **No Fake Data:** All numbers originate directly from the provided CSV files.
2. **Reservoir Data Integrity:** Dataset 4 contains records **exclusively for Krishna basin projects** (Srisailam, Sunkesula, Bhairavanitippa). When Godavari is selected, reservoir storage is transparently reported as `0.0 mcm` with an explicit notice, without fabricating fake Godavari reservoir numbers.
3. **Transparent Unit Handling:** Original physical units are preserved in the UI:
   - Rainfall: `mm`
   - Reservoir Storage: `mcm` ($10^6\ \text{m}^3$)
   - River Discharge: `m³/sec`
   For mathematical optimization, non-dimensional indexing maps these onto a common **Allocation Units (AU)** pool.

---

## 🧮 Mathematical & Algorithmic Foundation

### 1. Prototype Water Demand Model
$$D_{\text{raw}} = C_{\text{base}} \times f_{\text{stage}} \times f_{\text{dens}} \times f_{\text{temp}} \times f_{\text{hum}} \times (0.55 f_{\text{soil}} + 0.45 f_{\text{rain}}) \times f_{\text{WUE}} \times 100$$
- Influenced positively by crop base water need, vegetative/flowering growth stages, plant density, and temperature.
- Influenced inversely by soil moisture, local precipitation, and water usage efficiency.
- Clearly labelled as a prototype demonstration model.

### 2. Robust Water Availability Under Inflow Uncertainty
$$W_{\text{robust}} = W_{\text{avail}} - \text{Uncertainty Penalty}$$
where the penalty is derived from historical coefficient of variation ($\text{CV} = \sigma / \mu$) calculated from regional rainfall and river discharge time series.
- **Low Uncertainty:** Mild operational buffer (10% - 15%).
- **Medium Uncertainty:** Proportional to historical seasonal volatility (20% - 30%).
- **High Uncertainty:** Severe drought / delayed monsoon stress test (35% - 50%).

### 3. Genuine QUBO Matrix Formulation
- **Decision Variable:** $x_{i,j} \in \{0, 1\}$ where $i \in \{0, \dots, N-1\}$ is the crop zone and $j \in \{0, \dots, L-1\}$ is the discrete allocation fraction (e.g., $0\%, 50\%, 100\%$).
- **Objective Function:**
$$\min_{x} H_{\text{QUBO}}(x) = \lambda_{\text{short}} H_{\text{short}} + \lambda_{\text{cap}} H_{\text{cap}} + \lambda_{\text{fair}} H_{\text{fair}} + \lambda_{\text{eff}} H_{\text{eff}} + P_{\text{onehot}} H_{\text{onehot}}$$
- Generates a genuine upper-triangular coupling matrix $Q \in \mathbb{R}^{K \times K}$ where $K = N \times L$ qubits.

### 4. QAOA with Qiskit
- Converts QuadraticProgram to an Ising Hamiltonian $H = \sum_i h_i Z_i + \sum_{i < j} J_{ij} Z_i Z_j + \text{offset}$.
- Executes parameterized QAOA circuit on a local quantum simulator using Qiskit:
$$|\psi(\gamma, \beta)\rangle = \prod_{l=1}^p e^{-i \beta_l \sum_i X_i} e^{-i \gamma_l H_C} |+\rangle^{\otimes K}$$
- Optimizes parameters $(\gamma, \beta)$ with classical algorithms (`COBYLA`, `Nelder-Mead`).
- Samples measurement bitstrings over shots, decodes into discrete allocations, and displays gate counts and quantum circuit diagram.

### 5. Classical vs Quantum Benchmark
- Benchmarks against an exact combinatorial discrete solver and a greedy heuristic.
- Completely honest, un-fabricated metric comparisons: objective values, total allocated volume, remaining reserve, water shortage, weighted efficiency, and runtime.

---

## 💻 Installation & Running

### Requirements
- Python 3.10+ (tested on Python 3.13)
- Dependencies installed via:
```bash
pip install -r requirements.txt
```

### Launch Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 🖥️ Dashboard Pages
1. **Dashboard:** High-level KPIs, visual pipeline diagram, station map, and demand summary.
2. **Data Explorer:** Inspect all 5 datasets, sample rows, filter by River, Basin, District, Station.
3. **Water Demand:** Interactive crop selection, what-if parameter sliders, full 22-crop registry.
4. **Water Availability:** Regional filters (Krishna, Godavari, Combined), time-series trends, unit conversions.
5. **Uncertainty:** Low/Medium/High uncertainty controls, historical CV statistics, robust water gauge.
6. **QUBO Model:** Mathematical derivations, penalty weights, interactive $K \times K$ QUBO matrix heatmap.
7. **QAOA Optimization:** Parameter controls ($p$, shots, optimizer), quantum circuit visualization, real-time simulation progress bar, bitstring probability histogram.
8. **Final Allocation:** Decoded water allocation table, demand vs allocation chart, shortage breakdown.
9. **Classical vs QAOA:** Objective side-by-side benchmark comparing classical solvers against QAOA.
