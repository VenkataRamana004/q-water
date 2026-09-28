"""
QUBO Formulation Module for Q-WATER
Constructs the genuine Quadratic Unconstrained Binary Optimization (QUBO) model
for quantum-assisted irrigation water allocation under uncertain inflows.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from qiskit_optimization import QuadraticProgram
from qiskit.quantum_info import SparsePauliOp


class QUBOModel:
    """
    Formulates and analyzes the QUBO problem for Q-WATER.
    Decision variable x[i, j]:
      i = irrigation zone / crop index (0 <= i < N)
      j = discrete water allocation level index (0 <= j < L)
    Total qubits / variables: K = N * L.
    """

    def __init__(
        self,
        zones: List[str],
        demands: List[float],
        efficiencies: List[float],
        robust_available_water: float,
        levels: Optional[List[float]] = None,
        lambda_short: float = 1.2,
        lambda_cap: float = 2.0,
        lambda_fair: float = 1.5,
        lambda_eff: float = 1.0,
        p_onehot: float = 35.0,
    ):
        self.zones = zones
        self.demands = demands
        self.efficiencies = efficiencies
        self.robust_available_water = float(robust_available_water)
        self.N = len(zones)
        
        # Default discrete allocation levels: 0%, 50%, 100% of demand
        self.levels = levels if levels is not None else [0.0, 0.5, 1.0]
        self.L = len(self.levels)
        self.K = self.N * self.L  # Total binary variables / qubits

        # Penalty coefficients
        self.lambda_short = float(lambda_short)
        self.lambda_cap = float(lambda_cap)
        self.lambda_fair = float(lambda_fair)
        self.lambda_eff = float(lambda_eff)
        self.p_onehot = float(p_onehot)

        # Variable names mapping
        self.var_names = []
        self.var_lookup = {}
        idx = 0
        for i in range(self.N):
            for j in range(self.L):
                name = f"x_{i}_{j}"
                self.var_names.append(name)
                self.var_lookup[(i, j)] = idx
                idx += 1

        self.linear_terms: Dict[str, float] = {}
        self.quadratic_terms: Dict[Tuple[str, str], float] = {}
        self.constant_offset: float = 0.0
        self.qubo_matrix: Optional[np.ndarray] = None
        self.qp: Optional[QuadraticProgram] = None

        self._build_qubo()

    def _build_qubo(self):
        """Constructs all linear and quadratic terms from the objective components."""
        linear = {name: 0.0 for name in self.var_names}
        quadratic = {}
        offset = 0.0

        # -------------------------------------------------------------
        # 1. ONE-HOT CONSTRAINT: P_onehot * sum_i (sum_j x_ij - 1)^2
        # (sum_j x_ij - 1)^2 = -sum_j x_ij + 2 sum_{j < k} x_ij x_ik + 1
        # -------------------------------------------------------------
        for i in range(self.N):
            offset += self.p_onehot
            for j in range(self.L):
                v = f"x_{i}_{j}"
                linear[v] -= self.p_onehot
                for k in range(j + 1, self.L):
                    v2 = f"x_{i}_{k}"
                    pair = (v, v2)
                    quadratic[pair] = quadratic.get(pair, 0.0) + 2.0 * self.p_onehot

        # -------------------------------------------------------------
        # 2. SHORTAGE MINIMIZATION: lambda_short * sum_i (D_i - sum_j w_ij x_ij)^2
        # Under one-hot selection, this expands to:
        # lambda_short * sum_ij (D_i - w_ij)^2 * x_ij
        # -------------------------------------------------------------
        for i in range(self.N):
            for j in range(self.L):
                v = f"x_{i}_{j}"
                w_ij = self.levels[j] * self.demands[i]
                cost_short = self.lambda_short * ((self.demands[i] - w_ij) ** 2)
                linear[v] += cost_short

        # -------------------------------------------------------------
        # 3. CAPACITY / OVER-ALLOCATION: lambda_cap * (sum_ij w_ij x_ij - W_robust)^2
        # = lambda_cap [ sum_ij w_ij^2 x_ij + 2 sum_{(i,j)<(k,m)} w_ij w_km x_ij x_km
        #                - 2 W_robust sum_ij w_ij x_ij + W_robust^2 ]
        # -------------------------------------------------------------
        offset += self.lambda_cap * (self.robust_available_water ** 2)
        all_indices = [(i, j) for i in range(self.N) for j in range(self.L)]

        for idx1, (i1, j1) in enumerate(all_indices):
            v1 = f"x_{i1}_{j1}"
            w1 = self.levels[j1] * self.demands[i1]
            # Diagonal/Linear contribution
            linear[v1] += self.lambda_cap * (w1 ** 2 - 2.0 * self.robust_available_water * w1)

            # Quadratic cross terms
            for idx2 in range(idx1 + 1, self.K):
                i2, j2 = all_indices[idx2]
                v2 = f"x_{i2}_{j2}"
                w2 = self.levels[j2] * self.demands[i2]
                pair = (v1, v2)
                quadratic[pair] = quadratic.get(pair, 0.0) + 2.0 * self.lambda_cap * w1 * w2

        # -------------------------------------------------------------
        # 4. WATER-USE INEFFICIENCY PENALTY:
        # lambda_eff * sum_ij (1.0 - eta_i) * w_ij * x_ij
        # -------------------------------------------------------------
        for i in range(self.N):
            eff = max(0.1, min(1.0, self.efficiencies[i]))
            for j in range(self.L):
                v = f"x_{i}_{j}"
                w_ij = self.levels[j] * self.demands[i]
                linear[v] += self.lambda_eff * (1.0 - eff) * w_ij

        # -------------------------------------------------------------
        # 5. ALLOCATION FAIRNESS PENALTY:
        # lambda_fair * sum_{i < k} (sum_j f_j x_ij - sum_m f_m x_km)^2
        # -------------------------------------------------------------
        for i1 in range(self.N):
            for i2 in range(i1 + 1, self.N):
                for j1 in range(self.L):
                    v1 = f"x_{i1}_{j1}"
                    f1 = self.levels[j1]
                    linear[v1] += self.lambda_fair * (f1 ** 2)
                    for j2 in range(self.L):
                        v2 = f"x_{i2}_{j2}"
                        f2 = self.levels[j2]
                        # cross term -2 * f1 * f2
                        pair = (v1, v2) if self.var_lookup[(i1, j1)] < self.var_lookup[(i2, j2)] else (v2, v1)
                        quadratic[pair] = quadratic.get(pair, 0.0) - 2.0 * self.lambda_fair * f1 * f2
                for j2 in range(self.L):
                    v2 = f"x_{i2}_{j2}"
                    f2 = self.levels[j2]
                    linear[v2] += self.lambda_fair * (f2 ** 2)

        self.linear_terms = linear
        self.quadratic_terms = quadratic
        self.constant_offset = float(offset)

        # Build genuine K x K upper-triangular QUBO matrix
        matrix = np.zeros((self.K, self.K), dtype=float)
        for name, coeff in linear.items():
            u = self.var_names.index(name)
            matrix[u, u] = coeff
        for (v1, v2), coeff in quadratic.items():
            u = self.var_names.index(v1)
            v = self.var_names.index(v2)
            if u > v:
                u, v = v, u
            matrix[u, v] += coeff

        self.qubo_matrix = matrix

        # Build Qiskit QuadraticProgram
        qp = QuadraticProgram(name="Q-WATER-Allocation")
        for name in self.var_names:
            qp.binary_var(name=name)
        qp.minimize(
            constant=self.constant_offset,
            linear=self.linear_terms,
            quadratic=self.quadratic_terms
        )
        self.qp = qp

    def get_qubo_dataframe(self) -> pd.DataFrame:
        """Returns the QUBO matrix as a labeled pandas DataFrame."""
        return pd.DataFrame(
            self.qubo_matrix,
            index=self.var_names,
            columns=self.var_names
        )

    def to_ising_hamiltonian(self) -> Tuple[SparsePauliOp, float]:
        """Converts the QuadraticProgram into an Ising Hamiltonian (operator, offset)."""
        operator, offset = self.qp.to_ising()
        return operator, offset

    def evaluate_solution(self, bitstring: List[int]) -> Dict[str, Any]:
        """
        Evaluates an allocation bitstring against all objectives and constraints.
        """
        x = np.array(bitstring, dtype=float)
        val = float(x @ self.qubo_matrix @ x + self.constant_offset)

        allocations = []
        is_onehot_valid = True
        total_allocated = 0.0
        total_shortage = 0.0
        total_demand = sum(self.demands)

        for i in range(self.N):
            active_levels = []
            chosen_fraction = 0.0
            for j in range(self.L):
                idx = i * self.L + j
                if x[idx] > 0.5:
                    active_levels.append(j)
                    chosen_fraction = self.levels[j]

            if len(active_levels) != 1:
                is_onehot_valid = False

            alloc = chosen_fraction * self.demands[i]
            shortage = max(0.0, self.demands[i] - alloc)
            total_allocated += alloc
            total_shortage += shortage

            allocations.append({
                "zone_crop": self.zones[i],
                "demand": self.demands[i],
                "allocated": alloc,
                "shortage": shortage,
                "fraction": chosen_fraction,
                "efficiency": self.efficiencies[i],
                "satisfaction_pct": round((alloc / self.demands[i]) * 100.0, 1) if self.demands[i] > 0 else 100.0
            })

        over_allocation = max(0.0, total_allocated - self.robust_available_water)
        remaining_water = max(0.0, self.robust_available_water - total_allocated)

        return {
            "qubo_value": round(val, 2),
            "allocations": allocations,
            "total_demand": round(total_demand, 2),
            "total_allocated": round(total_allocated, 2),
            "total_shortage": round(total_shortage, 2),
            "robust_available": round(self.robust_available_water, 2),
            "remaining_water": round(remaining_water, 2),
            "over_allocation": round(over_allocation, 2),
            "is_onehot_valid": is_onehot_valid,
            "is_capacity_respected": over_allocation <= 0.01
        }
