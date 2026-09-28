"""
Classical Optimization Baseline for Q-WATER
Provides SciPy / Combinatorial exact search and Greedy heuristics
to benchmark against the QAOA quantum optimization results without fabricated improvements.
"""

import time
import itertools
from typing import Dict, List, Any, Optional
import numpy as np

from src.qubo_model import QUBOModel


class ClassicalOptimizer:
    """
    Classical optimization baseline for irrigation water allocation.
    """

    def __init__(self, qubo_model: QUBOModel):
        self.qubo = qubo_model
        self.N = self.qubo.N
        self.L = self.qubo.L
        self.levels = self.qubo.levels
        self.demands = self.qubo.demands
        self.efficiencies = self.qubo.efficiencies
        self.W_robust = self.qubo.robust_available_water

    def solve_exact_classical(self) -> Dict[str, Any]:
        """
        Solves the combinatorial allocation problem by evaluating all valid one-hot states.
        For N zones with L discrete levels, there are exactly L^N candidate allocations.
        Guarantees the true classical optimal baseline.
        """
        t0 = time.time()
        best_obj = float("inf")
        best_bitstring = None

        # Iterate over all L^N combinations of level indices
        # Each tuple represents (level_zone_0, level_zone_1, ..., level_zone_{N-1})
        for combo in itertools.product(range(self.L), repeat=self.N):
            # Construct K-dimensional one-hot bitstring
            bits = [0] * (self.N * self.L)
            for i, j in enumerate(combo):
                bits[i * self.L + j] = 1

            eval_res = self.qubo.evaluate_solution(bits)
            cost = eval_res["qubo_value"]
            # Add penalty for capacity violation if any
            if eval_res["over_allocation"] > 0:
                cost += eval_res["over_allocation"] * 10.0

            if cost < best_obj:
                best_obj = cost
                best_bitstring = bits

        t1 = time.time()
        decoded = self.qubo.evaluate_solution(best_bitstring)
        bitstring_str = "".join(str(b) for b in best_bitstring)

        return {
            "method": "Classical Exact / SciPy Discrete",
            "best_bitstring": best_bitstring,
            "best_bitstring_str": bitstring_str,
            "objective_value": decoded["qubo_value"],
            "execution_time_ms": round((t1 - t0) * 1000.0, 3),
            "decoded_allocation": decoded
        }

    def solve_greedy(self) -> Dict[str, Any]:
        """
        Greedy allocation heuristic:
        Sorts irrigation zones by water-use efficiency (highest first)
        and allocates maximum feasible water level without exceeding W_robust.
        """
        t0 = time.time()
        # Sort indices by efficiency descending
        sorted_indices = sorted(range(self.N), key=lambda i: self.efficiencies[i], reverse=True)

        chosen_levels = {i: 0 for i in range(self.N)}
        current_allocated = 0.0

        for i in sorted_indices:
            # Try highest level first down to 0
            best_j = 0
            for j in reversed(range(self.L)):
                frac = self.levels[j]
                alloc = frac * self.demands[i]
                if current_allocated + alloc <= self.W_robust + 1e-4:
                    best_j = j
                    current_allocated += alloc
                    break
            chosen_levels[i] = best_j

        # Build bitstring
        bits = [0] * (self.N * self.L)
        for i in range(self.N):
            bits[i * self.L + chosen_levels[i]] = 1

        t1 = time.time()
        decoded = self.qubo.evaluate_solution(bits)
        bitstring_str = "".join(str(b) for b in bits)

        return {
            "method": "Classical Greedy Heuristic",
            "best_bitstring": bits,
            "best_bitstring_str": bitstring_str,
            "objective_value": decoded["qubo_value"],
            "execution_time_ms": round((t1 - t0) * 1000.0, 3),
            "decoded_allocation": decoded
        }


def compare_classical_vs_qaoa(
    classical_res: Dict[str, Any],
    qaoa_res: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generates a technical, evidence-based comparison between Classical and QAOA solutions.
    Does NOT invent performance improvements.
    """
    c_dec = classical_res["decoded_allocation"]
    q_dec = qaoa_res["decoded_allocation"]

    c_obj = classical_res["objective_value"]
    q_obj = qaoa_res["objective_value"]

    c_alloc = c_dec["total_allocated"]
    q_alloc = q_dec["total_allocated"]

    c_short = c_dec["total_shortage"]
    q_short = q_dec["total_shortage"]

    # Efficiency metric: weighted average efficiency of allocated water
    def calc_eff_score(allocations):
        tot_w = sum(a["allocated"] for a in allocations)
        if tot_w <= 0:
            return 0.0
        return sum(a["allocated"] * a["efficiency"] for a in allocations) / tot_w

    c_eff = round(calc_eff_score(c_dec["allocations"]), 3)
    q_eff = round(calc_eff_score(q_dec["allocations"]), 3)

    return {
        "classical": {
            "method": classical_res["method"],
            "objective_value": c_obj,
            "total_allocated": c_alloc,
            "total_shortage": c_short,
            "weighted_efficiency": c_eff,
            "remaining_water": c_dec["remaining_water"],
            "is_capacity_respected": c_dec["is_capacity_respected"],
            "execution_time_ms": classical_res.get("execution_time_ms", 0.0)
        },
        "qaoa": {
            "method": f"QAOA Simulation (Qiskit, p={qaoa_res['reps']})",
            "objective_value": q_obj,
            "total_allocated": q_alloc,
            "total_shortage": q_short,
            "weighted_efficiency": q_eff,
            "remaining_water": q_dec["remaining_water"],
            "is_capacity_respected": q_dec["is_capacity_respected"],
            "execution_time_sec": qaoa_res["execution_time_sec"],
            "solution_probability": qaoa_res["solution_probability"],
            "shots": qaoa_res["shots"]
        },
        "deltas": {
            "objective_diff": round(q_obj - c_obj, 2),
            "allocated_diff": round(q_alloc - c_alloc, 2),
            "shortage_diff": round(q_short - c_short, 2),
            "efficiency_diff": round(q_eff - c_eff, 3)
        }
    }
