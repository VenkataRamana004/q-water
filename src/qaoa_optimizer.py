"""
QAOA Optimizer Module for Q-WATER
Executes Quantum Approximate Optimization Algorithm (QAOA) using Qiskit.
Runs on a local quantum simulator (No IBM credentials required).
"""

import time
from typing import Dict, List, Tuple, Any, Optional, Callable
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from scipy.optimize import minimize

from src.qubo_model import QUBOModel


class QAOAOptimizer:
    """
    Genuine Qiskit QAOA implementation using local quantum statevector simulator.
    """

    def __init__(
        self,
        qubo_model: QUBOModel,
        reps: int = 1,
        shots: int = 1024,
        optimizer_name: str = "COBYLA",
        maxiter: int = 15,
        seed: int = 42
    ):
        self.qubo = qubo_model
        self.reps = reps
        self.shots = shots
        self.optimizer_name = optimizer_name
        self.maxiter = maxiter
        self.seed = seed

        self.n_qubits = self.qubo.K
        self.operator: Optional[SparsePauliOp] = None
        self.offset: float = 0.0

        # Ising parameters
        self.linear_h: Dict[int, float] = {}
        self.coupling_J: Dict[Tuple[int, int], float] = {}

        # Circuit and results
        self.circuit: Optional[QuantumCircuit] = None
        self.optimal_params: Optional[np.ndarray] = None
        self.optimal_energy: float = 0.0
        self.best_bitstring: Optional[List[int]] = None
        self.best_bitstring_str: str = ""
        self.solution_probability: float = 0.0
        self.probabilities: Dict[str, float] = {}
        self.measurement_counts: Dict[str, int] = {}
        self.decoded_allocation: Optional[Dict[str, Any]] = None
        self.execution_time: float = 0.0

        self._convert_to_ising()
        self._build_qaoa_circuit()

    def _convert_to_ising(self):
        """Converts the QUBO QuadraticProgram to an Ising Hamiltonian."""
        op, off = self.qubo.to_ising_hamiltonian()
        self.operator = op
        self.offset = float(off)

        # Parse Pauli terms: Z_i and Z_i Z_j
        h = {i: 0.0 for i in range(self.n_qubits)}
        J = {}

        for pauli, coeff in zip(op.paulis, op.coeffs):
            c_real = float(coeff.real)
            z_indices = [idx for idx, char in enumerate(reversed(str(pauli))) if char == "Z"]
            if len(z_indices) == 1:
                h[z_indices[0]] += c_real
            elif len(z_indices) == 2:
                u, v = sorted(z_indices)
                J[(u, v)] = J.get((u, v), 0.0) + c_real

        self.linear_h = h
        self.coupling_J = J

    def _build_qaoa_circuit(self):
        """Constructs the parameterized QAOA quantum circuit."""
        n = self.n_qubits
        p = self.reps

        gamma = ParameterVector("gamma", p)
        beta = ParameterVector("beta", p)

        qc = QuantumCircuit(n)

        # Step 1: Initial Superposition |+>^n
        for i in range(n):
            qc.h(i)

        # Step 2: Alternating Cost and Mixer Layers
        for layer in range(p):
            # Cost Unitary: exp(-i * gamma * H_C)
            # Single-qubit Z rotations: Rz(2 * gamma * h_i)
            for i, h_val in self.linear_h.items():
                if abs(h_val) > 1e-6:
                    qc.rz(2.0 * gamma[layer] * h_val, i)

            # Two-qubit ZZ interactions: Rzz(2 * gamma * J_ij)
            for (i, j), j_val in self.coupling_J.items():
                if abs(j_val) > 1e-6:
                    qc.rzz(2.0 * gamma[layer] * j_val, i, j)

            # Mixer Unitary: exp(-i * beta * H_M)
            # Single-qubit X rotations: Rx(2 * beta)
            for i in range(n):
                qc.rx(2.0 * beta[layer], i)

        self.circuit = qc

    def get_circuit_figure(self) -> plt.Figure:
        """Draws the QAOA circuit using Matplotlib."""
        try:
            fig = self.circuit.draw(output="mpl", style="iqp")
            return fig
        except Exception:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(
                0.5, 0.5,
                f"QAOA Circuit ({self.n_qubits} Qubits, p={self.reps} Layers)\n"
                f"Gate counts: {dict(self.circuit.count_ops())}\n"
                f"Depth: {self.circuit.depth()}",
                ha="center", va="center", fontsize=12
            )
            ax.axis("off")
            return fig

    def run_simulation(
        self,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Executes QAOA simulation with step-by-step progress tracking:
          1. Building QUBO (done)
          2. Converting to quantum representation (done)
          3. Running QAOA parameter optimization
          4. Measuring on quantum simulator
          5. Decoding allocation
        """
        t0 = time.time()

        if progress_callback:
            progress_callback(10, "Building QUBO & Ising Hamiltonian...")
        time.sleep(0.05)

        if progress_callback:
            progress_callback(25, "Constructing Parameterized QAOA Quantum Circuit...")
        time.sleep(0.05)

        p = self.reps
        # Objective function for classical parameter optimization
        # Parameters vector: [gamma_0, ..., gamma_{p-1}, beta_0, ..., beta_{p-1}]
        iter_count = [0]

        def expectation_fn(params: np.ndarray) -> float:
            param_dict = {}
            for l in range(p):
                param_dict[self.circuit.parameters[l]] = params[l]        # gamma
                param_dict[self.circuit.parameters[p + l]] = params[p + l]  # beta

            bound_circ = self.circuit.assign_parameters(param_dict)
            sv = Statevector.from_instruction(bound_circ)
            energy = float(sv.expectation_value(self.operator).real)

            iter_count[0] += 1
            if progress_callback and iter_count[0] % 3 == 0:
                pct = min(80, 30 + int((iter_count[0] / self.maxiter) * 50))
                progress_callback(pct, f"QAOA Parameter Optimization ({self.optimizer_name}: eval {iter_count[0]})...")

            return energy

        if progress_callback:
            progress_callback(35, f"Optimizing QAOA Angles via {self.optimizer_name} (maxiter={self.maxiter})...")

        # Initial angles: small heuristic values
        np.random.seed(self.seed)
        initial_params = np.concatenate([
            np.full(p, 0.4),  # gamma
            np.full(p, 0.6)   # beta
        ])

        opt_res = minimize(
            expectation_fn,
            x0=initial_params,
            method="COBYLA" if self.optimizer_name == "COBYLA" else "Nelder-Mead",
            options={"maxiter": self.maxiter}
        )

        self.optimal_params = opt_res.x
        self.optimal_energy = float(opt_res.fun) + self.offset

        if progress_callback:
            progress_callback(85, f"Sampling {self.shots:,} Shots on Local Quantum Simulator...")
        time.sleep(0.05)

        # Evaluate final statevector with optimal parameters
        final_dict = {}
        for l in range(p):
            final_dict[self.circuit.parameters[l]] = self.optimal_params[l]
            final_dict[self.circuit.parameters[p + l]] = self.optimal_params[p + l]

        final_circuit = self.circuit.assign_parameters(final_dict)
        final_sv = Statevector.from_instruction(final_circuit)

        # Exact probability distribution
        raw_probs = final_sv.probabilities_dict()
        # Qiskit bitstring order is Little Endian (q_{n-1} ... q_0)
        # Re-order to index order 0 ... n-1
        ordered_probs = {}
        for bitstr, prob in raw_probs.items():
            reversed_bits = bitstr[::-1]  # bit 0 first
            ordered_probs[reversed_bits] = float(prob)

        self.probabilities = ordered_probs

        # Simulate shot measurements using multinomial sampling
        bitstring_keys = list(ordered_probs.keys())
        p_vals = list(ordered_probs.values())
        p_vals = np.array(p_vals) / sum(p_vals)  # normalize
        counts = np.random.multinomial(self.shots, p_vals)
        self.measurement_counts = {bitstring_keys[idx]: int(cnt) for idx, cnt in enumerate(counts) if cnt > 0}

        # Select best bitstring
        # Prefer the measured bitstring that minimizes the QUBO objective
        best_bits = None
        best_obj = float("inf")

        for b_str in self.measurement_counts.keys():
            b_list = [int(c) for c in b_str]
            eval_res = self.qubo.evaluate_solution(b_list)
            # Give priority to valid one-hot solutions
            cost = eval_res["qubo_value"]
            if not eval_res["is_onehot_valid"]:
                cost += 500.0  # heavy penalty for invalid one-hot
            if eval_res["over_allocation"] > 0:
                cost += eval_res["over_allocation"] * 20.0

            if cost < best_obj:
                best_obj = cost
                best_bits = b_list
                self.best_bitstring_str = b_str

        if best_bits is None:
            # Fallback to highest probability bitstring
            best_str = max(ordered_probs, key=ordered_probs.get)
            best_bits = [int(c) for c in best_str]
            self.best_bitstring_str = best_str

        self.best_bitstring = best_bits
        self.solution_probability = float(self.probabilities.get(self.best_bitstring_str, 0.0))

        if progress_callback:
            progress_callback(95, "Decoding Quantum Allocation Bitstring...")
        time.sleep(0.05)

        self.decoded_allocation = self.qubo.evaluate_solution(self.best_bitstring)
        self.execution_time = float(time.time() - t0)

        if progress_callback:
            progress_callback(100, "QAOA Optimization Complete!")

        return {
            "n_qubits": self.n_qubits,
            "reps": self.reps,
            "shots": self.shots,
            "optimizer": self.optimizer_name,
            "best_bitstring": self.best_bitstring,
            "best_bitstring_str": self.best_bitstring_str,
            "objective_value": self.decoded_allocation["qubo_value"],
            "solution_probability": round(self.solution_probability, 4),
            "execution_time_sec": round(self.execution_time, 2),
            "decoded_allocation": self.decoded_allocation,
            "top_probabilities": sorted(
                self.probabilities.items(), key=lambda x: x[1], reverse=True
            )[:10],
            "circuit_depth": self.circuit.depth(),
            "gate_counts": dict(self.circuit.count_ops())
        }
