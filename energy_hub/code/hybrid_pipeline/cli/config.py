"""Central configuration of the hybrid classical-quantum pipeline (default run settings and solver options)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


GLOBAL_SEED = 42

CONFIG: dict[str, Any] = {
    "seed": GLOBAL_SEED,
    "qaoa_layers": 1,
    "qaoa_steps": 100,
    "qaoa_shots": 5000,
    "top_k_candidates": 40,
    "top_eval_per_iter": 2,
    "max_iter": 30,
    "score_alpha": 1.0,
    "score_beta": 0.05,
    "multiseed_seeds": list(range(10)),
}

HAMILTONIAN_PENALTY = 1.0

SOLVER_NAME = "ipopt"

SOLVER_OPTIONS: dict[str, Any] = {}


def get_default_config() -> dict[str, Any]:
    """Return an independent copy of the default run configuration."""
    return deepcopy(CONFIG)


def build_solver(solver_name: str = SOLVER_NAME):
    """Build a Pyomo solver with the configured solver options applied."""
    from pyomo.environ import SolverFactory

    solver = SolverFactory(solver_name)
    for opt_key, opt_val in SOLVER_OPTIONS.items():
        solver.options[opt_key] = opt_val
    return solver

