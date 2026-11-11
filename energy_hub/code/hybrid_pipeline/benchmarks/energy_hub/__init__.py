"""Energy Hub benchmark instance builders and evaluators."""

from .instance import EnergyHubInstanceBuilder, build_energy_hub_instance
from .model import solve_energy_hub_support, structural_filter
from .surrogates import (
    enumerate_exact_k_energies,
    evaluate_surrogate_energy,
    build_energy_hub_s0,
    build_energy_hub_s0_legacy_corrected,
    build_energy_hub_s1,
    build_energy_hub_s1_economic_linear,
    build_energy_hub_s2,
    build_energy_hub_s2_temporal_coverage,
)

__all__ = [
    "EnergyHubInstanceBuilder",
    "build_energy_hub_s0",
    "build_energy_hub_s0_legacy_corrected",
    "build_energy_hub_s1",
    "build_energy_hub_s1_economic_linear",
    "build_energy_hub_s2",
    "build_energy_hub_s2_temporal_coverage",
    "build_energy_hub_instance",
    "enumerate_exact_k_energies",
    "evaluate_surrogate_energy",
    "solve_energy_hub_support",
    "structural_filter",
]
