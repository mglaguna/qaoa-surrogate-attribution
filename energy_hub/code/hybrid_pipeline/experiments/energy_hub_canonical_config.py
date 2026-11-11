"""Frozen scientific configuration for the Energy Hub canonical campaign.

This module is the single source of truth for every scientific parameter of
the EH_CANONICAL_V1 protocol: instance size, seeds, per-arm sampler
configuration, downstream evaluation budgets, and tolerances. It intentionally
imports only the core benchmark modules (``hybrid_pipeline.benchmarks.energy_hub``,
``hybrid_pipeline.core``), so the canonical campaign cannot silently inherit
behaviour from any other code.

Nothing in this module may be overridden from the command line. Only
non-scientific knobs (output path, --resume, --validate-only) are exposed as
CLI arguments by ``energy_hub_canonical_campaign.py``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from hybrid_pipeline.benchmarks.energy_hub import build_energy_hub_instance, structural_filter
from hybrid_pipeline.benchmarks.energy_hub.surrogates import bits_to_selection

CANONICAL_PROTOCOL_VERSION = "EH_CANONICAL_V1"
CAMPAIGN_NAME = "energy_hub_canonical_v1"

N = 18
K = 6
T = 6
N_TOTAL_SUPPORTS = 18564

VALIDATION_SEED = 100
SEEDS = list(range(101, 111))
ARMS = ["S0", "S1", "S2", "Uniform"]
EXPECTED_TOTAL_RUNS = len(ARMS) * len(SEEDS)

ARM_CONFIGS = {
    "S0": {
        "surrogate_type": "S0",
        "role": "instantaneous-capacity",
        "depth": 5,
        "restarts": 1,
        "shots_per_restart": 5000,
        "total_shots": 5000,
        "qaoa_steps": 10,
        "selection_policy": "frequency_ranked",
    },
    "S1": {
        "surrogate_type": "S1",
        "role": "additive-economic",
        "depth": 10,
        "restarts": 3,
        "shots_per_restart": 5000,
        "total_shots": 15000,
        "qaoa_steps": 10,
        "selection_policy": "frequency_ranked",
    },
    "S2": {
        "surrogate_type": "S2",
        "role": "temporal-physical-coverage",
        "depth": 10,
        "restarts": 1,
        "shots_per_restart": 5000,
        "total_shots": 5000,
        "qaoa_steps": 10,
        "selection_policy": "frequency_ranked",
    },
    "S1_5k": {
        "surrogate_type": "S1",
        "role": "additive-economic (5k budget-matched control)",
        "depth": 10,
        "restarts": 1,
        "shots_per_restart": 5000,
        "total_shots": 5000,
        "qaoa_steps": 10,
        "selection_policy": "frequency_ranked",
    },
    "Uniform": {
        "surrogate_type": None,
        "role": "exact-K random control",
        "depth": 0,
        "restarts": 0,
        "shots_per_restart": 5000,
        "total_shots": 5000,
        "qaoa_steps": 0,
        "selection_policy": "uniform_random_seeded",
    },
}

MAX_CONTINUOUS_EVALUATIONS_PER_RUN = 100
REPORTING_BUDGETS = [10, 30, 60, 100]
PRIMARY_REPORTING_BUDGET = 30

BUDGET_DESCRIPTION = "common downstream evaluation budget"
FORBIDDEN_BUDGET_DESCRIPTIONS = (
    "equal computational budget",
    "equal shot budget",
    "matched total cost",
)

TAC_TIE_REL_TOL = 1e-6

EVALUATION_MODE = "exhaustive_reference_lookup"
EXHAUSTIVE_REFERENCE_PATH = Path("inputs/energy_hub_exhaustive_classical_rows.csv")
EXPECTED_TOTAL_SUPPORTS_IN_REFERENCE = 18564
EXPECTED_STRUCTURAL_REJECTED = 12633
EXPECTED_IPOPT_ATTEMPTED = 5931
EXPECTED_IPOPT_FEASIBLE = 5801
EXPECTED_IPOPT_INFEASIBLE_OR_FAILED = 130

EXPECTED_SURROGATE_COEFFICIENT_HASHES = {
    "S0": "f6bdbc18c005e68abb00b9eaa76f589954735acda79bcb54efd3290c38298ba0",
    "S1": "f57f11a2573ec94a0926e058534a5a4021b13272ac2b54576dc4eb153dcee616",
    "S2": "089689535dfea9173c5d0948d6125f9cfed67553a4c3b56659f500f77661db95",
}

OUTPUT_ROOT = Path("outputs/energy_hub_canonical_v1")


def _stable_json(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def derive_seed(base_seed: int, arm: str, restart: int, stage: str) -> int:
    """Derive a deterministic 32-bit sub-seed for one stochastic source.

    Uses SHA-256 of a canonical string payload -- never Python's ``hash()``,
    which is salted per-process and not reproducible across runs/machines.
    """
    payload = f"{CANONICAL_PROTOCOL_VERSION}|{base_seed}|{arm}|{restart}|{stage}"
    digest = _sha256_hex(payload)
    return int(digest[:8], 16)


def canonical_config_payload() -> dict:
    """The full frozen scientific configuration, as a JSON-serialisable dict."""
    return {
        "protocol_version": CANONICAL_PROTOCOL_VERSION,
        "campaign_name": CAMPAIGN_NAME,
        "N": N,
        "K": K,
        "T": T,
        "n_total_supports": N_TOTAL_SUPPORTS,
        "seeds": SEEDS,
        "validation_seed": VALIDATION_SEED,
        "arms": ARMS,
        "arm_configs": ARM_CONFIGS,
        "max_continuous_evaluations_per_run": MAX_CONTINUOUS_EVALUATIONS_PER_RUN,
        "reporting_budgets": REPORTING_BUDGETS,
        "primary_reporting_budget": PRIMARY_REPORTING_BUDGET,
        "budget_description": BUDGET_DESCRIPTION,
        "tac_tie_rel_tol": TAC_TIE_REL_TOL,
        "evaluation_mode": EVALUATION_MODE,
        "exhaustive_reference_path": str(EXHAUSTIVE_REFERENCE_PATH),
        "expected_surrogate_coefficient_hashes": EXPECTED_SURROGATE_COEFFICIENT_HASHES,
        "expected_total_runs": EXPECTED_TOTAL_RUNS,
    }


def config_hash() -> str:
    return _sha256_hex(_stable_json(canonical_config_payload()))


def _energy_hub_data_payload(instance: dict) -> dict:
    """A JSON-serialisable, order-independent snapshot of the fixed instance data."""
    d = instance["ENERGY_HUB_DATA"]
    techs = d["techs"]
    periods = d["periods"]

    def series_dict(series):
        return {str(k): float(series[k]) for k in techs}

    def series_dict_periods(series):
        return {str(k): float(series[k]) for k in periods}

    return {
        "techs": list(techs),
        "groups": {g: list(members) for g, members in d["groups"].items()},
        "periods": list(periods),
        "etaE": series_dict(d["etaE"]),
        "etaQ": series_dict(d["etaQ"]),
        "etaC": series_dict(d["etaC"]),
        "Smax": series_dict(d["Smax"]),
        "a": series_dict(d["a"]),
        "b": series_dict(d["b"]),
        "alpha": series_dict(d["alpha"]),
        "beta": series_dict(d["beta"]),
        "rho": {t: {p: float(d["rho"].loc[t, p]) for p in periods} for t in techs},
        "tau": series_dict_periods(d["tau"]),
        "DE": series_dict_periods(d["DE"]),
        "DQ": series_dict_periods(d["DQ"]),
        "DC": series_dict_periods(d["DC"]),
        "cE": series_dict_periods(d["cE"]),
        "CRF": float(d["CRF"]),
    }


def instance_hash(instance: dict | None = None) -> str:
    inst = instance if instance is not None else build_energy_hub_instance()
    return _sha256_hex(_stable_json(_energy_hub_data_payload(inst)))


def verify_surrogate_hashes(surrogates: dict) -> None:
    for name, expected in EXPECTED_SURROGATE_COEFFICIENT_HASHES.items():
        observed = surrogates[name]["coefficient_hash"]
        if observed != expected:
            raise RuntimeError(
                f"Surrogate {name} coefficient_hash mismatch: expected {expected}, "
                f"got {observed}. The canonical campaign refuses to run against a "
                "surrogate construction that differs from the one this protocol "
                "was validated against."
            )


class ExhaustiveReferenceError(RuntimeError):
    pass


def load_and_verify_exhaustive_reference(instance: dict | None = None) -> pd.DataFrame:
    """Load the exhaustive exact-K classical reference and verify it against
    a fresh, independent recomputation of the fixed structural filter.

    This is a full (not sampled) verification: every one of the 18,564
    enumerated exact-K supports has its structural admissibility recomputed
    from the current instance parameters and compared against the reference
    CSV's own classification. Any mismatch means the instance parameters (or
    the structural filter) have drifted since the reference was produced, and
    the campaign must stop before producing results (section 8).
    """
    inst = instance if instance is not None else build_energy_hub_instance()
    path = EXHAUSTIVE_REFERENCE_PATH
    if not path.exists():
        raise ExhaustiveReferenceError(f"Exhaustive reference not found at {path}")

    df = pd.read_csv(path, dtype={"support_bitstring": str, "support_hash": str})
    if len(df) != EXPECTED_TOTAL_SUPPORTS_IN_REFERENCE:
        raise ExhaustiveReferenceError(
            f"Exhaustive reference has {len(df)} rows, expected {EXPECTED_TOTAL_SUPPORTS_IN_REFERENCE}."
        )

    required_cols = {"support_bitstring", "support_hash", "method", "feasible", "TAC"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ExhaustiveReferenceError(f"Exhaustive reference missing columns: {sorted(missing)}")

    n_struct_rejected = int((df["method"] == "structural_filter").sum())
    n_attempted = int((df["method"] == "ipopt").sum())
    n_feasible = int(((df["method"] == "ipopt") & (df["feasible"] == True)).sum())
    if n_struct_rejected != EXPECTED_STRUCTURAL_REJECTED:
        raise ExhaustiveReferenceError(
            f"structural_filter rejected count {n_struct_rejected} != expected {EXPECTED_STRUCTURAL_REJECTED}"
        )
    if n_attempted != EXPECTED_IPOPT_ATTEMPTED:
        raise ExhaustiveReferenceError(
            f"ipopt-attempted count {n_attempted} != expected {EXPECTED_IPOPT_ATTEMPTED}"
        )
    if n_feasible != EXPECTED_IPOPT_FEASIBLE:
        raise ExhaustiveReferenceError(
            f"ipopt-feasible count {n_feasible} != expected {EXPECTED_IPOPT_FEASIBLE}"
        )

    mismatches = 0
    struct_reject_bits = set(df.loc[df["method"] == "structural_filter", "support_bitstring"])
    for bitstring in df["support_bitstring"]:
        bits = tuple(int(ch) for ch in bitstring)
        selection = bits_to_selection(bits, inst)
        admissible = bool(structural_filter(selection, inst))
        expected_admissible = bitstring not in struct_reject_bits
        if admissible != expected_admissible:
            mismatches += 1
    if mismatches:
        raise ExhaustiveReferenceError(
            f"{mismatches} supports disagree between a fresh structural_filter recomputation "
            "and the exhaustive reference's own classification. Instance parameters or the "
            "structural filter have drifted since the reference was produced."
        )

    feasible_mask = df["feasible"] == True
    if not df.loc[feasible_mask, "TAC"].map(lambda x: pd.notna(x)).all():
        raise ExhaustiveReferenceError("A feasible reference row has a missing TAC.")
    if df.loc[~feasible_mask, "TAC"].notna().any():
        raise ExhaustiveReferenceError("A non-feasible reference row carries a numeric TAC.")

    return df


def exhaustive_reference_hash(df: pd.DataFrame) -> str:
    cols = ["support_bitstring", "support_hash", "method", "feasible", "TAC"]
    payload = df[cols].sort_values("support_bitstring").to_csv(index=False)
    return _sha256_hex(payload)


def support_hash(bitstring: str) -> str:
    return hashlib.sha256(str(bitstring).encode("utf-8")).hexdigest()[:16]
