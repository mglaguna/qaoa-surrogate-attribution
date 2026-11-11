"""Energy Hub surrogate QUBO builders and audit utilities."""

from __future__ import annotations

import hashlib
import itertools
import json
from typing import Iterable

import numpy as np
import pandas as pd


EPS = 1e-12


HISTORICAL_SUPPORTS = {
    "cogen_dominant": (
        "tech_01",
        "tech_02",
        "tech_03",
        "tech_04",
        "tech_05",
        "tech_15",
    ),
    "historical_optimal": (
        "tech_07",
        "tech_08",
        "tech_09",
        "tech_11",
        "tech_15",
        "tech_16",
    ),
    "best_uniform_moderate": (
        "tech_07",
        "tech_08",
        "tech_10",
        "tech_11",
        "tech_16",
        "tech_18",
    ),
    "best_uniform_B1": (
        "tech_03",
        "tech_07",
        "tech_08",
        "tech_12",
        "tech_17",
        "tech_18",
    ),
}


def _assets(instance: dict) -> list[str]:
    return list(instance["ASSETS"])


def _energy_data(instance: dict) -> dict:
    return instance["ENERGY_HUB_DATA"]


def support_to_bits(support, instance: dict) -> tuple[int, ...]:
    """Return a support as a tuple of bits in instance asset order."""
    assets = _assets(instance)
    if isinstance(support, dict):
        return tuple(int(support[a]) for a in assets)
    support_set = set(support)
    return tuple(1 if a in support_set else 0 for a in assets)


def bits_to_selection(bits: Iterable[int], instance: dict) -> dict:
    return {asset: int(bit) for asset, bit in zip(_assets(instance), bits)}


def selected_assets(bits: Iterable[int], instance: dict) -> list[str]:
    return [asset for asset, bit in zip(_assets(instance), bits) if int(bit) == 1]


def support_composition(support, instance: dict) -> dict:
    bits = support_to_bits(support, instance)
    counts = {group: 0 for group in instance["SECTORS"]}
    for asset, bit in zip(_assets(instance), bits):
        if int(bit) == 1:
            counts[instance["SECTOR_OF"][asset]] += 1
    return counts


def _series_to_float_dict(series: pd.Series) -> dict:
    return {str(k): float(v) for k, v in series.items()}


def _frame_to_float_dict(frame: pd.DataFrame) -> dict:
    return {
        str(idx): {str(col): float(frame.loc[idx, col]) for col in frame.columns}
        for idx in frame.index
    }


def _canonical_payload(surrogate: dict) -> dict:
    return {
        "surrogate_type": surrogate["surrogate_type"],
        "assets": surrogate["assets"],
        "linear": {k: float(v) for k, v in surrogate["linear"].items()},
        "quadratic": {
            f"{i}|{j}": float(v)
            for (i, j), v in sorted(surrogate["quadratic"].items())
        },
        "scaling": surrogate.get("scaling", {}),
    }


def surrogate_hash(surrogate: dict) -> str:
    payload = json.dumps(_canonical_payload(surrogate), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def evaluate_surrogate_energy(support, surrogate_or_linear: dict, quadratic: dict | None = None) -> float:
    """
    Evaluate E(y) = sum_i q_i y_i + sum_{i<j} Q_ij y_i y_j.
    """
    if quadratic is None and "linear" in surrogate_or_linear:
        surrogate = surrogate_or_linear
        linear = surrogate["linear"]
        quadratic = surrogate["quadratic"]
        assets = surrogate["assets"]
    else:
        linear = surrogate_or_linear
        quadratic = quadratic or {}
        assets = list(linear.keys())

    if isinstance(support, dict):
        y = {asset: int(support[asset]) for asset in assets}
    else:
        support_set = set(support)
        y = {asset: 1 if asset in support_set else 0 for asset in assets}

    energy = sum(float(q) * y[asset] for asset, q in linear.items())
    for (i, j), coeff in quadratic.items():
        energy += float(coeff) * y[i] * y[j]
    return float(energy)


def qubo_to_ising_coefficients(surrogate: dict) -> dict:
    """
    Convert QUBO coefficients with y_i=(1-Z_i)/2 to Ising coefficients.
    """
    assets = list(surrogate["assets"])
    h = {asset: -0.5 * float(surrogate["linear"].get(asset, 0.0)) for asset in assets}
    constant = 0.5 * sum(float(v) for v in surrogate["linear"].values())
    j_coeff = {}
    for (i, j), coeff in surrogate["quadratic"].items():
        coeff = float(coeff)
        constant += coeff / 4.0
        h[i] = h.get(i, 0.0) - coeff / 4.0
        h[j] = h.get(j, 0.0) - coeff / 4.0
        j_coeff[(i, j)] = coeff / 4.0
    return {"constant": float(constant), "linear_z": h, "quadratic_zz": j_coeff}


def evaluate_ising_energy(support, surrogate: dict, ising: dict | None = None) -> float:
    """Evaluate the Ising representation for a support."""
    if ising is None:
        ising = qubo_to_ising_coefficients(surrogate)
    bits = support_to_bits(support, {"ASSETS": surrogate["assets"]})
    z = {asset: 1 - 2 * bit for asset, bit in zip(surrogate["assets"], bits)}
    energy = float(ising["constant"])
    energy += sum(float(coeff) * z[asset] for asset, coeff in ising["linear_z"].items())
    for (i, j), coeff in ising["quadratic_zz"].items():
        energy += float(coeff) * z[i] * z[j]
    return float(energy)


def enumerate_exact_k_energies(instance: dict, surrogate: dict) -> pd.DataFrame:
    """Enumerate all exact-K supports and their surrogate energy."""
    assets = _assets(instance)
    rows = []
    for combo in itertools.combinations(assets, instance["K"]):
        selection = {asset: (1 if asset in combo else 0) for asset in assets}
        energy = evaluate_surrogate_energy(selection, surrogate)
        comp = support_composition(selection, instance)
        rows.append(
            {
                "energy": energy,
                "support": tuple(combo),
                "bits": "".join(str(selection[a]) for a in assets),
                "structural_feasible": None,
                "composition": tuple(comp[g] for g in instance["SECTORS"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["energy", "bits"]).reset_index(drop=True)


def exact_k_scaling(instance: dict, raw_linear: dict, raw_quadratic: dict) -> dict:
    raw_surrogate = {
        "surrogate_type": "raw",
        "assets": _assets(instance),
        "linear": raw_linear,
        "quadratic": raw_quadratic,
    }
    df = enumerate_exact_k_energies(instance, raw_surrogate)
    mean = float(df["energy"].mean())
    max_abs_centered = float((df["energy"] - mean).abs().max())
    return {
        "mean_exactK": mean,
        "max_abs_centered_exactK": max_abs_centered,
        "epsilon": EPS,
        "n_exactK": int(len(df)),
        "raw_min_exactK": float(df["energy"].min()),
        "raw_max_exactK": float(df["energy"].max()),
        "raw_std_exactK": float(df["energy"].std(ddof=0)),
    }


def scale_qubo(instance: dict, raw_linear: dict, raw_quadratic: dict) -> tuple[dict, dict, dict]:
    scaling = exact_k_scaling(instance, raw_linear, raw_quadratic)
    denom = scaling["max_abs_centered_exactK"] + EPS
    linear = {asset: float(value) / denom for asset, value in raw_linear.items()}
    quadratic = {pair: float(value) / denom for pair, value in raw_quadratic.items()}
    return linear, quadratic, scaling


def build_energy_hub_s0(instance: dict, k: int | None = None, diagnostics: bool = True) -> dict:
    """Build S0: instantaneous service-capacity surrogate with same-family redundancy."""
    d = _energy_data(instance)
    assets = _assets(instance)
    raw_linear = {}
    singleton = {}
    for asset in assets:
        a_e = d["Smax"][asset] * float(d["rho"].loc[asset].mean()) * max(0.0, d["etaE"][asset])
        a_q = d["Smax"][asset] * float(d["rho"].loc[asset].max()) * max(0.0, d["etaQ"][asset])
        a_c = d["Smax"][asset] * float(d["rho"].loc[asset].max()) * max(0.0, d["etaC"][asset])
        cost = d["CRF"] * d["a"][asset] + 0.1 * d["alpha"][asset]
        singleton[asset] = 0.5 * a_e + a_q + a_c - cost
    scale = max([abs(v) for v in singleton.values()] + [1.0])
    utility = {asset: singleton[asset] / scale for asset in assets}
    raw_linear = {asset: -utility[asset] for asset in assets}
    raw_quadratic = {}
    for pos, i in enumerate(assets):
        for j in assets[pos + 1 :]:
            if d["group_of"][i] == d["group_of"][j]:
                raw_quadratic[(i, j)] = 0.08

    linear, quadratic, scaling = scale_qubo(instance, raw_linear, raw_quadratic)
    result = {
        "surrogate_type": "S0",
        "assets": assets,
        "linear": linear,
        "quadratic": quadratic,
        "raw_linear": raw_linear,
        "raw_quadratic": raw_quadratic,
        "scaling": scaling,
        "diagnostics": {
            "singleton_raw": singleton,
            "utility": utility,
            "instantaneous_weights": {"E": 0.5, "Q": 1.0, "C": 1.0, "fixed": 1.0},
            "redundancy_lambda": 0.08,
        }
        if diagnostics
        else {},
    }
    result["hash"] = surrogate_hash(result)
    result["coefficient_hash"] = result["hash"]
    return result


def _service_matrices(instance: dict, use_proxy_capacity: bool = False, kappa=None, mu=None):
    d = _energy_data(instance)
    assets = _assets(instance)
    periods = d["periods"]
    services = {"E": d["etaE"], "Q": d["etaQ"], "C": d["etaC"]}
    matrices = {s: pd.DataFrame(index=assets, columns=periods, dtype=float) for s in services}
    for s, eta in services.items():
        for asset in assets:
            for t in periods:
                capacity = d["Smax"][asset]
                multiplier = 1.0
                if use_proxy_capacity:
                    capacity = kappa[asset] * d["Smax"][asset]
                    multiplier = mu.loc[asset, t]
                matrices[s].loc[asset, t] = (
                    multiplier
                    * d["rho"].loc[asset, t]
                    * capacity
                    * max(0.0, float(eta[asset]))
                )
    return matrices


def _demand_star(instance: dict) -> dict:
    d = _energy_data(instance)
    return {"E": 0.6 * d["DE"], "Q": d["DQ"], "C": d["DC"]}


def _service_pressure(instance: dict, matrices: dict) -> dict:
    dstar = _demand_star(instance)
    periods = _energy_data(instance)["periods"]
    chi = {}
    for s in ("E", "Q", "C"):
        chi[s] = pd.Series(index=periods, dtype=float)
        for t in periods:
            chi[s][t] = min(1.0, float(dstar[s][t]) / (float(matrices[s][t].sum()) + EPS))
    return chi


def _s1_mu_kappa(instance: dict) -> tuple[pd.DataFrame, pd.Series, dict, dict]:
    d = _energy_data(instance)
    assets = _assets(instance)
    periods = d["periods"]
    full_matrices = _service_matrices(instance)
    chi = _service_pressure(instance, full_matrices)
    mu = pd.DataFrame(index=assets, columns=periods, dtype=float)
    for asset in assets:
        for t in periods:
            outputs = np.array([full_matrices[s].loc[asset, t] for s in ("E", "Q", "C")])
            denom = float(outputs.sum())
            if denom <= EPS:
                mu.loc[asset, t] = 0.0
            else:
                mu.loc[asset, t] = sum(
                    (outputs[idx] / denom) * chi[s][t]
                    for idx, s in enumerate(("E", "Q", "C"))
                )

    kappa = pd.Series(index=assets, dtype=float)
    for asset in assets:
        denom = sum(d["tau"][t] * d["rho"].loc[asset, t] for t in periods)
        if denom <= EPS:
            kappa[asset] = 0.0
        else:
            kappa[asset] = min(
                1.0,
                max(
                    0.0,
                    float(
                        sum(d["tau"][t] * mu.loc[asset, t] * d["rho"].loc[asset, t] for t in periods)
                        / denom
                    ),
                ),
            )
    return mu, kappa, full_matrices, chi


def build_energy_hub_s1(instance: dict, k: int | None = None, diagnostics: bool = True) -> dict:
    """
    Build S1: economic linear annualized individual-cost surrogate.
    """
    d = _energy_data(instance)
    assets = _assets(instance)
    periods = d["periods"]
    mu, kappa, full_matrices, chi = _s1_mu_kappa(instance)
    proxy_matrices = _service_matrices(instance, use_proxy_capacity=True, kappa=kappa, mu=mu)

    benefits = {s: pd.Series(0.0, index=assets) for s in ("E", "Q", "C")}
    for t in periods:
        cap = d["tau"][t] * d["cE"][t] * (0.4 * d["DE"][t])
        denom = float(proxy_matrices["E"][t].sum()) + EPS
        for asset in assets:
            benefits["E"][asset] += cap * proxy_matrices["E"].loc[asset, t] / denom

    p_proxy = {}
    for asset in assets:
        s_proxy = kappa[asset] * d["Smax"][asset]
        for t in periods:
            p_proxy[(asset, t)] = mu.loc[asset, t] * d["rho"].loc[asset, t] * s_proxy

    internal_prices = {}
    dstar = _demand_star(instance)
    for s in ("Q", "C"):
        internal_prices[s] = pd.Series(index=periods, dtype=float)
        for t in periods:
            candidates = []
            for asset in assets:
                output = proxy_matrices[s].loc[asset, t]
                if output > EPS:
                    period_cost = (
                        d["alpha"][asset] * p_proxy[(asset, t)]
                        + d["beta"][asset] * p_proxy[(asset, t)] ** 2
                    )
                    candidates.append(period_cost / (output + EPS))
            internal_prices[s][t] = min(candidates) if candidates else 0.0
            cap = d["tau"][t] * internal_prices[s][t] * dstar[s][t]
            denom = float(proxy_matrices[s][t].sum()) + EPS
            for asset in assets:
                benefits[s][asset] += cap * proxy_matrices[s].loc[asset, t] / denom

    cost_blocks = {
        "fixed": pd.Series(index=assets, dtype=float),
        "capacity": pd.Series(index=assets, dtype=float),
        "op_linear": pd.Series(index=assets, dtype=float),
        "op_quadratic": pd.Series(index=assets, dtype=float),
    }
    raw_linear = {}
    for asset in assets:
        s_proxy = kappa[asset] * d["Smax"][asset]
        cost_blocks["fixed"][asset] = d["CRF"] * d["a"][asset]
        cost_blocks["capacity"][asset] = d["CRF"] * d["b"][asset] * s_proxy
        cost_blocks["op_linear"][asset] = sum(
            d["tau"][t] * d["alpha"][asset] * p_proxy[(asset, t)] for t in periods
        )
        cost_blocks["op_quadratic"][asset] = sum(
            d["tau"][t] * d["beta"][asset] * p_proxy[(asset, t)] ** 2 for t in periods
        )
        raw_linear[asset] = (
            cost_blocks["fixed"][asset]
            + cost_blocks["capacity"][asset]
            + cost_blocks["op_linear"][asset]
            + cost_blocks["op_quadratic"][asset]
            - benefits["E"][asset]
            - benefits["Q"][asset]
            - benefits["C"][asset]
        )

    raw_quadratic = {}
    linear, quadratic, scaling = scale_qubo(instance, raw_linear, raw_quadratic)
    result = {
        "surrogate_type": "S1",
        "assets": assets,
        "linear": linear,
        "quadratic": quadratic,
        "raw_linear": raw_linear,
        "raw_quadratic": raw_quadratic,
        "scaling": scaling,
        "kappa": _series_to_float_dict(kappa),
        "mu": _frame_to_float_dict(mu),
        "cost_blocks": {key: _series_to_float_dict(value) for key, value in cost_blocks.items()},
        "benefit_blocks": {key: _series_to_float_dict(value) for key, value in benefits.items()},
        "diagnostics": {
            "service_pressure": {key: _series_to_float_dict(value) for key, value in chi.items()},
            "internal_prices": {
                key: _series_to_float_dict(value) for key, value in internal_prices.items()
            },
            "electric_benefit_cap": float(
                sum(d["tau"][t] * d["cE"][t] * (0.4 * d["DE"][t]) for t in periods)
            ),
        }
        if diagnostics
        else {},
    }
    result["hash"] = surrogate_hash(result)
    result["coefficient_hash"] = result["hash"]
    return result


def _s2_time_weights(instance: dict) -> dict:
    d = _energy_data(instance)
    dstar = _demand_star(instance)
    weights = {}
    for s in ("E", "Q", "C"):
        denom = sum(d["tau"][t] * dstar[s][t] for t in d["periods"])
        if denom <= EPS:
            weights[s] = pd.Series(
                {t: 1.0 / len(d["periods"]) for t in d["periods"]}, dtype=float
            )
        else:
            weights[s] = pd.Series(
                {t: d["tau"][t] * dstar[s][t] / denom for t in d["periods"]},
                dtype=float,
            )
    return weights


def build_energy_hub_s2(instance: dict, k: int | None = None, diagnostics: bool = True) -> dict:
    """
    Build S2: temporal physical coverage surrogate.
    """
    d = _energy_data(instance)
    assets = _assets(instance)
    periods = d["periods"]
    matrices = _service_matrices(instance)
    dstar = _demand_star(instance)
    time_weights = _s2_time_weights(instance)
    service_weights = {"E": 1.0 / 3.0, "Q": 1.0 / 3.0, "C": 1.0 / 3.0}

    relative = {s: pd.DataFrame(index=assets, columns=periods, dtype=float) for s in ("E", "Q", "C")}
    for s in ("E", "Q", "C"):
        for asset in assets:
            for t in periods:
                if dstar[s][t] <= EPS:
                    relative[s].loc[asset, t] = 0.0
                else:
                    relative[s].loc[asset, t] = min(
                        1.0, matrices[s].loc[asset, t] / (dstar[s][t] + EPS)
                    )

    service_contrib = {
        s: pd.Series(index=assets, dtype=float) for s in ("E", "Q", "C")
    }
    raw_b = pd.Series(index=assets, dtype=float)
    for asset in assets:
        total = 0.0
        for s in ("E", "Q", "C"):
            service_contrib[s][asset] = sum(
                time_weights[s][t] * relative[s].loc[asset, t] for t in periods
            )
            total += service_weights[s] * service_contrib[s][asset]
        raw_b[asset] = total

    raw_linear = {asset: -float(raw_b[asset]) for asset in assets}
    raw_quadratic = {}
    linear, quadratic, scaling = scale_qubo(instance, raw_linear, raw_quadratic)
    result = {
        "surrogate_type": "S2",
        "assets": assets,
        "linear": linear,
        "quadratic": quadratic,
        "raw_linear": raw_linear,
        "raw_quadratic": raw_quadratic,
        "scaling": scaling,
        "service_weights": service_weights,
        "time_weights": {key: _series_to_float_dict(value) for key, value in time_weights.items()},
        "relative_contributions": {key: _frame_to_float_dict(value) for key, value in relative.items()},
        "diagnostics": {
            "service_contributions": {
                key: _series_to_float_dict(value) for key, value in service_contrib.items()
            },
            "b": _series_to_float_dict(raw_b),
            "zero_demand_periods": {
                s: [t for t in periods if float(dstar[s][t]) == 0.0]
                for s in ("E", "Q", "C")
            },
        }
        if diagnostics
        else {},
    }
    result["hash"] = surrogate_hash(result)
    result["coefficient_hash"] = result["hash"]
    return result


def build_energy_hub_s0_legacy_corrected(instance: dict, k: int | None = None, diagnostics: bool = True) -> dict:
    return build_energy_hub_s0(instance, k=k, diagnostics=diagnostics)


def build_energy_hub_s1_economic_linear(instance: dict, k: int | None = None, diagnostics: bool = True) -> dict:
    return build_energy_hub_s1(instance, k=k, diagnostics=diagnostics)


def build_energy_hub_s2_temporal_coverage(instance: dict, k: int | None = None, diagnostics: bool = True) -> dict:
    return build_energy_hub_s2(instance, k=k, diagnostics=diagnostics)
