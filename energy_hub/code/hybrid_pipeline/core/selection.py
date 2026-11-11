"""Candidate-pool construction for the QAOA sampler.

Only build_candidate_pool (used by build_qaoa_sampler in samplers/qaoa.py) is provided.
"""

from __future__ import annotations

import numpy as np


def _energy_hub_surrogate_score(selection: dict, instance: dict) -> float:
    d = instance["ENERGY_HUB_DATA"]
    selected = [i for i in d["techs"] if selection[i] == 1]
    K = instance["K"]
    if len(selected) != K:
        return -1e9

    group_counts = {g: 0 for g in d["groups"]}
    for i in selected:
        group_counts[d["group_of"][i]] += 1

    e_margins, q_margins, c_margins = [], [], []
    for t in d["periods"]:
        e_cap = sum(
            d["Smax"][i] * d["rho"].loc[i, t] * max(0.0, d["etaE"][i])
            for i in selected
        )
        q_cap = sum(
            d["Smax"][i] * d["rho"].loc[i, t] * max(0.0, d["etaQ"][i])
            for i in selected
        )
        c_cap = sum(
            d["Smax"][i] * d["rho"].loc[i, t] * max(0.0, d["etaC"][i])
            for i in selected
        )
        e_margins.append(e_cap + 0.4 * d["DE"][t] - d["DE"][t])
        q_margins.append(q_cap - d["DQ"][t])
        c_margins.append(c_cap - d["DC"][t])

    heat_margin = min(q_margins)
    cool_margin = min(c_margins)
    elec_margin = min(e_margins)
    elec_capacity = sum(
        d["Smax"][i] * float(d["rho"].loc[i].mean()) * max(0.0, d["etaE"][i])
        for i in selected
    )
    fixed_cost = sum(d["CRF"] * d["a"][i] for i in selected)
    variable_cost = sum(d["alpha"][i] for i in selected) / K

    missing_service_penalty = (
        2.0 * max(0.0, -elec_margin)
        + 2.4 * max(0.0, -heat_margin)
        + 2.4 * max(0.0, -cool_margin)
    )
    represented_groups = sum(1 for v in group_counts.values() if v > 0)
    diversity_bonus = 8.0 * represented_groups
    redundancy_penalty = 6.0 * sum(max(0, v - 2) ** 2 for v in group_counts.values())
    target_share = K / max(len(group_counts), 1)
    balance_penalty = 1.5 * sum((v - target_share) ** 2 for v in group_counts.values())

    return (
        0.6 * elec_margin
        + 0.8 * heat_margin
        + 0.8 * cool_margin
        + 0.25 * elec_capacity
        + diversity_bonus
        - missing_service_penalty
        - redundancy_penalty
        - balance_penalty
        - 0.05 * fixed_cost
        - variable_cost
    )


def _candidate_composition(selection: dict, instance: dict) -> dict:
    counts = {group: 0 for group in instance["SECTORS"]}
    for asset in instance["ASSETS"]:
        if selection[asset] == 1:
            counts[instance["SECTOR_OF"][asset]] += 1
    return counts


def _structural_feasible(selection: dict, instance: dict):
    if instance.get("BENCHMARK") != "energy_hub":
        return None
    from hybrid_pipeline.benchmarks.energy_hub.model import structural_filter

    return bool(structural_filter(selection, instance))


def build_candidate_pool(
    raw_counts: dict,
    shots: int,
    instance: dict,
    alpha: float,
    beta: float,
    surrogate: dict | None = None,
    surrogate_type: str | None = None,
) -> list:
    """
    Build a structured candidate pool from sampler counts.
    """
    ASSETS = instance["ASSETS"]
    K = instance["K"]
    N = instance["N"]

    if instance.get("BENCHMARK") == "energy_hub":
        if surrogate is not None:
            from hybrid_pipeline.benchmarks.energy_hub.surrogates import (
                evaluate_surrogate_energy,
            )

            candidate_pool = []
            for b, count in raw_counts.items():
                sel = {ASSETS[i]: b[i] for i in range(N)}
                card = sum(b)
                is_exactK = card == K
                energy = (
                    float(evaluate_surrogate_energy(sel, surrogate))
                    if is_exactK
                    else np.inf
                )
                candidate_pool.append(
                    dict(
                        selection=sel,
                        support=[ASSETS[i] for i in range(N) if b[i] == 1],
                        bits=b,
                        count=count,
                        sample_count=count,
                        freq=count / shots,
                        sample_probability=count / shots,
                        cardinality=card,
                        is_exactK=is_exactK,
                        surrogate_type=surrogate_type
                        or surrogate.get("surrogate_type", "unknown"),
                        surrogate_hash=surrogate.get("hash"),
                        surrogate_energy=energy,
                        structural_feasible=(
                            _structural_feasible(sel, instance) if is_exactK else False
                        ),
                        composition=_candidate_composition(sel, instance),
                        objective_score=-energy if np.isfinite(energy) else 0.0,
                        structural_score=-energy if np.isfinite(energy) else 0.0,
                        r_score=-energy if np.isfinite(energy) else 0.0,
                        risk_score=0.0,
                        hybrid_score=-energy if np.isfinite(energy) else 0.0,
                    )
                )
            return candidate_pool

        candidate_pool = []
        scores = []
        for b, count in raw_counts.items():
            sel = {ASSETS[i]: b[i] for i in range(N)}
            card = sum(b)
            is_exactK = card == K
            score = float(_energy_hub_surrogate_score(sel, instance)) if is_exactK else 0.0
            scores.append(score)
            candidate_pool.append((b, count, sel, card, is_exactK, score))
        scale = max([abs(s) for s in scores] + [1.0])
        return [
            dict(
                selection=sel,
                bits=b,
                count=count,
                freq=count / shots,
                cardinality=card,
                is_exactK=is_exactK,
                objective_score=(score / scale if is_exactK else 0.0),
                structural_score=(score / scale if is_exactK else 0.0),
                r_score=(score / scale if is_exactK else 0.0),
                risk_score=0.0,
                hybrid_score=(score / scale if is_exactK else 0.0),
            )
            for b, count, sel, card, is_exactK, score in candidate_pool
        ]

    mu = instance["mu"]
    cov = instance["cov"]
    R = instance["R"]

    mu_vals = mu.loc[ASSETS].astype(float).values
    mu_signal = np.maximum(mu_vals - R, 0.0)
    mu_scale = float(np.max(np.abs(mu_signal)))
    if mu_scale > 1e-12:
        mu_hat = mu_signal / mu_scale
    else:
        mu_hat = np.zeros_like(mu_signal)

    cov_vals = cov.loc[ASSETS, ASSETS].astype(float).values
    cov_scale = float(np.max(np.abs(cov_vals)))
    if cov_scale > 1e-12:
        cov_hat = cov_vals / cov_scale
    else:
        cov_hat = np.zeros_like(cov_vals)

    candidate_pool = []

    for b, count in raw_counts.items():
        freq = count / shots
        sel = {ASSETS[i]: b[i] for i in range(N)}
        card = sum(b)
        is_exactK = card == K

        if is_exactK:
            chosen_idx = [i for i in range(N) if b[i] == 1]
            r_score = float(sum(mu_hat[i] for i in chosen_idx))
            risk_score = float(
                sum(cov_hat[i, i] for i in chosen_idx)
                + 2.0
                * sum(
                    cov_hat[i, j]
                    for pos, i in enumerate(chosen_idx)
                    for j in chosen_idx[pos + 1 :]
                )
            )
            hybrid_score = alpha * r_score - beta * risk_score
        else:
            r_score = risk_score = hybrid_score = 0.0

        candidate_pool.append(
            dict(
                selection=sel,
                bits=b,
                count=count,
                freq=freq,
                cardinality=card,
                is_exactK=is_exactK,
                objective_score=hybrid_score,
                structural_score=hybrid_score,
                r_score=r_score,
                risk_score=risk_score,
                hybrid_score=hybrid_score,
            )
        )

    return candidate_pool
