"""Continuous evaluator and structural filter for Energy Hub supports."""

from __future__ import annotations

import time

import pyomo.environ as pyo

from hybrid_pipeline.cli.config import SOLVER_NAME, build_solver


def _selected(selection: dict, instance: dict) -> list:
    return [i for i in instance["ENERGY_HUB_DATA"]["techs"] if selection[i] == 1]


def structural_filter(selection: dict, instance: dict, import_cap_frac: float = 0.4) -> bool:
    """
    Cheap necessary Energy Hub structural feasibility screen.

    This is the extracted equivalent of ``EnergyHubBenchmark.structural_filter``.

    ``import_cap_frac`` is the fraction of period electricity demand that may
    be covered by grid imports (constraint (8) in the manuscript). It
    defaults to 0.4 -- the value used throughout the canonical campaign,
    the frozen exhaustive reference, and the S1/S2 surrogates -- so every
    existing call site is completely unaffected. It is only ever overridden
    explicitly, by the supplementary import-cap sensitivity appendix.
    """
    d = instance["ENERGY_HUB_DATA"]
    selected = _selected(selection, instance)
    if not selected:
        return False
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
        if e_cap + import_cap_frac * d["DE"][t] + 1e-9 < d["DE"][t]:
            return False
        if q_cap + 1e-9 < d["DQ"][t]:
            return False
        if c_cap + 1e-9 < d["DC"][t]:
            return False
    return True


def solve_energy_hub_support(selection, instance, tee: bool = False, solver=None, import_cap_frac: float = 0.4) -> dict:
    """Solve the continuous Energy Hub NLP for a fixed binary support."""
    d = instance["ENERGY_HUB_DATA"]
    selected = _selected(selection, instance)
    K = instance["K"]

    if len(selected) != K:
        return dict(
            feasible=False,
            TAC=None,
            selected_tech_ids=selected,
            installed_capacities=None,
            dispatch=None,
            grid_imports=None,
            solver_status="non_exact_K",
            tc="non_exact_K",
            runtime=0.0,
            objective_value=None,
            objective_score=None,
            variance=None,
            ret_slack=None,
            active_assets=None,
            active_count=None,
            selected_count=len(selected),
        )

    if not structural_filter(selection, instance, import_cap_frac=import_cap_frac):
        return dict(
            feasible=False,
            TAC=None,
            selected_tech_ids=selected,
            installed_capacities=None,
            dispatch=None,
            grid_imports=None,
            solver_status="structural_filter",
            tc="structural_filter",
            runtime=0.0,
            objective_value=None,
            objective_score=None,
            variance=None,
            ret_slack=None,
            infeasible_by_structure=True,
            active_assets=None,
            active_count=None,
            selected_count=K,
        )

    if solver is None:
        solver = build_solver()
    if not solver.available(False):
        return dict(
            feasible=False,
            TAC=None,
            selected_tech_ids=selected,
            installed_capacities=None,
            dispatch=None,
            grid_imports=None,
            solver_status=f"{SOLVER_NAME}_not_available",
            tc=f"{SOLVER_NAME}_not_available",
            runtime=0.0,
            objective_value=None,
            objective_score=None,
            variance=None,
            ret_slack=None,
            solver_unavailable=True,
            active_assets=None,
            active_count=None,
            selected_count=len(selected),
        )

    m = pyo.ConcreteModel()
    techs = d["techs"]
    periods = d["periods"]
    m.s = pyo.Var(techs, within=pyo.NonNegativeReals)
    m.P = pyo.Var(techs, periods, within=pyo.NonNegativeReals)
    m.G = pyo.Var(periods, within=pyo.NonNegativeReals)

    m.cap = pyo.ConstraintList()
    for i in techs:
        m.cap.add(m.s[i] <= d["Smax"][i] * selection[i])
        for t in periods:
            m.cap.add(m.P[i, t] <= d["rho"].loc[i, t] * m.s[i])

    m.e_balance = pyo.ConstraintList()
    m.q_cover = pyo.ConstraintList()
    m.c_cover = pyo.ConstraintList()
    for t in periods:
        m.e_balance.add(
            sum(d["etaE"][i] * m.P[i, t] for i in techs) + m.G[t] == d["DE"][t]
        )
        m.cap.add(m.G[t] <= import_cap_frac * d["DE"][t])
        m.q_cover.add(sum(d["etaQ"][i] * m.P[i, t] for i in techs) >= d["DQ"][t])
        m.c_cover.add(sum(d["etaC"][i] * m.P[i, t] for i in techs) >= d["DC"][t])

    capex = d["CRF"] * sum(
        d["a"][i] * selection[i] + d["b"][i] * m.s[i] for i in techs
    )
    opex = sum(
        d["tau"][t]
        * (
            sum(
                d["alpha"][i] * m.P[i, t] + d["beta"][i] * m.P[i, t] ** 2
                for i in techs
            )
            + d["cE"][t] * m.G[t]
        )
        for t in periods
    )
    m.obj = pyo.Objective(expr=capex + opex, sense=pyo.minimize)

    t0 = time.time()
    res = solver.solve(m, tee=tee)
    runtime = time.time() - t0
    tc = getattr(res.solver, "termination_condition", None)
    if tc != pyo.TerminationCondition.optimal:
        return dict(
            feasible=False,
            TAC=None,
            selected_tech_ids=selected,
            installed_capacities=None,
            dispatch=None,
            grid_imports=None,
            solver_status=str(tc),
            tc=str(tc),
            runtime=runtime,
            objective_value=None,
            objective_score=None,
            variance=None,
            ret_slack=None,
            qp_infeasible=True,
            active_count=None,
            selected_count=K,
        )

    capacities = {i: float(pyo.value(m.s[i])) for i in selected}
    dispatch = {
        i: {t: float(pyo.value(m.P[i, t])) for t in periods}
        for i in selected
    }
    grid = {t: float(pyo.value(m.G[t])) for t in periods}
    tac = float(pyo.value(m.obj))
    active = [i for i, val in capacities.items() if val > 1e-8]

    return dict(
        feasible=True,
        TAC=tac,
        objective_value=tac,
        objective_score=tac,
        selected_tech_ids=selected,
        installed_capacities=capacities,
        dispatch=dispatch,
        grid_imports=grid,
        solver_status=str(tc),
        runtime=runtime,
        variance=None,
        ret_slack=0.0,
        qp_optimal=True,
        active_assets=active,
        active_count=len(active),
        selected_count=K,
        tc=str(tc),
    )
