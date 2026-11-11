"""Solver tolerances and constraint residuals of the exhaustive Energy Hub reference.

Re-solves every support that passes the structural screen (5,931) with the same
fixed-support model as hybrid_pipeline.benchmarks.energy_hub.model.solve_energy_hub_support
and the same solver settings (Ipopt 3.14.19 with default options), and records for each
support:
  * the Ipopt exit message, its final (unscaled) dual infeasibility, constraint
    violation, complementarity and overall NLP error, and the iteration count;
  * the maximum absolute and relative constraint residuals recomputed from the returned
    point, and the most negative variable value (bound violation);
  * the TAC and its difference from the archived reference value.
For supports that Ipopt reports as infeasible, feasibility is checked independently
with a linear program (the constraints are linear): SciPy/HiGHS minimizes the total
violation of the constraint set.

Usage (from energy_hub/): python ipopt_residuals/resolve_reference.py OUT_DIR [n_workers]
"""
import json, os, re, sys, tempfile, time
sys.dont_write_bytecode = True
EH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(EH_ROOT, "code")
DATA = os.path.join(EH_ROOT, "data")
sys.path.insert(0, CODE)
import numpy as np


def _instance():
    from hybrid_pipeline.benchmarks.energy_hub import build_energy_hub_instance
    return build_energy_hub_instance()


def build_model(bits, d):
    """Identical to solve_energy_hub_support (model.py lines 118-156)."""
    import pyomo.environ as pyo
    techs, periods = d["techs"], d["periods"]
    selection = {i: int(b) for i, b in zip(techs, bits)}
    m = pyo.ConcreteModel()
    m.s = pyo.Var(techs, within=pyo.NonNegativeReals)
    m.P = pyo.Var(techs, periods, within=pyo.NonNegativeReals)
    m.G = pyo.Var(periods, within=pyo.NonNegativeReals)
    m.cap = pyo.ConstraintList()
    for i in techs:
        m.cap.add(m.s[i] <= d["Smax"][i] * selection[i])
        for t in periods:
            m.cap.add(m.P[i, t] <= d["rho"].loc[i, t] * m.s[i])
    m.e_balance = pyo.ConstraintList(); m.q_cover = pyo.ConstraintList(); m.c_cover = pyo.ConstraintList()
    for t in periods:
        m.e_balance.add(sum(d["etaE"][i] * m.P[i, t] for i in techs) + m.G[t] == d["DE"][t])
        m.cap.add(m.G[t] <= 0.4 * d["DE"][t])
        m.q_cover.add(sum(d["etaQ"][i] * m.P[i, t] for i in techs) >= d["DQ"][t])
        m.c_cover.add(sum(d["etaC"][i] * m.P[i, t] for i in techs) >= d["DC"][t])
    capex = d["CRF"] * sum(d["a"][i] * selection[i] + d["b"][i] * m.s[i] for i in techs)
    opex = sum(d["tau"][t] * (sum(d["alpha"][i] * m.P[i, t] + d["beta"][i] * m.P[i, t] ** 2 for i in techs)
                              + d["cE"][t] * m.G[t]) for t in periods)
    m.obj = pyo.Objective(expr=capex + opex, sense=pyo.minimize)
    return m, selection


def residuals(m):
    import pyomo.environ as pyo
    worst_abs, worst_rel, worst_name = 0.0, 0.0, ""
    for c in m.component_data_objects(pyo.Constraint, active=True):
        body = pyo.value(c.body)
        lo = pyo.value(c.lower) if c.has_lb() else None
        up = pyo.value(c.upper) if c.has_ub() else None
        v = 0.0; ref = 1.0
        if lo is not None and body < lo:
            v = lo - body; ref = max(1.0, abs(lo))
        if up is not None and body > up:
            v = body - up; ref = max(1.0, abs(up))
        if v > worst_abs:
            worst_abs, worst_name = v, c.name
        worst_rel = max(worst_rel, v / ref)
    neg = min([pyo.value(x) for x in m.component_data_objects(pyo.Var)] + [0.0])
    return worst_abs, worst_rel, worst_name, neg


def parse_log(txt):
    out = {}
    m = re.search(r"EXIT:\s*(.+)", txt); out["ipopt_exit"] = m.group(1).strip() if m else ""
    m = re.search(r"Number of Iterations\.+:\s*(\d+)", txt); out["iterations"] = int(m.group(1)) if m else None
    for key, lab in (("dual_inf", "Dual infeasibility"), ("constr_viol", "Constraint violation"),
                     ("complementarity", "Complementarity"), ("nlp_error", "Overall NLP error")):
        m = re.search(lab + r"\.*:\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)", txt)
        if m:
            out[key + "_scaled"] = float(m.group(1)); out[key + "_unscaled"] = float(m.group(2))
    return out


def lp_violation(bits, d):
    """Independent phase-1 LP: minimize total slack of the (linear) constraint set with HiGHS."""
    from scipy.optimize import linprog
    techs, periods = list(d["techs"]), list(d["periods"]); nI, nT = len(techs), len(periods)
    sel = [int(b) for b in bits]
    nx = nI + nI * nT + nT
    iS = lambda i: i; iP = lambda i, t: nI + i * nT + t; iG = lambda t: nI + nI * nT + t
    A, b, eqA, eqb = [], [], [], []
    for i, ti in enumerate(techs):
        r = np.zeros(nx); r[iS(i)] = 1; A.append(r); b.append(d["Smax"][ti] * sel[i])
        for t, tt in enumerate(periods):
            r = np.zeros(nx); r[iP(i, t)] = 1; r[iS(i)] = -d["rho"].loc[ti, tt]; A.append(r); b.append(0.0)
    for t, tt in enumerate(periods):
        r = np.zeros(nx); r[iG(t)] = 1; A.append(r); b.append(0.4 * d["DE"][tt])
        r = np.zeros(nx)
        for i, ti in enumerate(techs): r[iP(i, t)] = d["etaE"][ti]
        r[iG(t)] = 1; eqA.append(r); eqb.append(d["DE"][tt])
        r = np.zeros(nx)
        for i, ti in enumerate(techs): r[iP(i, t)] = -d["etaQ"][ti]
        A.append(r); b.append(-d["DQ"][tt])
        r = np.zeros(nx)
        for i, ti in enumerate(techs): r[iP(i, t)] = -d["etaC"][ti]
        A.append(r); b.append(-d["DC"][tt])
    A, b, eqA, eqb = np.array(A), np.array(b), np.array(eqA), np.array(eqb)
    nu, ne = len(b), len(eqb)
    c = np.concatenate([np.zeros(nx), np.ones(nu), np.ones(2 * ne)])
    Aub = np.hstack([A, -np.eye(nu), np.zeros((nu, 2 * ne))])
    Aeq = np.hstack([eqA, np.zeros((ne, nu)), np.eye(ne), -np.eye(ne)])
    res = linprog(c, A_ub=Aub, b_ub=b, A_eq=Aeq, b_eq=eqb, bounds=[(0, None)] * len(c), method="highs")
    return float(res.fun) if res.status == 0 else float("nan"), res.status


def task(args):
    bits, out_dir = args
    path = os.path.join(out_dir, f"{bits}.json")
    if os.path.exists(path):
        return path
    import pyomo.environ as pyo
    from hybrid_pipeline.cli.config import build_solver
    d = _instance()["ENERGY_HUB_DATA"]
    m, _ = build_model(bits, d)
    solver = build_solver()
    fd, logf = tempfile.mkstemp(suffix=".log"); os.close(fd)
    t0 = time.time()
    try:
        res = solver.solve(m, tee=False, logfile=logf, load_solutions=False)
        tc = str(res.solver.termination_condition)
        if res.solver.termination_condition == pyo.TerminationCondition.optimal:
            m.solutions.load_from(res)
    except Exception as e:
        tc = f"error: {e}"
    rt = time.time() - t0
    txt = open(logf, encoding="utf-8", errors="replace").read(); os.remove(logf)
    row = {"support_bitstring": bits, "termination": tc, "runtime_s": rt, **parse_log(txt)}
    if tc == "optimal":
        a, r, n, neg = residuals(m)
        row.update(TAC=float(pyo.value(m.obj)), max_abs_residual=a, max_rel_residual=r, worst_constraint=n,
                   min_var_value=neg)
    else:
        v, st = lp_violation(bits, d)
        row.update(lp_min_total_violation=v, lp_status=st)
    json.dump(row, open(path, "w"))
    return path


if __name__ == "__main__":
    import multiprocessing as mp
    import pandas as pd
    out_dir = sys.argv[1]; nw = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    os.makedirs(out_dir, exist_ok=True)
    ref = pd.read_csv(os.path.join(DATA, "reference", "energy_hub_exhaustive_classical_rows.csv"),
                      dtype={"support_bitstring": str})
    todo = ref[ref.method == "ipopt"].support_bitstring.tolist()
    print("supports to re-solve:", len(todo), flush=True)
    t0 = time.time(); n = 0
    with mp.Pool(nw) as pool:
        for _ in pool.imap_unordered(task, [(b, out_dir) for b in todo], chunksize=8):
            n += 1
            if n % 500 == 0:
                print(f"[{time.time() - t0:7.1f}s] {n}/{len(todo)}", flush=True)
    print(f"DONE {n} in {time.time() - t0:.0f} s", flush=True)
