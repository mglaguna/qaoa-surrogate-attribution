"""Initialization controls for the exact-K QAOA samplers (both benchmarks).

Three controls, for the Energy Hub main study (S0, S1, S2; seeds 101-110) and the
portfolio benchmark (QAOA-surrogate, p = 1; seeds 0-9):

  (i)   zero depth / initial support only: the sampler returns the initial support y0;
  (ii)  alternative initial supports: y0 replaced by other exact-K supports, same ring;
  (iii) consistent asset-to-qubit permutations: the ring ordering of the XY mixer and
        the "first K" initial support are permuted together (the cost Hamiltonian is
        unchanged because it is defined on the physical assets/technologies).

Everything else (surrogate, depth, optimizer, steps, training/sampling seeds, shots)
is identical to the archived runs. The "orig" variant (identity ordering, y0 = first K)
must reproduce the archived samples exactly; this is checked in `analyze`.
No continuous solves are issued: values come from the exhaustive references.

Usage (PennyLane 0.38.0; one BLAS/OpenMP thread):
  python initialization_controls.py eh  OUT_DIR [n_workers]
  python initialization_controls.py pf  OUT_DIR [n_workers]
  python initialization_controls.py analyze OUT_DIR
"""
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from math import comb

sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "energy_hub", "code")
DATA = os.path.join(ROOT, "energy_hub", "data")
PF_CODE = os.path.join(ROOT, "portfolio", "code")
IPOPT_BIN = os.path.join(os.path.dirname(sys.executable), "Library", "bin")
sys.path.insert(0, CODE)

import numpy as np

EH_N, EH_K = 18, 6
PF_N, PF_K = 15, 5
EH_ARMS = {"S0": "S0", "S1": "S1_5k", "S2": "S2"}
EH_SEEDS = list(range(101, 111))
PF_SEEDS = list(range(10))
N_EH_VARIANTS, N_PF_VARIANTS = 5, 10
VARIANT_SEED = 20261109


def _variants(n, k, m):
    """m alternative initial supports (identity ring) and m permutations, deterministic."""
    rng = np.random.default_rng(VARIANT_SEED + n)
    first = tuple(range(k))
    inits, perms = [], []
    while len(inits) < m:
        s = tuple(sorted(rng.choice(n, k, replace=False).tolist()))
        if s != first and s not in inits:
            inits.append(s)
    while len(perms) < m:
        p = tuple(rng.permutation(n).tolist())
        if p != tuple(range(n)) and p not in perms:
            perms.append(p)
    out = [("orig", 0, tuple(range(n)), first)]
    out += [("init", j, tuple(range(n)), s) for j, s in enumerate(inits)]
    out += [("perm", j, p, tuple(sorted(p[:k]))) for j, p in enumerate(perms)]
    return out


def make_circuits(n, p, H_cost, init_support, order, shots, pl_seed):
    """Same construction as hybrid_pipeline.samplers.qaoa.make_circuits, with explicit
    initial support and ring order (identity order + first-K support == original)."""
    import pennylane as qml
    init_state = np.zeros(n, dtype=int)
    init_state[list(init_support)] = 1
    pairs = [(order[i], order[(i + 1) % n]) for i in range(n)]
    coeffs, ops = [], []
    for i, j in pairs:
        coeffs.extend([1.0, 1.0])
        ops.append(qml.PauliX(wires=i) @ qml.PauliX(wires=j))
        ops.append(qml.PauliY(wires=i) @ qml.PauliY(wires=j))
    H_mix = qml.Hamiltonian(coeffs, ops)
    dev_exp = qml.device("default.qubit", wires=n, seed=pl_seed)
    dev_sample = qml.device("default.qubit", wires=n, shots=shots, seed=pl_seed)

    @qml.qnode(dev_exp, interface="autograd")
    def circuit_exp(params):
        qml.BasisState(init_state, wires=range(n))
        for layer in range(p):
            qml.ApproxTimeEvolution(H_cost, params[2 * layer], 1)
            qml.ApproxTimeEvolution(H_mix, params[2 * layer + 1], 1)
        return qml.expval(H_cost)

    @qml.qnode(dev_sample)
    def circuit_sample(params):
        qml.BasisState(init_state, wires=range(n))
        for layer in range(p):
            qml.ApproxTimeEvolution(H_cost, params[2 * layer], 1)
            qml.ApproxTimeEvolution(H_mix, params[2 * layer + 1], 1)
        return qml.sample(wires=range(n))

    return circuit_exp, circuit_sample


def train(circuit_exp, p, steps, draw):
    import pennylane as qml
    from pennylane import numpy as pnp
    params = pnp.array(draw(0, np.pi, 2 * p), requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=0.1)
    for _ in range(steps):
        params = opt.step(lambda v: circuit_exp(v), params)
    return params


def exact_counts(raw, k):
    c = Counter("".join(str(int(x)) for x in row) for row in raw)
    return {b: v for b, v in c.items() if b.count("1") == k}, len(raw)


def eh_task(args):
    arm, seed, var, out_dir = args
    vtype, vid, order, init = var
    path = os.path.join(out_dir, f"EH_{arm}_seed{seed}_{vtype}{vid}.json")
    if os.path.exists(path):
        return path, 0.0
    t0 = time.time()
    from hybrid_pipeline.experiments import energy_hub_canonical_config as cfg
    from hybrid_pipeline.benchmarks.energy_hub import (build_energy_hub_instance, build_energy_hub_s0,
                                                       build_energy_hub_s1, build_energy_hub_s2)
    from hybrid_pipeline.samplers.qaoa import build_hamiltonian_from_qubo
    inst = build_energy_hub_instance()
    assert (inst["N"], inst["K"]) == (EH_N, EH_K)
    ac = cfg.ARM_CONFIGS[EH_ARMS[arm]]
    p, steps, shots = ac["depth"], ac["qaoa_steps"], ac["shots_per_restart"]
    rc = json.load(open(os.path.join(DATA, "canonical_5k", "runs", f"{arm}_{seed}", "run_complete.json")))
    ts = int(rc["restart_seeds"][0]["training_seed"])
    sur = {"S0": build_energy_hub_s0, "S1": build_energy_hub_s1, "S2": build_energy_hub_s2}[arm](inst)
    assert sur["coefficient_hash"] == cfg.EXPECTED_SURROGATE_COEFFICIENT_HASHES[arm]
    H = build_hamiltonian_from_qubo(sur)
    c_exp, c_s = make_circuits(EH_N, p, H, init, order, shots, ts)
    params = train(c_exp, p, steps, np.random.RandomState(ts).uniform)
    counts, n = exact_counts(c_s(params), EH_K)
    json.dump({"bench": "EH", "arm": arm, "seed": seed, "variant_type": vtype, "variant_id": vid,
               "order": list(order), "init_support": list(init), "training_seed": ts, "depth": p,
               "steps": steps, "params": [float(v) for v in params], "n_shots": n, "counts": counts,
               "seconds": time.time() - t0}, open(path, "w"))
    return path, time.time() - t0


_PF = None


def _pf_module():
    global _PF
    if _PF is None:
        os.environ["PATH"] = IPOPT_BIN + os.pathsep + os.environ["PATH"]
        spec = importlib.util.spec_from_file_location("pf_legacy", os.path.join(PF_CODE, "Quantico-cardinalidad.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["pf_legacy"] = mod
        spec.loader.exec_module(mod)
        _PF = mod
    return _PF


def pf_task(args):
    seed, var, out_dir = args
    vtype, vid, order, init = var
    path = os.path.join(out_dir, f"PF_seed{seed}_{vtype}{vid}.json")
    if os.path.exists(path):
        return path, 0.0
    t0 = time.time()
    m = _pf_module()
    inst = m.INSTANCE_15
    assert (inst["N"], inst["K"]) == (PF_N, PF_K)
    p, steps, shots = m.CONFIG["qaoa_layers"], m.CONFIG["qaoa_steps"], m.CONFIG["qaoa_shots"]
    m.set_global_seed(seed)
    H = m.build_support_surrogate_hamiltonian(PF_N, PF_K, penalty=m.HAMILTONIAN_PENALTY, instance=inst)
    c_exp, c_s = make_circuits(PF_N, p, H, init, order, shots, seed)
    params = train(c_exp, p, steps, np.random.uniform)
    counts, n = exact_counts(c_s(params), PF_K)
    json.dump({"bench": "PF", "seed": seed, "variant_type": vtype, "variant_id": vid, "order": list(order),
               "init_support": list(init), "depth": p, "steps": steps, "params": [float(v) for v in params],
               "n_shots": n, "counts": counts, "seconds": time.time() - t0}, open(path, "w"))
    return path, time.time() - t0


def _run(tasks, fn, nw, out_dir):
    import multiprocessing as mp
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    with mp.Pool(nw) as pool:
        for path, dt in pool.imap_unordered(fn, tasks):
            print(f"[{time.time() - t0:8.1f}s] {os.path.basename(path)} ({dt:.0f}s)", flush=True)
    print(f"DONE {len(tasks)} tasks in {time.time() - t0:.0f} s", flush=True)


def analyze(out_dir):
    import pandas as pd
    from hybrid_pipeline.benchmarks.energy_hub import (build_energy_hub_instance, build_energy_hub_s0,
                                                       build_energy_hub_s1, build_energy_hub_s2,
                                                       enumerate_exact_k_energies, structural_filter)
    from hybrid_pipeline.benchmarks.energy_hub.surrogates import bits_to_selection

    def bits_of(support, n):
        return "".join("1" if i in set(support) else "0" for i in range(n))

    rows = []
    inst = build_energy_hub_instance()
    ref = pd.read_csv(os.path.join(DATA, "reference", "energy_hub_exhaustive_classical_rows.csv"),
                      dtype={"support_bitstring": str})
    ref["TAC"] = pd.to_numeric(ref["TAC"], errors="coerce")
    feas = ref[ref.feasible == True].set_index("support_bitstring")["TAC"]
    opt = feas.min()
    energies = {a: enumerate_exact_k_energies(inst, b(inst)).set_index("bits")["energy"].to_dict()
                for a, b in (("S0", build_energy_hub_s0), ("S1", build_energy_hub_s1), ("S2", build_energy_hub_s2))}
    elig_cache = {}

    def eligible(b):
        if b not in elig_cache:
            elig_cache[b] = bool(structural_filter(bits_to_selection(tuple(int(c) for c in b), inst), inst))
        return elig_cache[b]

    repro = []
    for f in sorted(os.listdir(out_dir)):
        if not (f.startswith("EH_") and f.endswith(".json")):
            continue
        d = json.load(open(os.path.join(out_dir, f)))
        arm, counts, n = d["arm"], d["counts"], d["n_shots"]
        E = energies[arm]
        ordered = sorted(counts, key=lambda b: (-counts[b] / n, E[b], b))
        elig = [b for b in ordered if eligible(b)]
        tacs = [feas.get(b, np.nan) for b in elig]

        def best_at(B):
            x = [t for t in tacs[:B] if not np.isnan(t)]
            return min(x) if x else np.nan
        y0 = bits_of(d["init_support"], EH_N)
        rank0 = ordered.index(y0) + 1 if y0 in counts else None
        row = {"bench": "EH", "arm": arm, "seed": d["seed"], "variant_type": d["variant_type"],
               "variant_id": d["variant_id"], "init_bits": y0, "init_eligible": eligible(y0),
               "init_TAC": feas.get(y0, np.nan), "init_prob": counts.get(y0, 0) / n, "init_freq_rank": rank0,
               "init_in_B100": y0 in elig[:100], "top1_bits": ordered[0], "top1_is_init": ordered[0] == y0,
               "n_unique": len(counts), "n_eligible": len(elig), "pool_best": np.nanmin(tacs) if tacs else np.nan}
        for B in (10, 30, 60, 100):
            row[f"best_B{B}"] = best_at(B)
        rows.append(row)
        if d["variant_type"] == "orig":
            u = pd.read_csv(os.path.join(DATA, "canonical_5k", "runs", f"{arm}_{d['seed']}", "unique_supports.csv"),
                            dtype={"support_bitstring": str})
            archived = dict(zip(u.support_bitstring, u.sample_count))
            repro.append({"arm": arm, "seed": d["seed"], "identical_counts": archived == counts})
    be = pd.read_csv(os.path.join(PF_CODE, "baseline_enumeration.csv"), dtype={"bits": str})
    be["bits"] = be["bits"].str.zfill(PF_N)
    var = be[be.feasible == True].set_index("bits")["variance"]
    pf_opt = var.min()
    for f in sorted(os.listdir(out_dir)):
        if not (f.startswith("PF_") and f.endswith(".json")):
            continue
        d = json.load(open(os.path.join(out_dir, f)))
        counts, n = d["counts"], d["n_shots"]
        ordered = sorted(counts, key=lambda b: (-counts[b], b))
        vals = [var.get(b, np.nan) for b in ordered]
        y0 = bits_of(d["init_support"], PF_N)
        fin = [v for v in vals if not np.isnan(v)]
        top20 = [v for v in vals[:20] if not np.isnan(v)]
        rows.append({"bench": "PF", "arm": "QAOA-surrogate", "seed": d["seed"], "variant_type": d["variant_type"],
                     "variant_id": d["variant_id"], "init_bits": y0, "init_eligible": y0 in var.index,
                     "init_TAC": var.get(y0, np.nan), "init_prob": counts.get(y0, 0) / n,
                     "init_freq_rank": ordered.index(y0) + 1 if y0 in counts else None,
                     "top1_bits": ordered[0], "top1_is_init": ordered[0] == y0, "n_unique": len(counts),
                     "n_eligible": len(fin), "pool_best": min(fin) if fin else np.nan,
                     "best_B20": min(top20) if top20 else np.nan})
    df = pd.DataFrame(rows)
    ref_val = {"EH": opt, "PF": pf_opt}
    for c in ("init_TAC", "pool_best", "best_B10", "best_B30", "best_B60", "best_B100", "best_B20"):
        if c in df:
            df[c.replace("best", "gap").replace("init_TAC", "init_gap").replace("pool_best", "pool_gap")] = \
                100 * (df[c] / df.bench.map(ref_val) - 1)
    df.to_csv(os.path.join(out_dir, "runs.csv"), index=False)
    pd.DataFrame(repro).to_csv(os.path.join(out_dir, "reproduction_check.csv"), index=False)
    print("reference optima: EH TAC* =", opt, "| PF variance* =", pf_opt)
    if repro:
        r = pd.DataFrame(repro)
        print(f"EH reproduction of archived samples (orig variant): {r.identical_counts.sum()}/{len(r)} identical")
    return df


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mode, out = sys.argv[1], sys.argv[2]
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    if mode == "eh":
        V = _variants(EH_N, EH_K, N_EH_VARIANTS)
        tasks = [(a, s, v, out) for v in V for a in EH_ARMS for s in EH_SEEDS]
        _run(tasks, eh_task, nw, out)
    elif mode == "pf":
        V = _variants(PF_N, PF_K, N_PF_VARIANTS)
        tasks = [(s, v, out) for v in V for s in PF_SEEDS]
        _run(tasks, pf_task, nw, out)
    elif mode == "analyze":
        analyze(out)
