"""Mixer topology x normalization extension (QAOA depth p = 1, seeds 0-9).

usage: python recompute_mixer_ext.py <tag> <topology:normalization> [<topology:normalization> ...]
  topology in {ring, line, fully_connected}; normalization in {none, by_pairs}
  by_pairs: every XY pair term of the mixer is scaled by 1/|pairs| (|pairs| = number of mixer edges).

The circuit construction of Quantico-cardinalidad.py is used unchanged except for the pair list
(fully_connected = all C(n,2) pairs) and the optional 1/|pairs| scaling.
Outputs (working directory): mixer_ext_<tag>_{distribution,performance,top10}.csv
"""
import importlib.util, os, sys, itertools
from math import comb, log
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod; spec.loader.exec_module(mod); return mod

f3 = _load(os.path.join(HERE, "Fase3_separacion_efectos.py"), "fase3")
q = sys.modules["quantico_cardinalidad"]; qml = q.qml
inst, cfg = q.INSTANCE_15, q.CONFIG
tag = sys.argv[1]; conds = [c.split(":") for c in sys.argv[2:]]


def make_circuits_ext(n, p, H_cost, k, shots, topology, normalization, pl_seed):
    init_state = np.array([1] * k + [0] * (n - k), dtype=int)
    if topology == "ring": pairs = [(i, (i + 1) % n) for i in range(n)]
    elif topology == "line": pairs = [(i, i + 1) for i in range(n - 1)]
    elif topology == "fully_connected": pairs = list(itertools.combinations(range(n), 2))
    else: raise ValueError(topology)
    w = 1.0 / len(pairs) if normalization == "by_pairs" else 1.0
    coeffs, ops = [], []
    for i, j in pairs:
        coeffs.extend([w, w]); ops.append(qml.PauliX(wires=i) @ qml.PauliX(wires=j)); ops.append(qml.PauliY(wires=i) @ qml.PauliY(wires=j))
    H_mix = qml.Hamiltonian(coeffs, ops)
    dev_exp = qml.device("default.qubit", wires=n, seed=pl_seed)
    dev_sample = qml.device("default.qubit", wires=n, shots=shots, seed=pl_seed)

    @qml.qnode(dev_exp, interface="autograd")
    def circuit_exp(params):
        qml.BasisState(init_state, wires=range(n))
        for layer in range(p):
            qml.ApproxTimeEvolution(H_cost, params[2 * layer], 1); qml.ApproxTimeEvolution(H_mix, params[2 * layer + 1], 1)
        return qml.expval(H_cost)

    @qml.qnode(dev_sample)
    def circuit_sample(params):
        qml.BasisState(init_state, wires=range(n))
        for layer in range(p):
            qml.ApproxTimeEvolution(H_cost, params[2 * layer], 1); qml.ApproxTimeEvolution(H_mix, params[2 * layer + 1], 1)
        return qml.sample(wires=range(n))
    return circuit_exp, circuit_sample


q.set_global_seed(cfg["seed"])
b = pd.read_csv(os.path.join(HERE, "baseline_enumeration.csv"), dtype={"bits": str})
base = dict(best_var=float(b.loc[b.feasible == True, "variance"].min()),
            n_ipopt_calls=int((b.method == "ipopt").sum()), n_feasible=int((b.feasible == True).sum()))
N, K = inst["N"], inst["K"]; Hu = log(comb(N, K))
dist, perf, top = [], [], []
for topo, norm in conds:
    for seed in range(10):
        q.set_global_seed(seed)
        H = q.build_support_surrogate_hamiltonian(N, K, penalty=q.HAMILTONIAN_PENALTY, instance=inst)
        c_exp, c_samp = make_circuits_ext(N, 1, H, K, cfg["qaoa_shots"], topo, norm, seed)
        params = q.train_qaoa(c_exp, p=1, n_steps=cfg["qaoa_steps"], verbose=False)
        raw = c_samp(params); counts = {}
        for row in raw:
            t = tuple(int(x) for x in row)
            if sum(t) == K: counts[t] = counts.get(t, 0) + 1
        tot = sum(counts.values())
        if tot == 0:
            dist.append(dict(topology=topo, normalization=norm, seed=seed, n_unique_exactK=0, entropy_ratio=0.0, top1_freq=1.0, frac_exactK=0.0))
            perf.append(dict(topology=topo, normalization=norm, seed=seed, best_var=float('nan'), gap_pct=float('nan'), ipopt_total=0))
            continue
        pr = np.array(sorted(counts.values(), reverse=True), float) / tot
        H_q = float(-np.sum(pr * np.log(pr + 1e-15)))
        dist.append(dict(topology=topo, normalization=norm, seed=seed, n_unique_exactK=len(counts), entropy_ratio=round(H_q / Hu, 4),
                         top1_freq=round(float(pr[0]), 4), frac_exactK=tot / len(raw)))
        for r, v in enumerate(pr[:10], 1): top.append(dict(topology=topo, normalization=norm, seed=seed, rank=r, freq=float(v)))
        sampler = q.build_qaoa_sampler(c_samp, inst)
        df = f3.run_hybrid_v2(sampler=sampler, instance=inst, params=params, max_iter=cfg["max_iter"],
                              top_k_candidates=cfg["top_k_candidates"], shots=cfg["qaoa_shots"], top_eval_per_iter=cfg["top_eval_per_iter"],
                              score_alpha=cfg["score_alpha"], score_beta=cfg["score_beta"], baseline_stats=base,
                              label=f"QAOA_{topo}_{norm}", selection_policy="random_exactK")
        m = f3.summarize_run(df, base); m.update(topology=topo, normalization=norm, seed=seed); perf.append(m)
        print(f"{topo}/{norm} seed={seed}: unique={len(counts)} ent={H_q/Hu:.4f} top1={pr[0]:.4f} gap={m['gap_pct']:.3f}", flush=True)
pd.DataFrame(dist).to_csv(f"mixer_ext_{tag}_distribution.csv", index=False)
pd.DataFrame(perf).to_csv(f"mixer_ext_{tag}_performance.csv", index=False)
pd.DataFrame(top).to_csv(f"mixer_ext_{tag}_top10.csv", index=False)
print("DONE", flush=True)
