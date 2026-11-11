"""Circuit-depth diagnostics of the portfolio benchmark with 10 seeds.

Produces (in the working directory):
  exp5_distribution_stats.csv     QAOA distribution statistics for p in {1,2,3}, seeds 0-9
  exp5_depth_ablation.csv         pipeline performance (random_exactK) per p and seed
  uniform_distribution_stats.csv  Uniform exact-K sampler statistics, 5,000 draws, seeds 0-9

Uses Quantico-cardinalidad.py and Fase3_separacion_efectos.py unchanged.
"""
import importlib.util
import os
import sys
from math import comb, log

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


f3 = _load(os.path.join(HERE, "Fase3_separacion_efectos.py"), "fase3")
q = sys.modules["quantico_cardinalidad"]

instance = q.INSTANCE_15
config = q.CONFIG
q.set_global_seed(config["seed"])

df_base = pd.read_csv(os.path.join(HERE, "baseline_enumeration.csv"), dtype={"bits": str})
baseline_stats = dict(
    best_var=float(df_base.loc[df_base["feasible"] == True, "variance"].min()),
    n_ipopt_calls=int((df_base["method"] == "ipopt").sum()),
    n_feasible=int((df_base["feasible"] == True).sum()),
)
print("baseline best_var", baseline_stats["best_var"], "n_feasible", baseline_stats["n_feasible"], flush=True)

SEEDS = list(range(10))

N, K = instance["N"], instance["K"]
n_possible = comb(N, K)
H_unif = log(n_possible)
rows = []
for seed in SEEDS:
    rng = np.random.RandomState(seed)
    counts = {}
    for _ in range(config["qaoa_shots"]):
        idx = tuple(sorted(rng.choice(N, K, replace=False)))
        counts[idx] = counts.get(idx, 0) + 1
    c = np.array(list(counts.values()), dtype=float)
    probs = c / c.sum()
    H = float(-np.sum(probs * np.log(probs + 1e-15)))
    rows.append(dict(seed=seed, n_unique_exactK=len(counts), entropy_ratio=round(H / H_unif, 4),
                     top1_freq=round(float(probs.max()), 4), total_samples=int(c.sum())))
pd.DataFrame(rows).to_csv("uniform_distribution_stats.csv", index=False)
print("uniform stats done", flush=True)

f3.exp5_depth_ablation(instance, baseline_stats, config, depths=[1, 2, 3], seeds=SEEDS)
print("DONE", flush=True)
