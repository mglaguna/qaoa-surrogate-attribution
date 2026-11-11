"""Summary tables for SI Section S2 from the canonical recompute
(mean, sample SD, min, max and count over seeds 0-9).

Usage: python summarize_canonical.py OUT_DIR   (writes OUT_DIR/summary/*.csv)
"""
import os, sys
from math import comb
import numpy as np, pandas as pd

out = sys.argv[1]
S = os.path.join(out, "summary"); os.makedirs(S, exist_ok=True)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)


def agg(df, by, cols):
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    g = df.groupby(by)[cols].agg(["mean", "std", "min", "max", "count"]).round(4)
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    return g.reset_index()


def show(name, df):
    df.to_csv(os.path.join(S, name + ".csv"), index=False)
    print(f"\n=== {name}\n{df.to_string(index=False)}")


def p(*a):
    return os.path.join(out, *a)


if os.path.exists(p("frequency_policy", "exp4_frequency_policy.csv")):
    d = pd.read_csv(p("frequency_policy", "exp4_frequency_policy.csv"))
    show("tableS9_factorial", agg(d.rename(columns={"ipopt_total": "ipopt_used"}),
                                  ["sampler_type", "selection_policy"], ["best_var", "gap_pct", "ipopt_used"]))
if os.path.exists(p("capped_budget", "comparative_matrix_budgeted.csv")):
    d = pd.read_csv(p("capped_budget", "comparative_matrix_budgeted.csv"))
    show("tableS10_capped_budget", agg(d, ["sampler_type", "selection_policy"], ["best_var", "gap_pct", "ipopt_used"]))
if os.path.exists(p("hamiltonian_ablation", "ablation_hamiltonian.csv")):
    d = pd.read_csv(p("hamiltonian_ablation", "ablation_hamiltonian.csv"))
    show("tableS12_hamiltonian_ablation", agg(d.rename(columns={"ipopt_total": "ipopt_used"}),
                                             ["hamiltonian", "selection_policy"], ["best_var", "gap_pct", "ipopt_used"]))
if os.path.exists(p("depth_and_uniform", "exp5_distribution_stats.csv")):
    q = pd.read_csv(p("depth_and_uniform", "exp5_distribution_stats.csv"))
    u = pd.read_csv(p("depth_and_uniform", "uniform_distribution_stats.csv"))
    rows = []
    for name, df in (("QAOA-surrogate (p = 1)", q[q.p == 1]), ("Uniform-random", u)):
        r = {"sampler": name}
        for c in ("entropy_ratio", "top1_freq", "n_unique_exactK"):
            r[c + "_mean"] = round(df[c].mean(), 4); r[c + "_sd"] = round(df[c].std(), 4)
        rows.append(r)
    show("tableS11_distribution", pd.DataFrame(rows))
    show("depth_distribution_by_p", agg(q, ["p"], ["n_unique_exactK", "entropy_ratio", "top1_freq"]))
    dp = pd.read_csv(p("depth_and_uniform", "exp5_depth_ablation.csv"))
    show("depth_performance_by_p", agg(dp.rename(columns={"ipopt_total": "ipopt_used"}), ["p"],
                                       ["best_var", "gap_pct", "ipopt_used"]))
mix = [t for t in "ABC" if os.path.exists(p(f"mixer_{t}", f"mixer_ext_{t}_performance.csv"))]
if mix and os.path.exists(p("depth_and_uniform", "exp5_depth_ablation.csv")):
    perf = pd.concat([pd.read_csv(p(f"mixer_{t}", f"mixer_ext_{t}_performance.csv")) for t in mix], ignore_index=True)
    dist = pd.concat([pd.read_csv(p(f"mixer_{t}", f"mixer_ext_{t}_distribution.csv")) for t in mix], ignore_index=True)
    a5 = pd.read_csv(p("depth_and_uniform", "exp5_depth_ablation.csv")); a5 = a5[a5.p == 1].assign(topology="ring", normalization="none")
    d5 = pd.read_csv(p("depth_and_uniform", "exp5_distribution_stats.csv")); d5 = d5[d5.p == 1].assign(topology="ring", normalization="none")
    perf = pd.concat([perf, a5], ignore_index=True); dist = pd.concat([dist, d5], ignore_index=True)
    perf.to_csv(os.path.join(S, "mixer_all_performance.csv"), index=False)
    dist.to_csv(os.path.join(S, "mixer_all_distribution.csv"), index=False)
    perf["finite"] = np.isfinite(pd.to_numeric(perf.gap_pct, errors="coerce"))
    g = agg(perf.rename(columns={"ipopt_total": "ipopt_used"}), ["topology", "normalization"], ["gap_pct", "ipopt_used"])
    g = g.merge(perf.groupby(["topology", "normalization"]).finite.sum().reset_index(name="n_finite"))
    show("mixer_performance", g)
    show("mixer_distribution", agg(dist, ["topology", "normalization"], ["n_unique_exactK", "entropy_ratio", "top1_freq"]))
if os.path.exists(p("frozen_distribution", "exp3_distribution_injection.csv")):
    d = pd.read_csv(p("frozen_distribution", "exp3_distribution_injection.csv"))
    by = [c for c in ("sampler_type", "condition", "selection_policy") if c in d.columns]
    cols = [c for c in ("best_var", "gap_pct", "ipopt_total", "ipopt_used") if c in d.columns]
    show("frozen_distribution", agg(d, by, cols))
f2 = p("fase2_attractor_matrix_convergence")
if os.path.exists(os.path.join(f2, "comparative_matrix_v2.csv")):
    d = pd.read_csv(os.path.join(f2, "comparative_matrix_v2.csv"))
    by = [c for c in ("sampler_type", "sampler", "selection_policy") if c in d.columns]
    show("comparative_matrix_v2", agg(d.rename(columns={"ipopt_total": "ipopt_used"}), by, ["best_var", "gap_pct", "ipopt_used"]))
if os.path.exists(os.path.join(f2, "attractor_distribution.csv")):
    a = pd.read_csv(os.path.join(f2, "attractor_distribution.csv"), dtype={"bits": str})
    show("attractor_top10", a.head(10)[[c for c in ("bits", "assets", "count", "freq_qaoa", "ipopt_variance", "rank_by_variance") if c in a]])
