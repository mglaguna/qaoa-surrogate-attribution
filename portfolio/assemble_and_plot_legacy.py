"""Assemble the figure-data directory from the canonical outputs and draw SI Figures S1-S9.

Figures produced by code/regenerate_manuscript_outputs_legacy.py:
  main_fig01_feasible_support_landscape -> Figure S1
  main_fig02_surrogate_alignment        -> Figure S2
  sup_fig08_ranked_mu_orthogonality     -> Figure S3
  main_fig03_sampler_policy_gap         -> Figure S4   (comparative_matrix_v2.csv)
  main_fig04_convergence_...            -> Figure S5   (convergence_curves.csv)
  sup_fig01_capped_budget               -> Figure S7   (summary_capped_budget.csv)
  sup_fig04_frozen_vs_live_qaoa         -> Figure S8   (exp3_distribution_injection.csv)
  sup_fig05_hamiltonian_ablation        -> Figure S9   (ablation_hamiltonian.csv)
The classical inputs (baseline_enumeration.csv, surrogate_vs_ipopt_full.csv and
data/surrogate_correlation_table.csv) do not depend on QAOA.

Usage (from portfolio/): python assemble_and_plot_legacy.py OUT_DIR
The SI versions use a minimum annotation size of 8 pt; to reproduce them run the figure
script through ../energy_hub/figures/scripts/enlarge_fonts.py (see README.md).
"""
import os, shutil, subprocess, sys
import numpy as np, pandas as pd

out = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
dd = os.path.join(out, "legacy_figure_data"); os.makedirs(dd, exist_ok=True)
LEG = os.path.join(here, "data")
f2 = os.path.join(out, "fase2_attractor_matrix_convergence")


def summ(df, by, cols):
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    g = df.groupby(by)[cols].agg(["mean", "std", "min", "max", "count"]).round(4)
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    return g.reset_index()


copies = {
    os.path.join(here, "code", "baseline_enumeration.csv"): "baseline_enumeration.csv",
    os.path.join(f2, "surrogate_vs_ipopt_full.csv"): "surrogate_vs_ipopt_full.csv",
    os.path.join(LEG, "surrogate_correlation_table.csv"): "surrogate_correlation_table.csv",
    os.path.join(f2, "attractor_distribution.csv"): "attractor_distribution.csv",
    os.path.join(f2, "convergence_curves.csv"): "convergence_curves.csv",
    os.path.join(f2, "comparative_matrix_v2.csv"): "comparative_matrix_v2.csv",
    os.path.join(out, "frozen_distribution", "exp3_distribution_injection.csv"): "exp3_distribution_injection.csv",
    os.path.join(out, "hamiltonian_ablation", "ablation_hamiltonian.csv"): "ablation_hamiltonian.csv",
    os.path.join(out, "depth_and_uniform", "exp5_depth_ablation.csv"): "exp5_depth_ablation.csv",
    os.path.join(out, "depth_and_uniform", "exp5_distribution_stats.csv"): "exp5_distribution_stats.csv",
}
for src, dst in copies.items():
    shutil.copy2(src, os.path.join(dd, dst))
cap = pd.read_csv(os.path.join(out, "capped_budget", "comparative_matrix_budgeted.csv"))
summ(cap, ["sampler_type", "selection_policy"], ["best_var", "gap_pct", "ipopt_used"]).to_csv(os.path.join(dd, "summary_capped_budget.csv"), index=False)
fr = pd.read_csv(os.path.join(dd, "exp3_distribution_injection.csv")).rename(columns={"ipopt_total": "ipopt_used"})
summ(fr, ["sampler_type"], ["best_var", "gap_pct", "ipopt_used"]).to_csv(os.path.join(dd, "summary_exp3_frozen_distribution.csv"), index=False)
ab = pd.read_csv(os.path.join(dd, "ablation_hamiltonian.csv")).rename(columns={"ipopt_total": "ipopt_used"})
summ(ab, ["hamiltonian", "selection_policy"], ["best_var", "gap_pct", "ipopt_used"]).to_csv(os.path.join(dd, "summary_hamiltonian_ablation.csv"), index=False)
dp = pd.read_csv(os.path.join(dd, "exp5_depth_ablation.csv")).rename(columns={"ipopt_total": "ipopt_used"})
summ(dp, ["p"], ["best_var", "gap_pct", "ipopt_used"]).to_csv(os.path.join(dd, "summary_exp5_depth_performance.csv"), index=False)
ds = pd.read_csv(os.path.join(dd, "exp5_distribution_stats.csv"))
summ(ds, ["p"], ["entropy_ratio", "top1_freq", "n_unique_exactK"]).to_csv(os.path.join(dd, "summary_exp5_distribution_stats.csv"), index=False)

env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
r = subprocess.run([sys.executable, os.path.join(here, "code", "regenerate_manuscript_outputs_legacy.py"), "--all",
                    "--data-dir", dd, "--out-dir", os.path.join(out, "legacy_figures")], env=env, capture_output=True, text=True)
print(r.stdout[-3000:]); print(r.stderr[-3000:])
