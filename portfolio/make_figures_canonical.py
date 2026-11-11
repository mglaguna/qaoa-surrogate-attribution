"""SI Figures S6, S10a, S10b, S11, S12 and S13 (initialization controls) from the canonical
recompute and the initialization-control outputs.

Usage (from portfolio/): python make_figures_canonical.py OUT_DIR INIT_OUT_DIR FIG_DIR
"""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out, m2, figdir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(figdir, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.9, "xtick.direction": "in",
                     "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "savefig.dpi": 300,
                     "savefig.facecolor": "white"})
RED, BLUE = "#C96B6B", "#6F98CF"


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figdir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


d = pd.read_csv(os.path.join(out, "depth_and_uniform", "exp5_distribution_stats.csv"))
a = pd.read_csv(os.path.join(out, "depth_and_uniform", "exp5_depth_ablation.csv"))
cols = {1: "#9DBBD9", 2: "#A8D1B0", 3: "#D9A8A8"}


def panel(ax, df, col, title, ylabel, ylim=None):
    rng = np.random.RandomState(0)
    ps = [1, 2, 3]; m = []; se = []
    for p in ps:
        v = pd.to_numeric(df.loc[df.p == p, col], errors="coerce").values
        m.append(v.mean()); se.append(v.std(ddof=1) / np.sqrt(len(v)))
        ax.scatter(p + rng.uniform(-0.06, 0.06, len(v)), v, s=14, color=cols[p], alpha=0.8, zorder=2, linewidths=0)
    m = np.array(m); se = np.array(se)
    ax.fill_between(ps, m - se, m + se, color="#9AA0AB", alpha=0.35, label="±1 SE", linewidth=0, zorder=1)
    ax.plot(ps, m, "-o", color="#111827", lw=1.6, ms=5, label="Mean", zorder=3)
    ax.set_xticks(ps); ax.set_xlabel("Circuit depth p"); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=9)
    ax.grid(axis="y", color="#E5E7EB", lw=0.6); ax.set_xlim(0.7, 3.3)
    if ylim: ax.set_ylim(*ylim)


fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
panel(axs[0], a, "gap_pct", "Gap to exhaustive optimum (%)", "Gap (%)")
panel(axs[1], a, "best_var", "Best variance (σ²)", "Best variance (σ², %²)")
axs[0].legend(loc="upper left", fontsize=7, frameon=True, edgecolor="black", fancybox=False)
fig.suptitle("QAOA depth performance (random-exact-K ordering; 10 seeds, one dot per seed)", fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.93)); save(fig, "FigS10a_depth_performance")

fig, axs = plt.subplots(1, 3, figsize=(9.0, 3.0))
panel(axs[0], d, "entropy_ratio", "Entropy ratio", "Entropy ratio", (0, 1.02))
panel(axs[1], d, "top1_freq", "Top-1 support frequency", "Top-1 frequency", (0, 1.02))
panel(axs[2], d, "n_unique_exactK", "Unique exact-K supports", "Unique supports")
axs[0].legend(loc="upper right", fontsize=7, frameon=True, edgecolor="black", fancybox=False)
fig.suptitle("QAOA distributional structure versus circuit depth (5,000 shots; 10 seeds)", fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.93)); save(fig, "FigS10b_depth_distribution")

S = os.path.join(out, "summary")
perf = pd.read_csv(os.path.join(S, "mixer_all_performance.csv"))
dist = pd.read_csv(os.path.join(S, "mixer_all_distribution.csv"))
perf["gap_pct"] = pd.to_numeric(perf.gap_pct, errors="coerce").replace([np.inf, -np.inf], np.nan)
topos, norms = ["line", "ring", "fully_connected"], ["none", "by_pairs"]
style = {"none": dict(color=RED, hatch="//", edgecolor="#8B4A4A"), "by_pairs": dict(color=BLUE, edgecolor="#4A6A94")}


def bars(ax, df, col, ylabel, title, cap=None):
    x = np.arange(len(topos)); w = 0.38
    for k, nm in enumerate(norms):
        mu, sd = [], []
        for t in topos:
            v = df[(df.topology == t) & (df.normalization == nm)][col].dropna()
            mu.append(v.mean()); sd.append(v.std(ddof=1) if len(v) > 1 else 0)
        mu, sd = np.array(mu), np.array(sd)
        shown = np.minimum(mu, cap) if cap else mu
        ax.bar(x + (k - 0.5) * w, shown, w, label=nm, alpha=0.85, linewidth=0.8, **style[nm])
        ax.errorbar(x + (k - 0.5) * w, shown, yerr=np.where(mu > (cap or np.inf), 0, sd), fmt="none", ecolor="black",
                    capsize=3, lw=0.9)
        if cap:
            for xi, v in zip(x + (k - 0.5) * w, mu):
                if v > cap:
                    ax.annotate(f"{v:.1f}%", (xi, cap * 0.97), ha="center", va="top", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(topos, rotation=20); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3, lw=0.5)
    if cap: ax.set_ylim(0, cap)


fig, axs = plt.subplots(2, 2, figsize=(10, 7.6))
bars(axs[0, 0], perf, "gap_pct", "Mean downstream gap (%)", "Downstream gap (mean ± SD, finite runs)", cap=12)
nf = perf[(perf.topology == "fully_connected") & (perf.normalization == "none")].gap_pct.notna().sum()
axs[0, 0].text(0.02, 0.97, f"fully_connected/none: finite outcome\nin {nf} of 10 seeds", transform=axs[0, 0].transAxes,
               va="top", fontsize=7, bbox=dict(boxstyle="round", fc="white", ec="black", lw=0.6))
bars(axs[0, 1], dist, "entropy_ratio", "Mean entropy ratio", "Distributional spread")
bars(axs[1, 0], dist, "n_unique_exactK", "Mean unique exact-K supports", "Support diversity")
mk = {"line": "o", "ring": "D", "fully_connected": "s"}
ax = axs[1, 1]
for t in topos:
    for nm in norms:
        p_ = perf[(perf.topology == t) & (perf.normalization == nm)][["seed", "gap_pct"]]
        q_ = dist[(dist.topology == t) & (dist.normalization == nm)][["seed", "n_unique_exactK"]]
        j = p_.merge(q_, on="seed").dropna()
        ax.scatter(j.n_unique_exactK, j.gap_pct, marker=mk[t], color=style[nm]["color"], edgecolor="#333", lw=0.4, s=26,
                   label=f"{t}/{nm}")
ax.set_xscale("log"); ax.set_xlabel("Unique exact-K supports (per seed)"); ax.set_ylabel("Gap (%)")
ax.set_title("Diversity versus quality (one point per seed)", fontsize=9); ax.grid(alpha=0.3, lw=0.5)
ax.legend(fontsize=6, ncol=2, frameon=True)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, title="normalization", loc="lower center", ncol=2, frameon=False)
for k, a_ in zip("abcd", axs.flat):
    a_.text(-0.08, 1.04, f"({k})", transform=a_.transAxes, fontweight="bold", fontsize=11)
fig.tight_layout(rect=(0, 0.04, 1, 1)); save(fig, "FigS11_mixer_topology")

top = pd.concat([pd.read_csv(os.path.join(out, f"mixer_{t}", f"mixer_ext_{t}_top10.csv")) for t in "ABC"])
rows = []
for s in range(10):
    c = json.load(open(os.path.join(m2, f"PF_seed{s}_orig0.json")))["counts"]; n = sum(c.values())
    for r, v in enumerate(sorted(c.values(), reverse=True)[:10], 1):
        rows.append(dict(topology="ring", normalization="none", seed=s, rank=r, freq=v / n))
top = pd.concat([top, pd.DataFrame(rows)], ignore_index=True)
top.to_csv(os.path.join(S, "mixer_all_top10.csv"), index=False)
fig, axs = plt.subplots(3, 2, figsize=(9, 10), sharey=True)
for i, t in enumerate(["fully_connected", "line", "ring"]):
    for j, nm in enumerate(["by_pairs", "none"]):
        g = top[(top.topology == t) & (top.normalization == nm)]
        mf = g.pivot_table(index="seed", columns="rank", values="freq").reindex(columns=range(1, 11)).fillna(0).mean()
        axs[i, j].bar(mf.index, mf.values, alpha=0.85, linewidth=0.8, **style[nm])
        axs[i, j].set_title(f"{t} / {nm}", fontsize=9); axs[i, j].set_xticks(range(1, 11))
        axs[i, j].set_xlabel("top-k rank"); axs[i, j].set_ylabel("mean frequency"); axs[i, j].grid(alpha=0.3, lw=0.5)
        axs[i, j].set_ylim(0, 1.03)
fig.suptitle("Top-10 support frequencies by mixer topology and normalization (mean over 10 seeds)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.97)); save(fig, "FigS12_mixer_top10")

be = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "code", "baseline_enumeration.csv"),
                 dtype={"bits": str})
be["bits"] = be["bits"].str.zfill(15)
var = be[be.feasible == True].set_index("bits")["variance"]; vopt = var.min(); best_bits = var.idxmin()
pool = {}; tot = 0
for s in range(10):
    c = json.load(open(os.path.join(m2, f"PF_seed{s}_orig0.json")))["counts"]; tot += sum(c.values())
    for b, v in c.items():
        pool[b] = pool.get(b, 0) + v
topb = sorted(pool, key=pool.get, reverse=True)[:10]
fig, ax = plt.subplots(figsize=(8, 4.6))
y = np.arange(len(topb))[::-1]
for yi, b in zip(y, topb):
    f = pool[b] / tot; v = var.get(b, np.nan)
    col = "#111827" if b == "111110000000000" else ("#2E7D32" if b == best_bits else BLUE)
    ax.barh(yi, f, color=col, alpha=0.8, edgecolor="black", lw=0.5)
    lab = "infeasible" if np.isnan(v) else f"σ² = {v:.4f} (gap {100 * (v / vopt - 1):.2f}%)"
    ax.text(f + 0.005, yi, lab, va="center", fontsize=7)
ax.set_yticks(y); ax.set_yticklabels(topb, fontfamily="monospace", fontsize=7.5)
ax.set_xlabel("Share of all 50,000 shots (seeds 0–9 pooled)"); ax.set_xlim(0, max(pool[b] / tot for b in topb) * 1.45)
ax.set_title("QAOA-surrogate (p = 1): ten most frequent exact-K supports", fontsize=9)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#111827", label="initial support y0 (first five assets)"),
                   Patch(color=BLUE, label="other sampled support")] +
                  ([Patch(color="#2E7D32", label="global optimum")] if best_bits in topb else []),
          fontsize=7, loc="lower right")
ax.grid(axis="x", alpha=0.3, lw=0.5)
fig.tight_layout(); save(fig, "FigS6_top10_supports")
print("global optimum sampled in any seed:", best_bits in pool, "| y0 share:", round(pool.get("111110000000000", 0) / tot, 4))

r = pd.read_csv(os.path.join(m2, "runs.csv"), dtype={"init_bits": str})
pf, eh = r[r.bench == "PF"], r[r.bench == "EH"]
fig, axs = plt.subplots(1, 2, figsize=(10, 4.0), gridspec_kw={"width_ratios": [1, 1.25]})
ax = axs[0]
vv = pf.groupby(["variant_type", "variant_id"]).agg(y0=("init_gap", "first"), best=("pool_gap", "median")).reset_index()
cmap = {"orig": "#111827", "init": BLUE, "perm": RED}
lab = {"orig": "original y0 (first five assets)", "init": "alternative y0, same ring", "perm": "consistent permutation"}
for vt, g in vv.groupby("variant_type"):
    g = g.dropna(subset=["y0"])
    ax.scatter(g.y0, g.best.fillna(120), s=40 if vt == "orig" else 26, color=cmap[vt], edgecolor="black", lw=0.4,
               marker="*" if vt == "orig" else "o", label=lab[vt], zorder=3)
ax.plot([0, 115], [0, 115], ls="--", color="grey", lw=0.8, label="best sampled = y0")
ax.axhline(120, color="grey", lw=0.5, ls=":")
ax.text(113, 117, "plotted at 120: no feasible\nsupport sampled (median)", fontsize=6.5, va="top", ha="right", color="dimgrey")
ax.set_xlabel("Gap of the initial support y0 (%)  [zero-depth baseline]")
ax.set_ylabel("Median gap of best sampled support (%)")
ax.set_title("(a) Portfolio, QAOA p = 1 (feasible y0; median over 10 seeds)", fontsize=9)
ax.set_xlim(-3, 115); ax.set_ylim(-3, 125); ax.grid(alpha=0.3, lw=0.5); ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(0.0, 0.62))
ax = axs[1]
uni = 38.4
pos = 0; ticks = []; tl = []
for arm in ["S0", "S1", "S2"]:
    for vt, c in (("orig", "#111827"), ("init", BLUE), ("perm", RED)):
        v = eh[(eh.arm == arm) & (eh.variant_type == vt)].gap_B100.dropna().values
        bp = ax.boxplot(v, positions=[pos], widths=0.6, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black"))
        bp["boxes"][0].set(facecolor=c, alpha=0.55)
        ax.scatter(pos + np.random.RandomState(1).uniform(-0.15, 0.15, len(v)), v, s=6, color=c, alpha=0.7, zorder=3)
        ticks.append(pos); tl.append(f"{arm}\n{vt}"); pos += 1
    pos += 0.6
ax.axhline(uni, color="green", ls="--", lw=1, label="Uniform, main study (median)")
ax.set_xticks(ticks); ax.set_xticklabels(tl, fontsize=7); ax.set_ylabel("Gap of best-so-far TAC at B = 100 (%)")
ax.set_title("(b) Energy Hub, B = 100 (orig: 10 seeds; init/perm: 5 variants × 10 seeds)", fontsize=9)
ax.grid(axis="y", alpha=0.3, lw=0.5); ax.legend(fontsize=7, loc="upper right")
fig.tight_layout(); save(fig, "FigS13_initialization_controls")
