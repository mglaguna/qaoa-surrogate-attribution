"""
Regenerate manuscript tables and figures.

This script produces only the editorially selected main and supplementary
outputs for the portfolio-selection manuscript.
"""

from __future__ import annotations

import argparse
import math
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*findfont.*")
warnings.filterwarnings("ignore", message=".*Font family.*not found.*")


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR
OUT_DIR = BASE_DIR / "manuscript_outputs"

MAIN_FIG_DIR = OUT_DIR / "main_figures"
SUPP_FIG_DIR = OUT_DIR / "supplementary_figures"
MAIN_TABLE_DIR = OUT_DIR / "main_tables"
SUPP_TABLE_DIR = OUT_DIR / "supplementary_tables"
LOG_DIR = OUT_DIR / "logs"

STYLE = dict(
    fig_dpi=150,
    save_dpi=300,
    opt_color="#F59E0B",
    qaoa_color="#4F83C4",
    uniform_color="#C85C5C",
    neutral="#6B7280",
    mean="#111827",
    grid_alpha=0.22,
)

POLICY_ORDER = ["ranked_hybrid", "ranked_mu", "random_exactK", "ranked_frequency"]
POLICY_LABEL = {
    "ranked_hybrid": "ranked_hybrid",
    "ranked_mu": "ranked_mu",
    "random_exactK": "random_exactK",
    "ranked_frequency": "ranked_freq",
}

GENERATED = {
    "main_figures": [],
    "supplementary_figures": [],
    "main_tables": [],
    "supplementary_tables": [],
}


def configure_paths(data_dir: Path, out_dir: Path) -> None:
    global DATA_DIR, OUT_DIR, MAIN_FIG_DIR, SUPP_FIG_DIR, MAIN_TABLE_DIR, SUPP_TABLE_DIR, LOG_DIR
    DATA_DIR = data_dir.resolve()
    OUT_DIR = out_dir.resolve()
    MAIN_FIG_DIR = OUT_DIR / "main_figures"
    SUPP_FIG_DIR = OUT_DIR / "supplementary_figures"
    MAIN_TABLE_DIR = OUT_DIR / "main_tables"
    SUPP_TABLE_DIR = OUT_DIR / "supplementary_tables"
    LOG_DIR = OUT_DIR / "logs"


def ensure_dirs() -> None:
    for path in (MAIN_FIG_DIR, SUPP_FIG_DIR, MAIN_TABLE_DIR, SUPP_TABLE_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def set_publication_style() -> None:
    plt.rcParams.update({
        "figure.dpi": STYLE["fig_dpi"],
        "savefig.dpi": STYLE["save_dpi"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.grid": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_axes_origin(ax, minor: bool = True, keep_grid: bool = False, grid_axis: str = "both") -> None:
    ax.set_facecolor("white")
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("black")
    ax.tick_params(axis="both", which="major", direction="in", top=True, right=True, length=4.0, width=0.8)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True, length=2.5, width=0.6)
    if minor:
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(bool(keep_grid), axis=grid_axis, color="#B8C0CC", alpha=STYLE["grid_alpha"], linewidth=0.6)


def format_variance_axis(ax) -> None:
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))


def format_legend(ax=None, legend=None, **kwargs):
    if legend is None and ax is not None:
        legend = ax.legend(frameon=True, edgecolor="black", fancybox=False, framealpha=1.0, **kwargs)
    if legend is None:
        return None
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("black")
    frame.set_linewidth(0.8)
    frame.set_alpha(1.0)
    return legend


def require_columns(df: pd.DataFrame, columns: list[str], file_name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{file_name} is missing required columns: {missing}")


def normalize_bitstring(x, n: int = 15) -> str | None:
    if pd.isna(x):
        return None
    if isinstance(x, str):
        s = x.strip()
        if re.fullmatch(r"[01]+", s):
            return s.zfill(n)[-n:]
        if re.fullmatch(r"\d+(\.0+)?", s):
            return str(int(float(s))).zfill(n)[-n:]
        if re.fullmatch(r"\d+(\.\d+)?[eE][+-]?\d+", s):
            try:
                value = float(s)
                if not math.isfinite(value) or not value.is_integer():
                    warnings.warn(f"Cannot normalize non-integer bitstring value: {x!r}", RuntimeWarning)
                    return None
                return str(int(value)).zfill(n)[-n:]
            except ValueError:
                warnings.warn(f"Cannot normalize bitstring value: {x!r}", RuntimeWarning)
                return None
        warnings.warn(f"Unexpected bitstring format; returning None: {x!r}", RuntimeWarning)
        return None
    try:
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return str(int(x)).zfill(n)[-n:]
    except (TypeError, ValueError, OverflowError):
        return None


def load_csv(name: str, required: bool = True) -> pd.DataFrame | None:
    path = DATA_DIR / name
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        print(f"[WARN] Missing optional file: {path}")
        return None
    header = pd.read_csv(path, nrows=0).columns
    dtype = {"bits": str} if "bits" in header else None
    df = pd.read_csv(path, dtype=dtype)
    if "bits" in df.columns:
        df["bits"] = df["bits"].map(normalize_bitstring)
    print(f"[OK] {name}: {len(df)} rows")
    return df


def parse_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False}).fillna(False)


def load_all() -> dict:
    data = {
        "baseline": load_csv("baseline_enumeration.csv"),
        "surr_full": load_csv("surrogate_vs_ipopt_full.csv"),
        "surr_corr": load_csv("surrogate_correlation_table.csv"),
        "attractor": load_csv("attractor_distribution.csv"),
        "conv": load_csv("convergence_curves.csv"),
        "cv2": load_csv("comparative_matrix_v2.csv"),
        "cap": load_csv("summary_capped_budget.csv"),
        "frozen_sum": load_csv("summary_exp3_frozen_distribution.csv"),
        "frozen_raw": load_csv("exp3_distribution_injection.csv"),
        "ablation": load_csv("ablation_hamiltonian.csv"),
        "abl_sum": load_csv("summary_hamiltonian_ablation.csv"),
        "depth_raw": load_csv("exp5_depth_ablation.csv"),
        "depth_sum_p": load_csv("summary_exp5_depth_performance.csv"),
        "depth_dist": load_csv("exp5_distribution_stats.csv"),
        "depth_sum_d": load_csv("summary_exp5_distribution_stats.csv"),
    }

    base = data["baseline"].copy()
    require_columns(base, ["bits", "feasible", "variance", "assets"], "baseline_enumeration.csv")
    base["feasible"] = parse_bool(base["feasible"])
    base["variance"] = pd.to_numeric(base["variance"], errors="coerce")
    feas = base[base["feasible"]].copy()
    best = feas.loc[feas["variance"].idxmin()]

    data["baseline"] = base
    data["feas"] = feas
    data["n_total"] = len(base)
    data["n_feasible"] = len(feas)
    data["n_discarded"] = len(base) - len(feas)
    data["pct_feasible"] = 100.0 * len(feas) / len(base)
    data["global_best_var"] = float(best["variance"])
    data["global_best_bits"] = normalize_bitstring(best["bits"])
    data["global_best_assets"] = best["assets"]
    if "active_count" in best.index and not pd.isna(best["active_count"]):
        data["effective_active_count"] = int(float(best["active_count"]))
    else:
        data["effective_active_count"] = ""

    att = data["attractor"].copy()
    require_columns(att, ["bits", "freq_qaoa", "ipopt_variance"], "attractor_distribution.csv")
    for c in ["freq_qaoa", "ipopt_variance", "surrogate_energy", "hybrid_score", "mu_score"]:
        if c in att.columns:
            att[c] = pd.to_numeric(att[c], errors="coerce")
    data["attractor"] = att
    best_obs = att.loc[att["ipopt_variance"].idxmin()]
    data["qaoa_inc_var"] = float(best_obs["ipopt_variance"])
    data["qaoa_inc_bits"] = normalize_bitstring(best_obs["bits"])
    data["qaoa_inc_freq"] = float(best_obs["freq_qaoa"])
    data["global_opt_in_attractor"] = data["global_best_bits"] in set(att["bits"].astype(str))

    print(f"Global optimum: sigma^2={data['global_best_var']:.6f}, bits={data['global_best_bits']}")
    print(f"QAOA incumbent: sigma^2={data['qaoa_inc_var']:.6f}, bits={data['qaoa_inc_bits']}")
    return data


def save_table(df: pd.DataFrame, filename: str, out_dir: Path, label: str, sources: list[str], message: str, notes: str = "") -> None:
    path = out_dir / filename
    df.to_csv(path, index=False)
    key = "main_tables" if out_dir == MAIN_TABLE_DIR else "supplementary_tables"
    GENERATED[key].append((label, filename, sources, message, notes))
    print(f"[TABLE] {path.relative_to(OUT_DIR)}")


def save_figure(fig: plt.Figure, filename: str, out_dir: Path, label: str, sources: list[str], message: str, notes: str = "") -> None:
    png_path = out_dir / filename
    fig.savefig(png_path, bbox_inches="tight", dpi=STYLE["save_dpi"], facecolor="white")
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    key = "main_figures" if out_dir == MAIN_FIG_DIR else "supplementary_figures"
    GENERATED[key].append((label, filename, sources, message, notes))
    print(f"[FIG] {png_path.relative_to(OUT_DIR)}")


def build_table1(d: dict) -> None:
    rows = [
        ("Total exact-K supports", d["n_total"], ""),
        ("Structurally discarded supports", d["n_discarded"], ""),
        ("Feasible IPOPT-evaluated supports", d["n_feasible"], ""),
        ("Feasibility rate (%)", round(d["pct_feasible"], 2), ""),
        ("Global optimum variance", round(d["global_best_var"], 6), ""),
        ("Global optimum support bitstring", d["global_best_bits"], "Recovered from baseline_enumeration.csv."),
        ("Global optimum assets", d["global_best_assets"], ""),
        ("Effective active count", d["effective_active_count"], ""),
    ]
    df = pd.DataFrame(rows, columns=["metric", "value", "note"])
    save_table(
        df, "main_table01_benchmark_characterization.csv", MAIN_TABLE_DIR,
        "Table 1", ["baseline_enumeration.csv"],
        "Benchmark characterization and IPOPT reference optimum.",
        "Bitstring is recovered from baseline_enumeration.csv to avoid scientific notation truncation.",
    )


def build_table2(d: dict) -> None:
    corr = d["surr_corr"].copy()
    require_columns(corr, ["feature", "spearman_rho_vs_ipopt_var", "p_value", "rank_of_global_opt", "n_total_supports"], "surrogate_correlation_table.csv")
    interp = {
        "surrogate_energy": "Weak proxy for IPOPT variance in this instance.",
        "hybrid_score": "Weak-to-moderate proxy for IPOPT variance in this instance.",
        "mu_score": "Most informative proxy among the reported features for this instance.",
    }
    out = pd.DataFrame({
        "proxy_feature": corr["feature"],
        "spearman_rho": corr["spearman_rho_vs_ipopt_var"],
        "p_value": corr["p_value"],
        "rank_of_global_opt": corr["rank_of_global_opt"],
        "n_supports": corr["n_total_supports"],
        "interpretation": corr["feature"].map(interp).fillna("Reported surrogate proxy."),
    })
    save_table(
        out, "main_table02_surrogate_alignment.csv", MAIN_TABLE_DIR,
        "Table 2", ["surrogate_correlation_table.csv"],
        "Surrogate proxy alignment against IPOPT-refined variance.",
    )


def build_table3(d: dict) -> None:
    cv2 = d["cv2"].copy()
    require_columns(cv2, ["sampler_type", "selection_policy", "best_var", "gap_pct", "ipopt_total"], "comparative_matrix_v2.csv")
    for c in ["best_var", "gap_pct", "ipopt_total"]:
        cv2[c] = pd.to_numeric(cv2[c], errors="coerce")
    agg = (
        cv2.groupby(["sampler_type", "selection_policy"], sort=False)
        .agg(
            best_var_mean=("best_var", "mean"),
            best_var_std=("best_var", "std"),
            gap_pct_mean=("gap_pct", "mean"),
            gap_pct_std=("gap_pct", "std"),
            ipopt_calls_mean=("ipopt_total", "mean"),
            n_seeds=("best_var", "count"),
        )
        .reset_index()
    )
    agg["note"] = ""
    qaoa = agg["sampler_type"].eq("QAOA_surrogate")
    agg.loc[qaoa, "note"] = (
        "Zero or near-zero across-seed variability should not be interpreted as statistical robustness; "
        "it reflects concentrated sampling behavior under this configuration."
    )
    order = {p: i for i, p in enumerate(POLICY_ORDER)}
    agg["_policy_order"] = agg["selection_policy"].map(order)
    agg["_sampler_order"] = agg["sampler_type"].map({"QAOA_surrogate": 0, "uniform_random": 1})
    agg = agg.sort_values(["_sampler_order", "_policy_order"]).drop(columns=["_policy_order", "_sampler_order"])
    for c in ["best_var_mean", "best_var_std"]:
        agg[c] = agg[c].round(6)
    for c in ["gap_pct_mean", "gap_pct_std"]:
        agg[c] = agg[c].round(4)
    agg["ipopt_calls_mean"] = agg["ipopt_calls_mean"].round(2)
    save_table(
        agg, "main_table03_factorial_sampler_policy_summary.csv", MAIN_TABLE_DIR,
        "Table 3", ["comparative_matrix_v2.csv"],
        "Sampler x selection-policy factorial summary.",
    )


def build_supp_tables(d: dict) -> None:
    save_table(
        d["cap"].copy(), "sup_table01_capped_budget.csv", SUPP_TABLE_DIR,
        "Supplementary Table 1", ["summary_capped_budget.csv"],
        "Capped-budget summary at B_max = 20 IPOPT calls.",
    )
    save_table(
        d["frozen_sum"].copy(), "sup_table02_frozen_distribution.csv", SUPP_TABLE_DIR,
        "Supplementary Table 2", ["summary_exp3_frozen_distribution.csv"],
        "Frozen versus live QAOA distribution summary.",
    )

    abl = d["ablation"].copy()
    require_columns(abl, ["hamiltonian", "selection_policy", "best_var", "gap_pct", "ipopt_total"], "ablation_hamiltonian.csv")
    for c in ["best_var", "gap_pct", "ipopt_total"]:
        abl[c] = pd.to_numeric(abl[c], errors="coerce")
    group_cols = ["hamiltonian", "selection_policy"]
    out = (
        abl.groupby(group_cols, sort=False)
        .agg(
            best_var_mean=("best_var", "mean"),
            best_var_std=("best_var", "std"),
            gap_pct_mean=("gap_pct", "mean"),
            gap_pct_std=("gap_pct", "std"),
            ipopt_calls_mean=("ipopt_total", "mean"),
            ipopt_calls_std=("ipopt_total", "std"),
            n_seeds=("best_var", "count"),
        )
        .reset_index()
    )
    if "bits" in abl.columns:
        out_supports = abl.groupby(group_cols)["bits"].nunique().reset_index(name="n_unique_supports_evaluated")
        out = out.merge(out_supports, on=group_cols)
        notes = "Unique supports counted from bitstrings."
    else:
        out_proxy = abl.groupby(group_cols)["best_var"].nunique().reset_index(name="n_unique_best_var_values")
        out = out.merge(out_proxy, on=group_cols)
        notes = "No bitstring column is available; n_unique_best_var_values is a weak proxy, not unique supports."
    save_table(
        out, "sup_table03_hamiltonian_ablation.csv", SUPP_TABLE_DIR,
        "Supplementary Table 3", ["ablation_hamiltonian.csv", "summary_hamiltonian_ablation.csv"],
        "Hamiltonian ablation summary.",
        notes,
    )

    perf = d["depth_sum_p"].copy()
    dist = d["depth_sum_d"].copy()
    require_columns(perf, ["p"], "summary_exp5_depth_performance.csv")
    require_columns(dist, ["p"], "summary_exp5_distribution_stats.csv")
    perf["p"] = pd.to_numeric(perf["p"], errors="coerce")
    dist["p"] = pd.to_numeric(dist["p"], errors="coerce")
    merged = perf.merge(dist, on="p", suffixes=("_performance", "_distribution"))
    save_table(
        merged, "sup_table04_depth_ablation.csv", SUPP_TABLE_DIR,
        "Supplementary Table 4", ["summary_exp5_depth_performance.csv", "summary_exp5_distribution_stats.csv"],
        "Depth-ablation performance and distribution diagnostics.",
    )


def fig1_feasible_landscape(d: dict) -> None:
    feas = d["feas"].copy()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.hist(feas["variance"], bins=50, color="#3B82F6", alpha=0.75, edgecolor="none", label=f"Feasible supports (n={d['n_feasible']})")
    ax.axvline(d["global_best_var"], color=STYLE["opt_color"], ls="--", lw=1.8, label=fr"Global optimum $\sigma^2$={d['global_best_var']:.4f}")
    ax.text(
        0.98, 0.95,
        f"C(15,5) = {d['n_total']}\nIPOPT-evaluated = {d['n_feasible']}\nStructurally discarded = {d['n_discarded']}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(facecolor="white", edgecolor="black", linewidth=0.6),
    )
    ax.set_xlabel(r"IPOPT-refined portfolio variance ($\sigma^2$)")
    ax.set_ylabel("Frequency")
    ax.set_title("Feasible Support Landscape")
    style_axes_origin(ax, minor=True, keep_grid=False)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    format_legend(ax, loc="upper left", fontsize=7.5)
    fig.tight_layout()
    save_figure(
        fig, "main_fig01_feasible_support_landscape.png", MAIN_FIG_DIR,
        "Figure 1", ["baseline_enumeration.csv"],
        "Feasible-support IPOPT variance landscape.",
        "Histogram includes only the 2211 feasible supports evaluated by IPOPT, not all 3003 exact-K supports.",
    )


def fig2_surrogate_alignment(d: dict) -> None:
    surr = d["surr_full"].copy()
    require_columns(surr, ["bits", "surrogate_energy", "hybrid_score", "mu_score", "ipopt_variance"], "surrogate_vs_ipopt_full.csv")
    for c in ["surrogate_energy", "hybrid_score", "mu_score", "ipopt_variance"]:
        surr[c] = pd.to_numeric(surr[c], errors="coerce")
    surr = surr.dropna(subset=["ipopt_variance"])

    corr = d["surr_corr"].copy()
    corr_map = {
        row["feature"]: row
        for _, row in corr.iterrows()
        if "feature" in row.index
    }
    features = [
        ("surrogate_energy", "Surrogate energy"),
        ("hybrid_score", "Hybrid score"),
        ("mu_score", "mu score (return-based)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), sharey=True)
    opt_rows = surr[surr["bits"].eq(d["global_best_bits"])]
    if opt_rows.empty:
        opt_rows = surr[np.isclose(surr["ipopt_variance"], d["global_best_var"], rtol=0, atol=1e-8)]
    opt_row = opt_rows.iloc[0] if not opt_rows.empty else None

    for ax, (feature, xlabel) in zip(axes, features):
        clean = surr[[feature, "ipopt_variance"]].dropna()
        x = clean[feature].to_numpy(dtype=float)
        y = clean["ipopt_variance"].to_numpy(dtype=float)
        ax.scatter(x, y, s=6, alpha=0.25, color=STYLE["neutral"], linewidths=0, rasterized=True, zorder=1)
        if len(clean) >= 2:
            coef = np.polyfit(x, y, 1)
            xr = np.linspace(np.nanmin(x), np.nanmax(x), 200)
            ax.plot(xr, np.polyval(coef, xr), color="#1D4ED8", lw=1.4, alpha=0.9, zorder=2)
        if opt_row is not None:
            ax.scatter([opt_row[feature]], [opt_row["ipopt_variance"]], marker="*", s=90, color=STYLE["opt_color"], edgecolor="black", linewidth=0.5, zorder=5, label="Global optimum")
        row = corr_map.get(feature)
        if row is not None:
            rho = float(row["spearman_rho_vs_ipopt_var"])
            pval = float(row["p_value"])
            rank = int(row["rank_of_global_opt"])
            n = int(row["n_total_supports"])
            ax.set_title(f"Spearman rho = {rho:.3f}  p = {pval:.1e}\nGlobal opt rank: {rank}/{n}", fontsize=7.5)
        ax.set_xlabel(xlabel)
        style_axes_origin(ax, minor=True, keep_grid=False)
    axes[0].set_ylabel(r"IPOPT variance ($\sigma^2$)")
    format_variance_axis(axes[0])
    format_legend(axes[0], loc="upper left", fontsize=7, markerscale=0.8)
    fig.suptitle("Surrogate Alignment with IPOPT-Refined Portfolio Variance", fontsize=9, y=1.02)
    fig.tight_layout()
    save_figure(
        fig, "main_fig02_surrogate_alignment.png", MAIN_FIG_DIR,
        "Figure 2", ["surrogate_vs_ipopt_full.csv", "surrogate_correlation_table.csv"],
        "Three surrogate proxies compared with IPOPT-refined variance.",
        "The global optimum marker is matched by bitstring when available.",
    )


def fig3_sampler_policy_gap(d: dict) -> None:
    cv2 = d["cv2"].copy()
    require_columns(cv2, ["sampler_type", "selection_policy", "gap_pct"], "comparative_matrix_v2.csv")
    cv2["gap_pct"] = pd.to_numeric(cv2["gap_pct"], errors="coerce")
    agg = (
        cv2.groupby(["sampler_type", "selection_policy"])
        .agg(gap_mean=("gap_pct", "mean"), gap_std=("gap_pct", "std"))
        .reset_index()
    )
    panels = [
        ("QAOA_surrogate", "QAOA sampler: effect of selection policy", STYLE["qaoa_color"], (-0.05, 0.50)),
        ("uniform_random", "Uniform sampler: effect of selection policy", STYLE["uniform_color"], None),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8), sharey=False)
    for ax, (sampler, title, color, ylim) in zip(axes, panels):
        sub = agg[agg["sampler_type"].eq(sampler)].set_index("selection_policy")
        x = np.arange(len(POLICY_ORDER))
        means = np.array([sub.loc[p, "gap_mean"] if p in sub.index else np.nan for p in POLICY_ORDER], dtype=float)
        stds = np.array([sub.loc[p, "gap_std"] if p in sub.index else np.nan for p in POLICY_ORDER], dtype=float)
        ax.bar(x, means, width=0.58, color=color, alpha=0.78, edgecolor="black", linewidth=0.6, zorder=3)
        ax.errorbar(x, means, yerr=stds, fmt="none", color="black", capsize=3, lw=0.8, zorder=4)
        ax.axhline(0.0, color=STYLE["opt_color"], ls="--", lw=1.2, label="Global optimum (gap = 0%)", zorder=2)
        ax.axhline(0.288, color=STYLE["neutral"], ls=":", lw=1.1, label="QAOA incumbent (gap = 0.288%)", zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([POLICY_LABEL[p] for p in POLICY_ORDER], fontsize=8)
        ax.set_xlabel("Selection policy")
        ax.set_ylabel("Mean gap vs IPOPT baseline (%)")
        ax.set_title(title)
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(min(-0.25, np.nanmin(means - stds) - 0.5), np.nanmax(means + stds) * 1.15)
        style_axes_origin(ax, minor=True, keep_grid=True, grid_axis="y")
    format_legend(axes[0], loc="best", fontsize=7)
    fig.tight_layout()
    save_figure(
        fig, "main_fig03_sampler_policy_gap.png", MAIN_FIG_DIR,
        "Figure 3", ["comparative_matrix_v2.csv"],
        "Sampler x selection-policy interaction in final gap.",
        "The QAOA and Uniform panels intentionally use independent y-axis ranges; compare patterns within each panel, not absolute visual bar heights across panels.",
    )


def _plot_convergence(d: dict, filename: str, out_dir: Path, label: str, title: str, ylim=None, zoom: bool = False) -> None:
    conv = d["conv"].copy()
    require_columns(conv, ["sampler_type", "selection_policy", "ipopt_calls", "best_var_mean"], "convergence_curves.csv")
    for c in ["best_var_mean", "best_var_q25", "best_var_q75", "ipopt_calls"]:
        if c in conv.columns:
            conv[c] = pd.to_numeric(conv[c], errors="coerce")
    curves = [
        ("QAOA_surrogate", "ranked_hybrid", STYLE["qaoa_color"], "-", "QAOA + ranked_hybrid", 1.0),
        ("QAOA_surrogate", "random_exactK", STYLE["qaoa_color"], ":", "QAOA + random_exactK", 1.0),
        ("uniform_random", "ranked_mu", STYLE["uniform_color"], "-", "Uniform + ranked_mu", 1.0),
        ("uniform_random", "ranked_hybrid", STYLE["uniform_color"], "--", "Uniform + ranked_hybrid", 0.75),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for sampler, policy, color, ls, curve_label, alpha_line in curves:
        sub = conv[(conv["sampler_type"].eq(sampler)) & (conv["selection_policy"].eq(policy))].sort_values("ipopt_calls")
        if sub.empty:
            continue
        x = sub["ipopt_calls"].to_numpy(dtype=float)
        y = sub["best_var_mean"].to_numpy(dtype=float)
        ax.step(x, y, where="post", color=color, ls=ls, lw=1.45, alpha=alpha_line, label=curve_label, zorder=4)
        if {"best_var_q25", "best_var_q75"}.issubset(sub.columns):
            ax.fill_between(x, sub["best_var_q25"].to_numpy(dtype=float), sub["best_var_q75"].to_numpy(dtype=float), step="post", color=color, alpha=0.08, linewidth=0, zorder=1)
    ax.axhline(d["global_best_var"], color=STYLE["opt_color"], ls="--", lw=1.2, label=fr"Global optimum $\sigma^2$={d['global_best_var']:.4f}", zorder=3)
    if zoom:
        ax.axhline(d["qaoa_inc_var"], color=STYLE["neutral"], ls=":", lw=1.1, label=fr"QAOA incumbent $\sigma^2$={d['qaoa_inc_var']:.4f}", zorder=3)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Cumulative IPOPT calls")
    ax.set_ylabel(r"Best incumbent variance ($\sigma^2$)")
    ax.set_title(title)
    style_axes_origin(ax, minor=True, keep_grid=True, grid_axis="y")
    format_variance_axis(ax)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=3, frameon=True, edgecolor="black", fancybox=False, framealpha=1.0, fontsize=7.2)
    leg.get_frame().set_linewidth(0.8)
    fig.tight_layout()
    key_sources = ["convergence_curves.csv"]
    message = "Incumbent convergence by cumulative IPOPT evaluations."
    notes = "Zoomed panel includes QAOA incumbent reference." if zoom else "Full-scale convergence view."
    save_figure(fig, filename, out_dir, label, key_sources, message, notes)


def fig4_convergence_main(d: dict) -> None:
    _plot_convergence(
        d, "main_fig04_convergence_ipopt_evaluations.png", MAIN_FIG_DIR,
        "Figure 4", "Incumbent Convergence by IPOPT Evaluations", ylim=None, zoom=False,
    )


def figS1_capped_budget(
    d: dict,
    out_dir: Path | None = None,
    label: str = "Supplementary Figure S1",
    filename: str = "sup_fig01_capped_budget.png",
) -> None:
    if out_dir is None:
        out_dir = SUPP_FIG_DIR
    cap = d["cap"].copy()
    require_columns(cap, ["sampler_type", "selection_policy", "gap_pct_mean"], "summary_capped_budget.csv")
    cap["gap_pct_mean"] = pd.to_numeric(cap["gap_pct_mean"], errors="coerce")
    if "gap_pct_std" in cap.columns:
        cap["gap_pct_std"] = pd.to_numeric(cap["gap_pct_std"], errors="coerce")
    labels = []
    colors = []
    for _, row in cap.iterrows():
        labels.append(f"{row['sampler_type'].replace('QAOA_surrogate', 'QAOA').replace('uniform_random', 'Uniform')}\n{POLICY_LABEL.get(row['selection_policy'], row['selection_policy'])}")
        colors.append(STYLE["qaoa_color"] if row["sampler_type"] == "QAOA_surrogate" else STYLE["uniform_color"])
    x = np.arange(len(cap))
    y = cap["gap_pct_mean"].to_numpy(dtype=float)
    yerr = cap["gap_pct_std"].to_numpy(dtype=float) if "gap_pct_std" in cap.columns else None
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.bar(x, y, color=colors, alpha=0.78, width=0.6, edgecolor="black", linewidth=0.5, zorder=3)
    if yerr is not None:
        ax.errorbar(x, y, yerr=yerr, fmt="none", color="black", capsize=3, lw=0.8, zorder=4)
    ax.axhline(0, color=STYLE["opt_color"], ls="--", lw=1.2, label="Global optimum (gap = 0%)")
    ax.axhline(0.288, color=STYLE["neutral"], ls=":", lw=1.1, label="QAOA incumbent (gap = 0.288%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Mean gap vs IPOPT baseline (%)")
    ax.set_title("Capped Budget Comparison")
    style_axes_origin(ax, minor=True, keep_grid=True, grid_axis="y")
    format_legend(ax, fontsize=7.2, loc="best")
    fig.tight_layout()
    save_figure(fig, filename, out_dir, label, ["summary_capped_budget.csv"], "Capped-budget comparison at B_max = 20 IPOPT calls.")


def figS2_convergence_zoom(d: dict) -> None:
    _plot_convergence(
        d, "sup_fig02_convergence_zoom.png", SUPP_FIG_DIR,
        "Supplementary Figure S2", "Zoomed Incumbent Convergence near the Optimum",
        ylim=(11.75, 13.20), zoom=True,
    )


def figS3_attractor(d: dict) -> None:
    att = d["attractor"].copy()
    require_columns(att, ["bits", "freq_qaoa", "ipopt_variance"], "attractor_distribution.csv")
    top10 = att.sort_values("freq_qaoa", ascending=False).head(10).reset_index(drop=True)
    inc_bits = d["qaoa_inc_bits"]
    colors = ["#7C6FA8" if b == inc_bits else "#8FB7D7" for b in top10["bits"]]
    y_pos = np.arange(len(top10))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.barh(y_pos, top10["freq_qaoa"], color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    for i, row in top10.iterrows():
        ax.text(row["freq_qaoa"] + 0.002, i, fr"$\sigma^2$={row['ipopt_variance']:.4f}", va="center", fontsize=7)
    labels = [str(b) for b in top10["bits"]]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7, family="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("QAOA sampling frequency")
    ax.set_title("QAOA Attractor Distribution: Top 10 by Frequency")
    if len(top10):
        ax.text(0.002, -0.28, "dominant attractor", color="#2F5D8A", fontsize=7, va="center")
    if inc_bits in set(top10["bits"]):
        idx = top10.index[top10["bits"].eq(inc_bits)][0]
        ax.text(0.002, idx + 0.30, fr"QAOA incumbent, freq={d['qaoa_inc_freq']:.4f}", color="#6D5FA4", fontsize=7)
    if not d["global_opt_in_attractor"]:
        ax.text(0.98, 0.04, "Global optimum absent from reported QAOA supports", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, bbox=dict(facecolor="white", edgecolor="black", linewidth=0.6))
    ax.axvline(d["qaoa_inc_freq"], color="#6D5FA4", ls=":", lw=0.9, alpha=0.75)
    style_axes_origin(ax, minor=True, keep_grid=True, grid_axis="x")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#8FB7D7", alpha=0.85, label="Sampled support"),
        plt.Rectangle((0, 0), 1, 1, color="#7C6FA8", alpha=0.85, label="QAOA incumbent"),
    ]
    format_legend(ax, handles=handles, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "sup_fig03_qaoa_attractor_distribution.png", SUPP_FIG_DIR, "Supplementary Figure S3", ["attractor_distribution.csv", "baseline_enumeration.csv"], "Top-10 QAOA attractor distribution.")


def figS4_frozen(d: dict) -> None:
    df = d["frozen_sum"].copy()
    require_columns(df, ["sampler_type", "gap_pct_mean"], "summary_exp3_frozen_distribution.csv")
    df["gap_pct_mean"] = pd.to_numeric(df["gap_pct_mean"], errors="coerce")
    if "gap_pct_std" in df.columns:
        df["gap_pct_std"] = pd.to_numeric(df["gap_pct_std"], errors="coerce")
    colors = {"frozen_qaoa": "#8FB7D7", "live_qaoa": STYLE["qaoa_color"], "uniform": STYLE["uniform_color"]}
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.bar(x, df["gap_pct_mean"], color=[colors.get(s, "#A8B0BA") for s in df["sampler_type"]], width=0.55, alpha=0.78, edgecolor="black", linewidth=0.5, zorder=3)
    if "gap_pct_std" in df.columns:
        ax.errorbar(x, df["gap_pct_mean"], yerr=df["gap_pct_std"], fmt="none", color="black", capsize=3, lw=0.8, zorder=4)
    ax.axhline(0, color=STYLE["opt_color"], ls="--", lw=1.1, label="Global optimum")
    ax.set_xticks(x)
    ax.set_xticklabels(df["sampler_type"], fontsize=8)
    ax.set_ylabel("Mean gap (%)")
    ax.set_title("Frozen vs Live QAOA Distribution")
    style_axes_origin(ax, minor=True, keep_grid=True, grid_axis="y")
    format_legend(ax, fontsize=7.5)
    fig.tight_layout()
    save_figure(fig, "sup_fig04_frozen_vs_live_qaoa.png", SUPP_FIG_DIR, "Supplementary Figure S4", ["exp3_distribution_injection.csv", "summary_exp3_frozen_distribution.csv"], "Frozen, live-QAOA, and uniform distribution comparison.")


def figS5_hamiltonian(d: dict) -> None:
    abl = d["ablation"].copy()
    require_columns(abl, ["hamiltonian", "selection_policy", "ipopt_total", "gap_pct"], "ablation_hamiltonian.csv")
    for c in ["ipopt_total", "gap_pct"]:
        abl[c] = pd.to_numeric(abl[c], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.0))
    metrics = [("ipopt_total", "IPOPT calls"), ("gap_pct", "Gap (%)")]
    for ax, (metric, ylabel) in zip(axes, metrics):
        for ham, color, marker in [("H_full", STYLE["qaoa_color"], "o"), ("H_retonly", "#D08A4E", "s")]:
            for policy, xpos in [("random_exactK", 0), ("ranked_hybrid", 1)]:
                sub = abl[(abl["hamiltonian"].eq(ham)) & (abl["selection_policy"].eq(policy))][metric]
                jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(sub))
                ax.scatter(np.full(len(sub), xpos) + jitter, sub, color=color, marker=marker, s=24, alpha=0.70, edgecolor="black", linewidth=0.3, zorder=3)
                ax.hlines(sub.mean(), xpos - 0.15, xpos + 0.15, color=color, lw=1.5, zorder=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["random_exactK", "ranked_hybrid"], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        style_axes_origin(ax, minor=True, keep_grid=True, grid_axis="y")
    handles = [
        plt.Line2D([0], [0], marker="o", color=STYLE["qaoa_color"], ls="", markersize=6, label="H_full"),
        plt.Line2D([0], [0], marker="s", color="#D08A4E", ls="", markersize=6, label="H_retonly"),
    ]
    leg = fig.legend(handles=handles, loc="upper right", fontsize=7.5, frameon=True, edgecolor="black", fancybox=False, framealpha=1.0)
    leg.get_frame().set_linewidth(0.8)
    fig.suptitle("Hamiltonian Ablation: H_full vs H_retonly", fontsize=8.5, y=1.02)
    fig.text(0.5, 0.01, "H_retonly collapses onto a single support (one IPOPT call per run); a lower or equal gap does not imply that the risk term is dispensable.", ha="center", fontsize=7.2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    save_figure(fig, "sup_fig05_hamiltonian_ablation.png", SUPP_FIG_DIR, "Supplementary Figure S5", ["ablation_hamiltonian.csv"], "Hamiltonian ablation controls.", "Warning displayed in the figure as requested.")


def _layered_depth_plot(raw: pd.DataFrame, summary: pd.DataFrame, specs, filename: str, label: str, title: str, sources: list[str]) -> None:
    for df in (raw, summary):
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    raw = raw.sort_values(["p", "seed"]) if "seed" in raw.columns else raw.sort_values("p")
    summary = summary.sort_values("p")
    p_vals = summary["p"].to_numpy(dtype=float)
    colors = {1: "#8FB7D7", 2: "#9BC7A4", 3: "#D8A7A7"}
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, len(specs), figsize=(3.7 * len(specs), 3.6))
    axes = np.atleast_1d(axes)
    for ax, spec in zip(axes, specs):
        raw_col, mean_col, std_col, count_col, ylabel = spec
        means = summary[mean_col].to_numpy(dtype=float)
        se = summary[std_col].to_numpy(dtype=float) / np.sqrt(summary[count_col].to_numpy(dtype=float))
        ax.fill_between(p_vals, means - se, means + se, color=STYLE["neutral"], alpha=0.25, linewidth=0, zorder=1, label="+/- 1 SE")
        for p in p_vals:
            vals = raw.loc[raw["p"].eq(p), raw_col].dropna().to_numpy(dtype=float)
            jitter = rng.uniform(-0.055, 0.055, size=len(vals))
            ax.scatter(np.full(len(vals), p) + jitter, vals, s=24, color=colors.get(int(p), "#A8B0BA"), alpha=0.72, edgecolors="none", zorder=2)
        ax.plot(p_vals, means, color=STYLE["mean"], lw=1.35, marker="o", markersize=4.2, zorder=3, label="Mean")
        ax.set_xlabel("Circuit depth p")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.set_xticks([1, 2, 3])
        ax.set_xlim(0.75, 3.25)
        if raw_col in ("entropy_ratio", "top1_freq"):
            ax.set_ylim(0, 1.02)
        style_axes_origin(ax, minor=True, keep_grid=True, grid_axis="y")
    format_legend(axes[0], loc="best", fontsize=7)
    fig.suptitle(title, fontsize=8.5, y=1.02)
    fig.tight_layout()
    save_figure(fig, filename, SUPP_FIG_DIR, label, sources, title)


def figS6_depth_performance(d: dict) -> None:
    specs = [
        ("gap_pct", "gap_pct_mean", "gap_pct_std", "gap_pct_count", "Mean gap (%)"),
        ("best_var", "best_var_mean", "best_var_std", "best_var_count", r"Mean best variance ($\sigma^2$)"),
    ]
    _layered_depth_plot(d["depth_raw"].copy(), d["depth_sum_p"].copy(), specs, "sup_fig06_depth_performance.png", "Supplementary Figure S6", "QAOA Depth Performance", ["exp5_depth_ablation.csv", "summary_exp5_depth_performance.csv"])


def figS7_depth_distribution(d: dict) -> None:
    specs = [
        ("entropy_ratio", "entropy_ratio_mean", "entropy_ratio_std", "entropy_ratio_count", "Entropy ratio"),
        ("top1_freq", "top1_freq_mean", "top1_freq_std", "top1_freq_count", "Top-1 bitstring frequency"),
        ("n_unique_exactK", "n_unique_exactK_mean", "n_unique_exactK_std", "n_unique_exactK_count", "Unique exact-K bitstrings"),
    ]
    _layered_depth_plot(d["depth_dist"].copy(), d["depth_sum_d"].copy(), specs, "sup_fig07_depth_distribution.png", "Supplementary Figure S7", "QAOA Distributional Structure vs Circuit Depth", ["exp5_distribution_stats.csv", "summary_exp5_distribution_stats.csv"])


def figS8_orthogonality(d: dict) -> None:
    surr = d["surr_full"].copy()
    require_columns(surr, ["bits", "mu_score", "hybrid_score", "ipopt_variance"], "surrogate_vs_ipopt_full.csv")
    for c in ["mu_score", "hybrid_score", "ipopt_variance"]:
        surr[c] = pd.to_numeric(surr[c], errors="coerce")
    surr = surr.dropna(subset=["mu_score", "hybrid_score"])
    rho, pval = spearmanr(surr["mu_score"], surr["hybrid_score"])
    opt_rows = surr[surr["bits"].eq(d["global_best_bits"])]
    if opt_rows.empty:
        opt_rows = surr[np.isclose(surr["ipopt_variance"], d["global_best_var"], rtol=0, atol=1e-8)]
    opt_row = opt_rows.iloc[0] if not opt_rows.empty else None
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.scatter(surr["mu_score"], surr["hybrid_score"], s=6, alpha=0.25, color=STYLE["neutral"], linewidths=0, rasterized=True, zorder=1)
    coef = np.polyfit(surr["mu_score"], surr["hybrid_score"], 1)
    xr = np.linspace(surr["mu_score"].min(), surr["mu_score"].max(), 200)
    ax.plot(xr, np.polyval(coef, xr), color="#1D4ED8", lw=1.4, alpha=0.9, label=f"Linear fit, rho={rho:.3f}")
    if opt_row is not None:
        ax.scatter([opt_row["mu_score"]], [opt_row["hybrid_score"]], marker="*", s=90, color=STYLE["opt_color"], edgecolor="black", linewidth=0.5, zorder=5, label="Global optimum")
    ax.set_xlabel("mu score (return-based)")
    ax.set_ylabel("Hybrid score")
    ax.set_title(f"ranked_mu vs ranked_hybrid\nSpearman rho = {rho:.3f}  p = {pval:.1e}")
    style_axes_origin(ax, minor=True, keep_grid=False)
    format_legend(ax, fontsize=7.5)
    fig.tight_layout()
    save_figure(fig, "sup_fig08_ranked_mu_orthogonality.png", SUPP_FIG_DIR, "Supplementary Figure S8", ["surrogate_vs_ipopt_full.csv"], "Orthogonality diagnostic for ranked_mu and ranked_hybrid.", "Global optimum is matched by bitstring when available.")


def generate_main_tables(d: dict) -> None:
    build_table1(d)
    build_table2(d)
    build_table3(d)


def generate_supp_tables(d: dict) -> None:
    build_supp_tables(d)


def generate_main_figures(d: dict, include_optional_main: bool = False) -> None:
    fig1_feasible_landscape(d)
    fig2_surrogate_alignment(d)
    fig3_sampler_policy_gap(d)
    fig4_convergence_main(d)
    if include_optional_main:
        figS1_capped_budget(d, out_dir=MAIN_FIG_DIR, label="Optional Main Figure 5", filename="main_fig05_capped_budget_optional.png")


def generate_supp_figures(d: dict, skip_capped_budget: bool = False) -> None:
    if not skip_capped_budget:
        figS1_capped_budget(d)
    figS2_convergence_zoom(d)
    figS3_attractor(d)
    figS4_frozen(d)
    figS5_hamiltonian(d)
    figS6_depth_performance(d)
    figS7_depth_distribution(d)
    figS8_orthogonality(d)


def write_index() -> Path:
    path = OUT_DIR / "figure_table_index.md"

    lines = [
        "# Manuscript Output Index",
        "",
        "## Package Notes",
        "",
        "- Minimum main package: Figures 1-4 and Tables 1-3.",
        "- Optional main figure: `main_fig05_capped_budget_optional.png`, generated only with `--include-optional-main`.",
        "- When the optional main capped-budget figure is generated, `sup_fig01_capped_budget.png` is not generated to avoid duplication.",
        "- Supplementary figures/tables contain controls and diagnostics supporting the attribution narrative.",
        "",
    ]

    sections = [
        ("Main figures", "main_figures"),
        ("Supplementary figures", "supplementary_figures"),
        ("Main tables", "main_tables"),
        ("Supplementary tables", "supplementary_tables"),
    ]

    for title, key in sections:
        lines.extend([f"## {title}", ""])

        items = GENERATED.get(key, [])
        if not items:
            lines.extend(["None generated.", ""])
            continue

        for item in items:
            label, filename, sources, message, notes = item

            if sources is None:
                sources_txt = "None"
            elif isinstance(sources, (list, tuple, set)):
                sources_txt = ", ".join(map(str, sources))
            else:
                sources_txt = str(sources)

            lines.append(f"- **{label}**: `{filename}`")
            lines.append(f"  - Sources: {sources_txt}")
            lines.append(f"  - Message: {message}")

            if notes:
                lines.append(f"  - Notes/warnings: {notes}")

        lines.append("")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate selected manuscript outputs.")
    parser.add_argument("--all", action="store_true", help="Generate all selected outputs.")
    parser.add_argument("--main", action="store_true", help="Generate main outputs.")
    parser.add_argument("--supp", action="store_true", help="Generate supplementary outputs.")
    parser.add_argument("--tables", action="store_true", help="Generate tables.")
    parser.add_argument("--figures", action="store_true", help="Generate figures.")
    parser.add_argument("--include-optional-main", action="store_true", help="Also generate optional capped-budget figure as main Figure 5.")
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR, help="Directory containing CSV input files.")
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "manuscript_outputs", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    no_mode = not any([args.all, args.main, args.supp, args.tables, args.figures])
    if no_mode:
        args.all = True

    configure_paths(args.data_dir, args.out_dir)
    ensure_dirs()
    set_publication_style()

    package_selected = args.main or args.supp
    output_type_selected = args.tables or args.figures
    run_main = args.all or args.main or not package_selected
    run_supp = args.all or args.supp or not package_selected
    run_tables = args.all or args.tables or not output_type_selected
    run_figures = args.all or args.figures or not output_type_selected

    print(f"Data dir: {DATA_DIR}")
    print(f"Output dir: {OUT_DIR}")
    d = load_all()

    if run_tables and run_main:
        generate_main_tables(d)
    if run_tables and run_supp:
        generate_supp_tables(d)
    if run_figures and run_main:
        generate_main_figures(d, include_optional_main=args.include_optional_main)
    if run_figures and run_supp:
        generate_supp_figures(d, skip_capped_budget=args.include_optional_main and run_main)

    index_path = write_index()
    print("\n[DONE]")
    print(f"Main tables: {len(GENERATED['main_tables'])}")
    print(f"Supplementary tables: {len(GENERATED['supplementary_tables'])}")
    print(f"Main figures: {len(GENERATED['main_figures'])}")
    print(f"Supplementary figures: {len(GENERATED['supplementary_figures'])}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
