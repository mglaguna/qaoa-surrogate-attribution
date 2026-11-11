"""Manuscript Figures 7, 11 and 12 (seed-level robustness, technology composition, and
energy-probability-value alignment), recomputed from the B = 100 endpoint in
data/canonical_5k/continuous_evaluations.csv.
"""

from __future__ import annotations

import csv
import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from _common import COLORS, FIGURES_ROOT, LINESTYLES, MARKERS, OUTPUTS_DIR as CORE_OUTPUTS, PACKAGE_ROOT, light_grid, setup_style

SUPP_DIR = FIGURES_ROOT / "supplement"
PNG_DIR = SUPP_DIR / "png_inspection_300ppi"
DATA_DIR = SUPP_DIR / "source_data"
QC_DIR = SUPP_DIR / "quality_control"

ARMS = ["S0", "S1", "S2", "Uniform"]
SURROGATES = ["S0", "S1", "S2"]
GROUPS = {
    "cogen": [f"tech_{i:02d}" for i in range(1, 7)],
    "renewable": [f"tech_{i:02d}" for i in range(7, 11)],
    "thermal": [f"tech_{i:02d}" for i in range(11, 15)],
    "cooling": [f"tech_{i:02d}" for i in range(15, 19)],
}
GROUP_OF = {tech: group for group, techs in GROUPS.items() for tech in techs}
GROUP_LABEL = {"cogen": "Co", "renewable": "Re", "thermal": "Th", "cooling": "Cl"}
GROUP_COLORS = {
    "cogen": "#0072B2",
    "renewable": "#009E73",
    "thermal": "#D55E00",
    "cooling": "#999999",
}


def panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def outer_panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.14, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def validate_inputs() -> dict:
    config = json.loads((CORE_OUTPUTS / "canonical_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((CORE_OUTPUTS / "manifest.json").read_text(encoding="utf-8"))
    if config.get("protocol_version") != "EH_CANONICAL_V1":
        raise RuntimeError("canonical_config.json is not EH_CANONICAL_V1")
    if manifest.get("final_status") != "PASS":
        raise RuntimeError("canonical manifest final_status is not PASS")
    if config.get("config_hash") != manifest.get("config_hash"):
        raise RuntimeError("config_hash mismatch between canonical_config.json and manifest.json")
    return {"config_hash": config["config_hash"], "protocol_version": config["protocol_version"]}


def load_b100_best() -> pd.DataFrame:
    df = pd.read_csv(CORE_OUTPUTS / "continuous_evaluations.csv")
    df["evaluation_rank"] = pd.to_numeric(df["evaluation_rank"], errors="coerce")
    df["TAC"] = pd.to_numeric(df["TAC"], errors="coerce")
    df["nlp_feasible"] = df["nlp_feasible"].astype(str).str.lower().eq("true")
    eligible = df[(df["evaluation_rank"] <= 100) & df["nlp_feasible"] & df["TAC"].notna()].copy()
    idx = eligible.groupby(["arm", "seed"])["TAC"].idxmin()
    best = eligible.loc[idx].copy().sort_values(["arm", "seed"])
    best["TAC_million"] = best["TAC"] / 1e6
    best.to_csv(DATA_DIR / "canonical_B100_best_support_by_arm_seed.csv", index=False)
    return best


def load_best_by_budget(budgets: list[int]) -> pd.DataFrame:
    df = pd.read_csv(CORE_OUTPUTS / "continuous_evaluations.csv")
    df["evaluation_rank"] = pd.to_numeric(df["evaluation_rank"], errors="coerce")
    df["TAC"] = pd.to_numeric(df["TAC"], errors="coerce")
    df["nlp_feasible"] = df["nlp_feasible"].astype(str).str.lower().eq("true")
    eligible = df[df["nlp_feasible"] & df["TAC"].notna()].copy()

    frames = []
    for budget in budgets:
        sub = eligible[eligible["evaluation_rank"] <= budget].copy()
        idx = sub.groupby(["arm", "seed"])["TAC"].idxmin()
        best = sub.loc[idx].copy()
        best["budget"] = budget
        frames.append(best)

    best_by_budget = pd.concat(frames, ignore_index=True).sort_values(["budget", "arm", "seed"])
    best_by_budget["TAC_million"] = best_by_budget["TAC"] / 1e6
    best_by_budget.to_csv(DATA_DIR / "canonical_best_support_by_budget_arm_seed.csv", index=False)
    return best_by_budget


def save_figure(fig, stem: str, meta: dict, source_files: list[str], description: str) -> None:
    fig_dir = SUPP_DIR / stem
    fig_dir.mkdir(parents=True, exist_ok=True)
    png = fig_dir / f"{stem}.png"
    pdf = fig_dir / f"{stem}.pdf"
    svg = fig_dir / f"{stem}.svg"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

    Image.open(png).convert("L").save(fig_dir / "preview_grayscale_check.png")
    Image.open(png).save(PNG_DIR / f"{stem}.png", dpi=(300, 300))
    provenance = {
        "figure": stem,
        "description": description,
        "protocol_version": meta["protocol_version"],
        "config_hash": meta["config_hash"],
        "source_files": source_files,
        "outputs": [
            str(png.relative_to(PACKAGE_ROOT)),
            str(pdf.relative_to(PACKAGE_ROOT)),
            str(svg.relative_to(PACKAGE_ROOT)),
            str((fig_dir / "preview_grayscale_check.png").relative_to(PACKAGE_ROOT)),
            str((PNG_DIR / f"{stem}.png").relative_to(PACKAGE_ROOT)),
        ],
    }
    (fig_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")


def figure_s1_seed_robustness(best: pd.DataFrame, meta: dict) -> None:
    best_by_budget = load_best_by_budget([30, 60, 100])
    rows = []
    for budget in [30, 60, 100]:
        budget_best = best_by_budget[best_by_budget["budget"] == budget]
        uniform = budget_best[budget_best["arm"] == "Uniform"][["seed", "TAC"]].rename(columns={"TAC": "TAC_Uniform"})
        for arm in SURROGATES:
            sub = budget_best[budget_best["arm"] == arm][["seed", "TAC"]].rename(columns={"TAC": f"TAC_{arm}"})
            merged = uniform.merge(sub, on="seed", how="inner")
            merged["budget"] = budget
            merged["surrogate"] = arm
            merged["delta_uniform_minus_surrogate"] = merged["TAC_Uniform"] - merged[f"TAC_{arm}"]
            merged["delta_TAC_million"] = merged["delta_uniform_minus_surrogate"] / 1e6
            rows.append(
                merged[
                    [
                        "budget",
                        "surrogate",
                        "seed",
                        "delta_uniform_minus_surrogate",
                        "delta_TAC_million",
                        "TAC_Uniform",
                        f"TAC_{arm}",
                    ]
                ].rename(columns={f"TAC_{arm}": "TAC_surrogate"})
            )
    deltas = pd.concat(rows, ignore_index=True)
    deltas.to_csv(DATA_DIR / "figureS1_seed_robustness_canonical_budget_data.csv", index=False)

    summary_rows = []
    for (budget, arm), sub in deltas.groupby(["budget", "surrogate"]):
        values = sub["delta_uniform_minus_surrogate"].to_numpy()
        wins = int((values < -1e-6).sum())
        ties = int((np.abs(values) <= 1e-6).sum())
        losses = int((values > 1e-6).sum())
        summary_rows.append(
            {
                "budget": budget,
                "surrogate": arm,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "median_delta_TAC_million": float(sub["delta_TAC_million"].median()),
                "q25_delta_TAC_million": float(sub["delta_TAC_million"].quantile(0.25)),
                "q75_delta_TAC_million": float(sub["delta_TAC_million"].quantile(0.75)),
            }
        )
    comp_summary = pd.DataFrame(summary_rows).sort_values(["budget", "surrogate"])
    comp_summary.to_csv(DATA_DIR / "figureS1_seed_robustness_canonical_budget_summary.csv", index=False)

    b100 = best[best["arm"].isin(ARMS)].copy()
    pair_rows = []
    for first, second in [
        ("S0", "Uniform"),
        ("S1", "Uniform"),
        ("S2", "Uniform"),
        ("S0", "S1"),
        ("S1", "S2"),
    ]:
        left = b100[b100["arm"] == first][["seed", "TAC"]].rename(columns={"TAC": "TAC_first"})
        right = b100[b100["arm"] == second][["seed", "TAC"]].rename(columns={"TAC": "TAC_second"})
        merged = left.merge(right, on="seed", how="inner")
        diffs = (merged["TAC_first"] - merged["TAC_second"]) / 1e6
        pair_rows.append(
            {
                "comparison": f"{first} - {second}",
                "median_diff_TAC_million": float(diffs.median()),
                "q25_diff_TAC_million": float(diffs.quantile(0.25)),
                "q75_diff_TAC_million": float(diffs.quantile(0.75)),
            }
        )
    pair = pd.DataFrame(pair_rows)
    pair.to_csv(DATA_DIR / "figureS1_pairwise_effects_canonical_B100.csv", index=False)

    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.35), constrained_layout=True)
    axes_abc = [axes[0, 0], axes[0, 1], axes[1, 0]]
    ax_d = axes[1, 1]

    for ax, budget, label in zip(axes_abc, [30, 60, 100], ["A", "B", "C"]):
        sub = deltas[deltas["budget"] == budget]
        seeds_sorted = sorted(sub["seed"].unique())
        x_positions = np.arange(1, len(seeds_sorted) + 1)
        seed_to_x = {seed: i + 1 for i, seed in enumerate(seeds_sorted)}
        for arm in SURROGATES:
            arm_data = sub[sub["surrogate"] == arm].sort_values("seed")
            xs = [seed_to_x[seed] for seed in arm_data["seed"]]
            ys = arm_data["delta_TAC_million"]
            ax.scatter(
                xs,
                ys,
                s=25,
                marker=MARKERS[arm],
                color=COLORS[arm],
                label=arm,
                alpha=0.92,
                edgecolors="white",
                linewidths=0.3,
                zorder=3,
            )
            median = float(ys.median())
            ax.hlines(
                median,
                0.65,
                x_positions[-1] + 0.35,
                colors=COLORS[arm],
                linestyles=LINESTYLES[arm],
                linewidth=1.0,
                alpha=0.9,
                zorder=2,
            )

        ax.axhline(0, color="#555555", linewidth=0.8, zorder=1)
        ax.set_title(f"B={budget}", fontsize=8.5, fontweight="normal")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(seed) for seed in seeds_sorted], rotation=45)
        if ax is axes_abc[0]:
            ax.set_ylabel("Delta TAC, Uniform minus surrogate\n(million currency units)")
        light_grid(ax)
        panel_label(ax, label)

        summary_lines = []
        for arm in SURROGATES:
            row = comp_summary[(comp_summary["budget"] == budget) & (comp_summary["surrogate"] == arm)].iloc[0]
            summary_lines.append(f"{arm}: {row['wins']}/{row['ties']}/{row['losses']} W/T/L")
        ax.text(0.03, 0.03, "\n".join(summary_lines), transform=ax.transAxes, fontsize=7.0, va="bottom", ha="left")

    y = np.arange(len(pair))
    vals = pair["median_diff_TAC_million"].to_numpy()
    lower = vals - pair["q25_diff_TAC_million"].to_numpy()
    upper = pair["q75_diff_TAC_million"].to_numpy() - vals
    ax_d.hlines(y, 0, vals, color="#9a9a9a", lw=1.0)
    ax_d.errorbar(
        vals,
        y,
        xerr=[lower, upper],
        fmt="o",
        color="#555555",
        ecolor="#8a8a8a",
        markersize=4.8,
        linewidth=1.0,
        capsize=2.5,
        zorder=3,
    )
    ax_d.axvline(0, color="#333333", lw=0.75)
    ax_d.set_yticks(y)
    ax_d.set_yticklabels(pair["comparison"])
    ax_d.set_xlabel("Median TAC diff., first - second at B=100\n(million currency units)")
    light_grid(ax_d)
    panel_label(ax_d, "D")

    handles = [
        ax_d.plot([], [], marker=MARKERS[arm], color=COLORS[arm], linestyle=LINESTYLES[arm], label=arm)[0]
        for arm in SURROGATES
    ]
    axes_abc[1].legend(handles=handles, frameon=False, loc="upper right")

    save_figure(
        fig,
        "FigureS1_seed_robustness_B100",
        meta,
        [
            "data/canonical_5k/continuous_evaluations.csv",
            "figures/supplement/source_data/canonical_best_support_by_budget_arm_seed.csv",
        ],
        "Canonical budget-series seed-level robustness. Panels A-C show Uniform minus each surrogate at B=30, 60, and 100; panel D shows canonical B=100 pairwise median effects with IQR.",
    )


def figure_s2_composition(best: pd.DataFrame, meta: dict) -> None:
    rows = []
    for _, row in best.iterrows():
        support = str(row["support_ids"]).split("|")
        for tech in support:
            rows.append({"arm": row["arm"], "seed": int(row["seed"]), "technology": tech, "group": GROUP_OF.get(tech, "")})
    long = pd.DataFrame(rows)
    counts = (
        long.groupby(["technology", "group", "arm"])
        .size()
        .rename("count")
        .reset_index()
    )
    counts["frequency"] = counts["count"] / 10.0
    counts.to_csv(DATA_DIR / "figureS2_technology_composition_B100_long.csv", index=False)

    tech_order = [tech for group in ["cogen", "renewable", "thermal", "cooling"] for tech in GROUPS[group]]
    matrix = (
        counts.pivot_table(index="technology", columns="arm", values="count", fill_value=0)
        .reindex(index=tech_order, columns=ARMS, fill_value=0)
        .astype(int)
    )
    matrix.to_csv(DATA_DIR / "figureS2_technology_composition_B100_matrix.csv")

    figure_s2_option3_bubble(matrix, meta)


def _grouped_tech_labels(index) -> list[str]:
    return [f"{tech.replace('tech_', 't')} ({GROUP_LABEL[GROUP_OF[tech]]})" for tech in index]


def _draw_group_separators(ax) -> None:
    boundaries = np.cumsum([len(GROUPS[g]) for g in ["cogen", "renewable", "thermal", "cooling"]])[:-1] - 0.5
    for y in boundaries:
        ax.axhline(y, color="#777777", linewidth=0.8)


def figure_s2_option3_bubble(matrix: pd.DataFrame, meta: dict) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(6.2, 5.9))

    for i, tech in enumerate(matrix.index):
        group = GROUP_OF[tech]
        for j, arm in enumerate(ARMS):
            value = int(matrix.loc[tech, arm])
            if value > 0:
                ax.scatter(
                    j,
                    i,
                    s=22 + value * 38,
                    color=GROUP_COLORS[group],
                    alpha=0.82,
                    edgecolor="black",
                    linewidth=0.45,
                    zorder=2,
                )
                ax.text(j, i, str(value), ha="center", va="center", color="black", fontsize=7, zorder=3)
            else:
                ax.text(j, i, "0", ha="center", va="center", color="#B0B0B0", fontsize=7, zorder=1)

    ax.set_xlim(-0.6, len(ARMS) - 0.4)
    ax.set_ylim(len(matrix.index) - 0.5, -0.5)
    ax.set_xticks(np.arange(len(ARMS)))
    ax.set_xticklabels(ARMS)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(_grouped_tech_labels(matrix.index))
    ax.set_xlabel("Arm")
    ax.set_ylabel("Technology")
    ax.set_title("B=100 technology frequency: bubble view")
    ax.grid(True, color="#D0D0D0", linewidth=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    _draw_group_separators(ax)

    legend_counts = [2, 5, 10]
    handles = [
        ax.scatter([], [], s=22 + c * 38, color="#BBBBBB", edgecolor="black", linewidth=0.45, label=str(c))
        for c in legend_counts
    ]
    group_handles = [
        ax.scatter([], [], s=70, color=GROUP_COLORS[g], edgecolor="black", linewidth=0.45, label=GROUP_LABEL[g])
        for g in ["cogen", "renewable", "thermal", "cooling"]
    ]
    leg1 = ax.legend(handles=handles, title="Seeds", frameon=False, loc="upper right", bbox_to_anchor=(1.18, 1.0))
    ax.add_artist(leg1)
    ax.legend(handles=group_handles, title="Group", frameon=False, loc="lower right", bbox_to_anchor=(1.2, 0.0))
    fig.tight_layout()

    save_figure(
        fig,
        "FigureS2_technology_composition_B100",
        meta,
        ["data/canonical_5k/continuous_evaluations.csv"],
        "Canonical B=100 technology-composition bubble plot (chosen final design). Bubble area encodes selection frequency and color encodes technology family.",
    )


def figure_s3_energy_probability_value_alignment(meta: dict) -> None:
    unique = pd.read_csv(CORE_OUTPUTS / "unique_supports.csv")
    evals = pd.read_csv(CORE_OUTPUTS / "continuous_evaluations.csv")

    unique = unique[unique["arm"].isin(SURROGATES)].copy()
    unique["empirical_probability"] = pd.to_numeric(unique["empirical_probability"], errors="coerce")
    unique["surrogate_energy"] = pd.to_numeric(unique["surrogate_energy"], errors="coerce")
    unique = unique[unique["empirical_probability"].notna() & unique["surrogate_energy"].notna()]

    evals = evals[evals["arm"].isin(SURROGATES)].copy()
    evals["evaluation_rank"] = pd.to_numeric(evals["evaluation_rank"], errors="coerce")
    evals["surrogate_energy"] = pd.to_numeric(evals["surrogate_energy"], errors="coerce")
    evals["TAC"] = pd.to_numeric(evals["TAC"], errors="coerce")
    evals["nlp_feasible"] = evals["nlp_feasible"].astype(str).str.lower().eq("true")
    evals = evals[
        (evals["evaluation_rank"] <= 100)
        & evals["nlp_feasible"]
        & evals["surrogate_energy"].notna()
        & evals["TAC"].notna()
    ].copy()
    evals["TAC_million"] = evals["TAC"] / 1e6

    corr_rows = []
    for arm in SURROGATES:
        sub = evals[evals["arm"] == arm]
        corr_rows.append(
            {
                "arm": arm,
                "n_evaluated_feasible": int(len(sub)),
                "spearman_energy_TAC": float(sub["surrogate_energy"].corr(sub["TAC"], method="spearman")),
            }
        )
    corr = pd.DataFrame(corr_rows)

    unique.to_csv(DATA_DIR / "figureS3_energy_probability_alignment_unique_supports.csv", index=False)
    evals.to_csv(DATA_DIR / "figureS3_energy_TAC_alignment_B100_evaluations.csv", index=False)
    corr.to_csv(DATA_DIR / "figureS3_energy_TAC_spearman_B100.csv", index=False)

    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.55))
    ax_a, ax_b, ax_c = axes

    for arm in SURROGATES:
        sampled = unique[unique["arm"] == arm]
        evaluated = evals[evals["arm"] == arm]
        ax_a.scatter(
            sampled["surrogate_energy"],
            sampled["empirical_probability"],
            s=10,
            alpha=0.48,
            color=COLORS[arm],
            marker=MARKERS[arm],
            label=arm,
            linewidth=0,
        )
        ax_b.scatter(
            evaluated["surrogate_energy"],
            evaluated["TAC_million"],
            s=11,
            alpha=0.52,
            color=COLORS[arm],
            marker=MARKERS[arm],
            label=arm,
            linewidth=0,
        )

    ax_a.set_xlabel("Surrogate energy")
    ax_a.set_ylabel("Sample probability")
    ax_a.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.50, 1.05), ncol=3)
    light_grid(ax_a)
    outer_panel_label(ax_a, "A")

    ax_b.set_xlabel("Surrogate energy")
    ax_b.set_ylabel("TAC (million)")
    light_grid(ax_b)
    outer_panel_label(ax_b, "B")

    vals = [corr.loc[corr["arm"] == arm, "spearman_energy_TAC"].iloc[0] for arm in SURROGATES]
    ax_c.bar(SURROGATES, vals, color=[COLORS[arm] for arm in SURROGATES], alpha=0.84, edgecolor="#333333", linewidth=0.35)
    ax_c.axhline(0, color="#333333", lw=0.7)
    ax_c.set_ylim(-1, 1)
    ax_c.set_ylabel("Spearman energy-TAC")
    ax_c.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax_c.set_axisbelow(True)
    outer_panel_label(ax_c, "C")

    fig.tight_layout(w_pad=2.0, rect=(0, 0, 1, 0.90))
    save_figure(
        fig,
        "FigureS3_energy_probability_value_alignment_B100",
        meta,
        [
            "data/canonical_5k/unique_supports.csv",
            "data/canonical_5k/continuous_evaluations.csv",
        ],
        "Canonical visual update of the energy-probability-value alignment diagnostic. Panel A plots surrogate energy versus empirical sample probability for unique sampled supports; panel B plots surrogate energy versus feasible TAC among B<=100 continuous evaluations; panel C reports per-arm Spearman energy-TAC correlations.",
    )


def write_readme() -> None:
    text = """# Energy Hub figures 7, 11 and 12

Recomputed from the B = 100 endpoint of `data/canonical_5k/` (protocol EH_CANONICAL_V1).

- `FigureS1_seed_robustness_B100` (manuscript Figure 7): seed-level deltas at B = 30, 60 and 100
  (panels A-C; Uniform minus each surrogate, negative values favor Uniform) and B = 100 pairwise
  median effects with IQR (panel D).
- `FigureS2_technology_composition_B100` (manuscript Figure 11): technology frequency of the B = 100
  best support across 10 seeds per arm; bubble area encodes frequency, color the technology family.
- `FigureS3_energy_probability_value_alignment_B100` (manuscript Figure 12): surrogate energy versus
  empirical sampling probability and versus feasible TAC (B <= 100), with per-arm Spearman correlations.

Each figure folder contains PNG/PDF/SVG, a grayscale preview and a provenance file.
"""
    (SUPP_DIR / "README.md").write_text(text, encoding="utf-8")


def write_manifest() -> None:
    rows = []
    for path in sorted(SUPP_DIR.rglob("*")):
        if path.is_file() and path.name != "package_manifest.csv":
            rel = path.relative_to(SUPP_DIR).as_posix()
            rows.append(
                {
                    "path": rel,
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    QC_DIR.mkdir(parents=True, exist_ok=True)
    with open(QC_DIR / "package_manifest.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    for path in [SUPP_DIR, PNG_DIR, DATA_DIR, QC_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    meta = validate_inputs()
    best = load_b100_best()
    figure_s1_seed_robustness(best, meta)
    figure_s2_composition(best, meta)
    figure_s3_energy_probability_value_alignment(meta)
    write_readme()
    write_manifest()
    print(f"Wrote canonical supplement candidates to {SUPP_DIR}")


if __name__ == "__main__":
    main()
