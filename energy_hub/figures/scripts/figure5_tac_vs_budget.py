"""Manuscript Figure 6: TAC versus downstream evaluation budget and stage alignment.

Reconstructs best-so-far curves for every prefix B = 1..100 from the per-candidate
continuous-evaluation log; the reporting budgets (10, 30, 60, 100) are nested
prefixes of one ordered per-run candidate list (no resampling per budget).
"""

from __future__ import annotations

import pandas as pd

from _common import ARMS_ORDER, COLORS, DATA_DIR, LINESTYLES, MARKERS, MAX_WIDTH_IN, OUTPUTS_DIR, light_grid, panel_label, require_file, save_all, setup_style


REPORTING_BUDGETS = [10, 30, 60, 100]
MAX_BUDGET = 100


def make_figure(config_hash: str) -> dict:
    setup_style()
    import matplotlib.pyplot as plt

    eval_path = require_file(OUTPUTS_DIR / "continuous_evaluations.csv")
    ref_path = require_file(OUTPUTS_DIR / "exhaustive_reference_summary.json")
    unique_path = require_file(OUTPUTS_DIR / "unique_supports.csv")
    evals = pd.read_csv(eval_path)
    unique = pd.read_csv(unique_path)
    reference = pd.read_json(ref_path, typ="series")
    optimum = float(reference["best_TAC"])

    evals["TAC"] = pd.to_numeric(evals["TAC"], errors="coerce")
    evals["evaluation_rank"] = pd.to_numeric(evals["evaluation_rank"], errors="coerce")
    evals["nlp_feasible"] = evals["nlp_feasible"].astype(str).str.lower().eq("true")

    rows = []
    for (arm, seed), sub in evals.groupby(["arm", "seed"], sort=False):
        sub = sub.sort_values("evaluation_rank")
        feasible = sub[sub["nlp_feasible"] & sub["TAC"].notna()]
        for budget in range(1, MAX_BUDGET + 1):
            prefix = feasible[feasible["evaluation_rank"] <= budget]
            if prefix.empty:
                continue
            best = float(prefix["TAC"].min())
            rows.append(
                {
                    "arm": arm,
                    "seed": int(seed),
                    "budget": budget,
                    "best_so_far_TAC": best,
                    "gap_pct": 100.0 * (best - optimum) / optimum,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "figure5_tac_vs_budget_seed_prefix_data.csv", index=False)

    agg = df.groupby(["arm", "budget"])["best_so_far_TAC"].agg(["median", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
    agg.columns = ["median", "q25", "q75"]
    agg = agg.reset_index()
    agg.to_csv(DATA_DIR / "figure5_tac_vs_budget_data.csv", index=False)

    endpoint = agg[agg["budget"] == MAX_BUDGET].copy()
    endpoint["median_gap_to_exactK_million"] = (endpoint["median"] - optimum) / 1e6
    endpoint["q25_gap_to_exactK_million"] = (endpoint["q25"] - optimum) / 1e6
    endpoint["q75_gap_to_exactK_million"] = (endpoint["q75"] - optimum) / 1e6
    endpoint.to_csv(DATA_DIR / "figure5_tac_vs_budget_B100_gap_to_exactK_data.csv", index=False)

    unique["empirical_probability"] = pd.to_numeric(unique["empirical_probability"], errors="coerce")
    unique["surrogate_energy"] = pd.to_numeric(unique["surrogate_energy"], errors="coerce")
    alignment_rows = []
    for arm in ["S0", "S1", "S2"]:
        sampled = unique[
            (unique["arm"] == arm)
            & unique["empirical_probability"].notna()
            & unique["surrogate_energy"].notna()
        ]
        evaluated = evals[
            (evals["arm"] == arm)
            & (evals["evaluation_rank"] <= MAX_BUDGET)
            & evals["nlp_feasible"]
            & evals["sampling_frequency"].notna()
            & evals["TAC"].notna()
        ]
        alignment_rows.append(
            {
                "arm": arm,
                "rho_E_p": float((-sampled["surrogate_energy"]).corr(sampled["empirical_probability"], method="spearman")),
                "rho_p_V": float(evaluated["sampling_frequency"].corr(-evaluated["TAC"], method="spearman")),
                "n_unique_sampled": int(len(sampled)),
                "n_evaluated_feasible_B100": int(len(evaluated)),
            }
        )
    alignment = pd.DataFrame(alignment_rows)
    alignment.to_csv(DATA_DIR / "figure5_energy_probability_value_alignment_data.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(MAX_WIDTH_IN, 3.0))
    ax_a, ax_b = axes

    for arm in ARMS_ORDER:
        sub = agg[agg["arm"] == arm].sort_values("budget")
        reported = sub[sub["budget"].isin(REPORTING_BUDGETS)]
        ax_a.plot(
            sub["budget"],
            sub["median"] / 1e6,
            color=COLORS[arm],
            linestyle=LINESTYLES[arm],
            label=arm,
            linewidth=1.6,
        )
        ax_a.scatter(reported["budget"], reported["median"] / 1e6, marker=MARKERS[arm], color=COLORS[arm], s=34, zorder=3)
        ax_a.fill_between(sub["budget"], sub["q25"] / 1e6, sub["q75"] / 1e6, color=COLORS[arm], alpha=0.15)
    exact_line = ax_a.axhline(optimum / 1e6, color="#333333", linestyle=(0, (1.2, 1.2)), linewidth=1.1, label="Exact-K")
    ax_a.text(
        3,
        optimum / 1e6 + 0.35,
        "Exact-K",
        va="bottom",
        ha="left",
        fontsize=7,
        color="#333333",
        clip_on=False,
    )
    ax_a.set_xlabel("Downstream evaluation budget B")
    ax_a.set_ylabel("Best-so-far TAC (million currency units)")
    ax_a.set_ylim(bottom=max(0.0, optimum / 1e6 - 0.8))
    ax_a.legend(frameon=False, handles=[*ax_a.get_legend_handles_labels()[0][:-1], exact_line])
    light_grid(ax_a)
    ax_a.text(-0.16, 1.08, "A", transform=ax_a.transAxes, fontsize=10, fontweight="bold", va="top")

    ax_b.set_axis_off()
    x_nodes = [0.16, 0.50, 0.84]
    y_nodes = [0.78, 0.50, 0.22]
    for y, (_, row) in zip(y_nodes, alignment.iterrows()):
        arm = row["arm"]
        color = COLORS[arm]
        ax_b.text(-0.01, y, arm, ha="right", va="center", fontsize=8.5, fontweight="bold", color=color, transform=ax_b.transAxes)
        for x, label in zip(x_nodes, ["E", r"$\hat{p}$", "V"]):
            node = plt.Circle((x, y), 0.046, transform=ax_b.transAxes, facecolor="white", edgecolor=color, linewidth=1.2)
            ax_b.add_patch(node)
            ax_b.text(x, y, label, ha="center", va="center", fontsize=8.2, color="#222222", transform=ax_b.transAxes)
        for x0, x1, value in [
            (x_nodes[0] + 0.055, x_nodes[1] - 0.055, row["rho_E_p"]),
            (x_nodes[1] + 0.055, x_nodes[2] - 0.055, row["rho_p_V"]),
        ]:
            ax_b.annotate(
                "",
                xy=(x1, y),
                xytext=(x0, y),
                xycoords=ax_b.transAxes,
                arrowprops={"arrowstyle": "->", "lw": 1.0, "color": color, "shrinkA": 0, "shrinkB": 0},
            )
            ax_b.text((x0 + x1) / 2, y + 0.058, f"{value:+.2f}", ha="center", va="bottom", fontsize=7.2, color="#333333", transform=ax_b.transAxes)
    ax_b.text(0.33, 0.96, r"$\rho(E,\hat{p})$", ha="center", va="center", fontsize=8.2, transform=ax_b.transAxes)
    ax_b.text(0.67, 0.96, r"$\rho(\hat{p},V)$", ha="center", va="center", fontsize=8.2, transform=ax_b.transAxes)
    ax_b.text(
        0.50,
        0.04,
        "Positive values indicate preserved ordering toward lower energy/TAC.",
        ha="center",
        va="center",
        fontsize=6.4,
        color="#555555",
        transform=ax_b.transAxes,
    )
    ax_b.text(-0.16, 1.08, "B", transform=ax_b.transAxes, fontsize=10, fontweight="bold", va="top")

    fig.tight_layout()
    return save_all(
        fig, "figure5_tac_vs_budget", config_hash,
        source_files=[
            "outputs/energy_hub_canonical_v1/continuous_evaluations.csv",
            "outputs/energy_hub_canonical_v1/exhaustive_reference_summary.json",
            "outputs/energy_hub_canonical_v1/unique_supports.csv",
        ],
        description="TAC outcomes versus the common downstream evaluation budget and stage-wise information alignment. Panel A uses every prefix B=1..100 from the ordered candidate list and shows the exhaustive Exact-K reference as a horizontal line; larger markers show the protocol reporting budgets B in {10,30,60,100}. Panel B reports canonical Spearman correlations for E(y)->p(y)->V(y), where positive values indicate preserved ordering toward lower energy/TAC. Distinct line styles support grayscale/black-and-white inspection.",
    )
