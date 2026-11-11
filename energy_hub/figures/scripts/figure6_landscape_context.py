"""Manuscript Figure 9: campaign outcomes within the complete feasible landscape.

Markers are the per-strategy medians of the seed-level best TAC at B = 100; the best
seed is shown dotted. Reads data/reference/energy_hub_exhaustive_classical_rows.csv
and data/canonical_5k/seed_arm_summary.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import ARMS_ORDER, COLORS, DATA_DIR, INPUTS_DIR, MAX_WIDTH_IN, OUTPUTS_DIR, light_grid, panel_label, require_file, save_all, setup_style


def make_figure(config_hash: str) -> dict:
    setup_style()
    import matplotlib.pyplot as plt

    ref_path = require_file(INPUTS_DIR / "energy_hub_exhaustive_classical_rows.csv")
    summary_path = require_file(OUTPUTS_DIR / "seed_arm_summary.csv")
    ref = pd.read_csv(ref_path)
    summary = pd.read_csv(summary_path)

    feasible = ref[ref["feasible"] == True].copy()
    feasible = feasible.sort_values("TAC").reset_index(drop=True)
    global_best = feasible["TAC"].iloc[0]

    med_by_arm = summary.groupby("arm")["best_TAC"].median().reindex(ARMS_ORDER)
    best_by_arm = summary.groupby("arm")["best_TAC"].min().reindex(ARMS_ORDER)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"median_best_TAC_B100": med_by_arm, "min_best_TAC_B100": best_by_arm}).to_csv(
        DATA_DIR / "figure6_landscape_context_data.csv")

    fig, axes = plt.subplots(1, 2, figsize=(MAX_WIDTH_IN, 3.0))
    ax_a, ax_b = axes

    ax_a.hist(feasible["TAC"] / 1e6, bins=80, color="#BBBBBB", edgecolor="none")
    ax_a.axvline(global_best / 1e6, color="black", linestyle="--", linewidth=1.2, label="Exhaustive optimum")
    for arm in ARMS_ORDER:
        ax_a.axvline(med_by_arm[arm] / 1e6, color=COLORS[arm], linewidth=1.4, alpha=0.9, label=f"{arm} median")
        ax_a.axvline(best_by_arm[arm] / 1e6, color=COLORS[arm], linewidth=1.0, alpha=0.8, linestyle=":")
    ax_a.plot([], [], color="grey", linestyle=":", linewidth=1.0, label="best seed (dotted)")
    ax_a.set_xlabel("TAC (million currency units)")
    ax_a.set_ylabel("Count (feasible exact-K supports)")
    ax_a.legend(frameon=False, fontsize=6)
    light_grid(ax_a)
    panel_label(ax_a, "A")

    lower_tail = feasible[feasible["TAC"] <= feasible["TAC"].quantile(0.25)]
    ax_b.hist(lower_tail["TAC"] / 1e6, bins=60, color="#BBBBBB", edgecolor="none")
    ax_b.axvline(global_best / 1e6, color="black", linestyle="--", linewidth=1.2)
    for arm in ARMS_ORDER:
        for val, ls, lw in ((med_by_arm[arm], "-", 1.4), (best_by_arm[arm], ":", 1.0)):
            if val <= lower_tail["TAC"].max():
                pct = 100.0 * (feasible["TAC"] <= val).sum() / len(feasible)
                ax_b.axvline(val / 1e6, color=COLORS[arm], linewidth=lw, alpha=0.9, linestyle=ls)
                if ls == "-":
                    ax_b.text(val / 1e6, ax_b.get_ylim()[1] * 0.9, f"{arm} {pct:.1f}%ile", color=COLORS[arm], fontsize=6,
                              rotation=90, va="top")
    ax_b.set_xlabel("TAC (million currency units) -- lower quartile")
    ax_b.set_ylabel("Count")
    light_grid(ax_b)
    panel_label(ax_b, "B")

    fig.tight_layout()
    return save_all(
        fig, "figure6_landscape_context", config_hash,
        source_files=["inputs/energy_hub_exhaustive_classical_rows.csv", "outputs/energy_hub_canonical_v1/seed_arm_summary.csv"],
        description="Per-strategy median (solid) and best-seed (dotted) TAC at B = 100 in the context of the complete feasible exact-K TAC distribution (5,801 supports): (A) full distribution, (B) lower-tail zoom with percentile annotations.",
    )
