"""Manuscript Figure 5: candidate flow (unique exact-K, structurally eligible, evaluated
by stored-value lookup within B <= 100, and feasible supports per seed).

Reads data/canonical_5k/candidate_flow.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import ARMS_ORDER, COLORS, DATA_DIR, MAX_WIDTH_IN, OUTPUTS_DIR, light_grid, panel_label, require_file, save_all, setup_style


def make_figure(config_hash: str) -> dict:
    setup_style()
    import matplotlib.pyplot as plt

    path = require_file(OUTPUTS_DIR / "candidate_flow.csv")
    df = pd.read_csv(path)

    stages = ["unique_exactK_supports", "eligible_supports", "attempted_continuous_evaluations", "feasible_continuous_evaluations"]
    stage_labels = ["Unique\nexact-K", "Structurally\neligible", "Evaluated\n(lookup,\n<=100)", "Feasible\n(reference\nvalue)"]
    medians = df.groupby("arm")[stages].median().reindex(ARMS_ORDER)
    medians.to_csv(DATA_DIR / "figure4_candidate_flow_data.csv")

    df["feasibility_fraction"] = df["eligible_supports"] / df["unique_exactK_supports"]
    frac_medians = df.groupby("arm")["feasibility_fraction"].median().reindex(ARMS_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(MAX_WIDTH_IN, 3.0))
    ax_a, ax_b = axes

    x = np.arange(len(stages))
    for arm in ARMS_ORDER:
        ax_a.plot(x, medians.loc[arm].to_numpy(), marker="o", color=COLORS[arm], label=arm, linewidth=1.4)
    ax_a.set_yscale("log")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(stage_labels)
    ax_a.set_ylabel("Median count per seed (log scale)")
    ax_a.legend(frameon=False)
    light_grid(ax_a)
    panel_label(ax_a, "A")

    ax_b.bar(ARMS_ORDER, [frac_medians[a] for a in ARMS_ORDER], color=[COLORS[a] for a in ARMS_ORDER])
    ax_b.set_ylabel("Fraction passing structural screen")
    ax_b.set_ylim(0, 1)
    light_grid(ax_b)
    panel_label(ax_b, "B")

    fig.tight_layout()
    return save_all(
        fig, "figure4_candidate_flow", config_hash,
        source_files=["outputs/energy_hub_canonical_v1/candidate_flow.csv"],
        description="Candidate flow: (A) median unique exact-K, structurally eligible, evaluated (stored-value lookup, <=100) and feasible evaluated supports per seed, log scale; (B) median fraction of unique supports passing the structural screen, by arm.",
    )
