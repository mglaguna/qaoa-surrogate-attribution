"""Manuscript Figure 4: sampling concentration and support coverage.

Reads data/canonical_5k/unique_supports.csv and seed_arm_summary.csv.
"""

from __future__ import annotations

import pandas as pd

from _common import ARMS_ORDER, COLORS, DATA_DIR, MAX_WIDTH_IN, OUTPUTS_DIR, light_grid, panel_label, require_file, save_all, setup_style


def make_figure(config_hash: str) -> dict:
    setup_style()
    import matplotlib.pyplot as plt

    unique_path = require_file(OUTPUTS_DIR / "unique_supports.csv")
    summary_path = require_file(OUTPUTS_DIR / "seed_arm_summary.csv")
    unique_df = pd.read_csv(unique_path, dtype={"support_bitstring": str})
    summary_df = pd.read_csv(summary_path)

    top10_rows = []
    for (arm, seed), grp in unique_df.groupby(["arm", "seed"]):
        top10 = grp.sort_values("empirical_probability", ascending=False).head(10)["empirical_probability"].sum()
        top10_rows.append({"arm": arm, "seed": seed, "top10_mass": top10})
    top10_df = pd.DataFrame(top10_rows)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged = summary_df.merge(top10_df, on=["arm", "seed"])
    merged.to_csv(DATA_DIR / "figure3_sampling_concentration_data.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(MAX_WIDTH_IN, 2.6))
    ax_a, ax_b, ax_c = axes

    for i, arm in enumerate(ARMS_ORDER):
        vals = merged.loc[merged["arm"] == arm, "unique_support_count"]
        ax_a.boxplot(vals, positions=[i], widths=0.6, patch_artist=True,
                     boxprops=dict(facecolor=COLORS[arm], alpha=0.35), medianprops=dict(color=COLORS[arm]))
    ax_a.set_xticks(range(len(ARMS_ORDER)))
    ax_a.set_xticklabels(ARMS_ORDER)
    ax_a.set_ylabel("Unique exact-K supports")
    ax_a.set_ylim(-146.75, 5281.75)
    light_grid(ax_a)
    panel_label(ax_a, "A")

    for i, arm in enumerate(ARMS_ORDER):
        vals = merged.loc[merged["arm"] == arm, "top10_mass"]
        ax_b.boxplot(vals, positions=[i], widths=0.6, patch_artist=True,
                     boxprops=dict(facecolor=COLORS[arm], alpha=0.35), medianprops=dict(color=COLORS[arm]))
    ax_b.set_xticks(range(len(ARMS_ORDER)))
    ax_b.set_xticklabels(ARMS_ORDER)
    ax_b.set_ylabel("Top-10 support probability mass")
    light_grid(ax_b)
    panel_label(ax_b, "B")

    for i, arm in enumerate(ARMS_ORDER):
        vals = merged.loc[merged["arm"] == arm, "entropy_ratio"]
        ax_c.boxplot(vals, positions=[i], widths=0.6, patch_artist=True,
                     boxprops=dict(facecolor=COLORS[arm], alpha=0.35), medianprops=dict(color=COLORS[arm]))
    ax_c.set_xticks(range(len(ARMS_ORDER)))
    ax_c.set_xticklabels(ARMS_ORDER)
    ax_c.set_ylabel("Entropy ratio vs. uniform")
    light_grid(ax_c)
    panel_label(ax_c, "C")

    fig.tight_layout()
    return save_all(
        fig, "figure3_sampling_concentration", config_hash,
        source_files=["outputs/energy_hub_canonical_v1/unique_supports.csv", "outputs/energy_hub_canonical_v1/seed_arm_summary.csv"],
        description="Sampling concentration and support coverage by arm across 10 seeds: (A) unique exact-K supports, (B) top-10 support probability mass, (C) entropy ratio vs. uniform.",
    )
