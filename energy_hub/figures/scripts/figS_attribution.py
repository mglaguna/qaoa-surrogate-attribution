"""Manuscript Figure 8: sampling coverage and S0-vs-S1 seed-paired outcomes for the
15,000-shot and the independent 5,000-shot S1 configurations.

Reads data/attribution/ (S1@15k campaign slice, Control A, Control B) and
data/canonical_5k/ (S0, 5,000 shots in every regime). Panel A: unique exact-K support
pool per seed; panel B: S0 win fraction against S1 across downstream budgets. The
figure shows an association between coverage and outcome; it does not isolate a cause.
"""

from __future__ import annotations

import pandas as pd

from _common import COLORS, FIGURES_ROOT, PACKAGE_ROOT, MAX_WIDTH_IN, OUTPUTS_DIR, light_grid, panel_label, require_file, save_all, setup_style

ATTR_DIR = PACKAGE_ROOT / "data" / "attribution"

VARIANT_COLOR = {
    "S0": COLORS["S0"],
    "S1_15k": COLORS["S1"],
    "Control_B": "#7A3B00",
}
VARIANT_LABEL = {
    "S0": "S0 (5k, reference)",
    "S1_15k": "S1 @ 15k shots (original)",
    "Control_B": "S1 @ 5k shots (Control B, independent)",
}


def make_figure(config_hash: str) -> dict:
    setup_style()
    import matplotlib.pyplot as plt

    s0_unique = pd.read_csv(require_file(OUTPUTS_DIR / "unique_supports.csv"), dtype={"support_bitstring": str})
    s0_unique = s0_unique[s0_unique["arm"] == "S0"]
    s1_15k_unique = pd.read_csv(require_file(ATTR_DIR / "S1_15k_campaign_slice" / "unique_supports.csv"), dtype={"support_bitstring": str})
    ctrl_b_unique = pd.read_csv(require_file(ATTR_DIR / "Control_B_full_output" / "unique_supports.csv"), dtype={"support_bitstring": str})

    pool_counts = {
        "S0": s0_unique.groupby("seed").size(),
        "S1_15k": s1_15k_unique.groupby("seed").size(),
        "Control_B": ctrl_b_unique.groupby("seed").size(),
    }

    wtl = pd.read_csv(require_file(ATTR_DIR / "attribution_pairwise_wtl.csv"))

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(MAX_WIDTH_IN, 3.0))

    variants = ["S0", "S1_15k", "Control_B"]
    for i, variant in enumerate(variants):
        vals = pool_counts[variant]
        ax_a.boxplot(
            vals, positions=[i], widths=0.6, patch_artist=True,
            boxprops=dict(facecolor=VARIANT_COLOR[variant], alpha=0.35),
            medianprops=dict(color=VARIANT_COLOR[variant]),
        )
    ax_a.set_xticks(range(len(variants)))
    ax_a.set_xticklabels(["S0\n(5k)", "S1\n(15k, original)", "S1\n(Control B, 5k)"])
    ax_a.set_ylabel("Unique exact-K supports")
    ax_a.annotate(
        f"median {pool_counts['S1_15k'].median():.0f}",
        xy=(1, pool_counts["S1_15k"].median()), xytext=(1.35, pool_counts["S1_15k"].median() - 150),
        fontsize=8, color=VARIANT_COLOR["S1_15k"],
    )
    ax_a.annotate(
        f"median {pool_counts['Control_B'].median():.0f}",
        xy=(2, pool_counts["Control_B"].median()), xytext=(2.35, pool_counts["Control_B"].median() - 100),
        fontsize=8, color=VARIANT_COLOR["Control_B"],
    )
    light_grid(ax_a)
    panel_label(ax_a, "A")

    for variant, marker, linestyle in [("S1_15k", "s", "--"), ("Control_B", "D", ":")]:
        sub = wtl[wtl["variant"] == variant].sort_values("budget")
        win_fraction = sub["S0_wins"] / (sub["S0_wins"] + sub["ties"] + sub["S0_losses"])
        ax_b.plot(
            sub["budget"], win_fraction, marker=marker, linestyle=linestyle,
            color=VARIANT_COLOR[variant], label=VARIANT_LABEL[variant], markersize=6,
        )
    ax_b.axhline(0.5, color="#333333", linewidth=0.8, linestyle="-", alpha=0.6)
    ax_b.text(102, 0.5, "0.5 (no detected\ndifference)", fontsize=7, va="center", color="#333333")
    ax_b.set_xlabel("Downstream evaluation budget B")
    ax_b.set_ylabel("S0 win fraction vs. S1 (10 seeds)")
    ax_b.set_ylim(-0.05, 1.05)
    ax_b.legend(frameon=False, loc="lower left", fontsize=7)
    light_grid(ax_b)
    panel_label(ax_b, "B")

    fig.tight_layout()
    return save_all(
        fig, "figS_attribution", config_hash,
        source_files=[
            "data/canonical_5k/unique_supports.csv",
            "data/attribution/S1_15k_campaign_slice/unique_supports.csv",
            "data/attribution/Control_B_full_output/unique_supports.csv",
            "data/attribution/attribution_pairwise_wtl.csv",
        ],
        description=(
            "Sampling coverage and S0-vs-S1 seed-paired outcomes (an association, not an isolated cause). "
            "(A) unique exact-K support pool size per seed changes from a median of 4,032.5 (S1 at "
            "15,000 shots) to 1,207.5 (Control B: S1 resampled independently at 5,000 shots, matching "
            "everyone else's budget). (B) S0 win fraction against S1 across downstream budget B: the "
            "crossover (S0 losing a majority of seeds) is seen in the 15k-shot configuration; Control B, "
            "the independent 5k-shot draw, gives a 5-0-5 split at B=60 and B=100 (no detected difference, not equivalence)."
        ),
        out_dir=FIGURES_ROOT / "supplement" / "figS_attribution",
    )


if __name__ == "__main__":
    import json

    config = json.loads((OUTPUTS_DIR / "canonical_config.json").read_text(encoding="utf-8"))
    make_figure(config["config_hash"])
