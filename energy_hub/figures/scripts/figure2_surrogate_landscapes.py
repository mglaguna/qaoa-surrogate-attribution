"""Manuscript Figure 3: exact-K surrogate energy landscapes.

Panel A shows the centered energies E_m(y) of Eq. (36) over all C(18,6) = 18,564
exact-K supports (counts); panel B the pairwise Spearman rank correlations; panel C
the top-k Jaccard overlaps. Deterministic enumeration over the fixed instance
(no QAOA, no IPOPT).
"""

from __future__ import annotations

import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from _common import COLORS, CODE_DIR, DATA_DIR, MAX_WIDTH_IN, panel_label, save_all, setup_style, light_grid

sys.path.insert(0, str(CODE_DIR))

from hybrid_pipeline.benchmarks.energy_hub import (
    build_energy_hub_instance,
    build_energy_hub_s0,
    build_energy_hub_s1,
    build_energy_hub_s2,
    evaluate_surrogate_energy,
)


def _enumerate_landscape(instance: dict, surrogates: dict) -> pd.DataFrame:
    assets = instance["ASSETS"]
    k = instance["K"]
    rows = []
    for combo in itertools.combinations(assets, k):
        selection = {a: (1 if a in combo else 0) for a in assets}
        bits = "".join(str(selection[a]) for a in assets)
        row = {"bitstring": bits}
        for name, surrogate in surrogates.items():
            row[f"energy_{name}"] = evaluate_surrogate_energy(selection, surrogate)
        rows.append(row)
    return pd.DataFrame(rows)


def make_figure(config_hash: str) -> dict:
    setup_style()
    instance = build_energy_hub_instance()
    surrogates = {"S0": build_energy_hub_s0(instance), "S1": build_energy_hub_s1(instance), "S2": build_energy_hub_s2(instance)}
    df = _enumerate_landscape(instance, surrogates)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_path = DATA_DIR / "figure2_surrogate_landscapes_data.csv"
    df.to_csv(data_path, index=False)

    pairs = [("S0", "S1"), ("S0", "S2"), ("S1", "S2")]
    corrs = {f"{a}-{b}": spearmanr(df[f"energy_{a}"], df[f"energy_{b}"]).correlation for a, b in pairs}

    top_ks = np.unique(np.geomspace(10, len(df), 12).astype(int))
    jaccard = {f"{a}-{b}": [] for a, b in pairs}
    ranks = {name: df[f"energy_{name}"].rank(method="first").to_numpy() for name in surrogates}
    for k in top_ks:
        for a, b in pairs:
            top_a = set(np.where(ranks[a] <= k)[0])
            top_b = set(np.where(ranks[b] <= k)[0])
            jaccard[f"{a}-{b}"].append(len(top_a & top_b) / len(top_a | top_b))

    fig, axes = plt.subplots(1, 3, figsize=(MAX_WIDTH_IN, 2.6))
    ax_a, ax_b, ax_c = axes

    bins = np.linspace(-1.0, 1.0, 61)
    for name in ("S0", "S1", "S2"):
        vals = df[f"energy_{name}"]
        centered = vals - vals.mean()
        assert centered.abs().max() <= 1.0 + 1e-9
        ax_a.hist(centered, bins=bins, histtype="step", color=COLORS[name], label=name, linewidth=1.3)
    ax_a.set_xlabel("Centered energy $E_m(y)$, Eq. (36)")
    ax_a.set_ylabel("Count (exact-K supports)")
    ax_a.legend(frameon=False)
    light_grid(ax_a)
    panel_label(ax_a, "A")

    labels = list(corrs.keys())
    vals = [corrs[k] for k in labels]
    bar_colors = ["#888888" if v < 0 else COLORS["S2"] for v in vals]
    ax_b.bar(labels, vals, color=bar_colors)
    ax_b.axhline(0, color="black", linewidth=0.8)
    ax_b.set_ylabel("Spearman rank correlation")
    ax_b.set_ylim(-1, 1)
    light_grid(ax_b)
    panel_label(ax_b, "B")

    for pair in jaccard:
        ax_c.plot(top_ks, jaccard[pair], marker="o", markersize=3, label=pair)
    ax_c.set_xscale("log")
    ax_c.set_xlabel("Top-k region")
    ax_c.set_ylabel("Jaccard overlap")
    ax_c.legend(frameon=False, fontsize=6.5)
    light_grid(ax_c)
    panel_label(ax_c, "C")

    fig.tight_layout()
    return save_all(
        fig, "figure2_surrogate_landscapes", config_hash,
        source_files=["code/hybrid_pipeline/benchmarks/energy_hub/* (fresh enumeration)"],
        description="Exact-K surrogate energy landscapes: (A) centered energy distributions (Eq. 36, counts), (B) pairwise Spearman rank correlation, (C) top-k Jaccard overlap.",
    )
