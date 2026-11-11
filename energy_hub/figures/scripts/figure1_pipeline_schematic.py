"""Manuscript Figure 2: stage-wise architecture of the surrogate-guided hybrid QAOA workflow.

Purely diagrammatic; no data dependency.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _common import COLORS, MAX_WIDTH_IN, save_all, setup_style


def _box(ax, xy, w, h, text, color, fontsize=7.4):
    box = FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.1, edgecolor=color, facecolor=color, alpha=0.15,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def _arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.2, color="#444444"))


def make_figure(config_hash: str) -> dict:
    setup_style()
    plt.rcParams.update({"font.family": "Times New Roman"})
    fig, ax = plt.subplots(figsize=(MAX_WIDTH_IN, 2.65))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.7)
    ax.axis("off")

    stages = [
        (0.20, "Surrogate\nHamiltonian\nS0/S1/S2", COLORS["S0"]),
        (2.18, "QAOA + uniform\nsampling\np(y)", COLORS["S1"]),
        (4.16, "Structural\nfilter\n+ no-good memory", COLORS["S2"]),
        (6.14, "Exact-K\nreference\nV(y)=TAC(y)", COLORS["Uniform"]),
        (8.12, "Budgeted\nreporting\nB=10,30,60,100", "#555555"),
    ]
    w, h, y = 1.55, 1.15, 0.75
    for x, text, color in stages:
        _box(ax, (x, y), w, h, text, color)
    for (x1, _, _), (x2, _, _) in zip(stages[:-1], stages[1:]):
        _arrow(ax, (x1 + w, y + h / 2), (x2, y + h / 2))

    ax.text(
        5,
        2.25,
        "E(y)  " + r"$\rightarrow$" + "  p(y)  " + r"$\rightarrow$" + "  eligibility  " + r"$\rightarrow$" + "  V(y)",
        ha="center",
        fontsize=8.8,
        fontweight="bold",
    )

    return save_all(
        fig, "figure1_pipeline_schematic", config_hash,
        source_files=[],
        description="Audited attribution pipeline: E(y) -> p(y) -> structural eligibility -> V(y), with the nested downstream budget.",
    )
