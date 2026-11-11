"""Manuscript Figure 10: ex ante structural characteristics of the sampled supports and
their relationship with downstream TAC.

Panel A summarizes structural feature shifts relative to Uniform; panel B shows the
evaluated supports in the nominal-adequacy / technology-family-diversity plane,
colored by downstream TAC (B <= 100 evaluations).
"""

from __future__ import annotations

import sys

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import COLORS, CODE_DIR, DATA_DIR, MAX_WIDTH_IN, OUTPUTS_DIR, light_grid, panel_label, require_file, save_all, setup_style

sys.path.insert(0, str(CODE_DIR))

from hybrid_pipeline.benchmarks.energy_hub import build_energy_hub_instance
from hybrid_pipeline.benchmarks.energy_hub.surrogates import bits_to_selection


ARMS_ABC = ["S0", "S1", "S2"]
ARMS_ALL = ["S0", "S1", "S2", "Uniform"]
PANEL_A_ORDER = ["nominal_adequacy_q10", "family_simpson", "temporal_cv_max"]
PANEL_A_SHORT = {
    "nominal_adequacy_q10": "Nominal adequacy q10",
    "family_simpson": "Family diversity",
    "temporal_cv_max": "Max temporal CV",
}
PANEL_B_X_LABEL = "Ex ante nominal adequacy q10"
PANEL_B_Y_LABEL = "Technology-family diversity (Simpson)"
TAC_LABEL = "Optimized TAC from downstream NLP (million currency units)"


def _support_features(bitstring: str, instance: dict) -> dict:
    d = instance["ENERGY_HUB_DATA"]
    selection = bits_to_selection(tuple(int(ch) for ch in bitstring), instance)
    selected = [asset for asset in instance["ASSETS"] if selection[asset] == 1]

    service_ratios = {"E": [], "Q": [], "C": []}
    for t in d["periods"]:
        service_ratios["E"].append(
            sum(d["Smax"][i] * d["rho"].loc[i, t] * max(float(d["etaE"][i]), 0.0) for i in selected)
            / float(d["DE"][t])
        )
        service_ratios["Q"].append(
            sum(d["Smax"][i] * d["rho"].loc[i, t] * max(float(d["etaQ"][i]), 0.0) for i in selected)
            / float(d["DQ"][t])
        )
        service_ratios["C"].append(
            sum(d["Smax"][i] * d["rho"].loc[i, t] * max(float(d["etaC"][i]), 0.0) for i in selected)
            / float(d["DC"][t])
        )

    all_ratios = np.asarray([v for values in service_ratios.values() for v in values], dtype=float)
    family_counts = pd.Series([d["group_of"][asset] for asset in selected]).value_counts().to_numpy(dtype=float)
    family_p = family_counts / family_counts.sum()

    temporal_cvs = []
    for values in service_ratios.values():
        arr = np.asarray(values, dtype=float)
        mean = arr.mean()
        temporal_cvs.append(float(arr.std(ddof=0) / mean) if mean > 1e-12 else np.nan)

    return {
        "nominal_adequacy_q10": float(np.percentile(all_ratios, 10)),
        "family_simpson": float(1.0 - np.sum(family_p**2)),
        "temporal_cv_max": float(np.nanmax(temporal_cvs)),
    }


def _feature_table(bitstrings: pd.Series, instance: dict) -> pd.DataFrame:
    clean = sorted({str(b).zfill(len(instance["ASSETS"])) for b in bitstrings.dropna().astype(str)})
    rows = [{"support_bitstring": bitstring, **_support_features(bitstring, instance)} for bitstring in clean]
    return pd.DataFrame(rows)


def _panel_a(unique: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    unique_arm_support = (
        unique[["arm", "support_bitstring"]]
        .drop_duplicates()
        .merge(features, on="support_bitstring", how="left")
    )
    unique_arm_support.to_csv(DATA_DIR / "figure7_engineering_fingerprints_unique_arm_support_features.csv", index=False)

    uniform = unique_arm_support[unique_arm_support["arm"] == "Uniform"]
    ref_median = {feature: float(uniform[feature].median()) for feature in PANEL_A_ORDER}
    ref_iqr = {
        feature: float(uniform[feature].quantile(0.75) - uniform[feature].quantile(0.25)) or 1.0
        for feature in PANEL_A_ORDER
    }

    rows = []
    for arm in ARMS_ABC:
        sub = unique_arm_support[unique_arm_support["arm"] == arm]
        for feature in PANEL_A_ORDER:
            median = float(sub[feature].median())
            q25 = float(sub[feature].quantile(0.25))
            q75 = float(sub[feature].quantile(0.75))
            rows.append(
                {
                    "panel": "A",
                    "aggregation": "unique_arm_support_records",
                    "arm": arm,
                    "feature": feature,
                    "feature_short": PANEL_A_SHORT[feature],
                    "median": median,
                    "q25": q25,
                    "q75": q75,
                    "reference_arm": "Uniform",
                    "reference_median": ref_median[feature],
                    "reference_iqr": ref_iqr[feature],
                    "standardized_delta_median": (median - ref_median[feature]) / ref_iqr[feature],
                    "standardized_delta_q25": (q25 - ref_median[feature]) / ref_iqr[feature],
                    "standardized_delta_q75": (q75 - ref_median[feature]) / ref_iqr[feature],
                }
            )
    return pd.DataFrame(rows)


def _panel_b(evals: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    evals = evals.copy()
    evals["evaluation_rank"] = pd.to_numeric(evals["evaluation_rank"], errors="coerce")
    evals["TAC"] = pd.to_numeric(evals["TAC"], errors="coerce")
    evals["nlp_feasible"] = evals["nlp_feasible"].astype(str).str.lower().eq("true")
    evals = evals[evals["evaluation_rank"] <= 100].copy()

    rows = []
    for (arm, support), sub in evals.groupby(["arm", "support_bitstring"]):
        feasible = sub[sub["nlp_feasible"] & sub["TAC"].notna()]
        rows.append(
            {
                "panel": "B",
                "aggregation": "unique_nlp_evaluated_arm_support_records",
                "arm": arm,
                "support_bitstring": support,
                "support_cardinality": int(str(support).count("1")),
                "nlp_feasible": bool(not feasible.empty),
                "tac_million": float(feasible["TAC"].min() / 1e6) if not feasible.empty else np.nan,
            }
        )
    panel_b = pd.DataFrame(rows).merge(features, on="support_bitstring", how="left")
    panel_b["x_feature"] = "nominal_adequacy_q10"
    panel_b["x_value"] = panel_b["nominal_adequacy_q10"]
    panel_b["y_feature"] = "family_simpson"
    panel_b["y_value"] = panel_b["family_simpson"]
    return panel_b


def _draw_panel_a(ax: plt.Axes, panel_a: pd.DataFrame) -> None:
    y_positions = np.arange(len(PANEL_A_ORDER))
    offsets = {"S0": -0.18, "S1": 0.0, "S2": 0.18}
    markers = {"S0": "o", "S1": "s", "S2": "^"}
    for arm in ARMS_ABC:
        sub = panel_a[panel_a["arm"] == arm].set_index("feature")
        xs, xerr_low, xerr_high = [], [], []
        for feature in PANEL_A_ORDER:
            row = sub.loc[feature]
            x = float(row["standardized_delta_median"])
            low = float(row["standardized_delta_q25"])
            high = float(row["standardized_delta_q75"])
            xs.append(x)
            xerr_low.append(max(0.0, x - low))
            xerr_high.append(max(0.0, high - x))
        ax.errorbar(
            xs,
            y_positions + offsets[arm],
            xerr=[xerr_low, xerr_high],
            fmt=markers[arm],
            ms=4.8,
            lw=1.1,
            capsize=2.5,
            color=COLORS[arm],
            markeredgecolor="white",
            markeredgewidth=0.3,
            label=arm,
        )
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_xlim(-1.15, 1.15)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([PANEL_A_SHORT[f] for f in PANEL_A_ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("Median shift vs Uniform\n(Uniform IQR units)")
    ax.legend(loc="upper left", ncol=1, frameon=False)
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    panel_label(ax, "A")


def _draw_panel_b(fig: plt.Figure, gs, panel_b: pd.DataFrame) -> None:
    feasible = panel_b[panel_b["nlp_feasible"]].copy()
    infeasible = panel_b[~panel_b["nlp_feasible"]].copy()
    x_min, x_max = panel_b["x_value"].min(), panel_b["x_value"].max()
    y_min, y_max = panel_b["y_value"].min(), panel_b["y_value"].max()
    x_pad = (x_max - x_min) * 0.06
    y_pad = max((y_max - y_min) * 0.12, 0.02)
    norm = mpl.colors.Normalize(vmin=float(feasible["tac_million"].min()), vmax=float(feasible["tac_million"].max()))
    cmap = plt.get_cmap("viridis_r")

    facet_axes = []
    for i, arm in enumerate(ARMS_ALL):
        ax = fig.add_subplot(gs[i // 2, 1 + i % 2])
        facet_axes.append(ax)
        fsub = feasible[feasible["arm"] == arm]
        isub = infeasible[infeasible["arm"] == arm]
        if not isub.empty:
            ax.scatter(isub["x_value"], isub["y_value"], marker="x", s=18, color="#777777", linewidths=0.8, zorder=2)
        ax.scatter(
            fsub["x_value"],
            fsub["y_value"],
            c=fsub["tac_million"],
            cmap=cmap,
            norm=norm,
            marker="o",
            s=16,
            edgecolors="none",
            alpha=0.78,
            zorder=3,
        )
        label = ax.text(
            0.04,
            0.95,
            arm,
            transform=ax.transAxes,
            fontsize=8.2,
            fontweight="bold",
            color=COLORS[arm],
            va="top",
            ha="left",
        )
        label.set_path_effects([path_effects.withStroke(linewidth=2.2, foreground="white")])
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        if i // 2 == 1:
            ax.set_xlabel(PANEL_B_X_LABEL)
        else:
            ax.set_xticklabels([])
        if i == 0:
            ax.set_ylabel(PANEL_B_Y_LABEL)
        elif i % 2 == 0:
            ax.set_ylabel("")
        else:
            ax.set_yticklabels([])
        light_grid(ax)
    panel_label(facet_axes[0], "B")
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=facet_axes, fraction=0.035, pad=0.03)
    cbar.set_label(TAC_LABEL)


def make_figure(config_hash: str) -> dict:
    setup_style()
    instance = build_energy_hub_instance()
    evals = pd.read_csv(require_file(OUTPUTS_DIR / "continuous_evaluations.csv"), dtype={"support_bitstring": str})
    unique = pd.read_csv(require_file(OUTPUTS_DIR / "unique_supports.csv"), dtype={"support_bitstring": str})
    all_bitstrings = pd.concat([evals["support_bitstring"], unique["support_bitstring"]], ignore_index=True)
    features = _feature_table(all_bitstrings, instance)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(DATA_DIR / "figure7_engineering_fingerprints_support_features.csv", index=False)
    panel_a = _panel_a(unique, features)
    panel_b = _panel_b(evals, features)
    long = pd.concat([panel_a, panel_b], ignore_index=True, sort=False)
    long.to_csv(DATA_DIR / "figure7_engineering_fingerprints_data.csv", index=False)

    fig = plt.figure(figsize=(MAX_WIDTH_IN, 4.9), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.15, 1.0, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.09,
        right=0.91,
        top=0.90,
        bottom=0.13,
        wspace=0.40,
        hspace=0.38,
    )
    ax_a = fig.add_subplot(gs[:, 0])
    _draw_panel_a(ax_a, panel_a)
    _draw_panel_b(fig, gs, panel_b)

    return save_all(
        fig,
        "figure7_engineering_fingerprints",
        config_hash,
        source_files=[
            "outputs/energy_hub_canonical_v1/unique_supports.csv",
            "outputs/energy_hub_canonical_v1/continuous_evaluations.csv",
            "code/hybrid_pipeline (fresh per-support feature computation)",
        ],
        description="Engineering fingerprints of surrogate-induced structural search bias, recomputed from EH_CANONICAL_V1 using the historical forest-plus-facets grammar: (A) median ex ante feature shifts relative to Uniform in Uniform-IQR units; (B) canonical B<=100 evaluated supports in nominal adequacy versus Simpson family-diversity space, colored by downstream TAC.",
    )
