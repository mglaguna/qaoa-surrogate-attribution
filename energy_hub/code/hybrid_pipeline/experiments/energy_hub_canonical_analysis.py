"""Energy Hub canonical campaign analysis (EH_CANONICAL_V1).

Reads exclusively from an ``outputs/energy_hub_canonical_v1/``-style directory
produced by ``energy_hub_canonical_campaign.py`` and derives budget-nested
statistics (best-so-far TAC, gap to the exhaustive reference, population
percentile, pairwise win/tie/loss, diversity/concentration, candidate flow)
at each reporting budget in ``REPORTING_BUDGETS``. Produces no figures.

Because the campaign samples/ranks/filters once per (arm, seed) and stores
the full ordered list of up to ``MAX_CONTINUOUS_EVALUATIONS_PER_RUN`` attempted
evaluations, every budget B in ``REPORTING_BUDGETS`` is simply the prefix
``evaluation_rank <= B`` of ``continuous_evaluations.csv`` -- no resampling,
reranking, or re-filtering happens here.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hybrid_pipeline.experiments import energy_hub_canonical_config as cfg


class CanonicalOutputError(RuntimeError):
    pass


def load_canonical_config(output: Path) -> dict:
    path = output / "canonical_config.json"
    if not path.exists():
        raise CanonicalOutputError(f"{path} not found; is this a canonical campaign output directory?")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("protocol_version") != cfg.CANONICAL_PROTOCOL_VERSION:
        raise CanonicalOutputError(
            f"canonical_config.json protocol_version={data.get('protocol_version')!r} does not match "
            f"the analysis script's expected {cfg.CANONICAL_PROTOCOL_VERSION!r}. Refusing to analyse "
            "outputs from a different protocol version."
        )
    return data


def load_continuous(output: Path) -> pd.DataFrame:
    path = output / "continuous_evaluations.csv"
    if not path.exists():
        raise CanonicalOutputError(f"{path} not found.")
    return pd.read_csv(path, dtype={"support_bitstring": str, "support_hash": str})


def load_unique(output: Path) -> pd.DataFrame:
    path = output / "unique_supports.csv"
    if not path.exists():
        raise CanonicalOutputError(f"{path} not found.")
    return pd.read_csv(path, dtype={"support_bitstring": str, "support_hash": str})


def load_exhaustive_best(output: Path) -> float:
    path = output / "exhaustive_reference_summary.json"
    if not path.exists():
        raise CanonicalOutputError(f"{path} not found.")
    return float(json.loads(path.read_text(encoding="utf-8"))["best_TAC"])


def tac_tie(a: float, b: float, tol: float = cfg.TAC_TIE_REL_TOL) -> bool:
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def seed_arm_budget_summary(continuous: pd.DataFrame, unique_df: pd.DataFrame, global_best: float) -> pd.DataFrame:
    rows = []
    for (arm, seed), grp in continuous.groupby(["arm", "seed"]):
        grp_sorted = grp.sort_values("evaluation_rank")
        n_available = int(len(grp_sorted))
        u = unique_df[(unique_df["arm"] == arm) & (unique_df["seed"] == seed)]
        probs = u["empirical_probability"].to_numpy(dtype=float)
        probs = probs[probs > 0]
        entropy = float(-(probs * np.log(probs)).sum()) if len(probs) else 0.0
        entropy_uniform = float(np.log(cfg.N_TOTAL_SUPPORTS))
        entropy_ratio = entropy / entropy_uniform if entropy_uniform > 0 else 0.0
        top1_freq = float(u["empirical_probability"].max()) if not u.empty else 0.0

        for budget in cfg.REPORTING_BUDGETS:
            prefix = grp_sorted[grp_sorted["evaluation_rank"] <= budget]
            feas = prefix[prefix["nlp_feasible"] == True]
            best_tac = float(feas["TAC"].min()) if not feas.empty else float("nan")
            best_pct = float(feas.loc[feas["TAC"].idxmin(), "exhaustive_percentile"]) if not feas.empty else float("nan")
            n_attempted = int(min(budget, n_available))
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "budget": budget,
                    "n_attempted": n_attempted,
                    "n_feasible": int(len(feas)),
                    "candidate_shortage": bool(n_available < budget),
                    "best_so_far_TAC": best_tac,
                    "gap_abs": (best_tac - global_best) if pd.notna(best_tac) else float("nan"),
                    "gap_pct": (100.0 * (best_tac - global_best) / global_best) if pd.notna(best_tac) else float("nan"),
                    "exhaustive_percentile": best_pct,
                    "unique_support_count": int(len(u)),
                    "entropy": entropy,
                    "entropy_ratio": entropy_ratio,
                    "top1_frequency": top1_freq,
                }
            )
    return pd.DataFrame(rows)


def arm_budget_summary(seed_budget_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, budget), grp in seed_budget_df.groupby(["arm", "budget"]):
        rows.append(
            {
                "arm": arm,
                "budget": budget,
                "n_seeds": int(len(grp)),
                "median_best_so_far_TAC": float(grp["best_so_far_TAC"].median()),
                "mean_best_so_far_TAC": float(grp["best_so_far_TAC"].mean()),
                "std_best_so_far_TAC": float(grp["best_so_far_TAC"].std(ddof=0)),
                "median_gap_pct": float(grp["gap_pct"].median()),
                "median_percentile": float(grp["exhaustive_percentile"].median()),
                "n_seeds_with_feasible_solution": int(grp["best_so_far_TAC"].notna().sum()),
                "n_seeds_with_candidate_shortage": int(grp["candidate_shortage"].sum()),
            }
        )
    return pd.DataFrame(rows)


def pairwise_wtl(seed_budget_df: pd.DataFrame, arms: list[str]) -> pd.DataFrame:
    rows = []
    for budget in cfg.REPORTING_BUDGETS:
        sub = seed_budget_df[seed_budget_df["budget"] == budget]
        pivot = sub.pivot(index="seed", columns="arm", values="best_so_far_TAC")
        for arm_a, arm_b in itertools.combinations(arms, 2):
            if arm_a not in pivot.columns or arm_b not in pivot.columns:
                continue
            wins = ties = losses = 0
            for seed, row in pivot.iterrows():
                a, b = row.get(arm_a), row.get(arm_b)
                if pd.isna(a) or pd.isna(b):
                    continue
                if tac_tie(a, b):
                    ties += 1
                elif a < b:
                    wins += 1
                else:
                    losses += 1
            rows.append(
                {
                    "budget": budget,
                    "arm_a": arm_a,
                    "arm_b": arm_b,
                    "arm_a_wins": wins,
                    "ties": ties,
                    "arm_a_losses": losses,
                    "n_seeds_compared": wins + ties + losses,
                    "tie_tolerance_rel": cfg.TAC_TIE_REL_TOL,
                }
            )
    return pd.DataFrame(rows)


def candidate_flow_summary(output: Path) -> pd.DataFrame:
    path = output / "candidate_flow.csv"
    if not path.exists():
        raise CanonicalOutputError(f"{path} not found.")
    flow = pd.read_csv(path)
    rows = []
    numeric_cols = [
        "total_shots",
        "unique_exactK_supports",
        "raw_exactK_events",
        "duplicates_skipped",
        "structural_rejects",
        "eligible_supports",
        "attempted_continuous_evaluations",
        "feasible_continuous_evaluations",
        "infeasible_continuous_evaluations",
    ]
    for arm, grp in flow.groupby("arm"):
        row = {"arm": arm, "n_seeds": int(len(grp))}
        for col in numeric_cols:
            row[f"median_{col}"] = float(grp[col].median())
        row["n_seeds_candidate_shortage"] = int(grp["candidate_shortage_flag"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def run_analysis(output: Path) -> dict:
    load_canonical_config(output)
    continuous = load_continuous(output)
    unique_df = load_unique(output)
    global_best = load_exhaustive_best(output)

    seed_budget_df = seed_arm_budget_summary(continuous, unique_df, global_best)
    arm_budget_df = arm_budget_summary(seed_budget_df)
    arms_present = sorted(continuous["arm"].unique().tolist())
    wtl_df = pairwise_wtl(seed_budget_df, arms_present)
    flow_df = candidate_flow_summary(output)

    analysis_dir = output / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    seed_budget_df.to_csv(analysis_dir / "seed_arm_budget_summary.csv", index=False)
    arm_budget_df.to_csv(analysis_dir / "arm_budget_summary.csv", index=False)
    wtl_df.to_csv(analysis_dir / "pairwise_win_tie_loss.csv", index=False)
    flow_df.to_csv(analysis_dir / "candidate_flow_summary.csv", index=False)

    return {
        "n_seed_arm_budget_rows": int(len(seed_budget_df)),
        "n_arm_budget_rows": int(len(arm_budget_df)),
        "n_pairwise_rows": int(len(wtl_df)),
        "arms": arms_present,
        "budgets": cfg.REPORTING_BUDGETS,
        "global_best_TAC": global_best,
    }


def parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Energy Hub canonical campaign analysis (EH_CANONICAL_V1)")
    parser.add_argument("--output-dir", default=str(cfg.OUTPUT_ROOT))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = run_analysis(Path(args.output_dir))
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
