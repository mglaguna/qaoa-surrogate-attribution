"""Energy Hub K-sensitivity sweep (protocol EH_K_SWEEP_APPENDIX_V1; Supporting Information S1.1).

Varies only the support cardinality, K in {2, 4, 6, 8}; N = 18 and T = 6 (technology
catalog and representative periods) are unchanged. The study uses its own protocol tag
and output directory, so seed derivation and file hashes cannot collide with the
canonical campaign. For K = 6 the canonical exhaustive reference and the canonical
Uniform B = 100 result are reused by lookup, never recomputed.

For each other K the script:
  1. enumerates every C(18, K) support, applies the structural filter and solves the
     fixed-support NLP for every structurally eligible support (a full exhaustive
     reference at that K);
  2. runs the Uniform exact-K control (same draw/order procedure as the canonical
     campaign) for seeds 101-110, evaluates up to B = 100 eligible candidates per seed
     by lookup against that reference, and reports the median best-so-far TAC at B = 100.

S0, S1 and S2 are not re-run at other K (that would require new surrogate Hamiltonians
for each K).

Usage (from energy_hub/, with code/ on PYTHONPATH):
    python code/hybrid_pipeline/experiments/energy_hub_k_sweep_appendix.py --output-dir OUT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from hybrid_pipeline.benchmarks.energy_hub.instance import EnergyHubInstanceBuilder
from hybrid_pipeline.benchmarks.energy_hub.model import solve_energy_hub_support, structural_filter
from hybrid_pipeline.benchmarks.energy_hub.surrogates import bits_to_selection
from hybrid_pipeline.cli.config import build_solver

PROTOCOL_VERSION = "EH_K_SWEEP_APPENDIX_V1"
N = 18
T = 6
SWEEP_K_VALUES = [2, 4, 6, 8]
SEEDS = list(range(101, 111))
UNIFORM_DRAWS = 5000
MAX_B = 100

OUTPUT_ROOT = Path("outputs/energy_hub_k_sweep_appendix")

_PKG_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_EXHAUSTIVE_REFERENCE = _PKG_ROOT / "data/reference/energy_hub_exhaustive_classical_rows.csv"
CANONICAL_ARM_BUDGET_SUMMARY = _PKG_ROOT / "data/canonical_5k/analysis/arm_budget_summary.csv"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def derive_seed(base_seed: int, arm: str, restart: int, stage: str) -> int:
    """Same SHA-256 derivation scheme as the canonical campaign, but keyed
    under this appendix's own protocol tag so sub-seeds never collide with
    (or silently reuse) EH_CANONICAL_V1's sub-seeds."""
    payload = f"{PROTOCOL_VERSION}|{base_seed}|{arm}|{restart}|{stage}"
    digest = _sha256_hex(payload)
    return int(digest[:8], 16)


def support_hash(bitstring: str) -> str:
    return hashlib.sha256(str(bitstring).encode("utf-8")).hexdigest()[:16]


def build_exhaustive_reference(K: int, instance: dict, solver, log_every: int = 2000) -> pd.DataFrame:
    techs = instance["ASSETS"]
    n = len(techs)
    rows = []
    t0 = time.time()
    n_total = 0
    for combo in combinations(range(n), K):
        n_total += 1
        bits = ["0"] * n
        for i in combo:
            bits[i] = "1"
        bitstring = "".join(bits)
        selection = {t: (1 if bitstring[i] == "1" else 0) for i, t in enumerate(techs)}

        if not structural_filter(selection, instance):
            rows.append(
                dict(
                    support_bitstring=bitstring,
                    support_hash=support_hash(bitstring),
                    method="structural_filter",
                    feasible=False,
                    TAC=float("nan"),
                    active_count=float("nan"),
                )
            )
            continue

        result = solve_energy_hub_support(selection, instance, solver=solver)
        rows.append(
            dict(
                support_bitstring=bitstring,
                support_hash=support_hash(bitstring),
                method="ipopt",
                feasible=bool(result["feasible"]),
                TAC=result["TAC"] if result["feasible"] else float("nan"),
                active_count=result["active_count"] if result["feasible"] else float("nan"),
            )
        )
        if log_every and n_total % log_every == 0:
            print(f"  [K={K}] {n_total} combos processed, {time.time()-t0:.1f}s elapsed", flush=True)

    df = pd.DataFrame(rows)
    print(
        f"[K={K}] exhaustive reference done: n_total={len(df)}, "
        f"structural_rejected={(df['method']=='structural_filter').sum()}, "
        f"ipopt_attempted={(df['method']=='ipopt').sum()}, "
        f"ipopt_feasible={((df['method']=='ipopt') & (df['feasible']==True)).sum()}, "
        f"runtime={time.time()-t0:.1f}s",
        flush=True,
    )
    return df


def load_canonical_k6_reference() -> pd.DataFrame:
    """Read-only load of the already-frozen K=6 reference. Never modified."""
    if not CANONICAL_EXHAUSTIVE_REFERENCE.exists():
        raise FileNotFoundError(
            f"Frozen K=6 reference not found at {CANONICAL_EXHAUSTIVE_REFERENCE}; "
            "run this script from the core/ directory."
        )
    df = pd.read_csv(CANONICAL_EXHAUSTIVE_REFERENCE, dtype={"support_bitstring": str, "support_hash": str})
    return df[["support_bitstring", "support_hash", "method", "feasible", "TAC", "active_count"]].copy()


def load_canonical_k6_uniform_b100() -> dict:
    """Read-only load of the already-published, frozen Uniform-B100 result
    for K=6 (median best-so-far TAC, gap%, percentile), from the same
    canonical_5k package the manuscript's main text already cites. Re-running
    a fresh Uniform draw for K=6 under this appendix's own protocol tag would
    give a legitimate but numerically DIFFERENT result (different derived
    sub-seeds) and create a spurious discrepancy against Table 3/Figure 5 in
    the main text, so K=6's row is looked up here instead of recomputed."""
    if not CANONICAL_ARM_BUDGET_SUMMARY.exists():
        raise FileNotFoundError(f"Canonical arm_budget_summary.csv not found at {CANONICAL_ARM_BUDGET_SUMMARY}.")
    df = pd.read_csv(CANONICAL_ARM_BUDGET_SUMMARY)
    row = df[(df["arm"] == "Uniform") & (df["budget"] == 100)]
    if len(row) != 1:
        raise RuntimeError("Expected exactly one Uniform/budget=100 row in arm_budget_summary.csv.")
    row = row.iloc[0]
    return dict(
        median_best_TAC=float(row["median_best_so_far_TAC"]),
        median_gap_pct=float(row["median_gap_pct"]),
        median_percentile=float(row["median_percentile"]),
        n_seeds=int(row["n_seeds"]),
        n_seeds_with_feasible_solution=int(row["n_seeds_with_feasible_solution"]),
    )


def sample_uniform_arm(base_seed: int, N: int, K: int) -> list[str]:
    """Identical draw/dedup/order logic to the canonical campaign's
    ``sample_uniform_arm``, re-derived here (not imported) because the
    canonical campaign module is frozen and intentionally does not expose a
    K-parameterized entry point."""
    sampling_seed = derive_seed(base_seed, "Uniform", 0, "uniform_draw")
    rng = np.random.default_rng(sampling_seed)
    seen: dict = {}
    order: list = []
    for _ in range(UNIFORM_DRAWS):
        idx = set(rng.choice(N, K, replace=False).tolist())
        bits = "".join("1" if i in idx else "0" for i in range(N))
        if bits not in seen:
            seen[bits] = 0
            order.append(bits)
        seen[bits] += 1
    return order


def build_percentile_lookup(exhaustive_df: pd.DataFrame) -> dict:
    feasible = exhaustive_df[exhaustive_df["feasible"] == True].copy()
    feasible = feasible.sort_values("TAC").reset_index(drop=True)
    n = len(feasible)
    if n == 0:
        return {}
    return {bitstring: 100.0 * (i + 1) / n for i, bitstring in enumerate(feasible["support_bitstring"])}


def run_uniform_seed(seed: int, K: int, instance: dict, lookup: dict, percentile_lookup: dict) -> dict:
    order = sample_uniform_arm(seed, instance["N"], K)
    techs = instance["ASSETS"]

    eligible = []
    for bitstring in order:
        selection = {t: (1 if bitstring[i] == "1" else 0) for i, t in enumerate(techs)}
        if structural_filter(selection, instance):
            eligible.append(bitstring)
        if len(eligible) >= MAX_B:
            break

    candidate_shortage = len(eligible) < MAX_B
    best_tac = float("nan")
    best_percentile = float("nan")
    n_feasible = 0
    for bitstring in eligible:
        ref = lookup.get(bitstring)
        if ref is None:
            raise RuntimeError(f"Support {bitstring} missing from the K={K} exhaustive reference.")
        if ref["feasible"]:
            n_feasible += 1
            if pd.isna(best_tac) or ref["TAC"] < best_tac:
                best_tac = ref["TAC"]
                best_percentile = percentile_lookup.get(bitstring, float("nan"))

    return dict(
        K=K,
        seed=seed,
        n_eligible=len(eligible),
        n_attempted=len(eligible),
        n_feasible=n_feasible,
        candidate_shortage=candidate_shortage,
        best_TAC=best_tac,
        best_percentile=best_percentile,
    )


def run_k(K: int, output: Path, solver) -> dict:
    k_dir = output / f"K{K}"
    k_dir.mkdir(parents=True, exist_ok=True)

    if K == 6:
        print(f"[K=6] reusing frozen canonical exhaustive reference (read-only, not recomputed)")
        exhaustive_df = load_canonical_k6_reference()
    else:
        instance = EnergyHubInstanceBuilder(K=K, allow_supplementary_k=True).generate_instance()
        exhaustive_df = build_exhaustive_reference(K, instance, solver)
        exhaustive_df.to_csv(k_dir / "exhaustive_reference.csv", index=False)

    instance = EnergyHubInstanceBuilder(K=K, allow_supplementary_k=True).generate_instance()

    lookup = {
        row.support_bitstring: {"feasible": bool(row.feasible), "TAC": float(row.TAC) if pd.notna(row.TAC) else float("nan")}
        for row in exhaustive_df.itertuples(index=False)
    }
    percentile_lookup = build_percentile_lookup(exhaustive_df)

    feasible_mask = exhaustive_df["feasible"] == True
    n_feasible_pop = int(feasible_mask.sum())
    tac_star = float(exhaustive_df.loc[feasible_mask, "TAC"].min()) if n_feasible_pop else float("nan")

    if K == 6:
        print(f"[K=6] reusing frozen canonical Uniform-B100 result (read-only, not recomputed)")
        canon = load_canonical_k6_uniform_b100()
        median_best_tac = canon["median_best_TAC"]
        median_gap_pct = canon["median_gap_pct"]
        median_percentile = canon["median_percentile"]
        n_seeds_feasible = canon["n_seeds_with_feasible_solution"]
    else:
        seed_rows = []
        for seed in SEEDS:
            seed_rows.append(run_uniform_seed(seed, K, instance, lookup, percentile_lookup))
        seed_df = pd.DataFrame(seed_rows)
        seed_df.to_csv(k_dir / "uniform_seed_results.csv", index=False)

        n_seeds_feasible = int(seed_df["best_TAC"].notna().sum())
        median_best_tac = float(seed_df["best_TAC"].median()) if n_seeds_feasible else float("nan")
        median_gap_pct = (
            float(100.0 * (median_best_tac - tac_star) / tac_star)
            if n_seeds_feasible and pd.notna(tac_star) and tac_star
            else float("nan")
        )
        median_percentile = float(seed_df["best_percentile"].median()) if n_seeds_feasible else float("nan")

    summary = dict(
        K=K,
        N=N,
        T=T,
        n_total_supports=int(len(exhaustive_df)),
        n_structural_rejected=int((exhaustive_df["method"] == "structural_filter").sum()),
        n_ipopt_attempted=int((exhaustive_df["method"] == "ipopt").sum()),
        n_feasible_population=n_feasible_pop,
        TAC_star=tac_star,
        n_seeds=len(SEEDS),
        n_seeds_with_feasible_solution=n_seeds_feasible,
        median_uniform_best_TAC_B100=median_best_tac,
        median_gap_to_TAC_star_pct=median_gap_pct,
        median_percentile_B100=median_percentile,
        uniform_b100_source=("canonical_5k (frozen, reused)" if K == 6 else "recomputed_this_appendix"),
    )
    (k_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"[K={K}] SUMMARY: {json.dumps(summary, default=str)}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Energy Hub K-sensitivity appendix sweep")
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    parser.add_argument("--k-values", type=int, nargs="+", default=SWEEP_K_VALUES)
    args = parser.parse_args(argv)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    solver = build_solver()
    if not solver.available(False):
        print("ERROR: ipopt solver not available on PATH.", file=sys.stderr)
        sys.exit(1)

    started_at = utc_now()
    all_summaries = []
    for K in args.k_values:
        all_summaries.append(run_k(K, output, solver))

    summary_df = pd.DataFrame(all_summaries).sort_values("K")
    summary_df.to_csv(output / "k_sweep_summary.csv", index=False)

    manifest = dict(
        protocol_version=PROTOCOL_VERSION,
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        k_values=args.k_values,
        seeds=SEEDS,
        note=(
            "Supplementary appendix study. S0/S1/S2 surrogate arms are not "
            "re-run at other K; only the exhaustive classical reference "
            "(TAC*) and the Uniform-random control at B=100 are computed "
            "per K. K=6 reuses the frozen EH_CANONICAL_V1 exhaustive "
            "reference by read-only lookup."
        ),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")

    print("\n=== K-SWEEP SUMMARY ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
