"""Energy Hub grid-import-cap sensitivity (protocol EH_IMPORT_CAP_SWEEP_APPENDIX_V1;
Supporting Information S1.2).

K, N and T stay at (6, 18, 6); only the import-cap fraction theta of constraint (8),
G_t <= theta * D_t^E, is varied over {0.2, 0.4, 0.6, 0.8}. theta = 0.4 is the value used
in the main study; its exhaustive reference and Uniform B = 100 result are reused by
lookup, so the baseline row equals the main-study values. For the other values only the
exhaustive classical reference (TAC*) and the Uniform control at B = 100 are recomputed,
through the import_cap_frac argument of structural_filter / solve_energy_hub_support
(default 0.4). S0, S1 and S2 are not re-run (their coefficients are built for theta = 0.4).

Usage (from energy_hub/, with code/ on PYTHONPATH):
    python code/hybrid_pipeline/experiments/energy_hub_import_cap_sensitivity_appendix.py --output-dir OUT
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from hybrid_pipeline.benchmarks.energy_hub import build_energy_hub_instance
from hybrid_pipeline.benchmarks.energy_hub.model import solve_energy_hub_support, structural_filter
from hybrid_pipeline.cli.config import build_solver

from hybrid_pipeline.experiments.energy_hub_k_sweep_appendix import (
    derive_seed as _derive_seed_base,
    support_hash,
    build_percentile_lookup,
    load_canonical_k6_reference,
    load_canonical_k6_uniform_b100,
    utc_now,
)

PROTOCOL_VERSION = "EH_IMPORT_CAP_SWEEP_APPENDIX_V1"
BASELINE_IMPORT_CAP_FRAC = 0.4
SWEEP_VALUES = [0.2, 0.4, 0.6, 0.8]
SEEDS = list(range(101, 111))
UNIFORM_DRAWS = 5000
MAX_B = 100

OUTPUT_ROOT = Path("outputs/energy_hub_import_cap_sweep_appendix")


def derive_seed(base_seed: int, arm: str, restart: int, stage: str) -> int:
    """Same SHA-256 scheme as the other appendix, but keyed under THIS
    appendix's own protocol tag, so sub-seeds never collide with either the
    canonical campaign's or the K-sweep appendix's."""
    import hashlib

    payload = f"{PROTOCOL_VERSION}|{base_seed}|{arm}|{restart}|{stage}"
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def build_exhaustive_reference(import_cap_frac: float, instance: dict, solver, log_every: int = 3000) -> pd.DataFrame:
    techs = instance["ASSETS"]
    n = len(techs)
    K = instance["K"]
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

        if not structural_filter(selection, instance, import_cap_frac=import_cap_frac):
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

        result = solve_energy_hub_support(selection, instance, solver=solver, import_cap_frac=import_cap_frac)
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
            print(f"  [cap={import_cap_frac}] {n_total} combos processed, {time.time()-t0:.1f}s elapsed", flush=True)

    df = pd.DataFrame(rows)
    print(
        f"[cap={import_cap_frac}] exhaustive reference done: n_total={len(df)}, "
        f"structural_rejected={(df['method']=='structural_filter').sum()}, "
        f"ipopt_attempted={(df['method']=='ipopt').sum()}, "
        f"ipopt_feasible={((df['method']=='ipopt') & (df['feasible']==True)).sum()}, "
        f"runtime={time.time()-t0:.1f}s",
        flush=True,
    )
    return df


def sample_uniform_arm(base_seed: int, N: int, K: int) -> list[str]:
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


def run_uniform_seed(seed: int, import_cap_frac: float, instance: dict, lookup: dict, percentile_lookup: dict) -> dict:
    order = sample_uniform_arm(seed, instance["N"], instance["K"])
    techs = instance["ASSETS"]

    eligible = []
    for bitstring in order:
        selection = {t: (1 if bitstring[i] == "1" else 0) for i, t in enumerate(techs)}
        if structural_filter(selection, instance, import_cap_frac=import_cap_frac):
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
            raise RuntimeError(f"Support {bitstring} missing from the cap={import_cap_frac} exhaustive reference.")
        if ref["feasible"]:
            n_feasible += 1
            if pd.isna(best_tac) or ref["TAC"] < best_tac:
                best_tac = ref["TAC"]
                best_percentile = percentile_lookup.get(bitstring, float("nan"))

    return dict(
        import_cap_frac=import_cap_frac,
        seed=seed,
        n_eligible=len(eligible),
        n_feasible=n_feasible,
        candidate_shortage=candidate_shortage,
        best_TAC=best_tac,
        best_percentile=best_percentile,
    )


def run_cap(import_cap_frac: float, output: Path, solver) -> dict:
    tag = f"cap{import_cap_frac:.2f}".replace(".", "p")
    cap_dir = output / tag
    cap_dir.mkdir(parents=True, exist_ok=True)

    instance = build_energy_hub_instance()

    is_baseline = abs(import_cap_frac - BASELINE_IMPORT_CAP_FRAC) < 1e-9
    if is_baseline:
        print(f"[cap={import_cap_frac}] baseline -- reusing frozen canonical exhaustive reference (read-only)")
        exhaustive_df = load_canonical_k6_reference()
    else:
        exhaustive_df = build_exhaustive_reference(import_cap_frac, instance, solver)
        exhaustive_df.to_csv(cap_dir / "exhaustive_reference.csv", index=False)

    lookup = {
        row.support_bitstring: {"feasible": bool(row.feasible), "TAC": float(row.TAC) if pd.notna(row.TAC) else float("nan")}
        for row in exhaustive_df.itertuples(index=False)
    }
    percentile_lookup = build_percentile_lookup(exhaustive_df)

    feasible_mask = exhaustive_df["feasible"] == True
    n_feasible_pop = int(feasible_mask.sum())
    tac_star = float(exhaustive_df.loc[feasible_mask, "TAC"].min()) if n_feasible_pop else float("nan")

    if is_baseline:
        print(f"[cap={import_cap_frac}] baseline -- reusing frozen canonical Uniform-B100 result (read-only)")
        canon = load_canonical_k6_uniform_b100()
        median_best_tac = canon["median_best_TAC"]
        median_gap_pct = canon["median_gap_pct"]
        median_percentile = canon["median_percentile"]
        n_seeds_feasible = canon["n_seeds_with_feasible_solution"]
    else:
        seed_rows = [run_uniform_seed(seed, import_cap_frac, instance, lookup, percentile_lookup) for seed in SEEDS]
        seed_df = pd.DataFrame(seed_rows)
        seed_df.to_csv(cap_dir / "uniform_seed_results.csv", index=False)

        n_seeds_feasible = int(seed_df["best_TAC"].notna().sum())
        median_best_tac = float(seed_df["best_TAC"].median()) if n_seeds_feasible else float("nan")
        median_gap_pct = (
            float(100.0 * (median_best_tac - tac_star) / tac_star)
            if n_seeds_feasible and pd.notna(tac_star) and tac_star
            else float("nan")
        )
        median_percentile = float(seed_df["best_percentile"].median()) if n_seeds_feasible else float("nan")

    summary = dict(
        import_cap_frac=import_cap_frac,
        is_baseline=is_baseline,
        K=instance["K"],
        N=instance["N"],
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
        source=("canonical_5k (frozen, reused)" if is_baseline else "recomputed_this_appendix"),
    )
    (cap_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"[cap={import_cap_frac}] SUMMARY: {json.dumps(summary, default=str)}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Energy Hub import-cap sensitivity appendix")
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    parser.add_argument("--cap-values", type=float, nargs="+", default=SWEEP_VALUES)
    args = parser.parse_args(argv)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    solver = build_solver()
    if not solver.available(False):
        print("ERROR: ipopt solver not available on PATH.", file=sys.stderr)
        sys.exit(1)

    started_at = utc_now()
    all_summaries = [run_cap(c, output, solver) for c in args.cap_values]

    summary_df = pd.DataFrame(all_summaries).sort_values("import_cap_frac")
    summary_df.to_csv(output / "import_cap_sweep_summary.csv", index=False)

    manifest = dict(
        protocol_version=PROTOCOL_VERSION,
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        cap_values=args.cap_values,
        baseline=BASELINE_IMPORT_CAP_FRAC,
        seeds=SEEDS,
        note=(
            "Supplementary appendix study. K=6, N=18, T=6 held fixed. "
            "S0/S1/S2 surrogate arms are not re-run at other import-cap "
            "values; only the exhaustive classical reference (TAC*) and "
            "the Uniform-random control at B=100 are recomputed per value. "
            "The 0.4 baseline reuses the frozen EH_CANONICAL_V1 exhaustive "
            "reference and canonical_5k Uniform-B100 result by read-only lookup."
        ),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")

    print("\n=== IMPORT-CAP SWEEP SUMMARY ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
