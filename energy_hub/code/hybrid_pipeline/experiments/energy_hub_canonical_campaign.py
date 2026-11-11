"""Energy Hub canonical campaign runner (EH_CANONICAL_V1).

Executes the 40 independent (arm, seed) units that make up the canonical
campaign: for each unit, samples/ranks/filters a support list exactly once,
then recovers V(y) for up to ``MAX_CONTINUOUS_EVALUATIONS_PER_RUN`` eligible
candidates via lookup against the audited exhaustive exact-K classical
reference. No IPOPT solve is issued by this script.

Imports only the core benchmark/sampler modules and this campaign's own frozen
configuration module.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from hybrid_pipeline.benchmarks.energy_hub import (
    build_energy_hub_instance,
    build_energy_hub_s0,
    build_energy_hub_s1,
    build_energy_hub_s2,
    structural_filter,
)
from hybrid_pipeline.benchmarks.energy_hub.surrogates import bits_to_selection, support_composition
from hybrid_pipeline.core.tabu import StructuralTabu
from hybrid_pipeline.samplers.qaoa import energy_hub_surrogate_factory
from hybrid_pipeline.experiments import energy_hub_canonical_config as cfg

SURROGATE_BUILDERS = {"S0": build_energy_hub_s0, "S1": build_energy_hub_s1, "S2": build_energy_hub_s2}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{uuid.uuid4().hex[:8]}")
    df.to_csv(tmp, index=False)
    _retry_fs_op(lambda: tmp.replace(path))


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def support_ids_from_bits(bitstring: str, instance: dict) -> str:
    return "|".join(a for a, ch in zip(instance["ASSETS"], bitstring) if ch == "1")


def composition_json(bitstring: str, instance: dict) -> str:
    selection = bits_to_selection(tuple(int(ch) for ch in bitstring), instance)
    return json.dumps(support_composition(selection, instance), sort_keys=True)


def sample_surrogate_arm(arm: str, base_seed: int, instance: dict) -> dict:
    arm_cfg = cfg.ARM_CONFIGS[arm]
    surrogate_type = arm_cfg["surrogate_type"]
    surrogate = SURROGATE_BUILDERS[surrogate_type](instance)
    if surrogate["coefficient_hash"] != cfg.EXPECTED_SURROGATE_COEFFICIENT_HASHES[surrogate_type]:
        raise RuntimeError(f"{arm} coefficient_hash drifted from the frozen protocol value.")

    qconf = {
        "qaoa_layers": arm_cfg["depth"],
        "qaoa_steps": arm_cfg["qaoa_steps"],
        "qaoa_shots": arm_cfg["shots_per_restart"],
    }
    counts: Counter = Counter()
    energies: dict = {}
    restart_seeds = []
    for restart in range(arm_cfg["restarts"]):
        training_seed = cfg.derive_seed(base_seed, arm, restart, "qaoa_train_and_sample")
        restart_seeds.append({"restart": restart, "training_seed": training_seed})
        sampler, params, factory_surrogate = energy_hub_surrogate_factory(training_seed, instance, qconf, surrogate_type)
        if factory_surrogate["coefficient_hash"] != surrogate["coefficient_hash"]:
            raise RuntimeError(f"{arm} factory-internal surrogate hash disagrees with the frozen builder.")
        pool, _smeta = sampler(
            params=params,
            shots=arm_cfg["shots_per_restart"],
            top_k=cfg.N_TOTAL_SUPPORTS,
            alpha=1.0,
            beta=0.05,
        )
        for cand in pool:
            if not cand["is_exactK"]:
                continue
            bits = "".join(str(int(b)) for b in cand["bits"])
            counts[bits] += int(cand["sample_count"])
            energies[bits] = float(cand["surrogate_energy"])

    total_shots = arm_cfg["restarts"] * arm_cfg["shots_per_restart"]
    ordered = [
        {
            "bitstring": bits,
            "sample_count": count,
            "empirical_probability": count / total_shots,
            "surrogate_energy": energies[bits],
        }
        for bits, count in counts.items()
    ]
    ordered.sort(key=lambda c: (-c["empirical_probability"], c["surrogate_energy"], c["bitstring"]))
    return {
        "ordered": ordered,
        "total_shots": total_shots,
        "restart_seeds": restart_seeds,
        "coefficient_hash": surrogate["coefficient_hash"],
        "raw_exactK_events": sum(counts.values()),
    }


UNIFORM_DRAWS = 5000
UNIFORM_POLICY_NAME = "uniform_random_seeded"


def sample_uniform_arm(base_seed: int, instance: dict) -> dict:
    sampling_seed = cfg.derive_seed(base_seed, "Uniform", 0, "uniform_draw")
    rng = np.random.default_rng(sampling_seed)
    n, k = instance["N"], instance["K"]

    seen: dict = {}
    order: list = []
    for _ in range(UNIFORM_DRAWS):
        idx = set(rng.choice(n, k, replace=False).tolist())
        bits = "".join("1" if i in idx else "0" for i in range(n))
        if bits not in seen:
            seen[bits] = 0
            order.append(bits)
        seen[bits] += 1

    ordered = [
        {
            "bitstring": bits,
            "sample_count": seen[bits],
            "empirical_probability": seen[bits] / UNIFORM_DRAWS,
            "surrogate_energy": float("nan"),
        }
        for bits in order
    ]
    return {
        "ordered": ordered,
        "total_shots": UNIFORM_DRAWS,
        "restart_seeds": [{"restart": 0, "training_seed": sampling_seed}],
        "coefficient_hash": "",
        "raw_exactK_events": UNIFORM_DRAWS,
    }


def build_lookup(exhaustive_df: pd.DataFrame) -> dict:
    lookup = {}
    for row in exhaustive_df.itertuples(index=False):
        lookup[row.support_bitstring] = {
            "method": row.method,
            "feasible": bool(row.feasible) if pd.notna(row.feasible) else False,
            "TAC": float(row.TAC) if pd.notna(row.TAC) else float("nan"),
            "active_count": float(row.active_count) if pd.notna(row.active_count) else float("nan"),
        }
    return lookup


def build_percentile_lookup(exhaustive_df: pd.DataFrame) -> dict:
    feasible = exhaustive_df[exhaustive_df["feasible"] == True].copy()
    feasible = feasible.sort_values("TAC").reset_index(drop=True)
    n = len(feasible)
    pct = {}
    for i, bitstring in enumerate(feasible["support_bitstring"]):
        pct[bitstring] = 100.0 * (i + 1) / n
    return pct


def run_arm_seed(
    arm: str,
    seed: int,
    instance: dict,
    lookup: dict,
    percentile_lookup: dict,
    reference_hash: str,
) -> dict:
    t0 = time.time()
    if arm == "Uniform":
        sampling = sample_uniform_arm(seed, instance)
        selection_policy = UNIFORM_POLICY_NAME
    else:
        sampling = sample_surrogate_arm(arm, seed, instance)
        selection_policy = cfg.ARM_CONFIGS[arm]["selection_policy"]

    tabu = StructuralTabu()
    unique_rows = []
    eligible = []
    for rank, cand in enumerate(sampling["ordered"], start=1):
        bitstring = cand["bitstring"]
        selection = bits_to_selection(tuple(int(ch) for ch in bitstring), instance)
        is_tabu, tabu_reason, _rule = tabu.classify(selection, bitstring, instance)
        if is_tabu:
            structural_pass = False
            status = "excluded"
            exclusion_reason = f"tabu:{tabu_reason}"
        else:
            structural_pass = bool(structural_filter(selection, instance))
            if structural_pass:
                status = "eligible"
                exclusion_reason = ""
            else:
                status = "excluded"
                exclusion_reason = "structural_reject"
        unique_rows.append(
            {
                "arm": arm,
                "seed": seed,
                "support_bitstring": bitstring,
                "support_ids": support_ids_from_bits(bitstring, instance),
                "support_hash": cfg.support_hash(bitstring),
                "sample_count": cand["sample_count"],
                "empirical_probability": cand["empirical_probability"],
                "surrogate_energy": cand["surrogate_energy"],
                "frequency_rank": rank,
                "structural_filter_pass": structural_pass,
                "selection_status": status,
                "exclusion_reason": exclusion_reason,
            }
        )
        if structural_pass and not is_tabu:
            eligible.append((bitstring, cand))

    available_budget = len(eligible)
    budget = min(cfg.MAX_CONTINUOUS_EVALUATIONS_PER_RUN, available_budget)
    selected = eligible[: cfg.MAX_CONTINUOUS_EVALUATIONS_PER_RUN]
    candidate_shortage = available_budget < cfg.MAX_CONTINUOUS_EVALUATIONS_PER_RUN

    continuous_rows = []
    n_feasible = 0
    n_infeasible = 0
    for eval_rank, (bitstring, cand) in enumerate(selected, start=1):
        tabu.register_visit(bitstring)
        ref = lookup.get(bitstring)
        lookup_status = "found" if ref is not None else "missing"
        if ref is None:
            raise RuntimeError(
                f"Support {bitstring} sampled for arm={arm} seed={seed} is not present in the "
                "exhaustive exact-K reference -- the reference is incomplete for this instance."
            )
        nlp_feasible = bool(ref["feasible"])
        tac = ref["TAC"] if nlp_feasible else float("nan")
        if nlp_feasible:
            n_feasible += 1
        else:
            n_infeasible += 1
        continuous_rows.append(
            {
                "arm": arm,
                "seed": seed,
                "evaluation_rank": eval_rank,
                "support_bitstring": bitstring,
                "support_ids": support_ids_from_bits(bitstring, instance),
                "support_hash": cfg.support_hash(bitstring),
                "sampling_frequency": cand["empirical_probability"],
                "surrogate_energy": cand["surrogate_energy"],
                "structural_status": "eligible",
                "lookup_status": lookup_status,
                "nlp_feasible": nlp_feasible,
                "TAC": tac,
                "realized_cardinality": ref["active_count"],
                "exhaustive_percentile": percentile_lookup.get(bitstring, float("nan")) if nlp_feasible else float("nan"),
                "source_reference_hash": reference_hash,
            }
        )

    n_struct_reject = sum(1 for r in unique_rows if r["exclusion_reason"] == "structural_reject")
    flow = {
        "arm": arm,
        "seed": seed,
        "selection_policy": selection_policy,
        "total_shots": sampling["total_shots"],
        "unique_exactK_supports": len(sampling["ordered"]),
        "raw_exactK_events": sampling["raw_exactK_events"],
        "duplicates_skipped": sampling["raw_exactK_events"] - len(sampling["ordered"]),
        "structural_rejects": n_struct_reject,
        "eligible_supports": available_budget,
        "attempted_continuous_evaluations": len(selected),
        "feasible_continuous_evaluations": n_feasible,
        "infeasible_continuous_evaluations": n_infeasible,
        "budget_exhausted_flag": candidate_shortage,
        "candidate_shortage_flag": candidate_shortage,
        "available_budget": available_budget,
        "runtime_s": time.time() - t0,
    }

    best_tac = min((r["TAC"] for r in continuous_rows if r["nlp_feasible"]), default=float("nan"))
    return {
        "arm": arm,
        "seed": seed,
        "unique_rows": unique_rows,
        "continuous_rows": continuous_rows,
        "flow": flow,
        "restart_seeds": sampling["restart_seeds"],
        "coefficient_hash": sampling["coefficient_hash"],
        "best_tac": best_tac,
        "runtime_s": time.time() - t0,
    }


def run_marker_path(run_dir: Path) -> Path:
    return run_dir / "run_complete.json"


def write_run_atomically(run_dir: Path, result: dict, config_hash: str) -> None:
    tmp_dir = run_dir.parent / f".tmp_{run_dir.name}_{uuid.uuid4().hex[:8]}"
    tmp_dir.mkdir(parents=True, exist_ok=False)
    try:
        atomic_write_csv(pd.DataFrame(result["unique_rows"]), tmp_dir / "unique_supports.csv")
        atomic_write_csv(pd.DataFrame(result["continuous_rows"]), tmp_dir / "continuous_evaluations.csv")
        atomic_write_csv(pd.DataFrame([result["flow"]]), tmp_dir / "candidate_flow.csv")
        marker = {
            "arm": result["arm"],
            "seed": result["seed"],
            "config_hash": config_hash,
            "protocol_version": cfg.CANONICAL_PROTOCOL_VERSION,
            "best_tac": result["best_tac"],
            "n_unique": len(result["unique_rows"]),
            "n_continuous": len(result["continuous_rows"]),
            "restart_seeds": result["restart_seeds"],
            "coefficient_hash": result["coefficient_hash"],
            "runtime_s": result["runtime_s"],
            "completed_at_utc": utc_now(),
        }
        run_marker_path(tmp_dir).write_text(json.dumps(marker, indent=2, sort_keys=True), encoding="utf-8")
        n_dup = pd.DataFrame(result["unique_rows"])["support_bitstring"].duplicated().sum() if result["unique_rows"] else 0
        if n_dup:
            raise RuntimeError(f"{result['arm']}/{result['seed']}: {n_dup} duplicate supports in unique_supports.csv")
        if len(result["continuous_rows"]) > cfg.MAX_CONTINUOUS_EVALUATIONS_PER_RUN:
            raise RuntimeError(f"{result['arm']}/{result['seed']}: exceeded max continuous evaluations budget")
        if run_dir.exists():
            _retry_fs_op(lambda: shutil.rmtree(run_dir))
        _retry_fs_op(lambda: tmp_dir.replace(run_dir))
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def _retry_fs_op(op, attempts: int = 8, delay_s: float = 0.5) -> None:
    """Retry a filesystem operation with backoff.

    On Windows, file-synchronization or antivirus services can transiently hold a
    handle on just-written files, which turns an atomic rename into an intermittent
    PermissionError (WinError 32). Each (arm, seed) unit touches only its own
    temporary/run directory, so a short retry with backoff is sufficient.
    """
    last_exc = None
    for attempt in range(attempts):
        try:
            op()
            return
        except (PermissionError, OSError) as exc:
            last_exc = exc
            time.sleep(delay_s * (attempt + 1))
    raise last_exc


def run_is_complete(run_dir: Path, config_hash: str) -> bool:
    marker = run_marker_path(run_dir)
    if not marker.exists():
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if data.get("protocol_version") != cfg.CANONICAL_PROTOCOL_VERSION:
        raise RuntimeError(f"{run_dir}: protocol_version mismatch; refusing to mix versions under --resume.")
    if data.get("config_hash") != config_hash:
        raise RuntimeError(f"{run_dir}: config_hash mismatch; refusing to mix configurations under --resume.")
    return True


def aggregate_campaign(output: Path, arms: list[str], seeds: list[int]) -> dict:
    all_unique, all_continuous, all_flow, registry = [], [], [], []
    for arm in arms:
        for seed in seeds:
            run_dir = output / "runs" / f"{arm}_{seed}"
            marker = json.loads(run_marker_path(run_dir).read_text(encoding="utf-8"))
            unique_df = pd.read_csv(run_dir / "unique_supports.csv", dtype={"support_bitstring": str, "support_hash": str})
            cont_df = pd.read_csv(run_dir / "continuous_evaluations.csv", dtype={"support_bitstring": str, "support_hash": str})
            flow_df = pd.read_csv(run_dir / "candidate_flow.csv")
            all_unique.append(unique_df)
            all_continuous.append(cont_df)
            all_flow.append(flow_df)
            registry.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "status": "completed",
                    "best_tac": marker["best_tac"],
                    "n_unique": marker["n_unique"],
                    "n_continuous": marker["n_continuous"],
                    "coefficient_hash": marker["coefficient_hash"],
                    "runtime_s": marker["runtime_s"],
                    "completed_at_utc": marker["completed_at_utc"],
                }
            )
    unique_all = pd.concat(all_unique, ignore_index=True) if all_unique else pd.DataFrame()
    continuous_all = pd.concat(all_continuous, ignore_index=True) if all_continuous else pd.DataFrame()
    flow_all = pd.concat(all_flow, ignore_index=True) if all_flow else pd.DataFrame()
    registry_df = pd.DataFrame(registry)

    atomic_write_csv(unique_all, output / "unique_supports.csv")
    atomic_write_csv(continuous_all, output / "continuous_evaluations.csv")
    atomic_write_csv(flow_all, output / "candidate_flow.csv")
    atomic_write_csv(registry_df, output / "run_registry.csv")

    occ_cols = [
        "arm",
        "seed",
        "support_bitstring",
        "support_hash",
        "sample_count",
        "empirical_probability",
        "surrogate_energy",
    ]
    occurrences = unique_all[occ_cols].copy() if not unique_all.empty else pd.DataFrame(columns=occ_cols)
    occurrences.insert(0, "protocol_version", cfg.CANONICAL_PROTOCOL_VERSION)
    atomic_write_csv(occurrences, output / "sampling_occurrences.csv")

    return {"n_runs": len(registry)}


def write_seed_arm_and_arm_summary(output: Path, exhaustive_df: pd.DataFrame) -> None:
    continuous = pd.read_csv(output / "continuous_evaluations.csv", dtype={"support_bitstring": str})
    unique_df = pd.read_csv(output / "unique_supports.csv", dtype={"support_bitstring": str})
    global_best = float(exhaustive_df.loc[exhaustive_df["feasible"] == True, "TAC"].min())

    seed_rows = []
    for (arm, seed), grp in continuous.groupby(["arm", "seed"]):
        feas = grp[grp["nlp_feasible"] == True]
        u = unique_df[(unique_df["arm"] == arm) & (unique_df["seed"] == seed)]
        probs = u["empirical_probability"].to_numpy(dtype=float)
        probs = probs[probs > 0]
        entropy = float(-(probs * np.log(probs)).sum()) if len(probs) else 0.0
        entropy_uniform = float(np.log(cfg.N_TOTAL_SUPPORTS))
        best_tac = float(feas["TAC"].min()) if not feas.empty else float("nan")
        best_support = feas.loc[feas["TAC"].idxmin(), "support_bitstring"] if not feas.empty else ""
        seed_rows.append(
            {
                "arm": arm,
                "seed": seed,
                "best_TAC": best_tac,
                "best_support": best_support,
                "median_TAC": float(feas["TAC"].median()) if not feas.empty else float("nan"),
                "mean_TAC": float(feas["TAC"].mean()) if not feas.empty else float("nan"),
                "iqr_TAC": float(feas["TAC"].quantile(0.75) - feas["TAC"].quantile(0.25)) if len(feas) > 1 else float("nan"),
                "n_attempted": int(len(grp)),
                "n_feasible": int(len(feas)),
                "best_gap_abs": (best_tac - global_best) if pd.notna(best_tac) else float("nan"),
                "best_gap_pct": (100.0 * (best_tac - global_best) / global_best) if pd.notna(best_tac) else float("nan"),
                "best_percentile": float(feas["exhaustive_percentile"].min()) if not feas.empty else float("nan"),
                "unique_support_count": int(len(u)),
                "entropy": entropy,
                "entropy_ratio": entropy / entropy_uniform if entropy_uniform > 0 else 0.0,
                "top1_frequency": float(u["empirical_probability"].max()) if not u.empty else 0.0,
            }
        )
    seed_arm_df = pd.DataFrame(seed_rows)
    atomic_write_csv(seed_arm_df, output / "seed_arm_summary.csv")

    arm_rows = []
    for arm, grp in seed_arm_df.groupby("arm"):
        n_feasible_seeds = int(grp["best_TAC"].notna().sum())
        arm_rows.append(
            {
                "arm": arm,
                "n_seeds": int(len(grp)),
                "median_best_TAC": float(grp["best_TAC"].median()),
                "mean_best_TAC": float(grp["best_TAC"].mean()),
                "std_best_TAC": float(grp["best_TAC"].std(ddof=0)),
                "iqr_best_TAC": float(grp["best_TAC"].quantile(0.75) - grp["best_TAC"].quantile(0.25)),
                "min_best_TAC": float(grp["best_TAC"].min()),
                "max_best_TAC": float(grp["best_TAC"].max()),
                "median_gap_pct": float(grp["best_gap_pct"].median()),
                "median_percentile": float(grp["best_percentile"].median()),
                "n_seeds_with_feasible_solution": n_feasible_seeds,
            }
        )
    atomic_write_csv(pd.DataFrame(arm_rows), output / "arm_summary.csv")


def git_info() -> dict:
    def run(cmd):
        try:
            return subprocess.check_output(cmd, cwd=str(Path.cwd()), text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""

    commit = run(["git", "rev-parse", "HEAD"])
    status = run(["git", "status", "--porcelain"])
    dirty = bool(status)
    diff_hash = ""
    if dirty:
        import hashlib

        diff = run(["git", "diff", "HEAD"])
        diff_hash = hashlib.sha256(diff.encode("utf-8", errors="ignore")).hexdigest()
    return {"commit": commit, "dirty": dirty, "diff_hash": diff_hash}


def package_versions() -> dict:
    versions = {}
    for pkg in ["numpy", "pandas", "pennylane", "pyomo", "scipy"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[pkg] = "not_installed"
    return versions


def write_environment(output: Path) -> None:
    env = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions(),
    }
    (output / "environment.json").write_text(json.dumps(env, indent=2, sort_keys=True), encoding="utf-8")


def write_source_hashes(output: Path) -> None:
    this_dir = Path(__file__).parent
    files = [
        this_dir / "energy_hub_canonical_config.py",
        this_dir / "energy_hub_canonical_campaign.py",
        this_dir / "energy_hub_canonical_analysis.py",
    ]
    hashes = {str(f.name): sha256_file(f) for f in files if f.exists()}
    (output / "source_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True), encoding="utf-8")


def append_audit_log(output: Path, event: dict) -> None:
    event = dict(event, timestamp_utc=utc_now())
    with open(output / "audit_log.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True, default=str) + "\n")


def write_readme(output: Path) -> None:
    text = (
        "# Energy Hub canonical campaign output (EH_CANONICAL_V1)\n\n"
        "Generated exclusively by `energy_hub_canonical_campaign.py` and "
        "`energy_hub_canonical_analysis.py`. See "
        "`energy_hub/README.md` in the release for the protocol.\n\n"
        "Do not hand-edit any file in this directory; regenerate via the canonical scripts.\n"
    )
    (output / "README.md").write_text(text, encoding="utf-8")


def run_campaign(output: Path, resume: bool, arms: list[str] | None = None, seeds: list[int] | None = None) -> dict:
    arms = arms if arms is not None else cfg.ARMS
    seeds = seeds if seeds is not None else cfg.SEEDS
    output.mkdir(parents=True, exist_ok=True)
    (output / "runs").mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    instance = build_energy_hub_instance()
    the_config_hash = cfg.config_hash()
    the_instance_hash = cfg.instance_hash(instance)

    (output / "canonical_config.json").write_text(
        json.dumps({**cfg.canonical_config_payload(), "config_hash": the_config_hash, "instance_hash": the_instance_hash}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    append_audit_log(output, {"event": "campaign_start", "config_hash": the_config_hash, "instance_hash": the_instance_hash})

    exhaustive_df = cfg.load_and_verify_exhaustive_reference(instance)
    reference_hash = cfg.exhaustive_reference_hash(exhaustive_df)
    lookup = build_lookup(exhaustive_df)
    percentile_lookup = build_percentile_lookup(exhaustive_df)

    (output / "exhaustive_reference_summary.json").write_text(
        json.dumps(
            {
                "path": str(cfg.EXHAUSTIVE_REFERENCE_PATH),
                "reference_hash": reference_hash,
                "n_total": int(len(exhaustive_df)),
                "n_structural_rejected": int((exhaustive_df["method"] == "structural_filter").sum()),
                "n_ipopt_attempted": int((exhaustive_df["method"] == "ipopt").sum()),
                "n_ipopt_feasible": int(((exhaustive_df["method"] == "ipopt") & (exhaustive_df["feasible"] == True)).sum()),
                "best_TAC": float(exhaustive_df.loc[exhaustive_df["feasible"] == True, "TAC"].min()),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    expected_runs = len(arms) * len(seeds)
    completed = 0
    reused = 0
    for arm in arms:
        for seed in seeds:
            run_dir = output / "runs" / f"{arm}_{seed}"
            if resume and run_is_complete(run_dir, the_config_hash):
                reused += 1
                append_audit_log(output, {"event": "run_reused", "arm": arm, "seed": seed})
                continue
            append_audit_log(output, {"event": "run_start", "arm": arm, "seed": seed})
            result = run_arm_seed(arm, seed, instance, lookup, percentile_lookup, reference_hash)
            write_run_atomically(run_dir, result, the_config_hash)
            completed += 1
            append_audit_log(
                output,
                {
                    "event": "run_complete",
                    "arm": arm,
                    "seed": seed,
                    "best_tac": result["best_tac"],
                    "runtime_s": result["runtime_s"],
                },
            )
            print(
                f"[{arm}/{seed}] eligible={result['flow']['eligible_supports']} "
                f"attempted={result['flow']['attempted_continuous_evaluations']} "
                f"feasible={result['flow']['feasible_continuous_evaluations']} "
                f"best_TAC={result['best_tac']:.2f} runtime={result['runtime_s']:.1f}s",
                flush=True,
            )

    aggregate_campaign(output, arms, seeds)
    write_seed_arm_and_arm_summary(output, exhaustive_df)
    write_environment(output)
    write_source_hashes(output)
    write_readme(output)

    n_run_dirs = sum(1 for _ in (output / "runs").iterdir())
    invariants = check_invariants(output, expected_runs)

    manifest = {
        "protocol_version": cfg.CANONICAL_PROTOCOL_VERSION,
        "campaign_name": cfg.CAMPAIGN_NAME,
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "git": git_info(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "config_hash": the_config_hash,
        "instance_hash": the_instance_hash,
        "exhaustive_reference_hash": reference_hash,
        "command_line": " ".join(sys.argv),
        "working_directory": str(Path.cwd()),
        "expected_runs": expected_runs,
        "completed_runs_this_invocation": completed,
        "reused_runs_this_invocation": reused,
        "total_run_dirs": n_run_dirs,
        "invariants": invariants,
        "final_status": "PASS" if invariants["all_passed"] else "FAIL",
    }
    output_file_hashes = {}
    for name in [
        "unique_supports.csv",
        "continuous_evaluations.csv",
        "candidate_flow.csv",
        "sampling_occurrences.csv",
        "run_registry.csv",
        "seed_arm_summary.csv",
        "arm_summary.csv",
    ]:
        p = output / name
        if p.exists():
            output_file_hashes[name] = sha256_file(p)
    manifest["output_file_hashes"] = output_file_hashes
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    append_audit_log(output, {"event": "campaign_end", "final_status": manifest["final_status"]})
    return manifest


def check_invariants(output: Path, expected_runs: int) -> dict:
    checks = {}
    unique_df = pd.read_csv(output / "unique_supports.csv", dtype={"support_bitstring": str})
    continuous_df = pd.read_csv(output / "continuous_evaluations.csv", dtype={"support_bitstring": str})
    registry_df = pd.read_csv(output / "run_registry.csv")

    checks["all_cardinality_K"] = bool(unique_df["support_bitstring"].map(lambda b: b.count("1") == cfg.K).all()) if not unique_df.empty else True
    checks["no_duplicate_arm_seed_support"] = bool(
        not unique_df.duplicated(subset=["arm", "seed", "support_bitstring"]).any() if not unique_df.empty else True
    )
    checks["max_100_attempted_per_run"] = bool(
        (continuous_df.groupby(["arm", "seed"]).size() <= cfg.MAX_CONTINUOUS_EVALUATIONS_PER_RUN).all() if not continuous_df.empty else True
    )
    if not continuous_df.empty and not unique_df.empty:
        merged = continuous_df.merge(
            unique_df[["arm", "seed", "support_bitstring"]].drop_duplicates(),
            on=["arm", "seed", "support_bitstring"],
            how="left",
            indicator=True,
        )
        checks["every_evaluation_was_sampled"] = bool((merged["_merge"] == "both").all())
    else:
        checks["every_evaluation_was_sampled"] = True
    checks["expected_total_runs"] = bool(len(registry_df) == expected_runs)

    if not continuous_df.empty:
        feas = continuous_df[continuous_df["nlp_feasible"] == True]
        tac_by_support = feas.groupby("support_bitstring")["TAC"].agg(lambda s: s.max() - s.min())
        checks["same_support_same_TAC"] = bool((tac_by_support.fillna(0.0) <= cfg.TAC_TIE_REL_TOL * 10).all())
    else:
        checks["same_support_same_TAC"] = True

    checks["all_passed"] = all(bool(v) for v in checks.values())
    return checks


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Energy Hub canonical campaign runner (EH_CANONICAL_V1)")
    parser.add_argument("--output-dir", default=str(cfg.OUTPUT_ROOT))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="Run a single reduced (S0, seed=100) validation unit outside the canonical output tree, then exit.")
    return parser.parse_args(argv)


def run_validation_only() -> dict:
    """Reduced infrastructure smoke check: arm=S0, seed=100. Never written
    into the canonical output tree (section 16)."""
    instance = build_energy_hub_instance()
    exhaustive_df = cfg.load_and_verify_exhaustive_reference(instance)
    reference_hash = cfg.exhaustive_reference_hash(exhaustive_df)
    lookup = build_lookup(exhaustive_df)
    percentile_lookup = build_percentile_lookup(exhaustive_df)
    result = run_arm_seed("S0", cfg.VALIDATION_SEED, instance, lookup, percentile_lookup, reference_hash)
    print(
        f"[VALIDATION S0/{cfg.VALIDATION_SEED}] eligible={result['flow']['eligible_supports']} "
        f"attempted={result['flow']['attempted_continuous_evaluations']} "
        f"feasible={result['flow']['feasible_continuous_evaluations']} best_TAC={result['best_tac']:.2f}"
    )
    return result


def main(argv=None):
    args = parse_args(argv)
    if args.validate_only:
        run_validation_only()
        print("VALIDATION OK -- infrastructure check passed, no output written to the canonical tree.")
        return
    manifest = run_campaign(Path(args.output_dir), args.resume)
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
