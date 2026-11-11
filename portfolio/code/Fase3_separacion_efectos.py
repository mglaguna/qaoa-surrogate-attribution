"""Experiments for effect separation in the hybrid portfolio pipeline (Phase 3).

Five targeted experiments; each addresses a different confound of the Phase 1-2
pipeline. They are independent and can be combined.

Confounds
---------
C1 - Hsurr and shybrid share the same normalized signals: bias and ranking are coupled.
C2 - Circuit depth p = 1: only 2 parameters, the distribution collapses on one support.
C3 - Asymmetric budget: QAOA uses far fewer IPOPT calls than uniform sampling.
C4 - No policy isolates the pure QAOA distributional bias (ranked_frequency).
C5 - The QAOA distribution is retrained per seed: zero spread reflects the
     collapse of training, not statistical robustness.

Experiments
-----------
EXP-1 Common capped IPOPT budget (C3)
EXP-2 Return-only Hamiltonian (C1)
EXP-3 Distribution injection (C5)
EXP-4 ranked_frequency as fourth policy (C4)
EXP-5 Circuit depth ablation p in {1, 2, 3} (C2)

Usage: python Fase3_separacion_efectos.py --exp N --seeds 10 --bmax 20
(baseline_enumeration.csv, written by Fase2.py, must be in the working directory).
"""

import importlib.util, sys, os, time, itertools
import numpy as np
import pandas as pd
import pennylane as qml
from scipy.stats import mannwhitneyu, spearmanr
from typing import Callable, Optional

_DIR = os.path.dirname(os.path.abspath(__file__))

def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_q  = _load(os.path.join(_DIR, "Quantico-cardinalidad.py"), "quantico_cardinalidad")
_f2 = _load(os.path.join(_DIR, "Fase2.py"),                "fase2")

from quantico_cardinalidad import (
    INSTANCE_15, CONFIG, HAMILTONIAN_PENALTY, GLOBAL_SEED,
    build_support_surrogate_hamiltonian, make_circuits, train_qaoa,
    build_qaoa_sampler, build_uniform_random_sampler,
    build_candidate_pool, candidate_pool_metadata,
    set_global_seed, StructuralTabu, DiscardReason,
    diagnose_infeasibility, infer_and_register_rule,
    solve_subproblem, _to_bits, _selected_names,
)
from fase2 import (
    run_hybrid_v2, select_candidates, build_budget_pool,
    mu_score_support, summarize_run, build_convergence_summary,
    extract_convergence_trajectory,
)


def tag_results(df, exp_id, confound, exp_name):
    df = df.copy()
    df["experiment_id"] = exp_id
    df["confound"] = confound
    df["experiment_name"] = exp_name
    return df


def _normalize_experiment_df(df, default_sampler_type=None,
                             default_selection_policy=None) -> pd.DataFrame:
    df = df.copy()

    if "ipopt_used" not in df.columns and "ipopt_total" in df.columns:
        df["ipopt_used"] = df["ipopt_total"]

    if "sampler_type" not in df.columns:
        if "hamiltonian" in df.columns:
            df["sampler_type"] = df["hamiltonian"]
        elif default_sampler_type is not None:
            df["sampler_type"] = default_sampler_type
        else:
            df["sampler_type"] = "NA"

    if "selection_policy" not in df.columns:
        if default_selection_policy is not None:
            df["selection_policy"] = default_selection_policy
        else:
            df["selection_policy"] = "NA"

    if "seed" not in df.columns:
        df["seed"] = np.nan

    return df


def run_exp_1(instance, baseline_stats, config, seeds, bmax):
    print("\n[EXP-1 | C1] return_only_hamiltonian")
    df = exp2_hamiltonian_ablation(instance, baseline_stats, config, seeds=seeds)
    df = _normalize_experiment_df(df)
    return tag_results(df, "EXP-1", "C1", "return_only_hamiltonian")


def run_exp_2(instance, baseline_stats, config, seeds, bmax):
    print("\n[EXP-2 | C2] depth_variation")
    _, df = exp5_depth_ablation(instance, baseline_stats, config,
                                depths=[1, 2, 3], seeds=seeds[:5])
    df = _normalize_experiment_df(
        df,
        default_sampler_type="QAOA_surrogate",
        default_selection_policy="random_exactK",
    )
    return tag_results(df, "EXP-2", "C2", "depth_variation")


def run_exp_3(instance, baseline_stats, config, seeds, bmax):
    print("\n[EXP-3 | C3] capped_budget")
    df = exp1_budget_equalization(instance, baseline_stats, config,
                                  B_max=bmax, seeds=seeds)
    df = _normalize_experiment_df(df)
    return tag_results(df, "EXP-3", "C3", "capped_budget")


def run_exp_4(instance, baseline_stats, config, seeds, bmax):
    print("\n[EXP-4 | C4] frequency_policy")
    df = exp4_frequency_policy(instance, baseline_stats, config, seeds=seeds)
    df = _normalize_experiment_df(df)
    return tag_results(df, "EXP-4", "C4", "frequency_policy")


def run_exp_5(instance, baseline_stats, config, seeds, bmax):
    print("\n[EXP-5 | C5] frozen_distribution")
    df = exp3_distribution_injection(instance, baseline_stats, config, seeds=seeds)
    df = _normalize_experiment_df(df, default_selection_policy="random_exactK")
    return tag_results(df, "EXP-5", "C5", "frozen_distribution")


def run_hybrid_budgeted(
    sampler: Callable,
    instance: dict,
    params=None,
    max_ipopt_budget: int = 20,
    top_k_candidates: int = 40,
    shots: int = 5000,
    top_eval_per_iter: int = 2,
    score_alpha: float = 1.0,
    score_beta: float  = 0.05,
    baseline_stats: dict = None,
    label: str = "",
    selection_policy: str = "ranked_hybrid",
    max_iter: int = 200,
) -> pd.DataFrame:
    """Hybrid loop with a common capped IPOPT budget (B_max).

    Difference from run_hybrid_v2: the loop stops when ipopt_total >= max_ipopt_budget,
    not when the space of unique observed supports is exhausted, so QAOA and uniform
    sampling are compared under a common capped budget. If the sampler exhausts its
    unique supports before B_max the loop stops anyway and the effective number of
    calls is recorded.
    """
    ASSETS = instance['ASSETS']
    K      = instance['K']
    N      = instance['N']
    R      = instance['R']
    sampler_name = label or getattr(sampler, '__name__', 'sampler')

    tabu        = StructuralTabu()
    log_rows    = []
    best_var    = np.inf
    best_sel    = None
    ipopt_total = 0

    for it in range(max_iter):
        if ipopt_total >= max_ipopt_budget:
            print(f"  Budget B_max={max_ipopt_budget} reached at iteration {it}. Stop.")
            break

        candidate_pool, sampling_meta = sampler(
            params=params, shots=shots, top_k=top_k_candidates,
            alpha=score_alpha, beta=score_beta,
        )

        valid = []
        for cand in candidate_pool:
            bits = _to_bits(cand['selection'], instance)
            is_tab, reason, _ = tabu.classify(cand['selection'], bits, instance)
            if not is_tab:
                valid.append(cand)

        if not valid:
            print(f"  Space exhausted at iteration {it} (ipopt_total={ipopt_total}).")
            break

        print(f"  Muestreo: unique={sampling_meta['n_unique_raw']} | "
              f"exactK={sampling_meta['n_exactK_raw']} | "
              f"entropy_exactK={sampling_meta.get('entropy_exactK', 0.0):.4f} | "
              f"top1_exactK={sampling_meta.get('top1_exactK_freq', 0.0):.4f}")

        budget_pool = build_budget_pool(
            valid, instance, selection_policy, top_k_candidates, rng=np.random)
        if not budget_pool:
            break

        remaining = max_ipopt_budget - ipopt_total
        to_evaluate = select_candidates(
            budget_pool, instance, selection_policy,
            min(top_eval_per_iter, remaining), rng=np.random)
        if not to_evaluate:
            break

        for chosen in to_evaluate:
            if ipopt_total >= max_ipopt_budget:
                break
            bits  = _to_bits(chosen['selection'], instance)
            names = _selected_names(chosen['selection'], instance)
            tabu.register_visit(bits)

            diag = diagnose_infeasibility(chosen['selection'], instance)
            if diag != 'unknown':
                infer_and_register_rule(tabu, diag, instance)
                log_rows.append({
                    'iter': it, 'ipopt_total': ipopt_total,
                    'bits': bits, 'variance': np.nan, 'best_cumul': best_var,
                    'sampler': sampler_name, 'policy': selection_policy,
                    'estado': 'infeasible_prescreen',
                })
                continue

            ipopt_total += 1
            result = solve_subproblem(chosen['selection'], instance)
            if result['feasible']:
                var = result['variance']
                if var < best_var - 1e-8:
                    best_var = var
                    best_sel = chosen['selection'].copy()
                log_rows.append({
                    'iter': it, 'ipopt_total': ipopt_total,
                    'bits': bits, 'variance': var, 'best_cumul': best_var,
                    'sampler': sampler_name, 'policy': selection_policy,
                    'estado': 'feasible',
                })
            else:
                log_rows.append({
                    'iter': it, 'ipopt_total': ipopt_total,
                    'bits': bits, 'variance': np.nan, 'best_cumul': best_var,
                    'sampler': sampler_name, 'policy': selection_policy,
                    'estado': 'infeasible_ipopt',
                })

    gap = (best_var - baseline_stats['best_var']) / baseline_stats['best_var'] * 100 \
          if baseline_stats and best_var < np.inf else np.nan
    print(f"  [{sampler_name}|{selection_policy}] best={best_var:.6f} gap={gap:.3f}% "
          f"ipopt={ipopt_total}/{max_ipopt_budget}")
    return pd.DataFrame(log_rows)


def exp1_budget_equalization(
    instance, baseline_stats, config,
    B_max: int = 20, seeds: list = None,
):
    """EXP-1: compare QAOA and uniform sampling under a common capped IPOPT budget B_max.

    H0: under the common budget, QAOA and uniform are indistinguishable (Mann-Whitney p > 0.05).
    H1: QAOA is better under the reduced budget.

    Output: comparative_matrix_budgeted.csv
    """
    if seeds is None:
        seeds = config.get('multiseed_seeds', list(range(10)))

    policies  = ['ranked_hybrid', 'ranked_mu', 'random_exactK']
    samplers  = ['QAOA_surrogate', 'uniform_random']
    rows = []

    for sampler_type in samplers:
        for policy in policies:
            for seed in seeds:
                set_global_seed(seed)
                if sampler_type == 'QAOA_surrogate':
                    H = build_support_surrogate_hamiltonian(
                        instance['N'], instance['K'],
                        penalty=HAMILTONIAN_PENALTY, instance=instance)
                    c_exp, c_sample = make_circuits(
                        instance['N'], config['qaoa_layers'], H,
                        k=instance['K'], shots=config['qaoa_shots'], pl_seed=seed)
                    p_trained = train_qaoa(c_exp, p=config['qaoa_layers'],
                                           n_steps=config['qaoa_steps'], verbose=False)
                    sampler = build_qaoa_sampler(c_sample, instance)
                    params  = p_trained
                else:
                    sampler = build_uniform_random_sampler(instance)
                    params  = None

                df_run = run_hybrid_budgeted(
                    sampler=sampler, instance=instance, params=params,
                    max_ipopt_budget=B_max,
                    top_k_candidates=config['top_k_candidates'],
                    shots=config['qaoa_shots'],
                    top_eval_per_iter=config['top_eval_per_iter'],
                    score_alpha=config['score_alpha'],
                    score_beta=config['score_beta'],
                    baseline_stats=baseline_stats,
                    label=sampler_type,
                    selection_policy=policy,
                )
                feasible_run = (df_run[df_run['estado'] == 'feasible']
                                if (not df_run.empty and 'estado' in df_run.columns)
                                else pd.DataFrame())
                best_v = (float(feasible_run['variance'].min())
                          if not feasible_run.empty else np.inf)
                gap    = ((best_v - baseline_stats['best_var']) / baseline_stats['best_var'] * 100
                          if np.isfinite(best_v) else np.nan)
                ipopt_used = (int(df_run['ipopt_total'].max())
                              if (not df_run.empty and 'ipopt_total' in df_run.columns)
                              else 0)
                rows.append(dict(
                    sampler_type=sampler_type, selection_policy=policy,
                    seed=seed, best_var=best_v, gap_pct=gap,
                    ipopt_used=ipopt_used,
                ))

    df = pd.DataFrame(rows)
    df.to_csv("comparative_matrix_budgeted.csv", index=False)

    print("\n=== EXP-1: Mann-Whitney QAOA vs Uniforme bajo common capped IPOPT budget B_max={} ===".format(B_max))
    for policy in policies:
        qa = df.query(f'sampler_type=="QAOA_surrogate" and selection_policy=="{policy}"')['best_var']
        un = df.query(f'sampler_type=="uniform_random"  and selection_policy=="{policy}"')['best_var']
        if len(qa) >= 2 and len(un) >= 2:
            stat, p = mannwhitneyu(qa, un, alternative='two-sided')
            print(f"  {policy}: QAOA mean={qa.mean():.4f} | Unif mean={un.mean():.4f} | p={p:.4f}")

    return df


def build_return_only_hamiltonian(n: int, k: int, penalty: float,
                                   instance: dict) -> qml.Hamiltonian:
    """Hamiltonian with the cardinality and return terms ONLY (lambda_risk = 0).

    Tests whether the exploration bias introduced by the covariance term of the
    surrogate Hamiltonian differs from that of a Hamiltonian that uses only the
    expected return.
    """
    return build_support_surrogate_hamiltonian(
        n=n, k=k, penalty=penalty, instance=instance,
        lambda_ret=1.0,
        lambda_risk=0.0,
    )


def exp2_hamiltonian_ablation(
    instance, baseline_stats, config,
    seeds: list = None,
):
    """EXP-2: compare Hsurr_full and Hsurr_return_only, crossed with two policies.

    2 x 2 design:
      H_full    x ranked_hybrid   (original Phase 1 configuration)
      H_full    x random_exactK
      H_retonly x ranked_hybrid
      H_retonly x random_exactK

    If H_full ~ H_retonly under random_exactK, the covariance term does not change
    which supports are visited. If H_full > H_retonly under ranked_hybrid, the
    covariance alignment between Hsurr and shybrid affects the final quality.

    Output: ablation_hamiltonian.csv
    """
    if seeds is None:
        seeds = config.get('multiseed_seeds', list(range(10)))

    hamiltonians = {
        'H_full':     lambda: build_support_surrogate_hamiltonian(
            instance['N'], instance['K'],
            penalty=HAMILTONIAN_PENALTY, instance=instance,
            lambda_ret=1.0, lambda_risk=1.0),
        'H_retonly':  lambda: build_return_only_hamiltonian(
            instance['N'], instance['K'],
            penalty=HAMILTONIAN_PENALTY, instance=instance),
    }
    policies = ['ranked_hybrid', 'random_exactK']
    rows = []

    for h_name, h_factory in hamiltonians.items():
        for policy in policies:
            for seed in seeds:
                set_global_seed(seed)
                H = h_factory()
                c_exp, c_sample = make_circuits(
                    instance['N'], config['qaoa_layers'], H,
                    k=instance['K'], shots=config['qaoa_shots'], pl_seed=seed)
                p_trained = train_qaoa(c_exp, p=config['qaoa_layers'],
                                       n_steps=config['qaoa_steps'], verbose=False)
                sampler   = build_qaoa_sampler(c_sample, instance)

                df_run = run_hybrid_v2(
                    sampler=sampler, instance=instance, params=p_trained,
                    max_iter=config['max_iter'],
                    top_k_candidates=config['top_k_candidates'],
                    shots=config['qaoa_shots'],
                    top_eval_per_iter=config['top_eval_per_iter'],
                    score_alpha=config['score_alpha'], score_beta=config['score_beta'],
                    baseline_stats=baseline_stats,
                    label=h_name, selection_policy=policy,
                )
                m = summarize_run(df_run, baseline_stats)
                m.update(dict(hamiltonian=h_name, selection_policy=policy, seed=seed))
                rows.append(m)
                print(f"  {h_name}|{policy}|seed={seed}: "
                      f"best={m['best_var']:.4f} gap={m['gap_pct']:.3f}%")

    df = pd.DataFrame(rows)
    df.to_csv("ablation_hamiltonian.csv", index=False)

    print("\n=== EXP-2: H_full vs H_retonly ===")
    print(df.groupby(['hamiltonian','selection_policy'])
          [['best_var','gap_pct','ipopt_total']].agg(['mean','std']).to_string())
    return df


def build_frozen_distribution_sampler(
    frozen_counts: dict,
    instance: dict,
) -> Callable:
    """Sampler that draws from a fixed (frozen) empirical distribution.

    frozen_counts: frequency table of a trained QAOA run. Every call produces a new
    sample of `shots` observations by multinomial resampling of the frozen
    distribution. This keeps the distributional bias of the trained QAOA while
    allowing shot-to-shot variability, separating the effect of P_QAOA from the
    effect of the training process.
    """
    total = sum(frozen_counts.values())
    keys  = list(frozen_counts.keys())
    probs = np.array([frozen_counts[k] / total for k in keys])

    def frozen_sampler(params=None, shots: int = 5000, top_k: int = 40,
                       alpha: float = 1.0, beta: float = 0.05) -> tuple:
        counts_new = {}
        sampled = np.random.multinomial(shots, probs)
        for k, c in zip(keys, sampled):
            if c > 0:
                counts_new[k] = int(c)

        full_pool = build_candidate_pool(counts_new, shots, instance, alpha, beta)
        meta      = candidate_pool_metadata(full_pool, shots)
        return full_pool, meta

    frozen_sampler.__name__ = "frozen_qaoa_sampler"
    return frozen_sampler


def exp3_distribution_injection(
    instance, baseline_stats, config,
    seeds: list = None,
):
    """EXP-3: frozen distribution vs. live QAOA vs. uniform sampling.

      1. Train QAOA with seed 0 and record the empirical distribution P_0.
      2. Build frozen_sampler, which resamples P_0 on every call.
      3. Run 10 seeds with live_qaoa, frozen_qaoa and uniform.
      4. Compare the spread of the results of the three samplers.

    frozen ~ live: the variability of QAOA comes from training, not shot noise.
    frozen > uniform: P_QAOA carries information of its own.
    frozen ~ uniform: P_QAOA adds no value over a flat distribution.

    Output: exp3_distribution_injection.csv
    """
    if seeds is None:
        seeds = config.get('multiseed_seeds', list(range(10)))

    set_global_seed(0)
    H_ref = build_support_surrogate_hamiltonian(
        instance['N'], instance['K'],
        penalty=HAMILTONIAN_PENALTY, instance=instance)
    c_exp_ref, c_sample_ref = make_circuits(
        instance['N'], config['qaoa_layers'], H_ref,
        k=instance['K'], shots=config['qaoa_shots'], pl_seed=0)
    p_ref = train_qaoa(c_exp_ref, p=config['qaoa_layers'],
                       n_steps=config['qaoa_steps'], verbose=False)

    raw_ref = c_sample_ref(p_ref)
    frozen_counts = {}
    for row in raw_ref:
        b = tuple(int(x) for x in row)
        frozen_counts[b] = frozen_counts.get(b, 0) + 1
    print(f"[EXP-3] Frozen distribution: {len(frozen_counts)} unique supports "
          f"| top freq: {max(frozen_counts.values())/len(raw_ref):.4f}")

    frozen_s = build_frozen_distribution_sampler(frozen_counts, instance)

    rows = []
    for sampler_type, (sampler_factory, use_params) in {
        'live_qaoa':    (lambda s: _build_live_qaoa(instance, config, s), True),
        'frozen_qaoa':  (lambda s: (frozen_s, None), False),
        'uniform':      (lambda s: (build_uniform_random_sampler(instance), None), False),
    }.items():
        for seed in seeds:
            set_global_seed(seed)
            sampler, params = sampler_factory(seed)
            df_run = run_hybrid_v2(
                sampler=sampler, instance=instance, params=params,
                max_iter=config['max_iter'],
                top_k_candidates=config['top_k_candidates'],
                shots=config['qaoa_shots'],
                top_eval_per_iter=config['top_eval_per_iter'],
                score_alpha=config['score_alpha'], score_beta=config['score_beta'],
                baseline_stats=baseline_stats,
                label=sampler_type, selection_policy='random_exactK',
            )
            m = summarize_run(df_run, baseline_stats)
            m.update(dict(sampler_type=sampler_type, seed=seed))
            rows.append(m)
            print(f"  {sampler_type}|seed={seed}: best={m['best_var']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv("exp3_distribution_injection.csv", index=False)

    print("\n=== EXP-3: Live QAOA vs Frozen QAOA vs Uniforme (random_exactK) ===")
    print(df.groupby('sampler_type')[['best_var','gap_pct']].agg(['mean','std']).to_string())
    return df


def _build_live_qaoa(instance, config, seed):
    set_global_seed(seed)
    H = build_support_surrogate_hamiltonian(
        instance['N'], instance['K'],
        penalty=HAMILTONIAN_PENALTY, instance=instance)
    c_exp, c_sample = make_circuits(
        instance['N'], config['qaoa_layers'], H,
        k=instance['K'], shots=config['qaoa_shots'], pl_seed=seed)
    p_trained = train_qaoa(c_exp, p=config['qaoa_layers'],
                           n_steps=config['qaoa_steps'], verbose=False)
    sampler = build_qaoa_sampler(c_sample, instance)
    return sampler, p_trained


def exp4_frequency_policy(
    instance, baseline_stats, config,
    seeds: list = None,
):
    """EXP-4: extend the comparative matrix with ranked_frequency as a fourth policy.

    2 x 4 design:
      QAOA_surrogate x {ranked_hybrid, ranked_mu, random_exactK, ranked_frequency}
      uniform_random x {ranked_hybrid, ranked_mu, random_exactK, ranked_frequency}

    QAOA + ranked_frequency > QAOA + random_exactK would indicate that the QAOA
    frequency concentration adds value even without hybrid_score.
    Uniform + ranked_frequency ~ uniform + random_exactK confirms that
    ranked_frequency is neutral under a flat distribution.

    Output: exp4_frequency_policy.csv
    """
    if seeds is None:
        seeds = config.get('multiseed_seeds', list(range(10)))


    policies = ['ranked_hybrid', 'ranked_mu', 'random_exactK', 'ranked_frequency']
    rows = []

    for sampler_type in ['QAOA_surrogate', 'uniform_random']:
        for policy in policies:
            for seed in seeds:
                set_global_seed(seed)
                if sampler_type == 'QAOA_surrogate':
                    sampler, params = _build_live_qaoa(instance, config, seed)
                else:
                    sampler = build_uniform_random_sampler(instance)
                    params  = None

                df_run = run_hybrid_v2(
                    sampler=sampler, instance=instance, params=params,
                    max_iter=config['max_iter'],
                    top_k_candidates=config['top_k_candidates'],
                    shots=config['qaoa_shots'],
                    top_eval_per_iter=config['top_eval_per_iter'],
                    score_alpha=config['score_alpha'], score_beta=config['score_beta'],
                    baseline_stats=baseline_stats,
                    label=sampler_type, selection_policy=policy,
                )
                m = summarize_run(df_run, baseline_stats)
                m.update(dict(sampler_type=sampler_type, selection_policy=policy, seed=seed))
                rows.append(m)

    df = pd.DataFrame(rows)
    df.to_csv("exp4_frequency_policy.csv", index=False)

    print("\n=== EXP-4: ranked_frequency as fourth policy ===")
    print(df.groupby(['sampler_type','selection_policy'])
          [['best_var','gap_pct','ipopt_total']].agg(['mean','std']).to_string())
    return df


def exp5_depth_ablation(
    instance, baseline_stats, config,
    depths: list = None,
    seeds: list  = None,
):
    """EXP-5: circuit-depth ablation, p in {1, 2, 3}.

    For each p: train the QAOA circuit (2p parameters), characterize the resulting
    distribution (entropy, TVD, n_unique), run the hybrid pipeline with
    ranked_hybrid and random_exactK, and record best_var and gap_pct.

    Key diagnostic: entropy_ratio = H(p_QAOA) / H(Uniform) in (0, 1). If it grows
    with p, deeper circuits are less concentrated; if it stays near 0, the attractor
    is structural rather than a p = 1 effect.

    Output: exp5_depth_ablation.csv, exp5_distribution_stats.csv
    """
    if depths is None:
        depths = [1, 2, 3]
    if seeds is None:
        seeds = config.get('multiseed_seeds', list(range(5)))

    from math import comb, log as mlog

    N = instance['N']
    K = instance['K']
    n_possible = comb(N, K)
    entropy_uniform = mlog(n_possible)

    distr_rows = []
    perf_rows  = []

    for p_depth in depths:
        print(f"\n{'='*60}")
        print(f"EXP-5: p={p_depth}")
        print(f"{'='*60}")

        for seed in seeds:
            set_global_seed(seed)
            H = build_support_surrogate_hamiltonian(
                N, K, penalty=HAMILTONIAN_PENALTY, instance=instance)
            c_exp, c_sample = make_circuits(
                N, p_depth, H, k=K,
                shots=config['qaoa_shots'], pl_seed=seed)
            p_trained = train_qaoa(c_exp, p=p_depth,
                                   n_steps=config['qaoa_steps'], verbose=False)

            raw = c_sample(p_trained)
            counts = {}
            for row in raw:
                b = tuple(int(x) for x in row)
                counts[b] = counts.get(b, 0) + 1
            exact_counts = {b: c for b, c in counts.items() if sum(b) == K}
            total_exact  = sum(exact_counts.values())
            n_unique = len(exact_counts)

            if total_exact > 0:
                probs = np.array(list(exact_counts.values()), dtype=float) / total_exact
                entropy_qaoa = float(-np.sum(probs * np.log(probs + 1e-15)))
                entropy_ratio = entropy_qaoa / entropy_uniform
                top1_freq = float(max(probs))
            else:
                entropy_ratio = 0.0
                top1_freq     = 1.0

            distr_rows.append(dict(
                p=p_depth, seed=seed,
                n_unique_exactK=n_unique,
                entropy_ratio=round(entropy_ratio, 4),
                top1_freq=round(top1_freq, 4),
                frac_exactK=total_exact / len(raw) if len(raw) > 0 else 0,
            ))
            print(f"  p={p_depth} seed={seed}: n_unique={n_unique} "
                  f"entropy_ratio={entropy_ratio:.4f} top1_freq={top1_freq:.4f}")

            sampler = build_qaoa_sampler(c_sample, instance)
            df_run  = run_hybrid_v2(
                sampler=sampler, instance=instance, params=p_trained,
                max_iter=config['max_iter'],
                top_k_candidates=config['top_k_candidates'],
                shots=config['qaoa_shots'],
                top_eval_per_iter=config['top_eval_per_iter'],
                score_alpha=config['score_alpha'], score_beta=config['score_beta'],
                baseline_stats=baseline_stats,
                label=f"QAOA_p{p_depth}", selection_policy='random_exactK',
            )
            m = summarize_run(df_run, baseline_stats)
            m.update(dict(p=p_depth, seed=seed, selection_policy='random_exactK'))
            perf_rows.append(m)

    df_distr = pd.DataFrame(distr_rows)
    df_perf  = pd.DataFrame(perf_rows)
    df_distr.to_csv("exp5_distribution_stats.csv", index=False)
    df_perf.to_csv( "exp5_depth_ablation.csv",     index=False)

    print("\n=== EXP-5: distribution by depth ===")
    print(df_distr.groupby('p')[['n_unique_exactK','entropy_ratio','top1_freq']]
          .agg(['mean','std']).to_string())
    print("\n=== EXP-5: performance by depth (random_exactK) ===")
    print(df_perf.groupby('p')[['best_var','gap_pct','ipopt_total']]
          .agg(['mean','std']).to_string())

    return df_distr, df_perf


if __name__ == '__main__':

    instance      = INSTANCE_15
    set_global_seed(CONFIG['seed'])

    _CSV = "baseline_enumeration.csv"
    if os.path.exists(_CSV):
        df_base = pd.read_csv(_CSV, dtype={'bits': str})
        baseline_stats = dict(
            best_var      = float(df_base.loc[df_base['feasible']==True, 'variance'].min()),
            n_ipopt_calls = int((df_base['method']=='ipopt').sum()),
            n_feasible    = int((df_base['feasible']==True).sum()),
        )
        print(f"[Phase 3] Baseline loaded: best_var={baseline_stats['best_var']:.6f}")
    else:
        raise FileNotFoundError("Run Fase2.py first to generate baseline_enumeration.csv")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, choices=['1', '2', '3', '4', '5', 'all'],
                        default='all',
                        help='Experimento a ejecutar (1-5 o all)')
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--bmax', type=int, default=20,
                        help='Common capped IPOPT budget for EXP-1')
    args = parser.parse_args()

    seeds = list(range(args.seeds))

    exp_runners = {
        '1': lambda: run_exp_1(instance, baseline_stats, CONFIG, seeds, args.bmax),
        '2': lambda: run_exp_2(instance, baseline_stats, CONFIG, seeds, args.bmax),
        '3': lambda: run_exp_3(instance, baseline_stats, CONFIG, seeds, args.bmax),
        '4': lambda: run_exp_4(instance, baseline_stats, CONFIG, seeds, args.bmax),
        '5': lambda: run_exp_5(instance, baseline_stats, CONFIG, seeds, args.bmax),
    }

    if args.exp == 'all':
        all_results = []
        for exp_key in ['1', '2', '3', '4', '5']:
            all_results.append(exp_runners[exp_key]())
        final_df = pd.concat(all_results, ignore_index=True)
    else:
        final_df = exp_runners[args.exp]()

    final_df.to_csv("fase3_all_results.csv", index=False)
    print(f"\n[Phase 3] Consolidated results saved to fase3_all_results.csv "
          f"({len(final_df)} filas)")

    print("\n[Phase 3] Complete.")
