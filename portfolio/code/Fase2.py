"""Phase 2 of the portfolio benchmark: (A) separation of the structural confound and
(B) diagnosis of the QAOA attractor.

Requires Quantico-cardinalidad.py in the same directory and the same dependencies
(IPOPT, PennyLane, Pyomo). The module reuses the infrastructure of that script via
importlib and adds only the Phase 2 components:

  Section A - ranked_mu policy (orthogonal to the Hamiltonian):
              extended select_candidates, run_comparative_matrix_v2
  Section B - QAOA attractor diagnosis:
              eval_hamiltonian_on_bitstring, diagnose_qaoa_attractor
  Execution block - writes attractor_distribution.csv, surrogate_vs_ipopt_full.csv,
              ranked_mu_orthogonality.csv, comparative_matrix_v2.csv and
              convergence_curves.csv to the working directory.
"""

import importlib.util
import sys
import os
import time
import itertools
import numpy as np
import pandas as pd
import pennylane as qml
from scipy.stats import spearmanr, kendalltau
from typing import Callable, Optional


_ORIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "Quantico-cardinalidad.py")

if not os.path.exists(_ORIG_PATH):
    raise FileNotFoundError(
        f"Quantico-cardinalidad.py not found at:\n  {_ORIG_PATH}\n"
        "Place it in the same directory as this script."
    )

_spec = importlib.util.spec_from_file_location("quantico_cardinalidad", _ORIG_PATH)
_mod  = importlib.util.module_from_spec(_spec)
sys.modules["quantico_cardinalidad"] = _mod
_spec.loader.exec_module(_mod)

from quantico_cardinalidad import (
    INSTANCE_15,
    CONFIG,
    HAMILTONIAN_PENALTY,
    GLOBAL_SEED,
    build_support_surrogate_hamiltonian,
    make_circuits,
    train_qaoa,
    build_qaoa_sampler,
    build_uniform_random_sampler,
    classical_baseline,
    run_hybrid,
    summarize_run,
    build_candidate_pool,
    candidate_pool_metadata,
    set_global_seed,
    StructuralTabu,
    DiscardReason,
    diagnose_infeasibility,
    infer_and_register_rule,
    select_candidates as _select_candidates_orig,
    build_budget_pool as _build_budget_pool_orig,
    _to_bits,
    _selected_names,
    _selection_sectors,
    _entry,
    solve_subproblem,
    support_risk_score,
)


def mu_score_support(sel: dict, instance: dict) -> float:
    """Pure financial score based on the expected returns of the support.

    Definition:
        mu_score(y) = sum_{i: y_i=1} mu_i

    Orthogonality with respect to the surrogate Hamiltonian:
      - uses neither the covariance nor a risk proxy;
      - neither normalizes nor truncates the return signal with respect to R;
      - does not depend on any parameter of the QAOA circuit.

    By construction the Hamiltonian uses max(mu_i - R, 0)/scale as the return signal
    (truncated, normalized) and adds a covariance ZZ term; mu_score uses the raw mu
    values without these transformations.

    Consequence for attribution: if QAOA + ranked_mu differs from uniform + ranked_mu,
    the difference is more attributable to the sampler than to a functional
    correlation between the ranking and the Hamiltonian.
    """
    ASSETS = instance['ASSETS']
    mu     = instance['mu']
    return float(sum(mu[a] for a in ASSETS if sel.get(a, 0) == 1))


def hybrid_score_support(sel: dict, instance: dict,
                          alpha: float = 1.0, beta: float = 0.05) -> float:
    """Recompute hybrid_score for an arbitrary support.

    Same computation as build_candidate_pool, for direct comparison with mu_score in
    the attractor diagnosis.
    """
    ASSETS   = instance['ASSETS']
    K        = instance['K']
    mu       = instance['mu']
    cov      = instance['cov']
    R        = instance['R']
    N        = instance['N']

    chosen = [a for a in ASSETS if sel.get(a, 0) == 1]
    if len(chosen) != K:
        return 0.0

    mu_vals   = mu.loc[ASSETS].astype(float).values
    mu_signal = np.maximum(mu_vals - R, 0.0)
    mu_scale  = float(np.max(np.abs(mu_signal)))
    mu_hat    = mu_signal / mu_scale if mu_scale > 1e-12 else np.zeros(N)

    cov_vals  = cov.loc[ASSETS, ASSETS].astype(float).values
    cov_scale = float(np.max(np.abs(cov_vals)))
    cov_hat   = cov_vals / cov_scale if cov_scale > 1e-12 else np.zeros((N, N))

    asset_idx = {a: i for i, a in enumerate(ASSETS)}
    idx       = [asset_idx[a] for a in chosen]

    r_score   = float(sum(mu_hat[i] for i in idx))
    risk_score = float(
        sum(cov_hat[i, i] for i in idx) +
        2.0 * sum(cov_hat[i, j]
                  for pos, i in enumerate(idx)
                  for j in idx[pos + 1:])
    )
    return alpha * r_score - beta * risk_score


def check_ranked_mu_orthogonality(
    df_full: pd.DataFrame,
    instance: dict,
    rho_warning_threshold: float = 0.70,
) -> dict:
    """Check whether ranked_mu and ranked_hybrid are functionally independent signals
    over the space of feasible supports.

    The orthogonality hypothesis states that mu_score (raw sum of returns) and
    hybrid_score (Hamiltonian proxy: truncated normalized return minus support risk)
    prioritize different orders, so that differences between the two policies are
    attributable to the ranking and not to the sampler. The hypothesis is
    structurally weak when the assets with mu_i >= R are the ones the Hamiltonian
    prioritizes, because on that subset both signals are monotone in mu_i.

    Metrics
    -------
    rho_mu_hybrid   : Spearman rho(mu_score, hybrid_score) over all feasible
                      supports. Above rho_warning_threshold the policies prioritize
                      essentially the same order.
    rho_mu_energy   : Spearman rho(mu_score, surrogate_energy). Negative if a larger
                      mu_score corresponds to a lower (more favorable) surrogate
                      energy, i.e. ranked_mu and the Hamiltonian are collinear.
    decile_table    : mean and std of hybrid_score per mu_score decile.
    verdict         : 'orthogonality_weak' if rho_mu_hybrid > threshold,
                      'orthogonality_acceptable' otherwise.

    Parameters
    ----------
    df_full : DataFrame with columns mu_score, hybrid_score, surrogate_energy,
              ipopt_variance (output of full_surrogate_vs_ipopt_analysis).
    instance : instance dict (used to print context).
    rho_warning_threshold : Spearman threshold that triggers the warning.

    Returns
    -------
    dict with rho_mu_hybrid, p_mu_hybrid, rho_mu_energy, p_mu_energy,
    decile_table, verdict, warning_message.
    """
    required = {'mu_score', 'hybrid_score', 'surrogate_energy'}
    missing = required - set(df_full.columns)
    if missing:
        raise ValueError(f"df_full falta columnas: {missing}")

    df = df_full.dropna(subset=['mu_score', 'hybrid_score', 'surrogate_energy'])
    n = len(df)

    rho_mh, p_mh = spearmanr(df['mu_score'], df['hybrid_score'])
    rho_me, p_me = spearmanr(df['mu_score'], df['surrogate_energy'])

    df = df.copy()
    df['mu_decile'] = pd.qcut(df['mu_score'], q=10, labels=False,
                               duplicates='drop')
    decile_table = (
        df.groupby('mu_decile')
          .agg(
              n=('mu_score', 'count'),
              mu_score_mean=('mu_score', 'mean'),
              hybrid_score_mean=('hybrid_score', 'mean'),
              hybrid_score_std=('hybrid_score', 'std'),
              surrogate_energy_mean=('surrogate_energy', 'mean'),
          )
          .reset_index()
    )

    verdict = ('orthogonality_weak'
               if abs(rho_mh) > rho_warning_threshold
               else 'orthogonality_acceptable')

    R = instance.get('R', '?')
    N_HIGH = instance.get('N_HIGH', '?')
    N = instance.get('N', '?')
    warning_message = ""
    if verdict == 'orthogonality_weak':
        warning_message = (
            f"WARNING: rho(mu_score, hybrid_score) = {rho_mh:.3f} > "
            f"{rho_warning_threshold}. "
            f"With R={R} and only {N_HIGH}/{N} assets with mu >= R, both "
            f"policies prioritize almost the same order. The differences "
            f"observed in the comparative matrix are not attributable "
            f"exclusively to the sampler - ranked_mu is NOT functionally "
            f"orthogonal to ranked_hybrid for this instance."
        )

    print("\n" + "=" * 70)
    print("ORTHOGONALITY CHECK: ranked_mu vs ranked_hybrid")
    print("=" * 70)
    print(f"  n supports analysed            : {n}")
    print(f"  rho(mu_score, hybrid_score)    : {rho_mh:.4f}  (p={p_mh:.2e})")
    print(f"  rho(mu_score, surrogate_energy): {rho_me:.4f}  (p={p_me:.2e})")
    print(f"  Warning threshold              : {rho_warning_threshold}")
    print(f"  Veredicto                      : {verdict}")
    if warning_message:
        print(f"\n  {warning_message}")
    print("\n  Table by mu_score decile:")
    print(decile_table.to_string(index=False))

    return dict(
        n_supports=n,
        rho_mu_hybrid=round(float(rho_mh), 4),
        p_mu_hybrid=round(float(p_mh), 6),
        rho_mu_energy=round(float(rho_me), 4),
        p_mu_energy=round(float(p_me), 6),
        decile_table=decile_table,
        verdict=verdict,
        warning_message=warning_message,
    )


def select_candidates(valid_candidates: list, instance: dict, policy: str,
                      top_eval_per_iter: int, rng=None) -> list:
    """Extended select_candidates with support for ranked_mu.

    Available policies:
      - ranked_hybrid    : hybrid_score desc (proxy shared with the Hamiltonian)
      - ranked_mu        : sum(mu) desc (pure return, orthogonal by design)
      - ranked_frequency : observed frequency desc (pure distributional bias)
      - random_exactK    : random selection among exact-K candidates
    """
    if top_eval_per_iter <= 0:
        return []

    if policy == "ranked_hybrid":
        ranked = sorted(valid_candidates, key=lambda c: (
            0 if c['is_exactK'] else 1,
            -c['hybrid_score'],
            -c['freq'],
        ))
        return ranked[:top_eval_per_iter]

    if policy == "ranked_mu":
        exact_candidates = [c for c in valid_candidates if c['is_exactK']]
        if not exact_candidates:
            return []
        ranked = sorted(
            exact_candidates,
            key=lambda c: -mu_score_support(c['selection'], instance)
        )
        return ranked[:top_eval_per_iter]

    if policy == "ranked_frequency":
        ranked = sorted(valid_candidates, key=lambda c: (
            0 if c['is_exactK'] else 1,
            -c['freq'],
        ))
        return ranked[:top_eval_per_iter]

    if policy == "random_exactK":
        exact_candidates = [c for c in valid_candidates if c['is_exactK']]
        if not exact_candidates:
            return []
        if rng is None:
            rng = np.random
        n_pick = min(top_eval_per_iter, len(exact_candidates))
        idx    = rng.choice(len(exact_candidates), size=n_pick, replace=False)
        return [exact_candidates[int(i)] for i in np.atleast_1d(idx)]

    raise ValueError(f"selection_policy no soportada: {policy}")


def build_budget_pool(valid_candidates: list, instance: dict,
                      selection_policy: str, top_k_candidates: int,
                      rng=None) -> list:
    """Extended build_budget_pool compatible with ranked_mu.

    Delegates to the extended select_candidates of this module.
    """
    return select_candidates(
        valid_candidates, instance, selection_policy, top_k_candidates,
        rng=rng)


def run_hybrid_v2(
    sampler: Callable,
    instance: dict,
    params=None,
    max_iter: int          = 30,
    top_k_candidates: int  = 40,
    shots: int             = 5000,
    top_eval_per_iter: int = 2,
    score_alpha: float     = 1.0,
    score_beta: float      = 0.05,
    baseline_stats: dict   = None,
    label: str             = "",
    selection_policy: str  = "ranked_hybrid",
) -> pd.DataFrame:
    """Phase 2 hybrid loop.

    Same logic as run_hybrid of Quantico-cardinalidad.py, but uses the extended
    select_candidates and build_budget_pool, which enables the ranked_mu policy
    without modifying the original module.
    """
    ASSETS      = instance['ASSETS']
    K           = instance['K']
    N           = instance['N']
    R           = instance['R']
    sampler_name = label or getattr(sampler, '__name__', 'sampler')

    print("=" * 70)
    print(f"HYBRID V2 | N={N}, K={K}, R={R}")
    print(f"  sampler={sampler_name} | top_eval={top_eval_per_iter}")
    print(f"  selection_policy={selection_policy}")
    print("=" * 70)

    tabu        = StructuralTabu()
    log_rows    = []
    best_var    = np.inf
    best_sel    = None
    best_active = None
    ipopt_total = 0

    for it in range(max_iter):
        t_iter = time.time()
        print(f"\n--- Iter {it} ---")

        candidate_pool, sampling_meta = sampler(
            params=params,
            shots=shots,
            top_k=top_k_candidates,
            alpha=score_alpha,
            beta=score_beta,
        )

        n_exactK_pre   = sum(1 for c in candidate_pool if c['is_exactK'])
        n_nonexact_pre = len(candidate_pool) - n_exactK_pre

        valid, discarded = [], []
        for cand in candidate_pool:
            bits   = _to_bits(cand['selection'], instance)
            is_tab, reason, rule = tabu.classify(cand['selection'], bits, instance)
            if is_tab:
                discarded.append((cand, reason, rule))
            else:
                valid.append(cand)

        n_no_good = sum(1 for _, r, _ in discarded if r == DiscardReason.NO_GOOD_EXACT)
        n_struct  = sum(1 for _, r, _ in discarded if r == DiscardReason.STRUCT_RULE)
        n_nonK    = sum(1 for _, r, _ in discarded if r == DiscardReason.NON_EXACT_K)
        n_exactK_valid = sum(1 for c in valid if c['is_exactK'])

        print(f"  Pool: {len(candidate_pool)} | exactK={n_exactK_pre} | "
              f"valid={len(valid)} | no-good={n_no_good} | struct={n_struct}")
        print(f"  Muestreo: unique={sampling_meta['n_unique_raw']} | "
              f"entropy_exactK={sampling_meta.get('entropy_exactK', 0.0):.4f} | "
              f"top1_exactK={sampling_meta.get('top1_exactK_freq', 0.0):.4f}")

        if not valid:
            print("  Space exhausted. Stopping.")
            break

        budget_pool = build_budget_pool(
            valid, instance, selection_policy, top_k_candidates, rng=np.random)

        if not budget_pool:
            print("  No candidates within the budget. Stopping.")
            break

        to_evaluate = select_candidates(
            budget_pool, instance, selection_policy, top_eval_per_iter,
            rng=np.random)

        if not to_evaluate:
            print("  Policy chose no candidates. Stopping.")
            break

        for rank, chosen in enumerate(to_evaluate):
            chosen_sel    = chosen['selection']
            chosen_freq   = chosen['freq']
            chosen_rscore = chosen['r_score']
            chosen_risk   = chosen['risk_score']
            chosen_hybrid = chosen['hybrid_score']
            chosen_mu     = mu_score_support(chosen_sel, instance)

            bits  = _to_bits(chosen_sel, instance)
            names = _selected_names(chosen_sel, instance)
            print(f"  [{rank}] {names} | freq={chosen_freq:.4f} | "
                  f"hybrid={chosen_hybrid:.3f} | mu_score={chosen_mu:.2f}")

            tabu.register_visit(bits)

            diag = diagnose_infeasibility(chosen_sel, instance)
            if diag != 'unknown':
                is_new = infer_and_register_rule(tabu, diag, instance)
                if is_new:
                    tabu.stats['structural_discoveries'] += 1
                tabu.stats['ipopt_avoided_prescreen'] += 1
                print(f"    Pre-screen: {diag} - IPOPT skipped"
                      + (" [new rule]" if is_new else ""))
                log_rows.append(_make_entry(
                    it, rank, bits, names, sampler_name, sampling_meta,
                    n_exactK_pre, n_nonexact_pre, n_exactK_valid,
                    n_no_good, n_struct, n_nonK,
                    chosen_freq, chosen_rscore, chosen_risk, chosen_hybrid,
                    chosen_mu, 'infeasible_prescreen', None, None,
                    None, None, best_var, ipopt_total, tabu,
                    round(time.time() - t_iter, 3)))
                continue

            ipopt_total += 1
            result = solve_subproblem(chosen_sel, instance)

            if not result['feasible']:
                print(f"    IPOPT: infeasible ({result['tc']})")
                log_rows.append(_make_entry(
                    it, rank, bits, names, sampler_name, sampling_meta,
                    n_exactK_pre, n_nonexact_pre, n_exactK_valid,
                    n_no_good, n_struct, n_nonK,
                    chosen_freq, chosen_rscore, chosen_risk, chosen_hybrid,
                    chosen_mu, 'infeasible_ipopt', None, None,
                    result.get('active_count'), result.get('selected_count'),
                    best_var, ipopt_total, tabu,
                    round(time.time() - t_iter, 3)))
                continue

            var   = result['variance']
            slack = result['ret_slack']
            print(f"    IPOPT: variance={var:.6f} | ret_slack={slack:.4f}")

            if var < best_var - 1e-8:
                best_var    = var
                best_sel    = chosen_sel.copy()
                best_active = result.get('active_count')
                print(f"    *** Best: {best_var:.6f} ***")

            log_rows.append(_make_entry(
                it, rank, bits, names, sampler_name, sampling_meta,
                n_exactK_pre, n_nonexact_pre, n_exactK_valid,
                n_no_good, n_struct, n_nonK,
                chosen_freq, chosen_rscore, chosen_risk, chosen_hybrid,
                chosen_mu, 'feasible', round(var, 8), round(slack, 6),
                result.get('active_count'), result.get('selected_count'),
                best_var, ipopt_total, tabu,
                round(time.time() - t_iter, 3)))

    print("\n" + "=" * 70)
    print(f"FINAL SOLUTION [{sampler_name} | {selection_policy}]")
    print("=" * 70)
    if best_sel:
        names   = _selected_names(best_sel, instance)
        sectors = _selection_sectors(best_sel, instance)
        print(f"  Assets   : {names}")
        print(f"  Sectores : {dict((s,v) for s,v in sectors.items() if v > 0)}")
        print(f"  Variance : {best_var:.6f}")
        if best_active is not None:
            print(f"  Effective assets: {best_active}")
        if baseline_stats and baseline_stats.get('best_var') is not None:
            ref  = baseline_stats['best_var']
            gap  = (best_var - ref) / ref * 100
            saved = baseline_stats['n_ipopt_calls'] - ipopt_total
            print(f"  Gap      : {gap:+.4f}%")
            print(f"  IPOPT    : {ipopt_total} (saved: {saved})")
    else:
        print("  No feasible solution found.")
    print(f"  Sampling/solver summary: ipopt_calls_used={ipopt_total}")

    return pd.DataFrame(log_rows)


def _make_entry(it, rank, bits, names, sampler_name, sampling_meta,
                n_exactK_pre, n_nonexact_pre, n_exactK_valid,
                n_no_good, n_struct, n_nonK,
                freq, rscore, risk, hybrid, mu_sc,
                estado, variance, ret_slack,
                active_count, selected_count,
                best_var, ipopt_total, tabu, iter_time):
    """Log entry extended with mu_score_chosen."""
    base = _entry(
        it, rank, bits, names, sampler_name,
        sampling_meta, n_exactK_pre, n_nonexact_pre,
        n_exactK_valid, n_no_good, n_struct, n_nonK,
        freq, rscore, risk, hybrid,
        estado, variance, ret_slack,
        active_count, selected_count,
        best_var, ipopt_total, tabu, iter_time,
    )
    base['mu_score_chosen'] = round(mu_sc, 4)
    return base


def run_comparative_matrix_v2(
    instance: dict,
    baseline_stats: dict,
    config: dict,
    seeds: list = None,
    selection_policies: list = None,
) -> pd.DataFrame:
    """Phase 2 comparative matrix.

    Factors:
      - sampler_type     : QAOA_surrogate, uniform_random
      - selection_policy : ranked_hybrid, ranked_mu, random_exactK, ranked_frequency

    ranked_mu takes the role of ranked_hybrid as the "informed" policy, without the
    structural dependence on the QAOA Hamiltonian. Comparing ranked_hybrid with
    ranked_mu within each sampler quantifies how much of the behavior is due to the
    functional correlation between proxy and ranking.
    """
    if seeds is None:
        seeds = config.get('multiseed_seeds', list(range(10)))
    if selection_policies is None:
        selection_policies = ['ranked_hybrid', 'ranked_mu',
                              'random_exactK', 'ranked_frequency']

    def _qaoa_factory(seed):
        set_global_seed(seed)
        H = build_support_surrogate_hamiltonian(
            instance['N'], instance['K'],
            penalty=HAMILTONIAN_PENALTY, instance=instance)
        c_exp, c_sample = make_circuits(
            instance['N'], config['qaoa_layers'], H, k=instance['K'],
            shots=config['qaoa_shots'], pl_seed=seed)
        p_trained = train_qaoa(c_exp, p=config['qaoa_layers'],
                               n_steps=config['qaoa_steps'], verbose=False)
        sampler = build_qaoa_sampler(c_sample, instance)
        return sampler, p_trained

    def _uniform_factory(seed):
        set_global_seed(seed)
        sampler = build_uniform_random_sampler(instance)
        return sampler, None

    sampler_factories = [
        ('QAOA_surrogate',  _qaoa_factory),
        ('uniform_random',  _uniform_factory),
    ]

    common_kwargs = dict(
        max_iter          = config['max_iter'],
        top_k_candidates  = config['top_k_candidates'],
        shots             = config['qaoa_shots'],
        top_eval_per_iter = config['top_eval_per_iter'],
        score_alpha       = config['score_alpha'],
        score_beta        = config['score_beta'],
        baseline_stats    = baseline_stats,
    )

    rows = []
    traj_rows = []
    print("\n" + "=" * 70)
    print("PHASE 2 - COMPARATIVE MATRIX V2: sampler_type x selection_policy")
    print("Policies:", selection_policies)
    print("=" * 70)

    for sampler_type, factory in sampler_factories:
        for policy in selection_policies:
            label = f"{sampler_type}__{policy}"
            print(f"\n--- {sampler_type} | {policy} ---")

            for seed in seeds:
                print(f"  Seed {seed}")
                sampler, params = factory(seed)
                df_run = run_hybrid_v2(
                    sampler          = sampler,
                    instance         = instance,
                    params           = params,
                    label            = label,
                    selection_policy = policy,
                    **common_kwargs,
                )
                metrics = summarize_run(df_run, baseline_stats)
                metrics['sampler_type']     = sampler_type
                metrics['selection_policy'] = policy
                metrics['seed']             = seed
                rows.append(metrics)

                traj = extract_convergence_trajectory(df_run)
                traj_rows.append(dict(
                    sampler_type=sampler_type,
                    selection_policy=policy,
                    seed=seed,
                    trajectory=traj,
                ))
                print(f"  -> best_var={metrics['best_var']:.6f} | "
                      f"gap={metrics['gap_pct']:.4f}% | "
                      f"ipopt={metrics['ipopt_total']}")

    df_matrix = pd.DataFrame(rows)
    ordered = ['sampler_type', 'selection_policy', 'seed',
               'best_var', 'gap_pct', 'ipopt_total', 'ipopt_saved',
               'iter_best', 'n_feasible']
    df_matrix = df_matrix[ordered]

    summary_cols = ['best_var', 'gap_pct', 'ipopt_total', 'ipopt_saved',
                    'iter_best', 'n_feasible']
    summary = (df_matrix
               .groupby(['sampler_type', 'selection_policy'])[summary_cols]
               .agg(['mean', 'std']))

    print("\n" + "=" * 70)
    print("MATRIX v2 SUMMARY")
    print("=" * 70)
    print(summary.to_string())

    return df_matrix, traj_rows


def extract_convergence_trajectory(df_run: pd.DataFrame) -> list:
    """Extract the incumbent trajectory indexed by cumulative IPOPT evaluations from the
    log of a single run.

    The trajectory is a monotone non-increasing step function: every time
    ipopt_total increases, the current incumbent is recorded.

    Parameters
    ----------
    df_run : DataFrame produced by run_hybrid_v2, with columns 'ipopt_total' and
             'best_cumul'.

    Returns
    -------
    List of tuples (ipopt_calls: int, best_var: float), one per increment of
    ipopt_total. The point (0, inf) is not included. Returns an empty list if
    df_run is empty or has no IPOPT calls.
    """
    if df_run.empty or 'ipopt_total' not in df_run.columns:
        return []

    df = df_run[['ipopt_total', 'best_cumul']].copy()
    df = df.dropna(subset=['ipopt_total'])
    df['ipopt_total'] = df['ipopt_total'].astype(int)

    traj_map = {}
    running_best = np.inf
    for _, row in df.iterrows():
        calls = int(row['ipopt_total'])
        val = row['best_cumul']
        if calls == 0:
            continue
        if pd.notna(val):
            running_best = min(running_best, float(val))
        if calls not in traj_map or running_best < traj_map[calls]:
            traj_map[calls] = running_best

    if not traj_map:
        return []

    max_calls = max(traj_map.keys())
    trajectory = []
    last_best = np.inf
    for c in range(1, max_calls + 1):
        if c in traj_map:
            last_best = traj_map[c]
        if last_best < np.inf:
            trajectory.append((c, last_best))

    return trajectory


def build_convergence_summary(
    traj_rows: list,
    max_ipopt: int,
    baseline_var: float,
) -> pd.DataFrame:
    """Aggregate convergence trajectories of several runs into a table of means and
    quantiles per number of IPOPT evaluations.

    Each run is interpolated on the grid [1, max_ipopt] with the last known
    observation (step function); runs that ended before max_ipopt keep their last
    incumbent.

    Parameters
    ----------
    traj_rows   : list of dicts with keys sampler_type, selection_policy, seed,
                  trajectory (list of tuples from extract_convergence_trajectory).
    max_ipopt   : integer, size of the x-axis grid.
    baseline_var: variance of the global optimum (for the cumulative gap).

    Returns
    -------
    DataFrame with columns:
        sampler_type, selection_policy, ipopt_calls,
        best_var_mean, best_var_q25, best_var_q75,
        gap_pct_mean, gap_pct_q25, gap_pct_q75,
        n_seeds_with_solution
    """
    def q25(x): return x.quantile(0.25)
    def q75(x): return x.quantile(0.75)

    grid = list(range(1, max_ipopt + 1))
    records = []

    for row in traj_rows:
        traj = row['trajectory']
        if not traj:
            interp = {c: np.inf for c in grid}
        else:
            traj_dict = dict(traj)
            interp = {}
            last = np.inf
            for c in grid:
                if c in traj_dict:
                    last = traj_dict[c]
                interp[c] = last

        for c in grid:
            records.append(dict(
                sampler_type=row['sampler_type'],
                selection_policy=row['selection_policy'],
                seed=row['seed'],
                ipopt_calls=c,
                best_var=interp[c],
            ))

    df = pd.DataFrame(records)

    df_fin = df[df['best_var'] < np.inf].copy()
    df_fin['gap_pct'] = (df_fin['best_var'] - baseline_var) / baseline_var * 100

    agg = (
        df_fin
        .groupby(['sampler_type', 'selection_policy', 'ipopt_calls'])
        .agg(
            best_var_mean        =('best_var', 'mean'),
            best_var_q25         =('best_var', q25),
            best_var_q75         =('best_var', q75),
            gap_pct_mean         =('gap_pct',  'mean'),
            gap_pct_q25          =('gap_pct',  q25),
            gap_pct_q75          =('gap_pct',  q75),
            n_seeds_with_solution=('best_var', 'count'),
        )
        .reset_index()
    )

    return agg


def eval_hamiltonian_on_bitstring(
    H: qml.Hamiltonian,
    bitstring: tuple,
) -> float:
    """Compute <b|H|b> for a computational basis state |b>.

    For Pauli-Z operators and their tensor products the expectation in |b> is the
    product of the eigenvalues Z_i = 1 - 2*b_i over the qubits in the support of
    the operator. The computation is exact and needs no circuit simulation, so the
    surrogate energy of every support can be evaluated without PennyLane.

    Parameters
    ----------
    H         : PennyLane Hamiltonian (PauliZ and PauliZ x PauliZ terms).
    bitstring : tuple of integers in {0,1} of length N.

    Returns
    -------
    Scalar energy of the Hamiltonian evaluated in |b>.
    """
    z_vals = [1 - 2 * b for b in bitstring]
    total  = 0.0

    for coeff, op in zip(H.coeffs, H.ops):
        wires = list(op.wires)
        ev    = 1.0
        for w in wires:
            ev *= z_vals[w]
        total += float(coeff) * ev

    return total


def diagnose_qaoa_attractor(
    circuit_sample,
    params,
    H_cost: qml.Hamiltonian,
    instance: dict,
    baseline_df: pd.DataFrame,
    shots: int = 20000,
    top_n: int = 30,
    verbose: bool = True,
) -> dict:
    """Full diagnosis of the support to which QAOA consistently converges.

    Questions addressed:
      1. Does the QAOA winning support correspond to the surrogate minimum in the
         exact-K subspace?
      2. How often does QAOA sample the truly optimal support?
      3. Is the attractor a local minimum of the surrogate or simply the most
         frequent support due to the distributional concentration of the ansatz?
      4. Is there a correlation between surrogate energy and IPOPT variance?

    Metrics:
      - for every observed exact-K support:
          freq_qaoa        : observed fraction of the support
          surrogate_energy : Hamiltonian energy evaluated on the basis state
          hybrid_score     : ranking score of Quantico-cardinalidad.py
          mu_score         : proposed ranking score (Phase 2)
          ipopt_variance   : QP variance (from baseline_df)
          rank_by_variance : nullable rank if the variance is not available
      - global statistics:
          surrogate vs. IPOPT correlations over rows with a valid variance,
          frequency of the attractor and of the true optimum,
          position of the true optimum in the QAOA distribution.

    Parameters
    ----------
    circuit_sample  : trained sampling circuit (parameters fixed).
    params          : parameters of the trained circuit.
    H_cost          : surrogate Hamiltonian used in training.
    instance        : instance dict.
    baseline_df     : baseline DataFrame with columns 'bits' and 'variance'.
    shots           : number of shots requested for the diagnosis; the actual
                      denominator is the number of observed samples.
    top_n           : number of most frequent supports in the detailed analysis.
    verbose         : print a summary to the console.

    Returns
    -------
    dict with:
      'support_df'      : detailed DataFrame per observed exact-K support
      'global_stats'    : dict with correlation and ranking metrics
      'attractor_bits'  : bitstring of the winning support
      'optimal_bits'    : bitstring of the true optimum
    """
    ASSETS = instance['ASSETS']
    K      = instance['K']
    N      = instance['N']
    R      = instance['R']
    mu     = instance['mu']
    alpha  = CONFIG.get('score_alpha', 1.0)
    beta   = CONFIG.get('score_beta',  0.05)

    print(f"\n[diagnose_qaoa_attractor] Muestreo QAOA (requested shots={shots}) ...")
    raw     = circuit_sample(params)
    n_samples = len(raw)
    if shots != n_samples:
        print(
            f"  Warning: requested shots={shots} but observed samples="
            f"{n_samples}. Observed samples are used for the diagnostic."
        )
    counts  = {}
    for row in raw:
        b           = tuple(int(x) for x in row)
        counts[b]   = counts.get(b, 0) + 1

    exact_counts = {b: c for b, c in counts.items() if sum(b) == K}
    total_exact  = sum(exact_counts.values())
    frac_exact   = total_exact / n_samples if n_samples > 0 else np.nan

    print(f"  Unique exact-K supports observed  : {len(exact_counts)}")
    print(f"  Observed fraction in exact-K      : {frac_exact:.4f}")

    feasible_baseline = baseline_df[baseline_df['feasible'] == True].copy()
    bits_to_var       = dict(zip(
        feasible_baseline['bits'].astype(str),
        feasible_baseline['variance'].astype(float)
    ))

    if not feasible_baseline.empty:
        opt_row      = feasible_baseline.loc[feasible_baseline['variance'].idxmin()]
        optimal_bits_str = str(opt_row['bits'])
        optimal_var      = float(opt_row['variance'])
        optimal_bits     = tuple(int(c) for c in optimal_bits_str)
    else:
        optimal_bits_str = None
        optimal_var      = np.inf
        optimal_bits     = None

    rows = []
    for b, count in exact_counts.items():
        freq   = count / n_samples if n_samples > 0 else np.nan
        sel    = {ASSETS[i]: b[i] for i in range(N)}
        bits_s = ''.join(str(x) for x in b)

        surr_energy = eval_hamiltonian_on_bitstring(H_cost, b)
        hy_score    = hybrid_score_support(sel, instance, alpha, beta)
        mu_sc       = mu_score_support(sel, instance)
        ipopt_var   = bits_to_var.get(bits_s, np.nan)
        names       = [ASSETS[i] for i in range(N) if b[i] == 1]

        rows.append(dict(
            bits            = bits_s,
            assets          = str(names),
            count           = count,
            freq_qaoa       = freq,
            surrogate_energy= surr_energy,
            hybrid_score    = hy_score,
            mu_score        = mu_sc,
            ipopt_variance  = ipopt_var,
        ))

    df = pd.DataFrame(rows)

    df['rank_by_freq'] = (
        df['freq_qaoa'].rank(ascending=False, method='min').astype('Int64')
    )
    df['rank_by_energy'] = (
        df['surrogate_energy'].rank(ascending=True, method='min').astype('Int64')
    )
    df['rank_by_hybrid'] = (
        df['hybrid_score'].rank(ascending=False, method='min').astype('Int64')
    )
    df['rank_by_mu'] = (
        df['mu_score'].rank(ascending=False, method='min').astype('Int64')
    )
    df['rank_by_variance'] = pd.Series(pd.NA, index=df.index, dtype='Int64')
    valid_var_mask = df['ipopt_variance'].notna()
    df.loc[valid_var_mask, 'rank_by_variance'] = (
        df.loc[valid_var_mask, 'ipopt_variance']
          .rank(ascending=True, method='min')
          .astype('Int64')
    )

    attractor_row  = df.loc[df['freq_qaoa'].idxmax()]
    attractor_bits = attractor_row['bits']

    attractor_var  = attractor_row['ipopt_variance']

    df_corr = df.dropna(subset=['ipopt_variance'])
    if len(df_corr) >= 5:
        rho, p_spear = spearmanr(df_corr['surrogate_energy'],
                                 df_corr['ipopt_variance'])
        tau, p_kend  = kendalltau(df_corr['surrogate_energy'],
                                  df_corr['ipopt_variance'])
    else:
        rho = p_spear = tau = p_kend = np.nan

    if optimal_bits_str in df['bits'].values:
        opt_row_df       = df[df['bits'] == optimal_bits_str].iloc[0]
        opt_freq_qaoa    = float(opt_row_df['freq_qaoa'])
        opt_rank_freq    = int(opt_row_df['rank_by_freq'])
        opt_rank_energy  = int(opt_row_df['rank_by_energy'])
        opt_observed     = True
    else:
        opt_freq_qaoa   = 0.0
        opt_rank_freq   = len(df) + 1
        opt_rank_energy = np.nan
        opt_observed    = False

    energy_min_bits   = df.loc[df['surrogate_energy'].idxmin(), 'bits']
    attractor_is_emin = (attractor_bits == energy_min_bits)

    attractor_var_phase1 = 11.922034
    n_same_var = len(df[abs(df['ipopt_variance'] - attractor_var_phase1) < 1e-4])

    global_stats = dict(
        n_unique_exactK         = len(df),
        frac_exact_shots        = frac_exact,
        requested_shots         = shots,
        observed_shots          = n_samples,
        shots_mismatch          = (shots != n_samples),
        spearman_rho_E_vs_var   = round(rho, 4)  if not np.isnan(rho) else np.nan,
        spearman_p              = round(p_spear, 4) if not np.isnan(p_spear) else np.nan,
        kendall_tau_E_vs_var    = round(tau, 4)  if not np.isnan(tau) else np.nan,
        kendall_p               = round(p_kend, 4) if not np.isnan(p_kend) else np.nan,
        attractor_bits          = attractor_bits,
        attractor_freq          = round(float(attractor_row['freq_qaoa']), 6),
        attractor_var_ipopt     = round(float(attractor_var), 6) if not np.isnan(attractor_var) else np.nan,
        attractor_rank_energy   = int(attractor_row['rank_by_energy']),
        attractor_is_energy_min = attractor_is_emin,
        attractor_coincides_phase1 = (abs(attractor_var - attractor_var_phase1) < 1e-4
                                      if not np.isnan(attractor_var) else False),
        energy_minimum_bits     = energy_min_bits,
        energy_minimum_var      = round(float(df.loc[df['surrogate_energy'].idxmin(),
                                                      'ipopt_variance']), 6)
                                  if not np.isnan(df.loc[df['surrogate_energy'].idxmin(),
                                                          'ipopt_variance']) else np.nan,
        optimal_bits            = optimal_bits_str,
        optimal_var_ipopt       = round(optimal_var, 6),
        optimal_observed_in_qaoa= opt_observed,
        optimal_freq_qaoa       = round(opt_freq_qaoa, 6),
        optimal_rank_by_freq    = opt_rank_freq,
        optimal_rank_by_energy  = int(opt_rank_energy) if not np.isnan(opt_rank_energy) else np.nan,
        n_supports_with_attractor_var = n_same_var,
    )

    if verbose:
        print("\n" + "=" * 70)
        print("QAOA ATTRACTOR DIAGNOSIS")
        print("=" * 70)

        print(f"\n  Observed exact-K supports    : {len(df)}")
        print(f"  Fraction of shots in exact-K : {frac_exact:.4f}")

        print(f"\n  === Attractor support (highest QAOA frequency) ===")
        print(f"  Bits              : {attractor_bits}")
        atr_names = [ASSETS[i] for i, c in enumerate(attractor_bits) if c == '1']
        print(f"  Assets            : {atr_names}")
        print(f"  QAOA frequency    : {attractor_row['freq_qaoa']:.6f}")
        print(f"  Surrogate energy  : {attractor_row['surrogate_energy']:.6f}")
        print(f"  hybrid_score      : {attractor_row['hybrid_score']:.6f}")
        print(f"  mu_score          : {attractor_row['mu_score']:.4f}")
        print(f"  IPOPT variance    : {attractor_var:.6f}")
        print(f"  Rank by energy    : {attractor_row['rank_by_energy']} of {len(df)}")
        print(f"  Is surrogate minimum?: {attractor_is_emin}")

        print(f"\n  === Surrogate minimum ===")
        emin_row = df.loc[df['surrogate_energy'].idxmin()]
        emin_names = [ASSETS[i] for i, c in enumerate(emin_row['bits']) if c == '1']
        print(f"  Bits              : {emin_row['bits']}")
        print(f"  Assets            : {emin_names}")
        print(f"  Surrogate energy  : {emin_row['surrogate_energy']:.6f}")
        print(f"  IPOPT variance    : {emin_row['ipopt_variance']:.6f}")
        print(f"  QAOA frequency    : {emin_row['freq_qaoa']:.6f}")
        print(f"  Rank by frequency : {emin_row['rank_by_freq']}")

        print(f"\n  === True optimal support ===")
        opt_names = [ASSETS[i] for i, c in enumerate(optimal_bits_str) if c == '1']
        print(f"  Bits              : {optimal_bits_str}")
        print(f"  Assets            : {opt_names}")
        print(f"  IPOPT variance    : {optimal_var:.6f}")
        print(f"  Observed by QAOA  : {opt_observed}")
        print(f"  QAOA frequency    : {opt_freq_qaoa:.6f}")
        print(f"  Rank by frequency : {opt_rank_freq}")
        if opt_observed:
            opt_e = df[df['bits'] == optimal_bits_str]['surrogate_energy'].values[0]
            print(f"  Surrogate energy  : {opt_e:.6f}")
            print(f"  Rank by energy    : {opt_rank_energy}")

        print(f"\n  === Surrogate ~ IPOPT correlation ===")
        print(f"  Spearman rho : {rho:.4f}  (p={p_spear:.4f})")
        print(f"  Kendall tau  : {tau:.4f}  (p={p_kend:.4f})")
        interp = ("significant positive correlation - the surrogate "
                  "is a useful proxy of the IPOPT variance"
                  if (not np.isnan(rho) and rho > 0.3 and p_spear < 0.05)
                  else "weak or non-significant correlation - the surrogate "
                       "is a limited proxy of the IPOPT variance")
        print(f"  Interpretation: {interp}")

        print(f"\n  === Top {min(top_n, len(df))} supports by QAOA frequency ===")
        show_cols = ['bits', 'freq_qaoa', 'surrogate_energy',
                     'hybrid_score', 'mu_score', 'ipopt_variance',
                     'rank_by_freq', 'rank_by_energy', 'rank_by_variance']
        top_df = df.sort_values('freq_qaoa', ascending=False).head(top_n)
        print(top_df[show_cols].to_string(index=False))

    return dict(
        support_df    = df.sort_values('freq_qaoa', ascending=False).reset_index(drop=True),
        global_stats  = global_stats,
        attractor_bits= attractor_bits,
        optimal_bits  = optimal_bits_str,
    )


def full_surrogate_vs_ipopt_analysis(
    H_cost: qml.Hamiltonian,
    instance: dict,
    baseline_df: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate the surrogate on ALL feasible exact-K supports.

    Measures the surrogate ~ IPOPT variance correlation independently of the QAOA
    sampling distribution, which separates two questions:
      - Is the surrogate a statistically useful proxy?
      - Is the QAOA problem that the circuit does not minimize the surrogate, or
        that the surrogate does not predict the QP variance well?

    Possible because the classical baseline has already evaluated every support;
    no additional IPOPT calls are needed.
    """
    ASSETS = instance['ASSETS']
    K      = instance['K']
    N      = instance['N']
    alpha  = CONFIG.get('score_alpha', 1.0)
    beta   = CONFIG.get('score_beta',  0.05)

    feasible_df = baseline_df[baseline_df['feasible'] == True].copy()
    print(f"\n[full_surrogate_vs_ipopt_analysis] "
          f"Evaluating the surrogate on {len(feasible_df)} feasible supports ...")

    energies, hybrids, mus, variances, bits_list = [], [], [], [], []

    for _, row in feasible_df.iterrows():
        bits_s = str(row['bits']).zfill(N)
        b      = tuple(int(c) for c in bits_s)
        sel    = {ASSETS[i]: b[i] for i in range(N)}

        energy = eval_hamiltonian_on_bitstring(H_cost, b)
        hy     = hybrid_score_support(sel, instance, alpha, beta)
        mu_sc  = mu_score_support(sel, instance)

        energies.append(energy)
        hybrids.append(hy)
        mus.append(mu_sc)
        variances.append(float(row['variance']))
        bits_list.append(bits_s)

    df_full = pd.DataFrame(dict(
        bits            = bits_list,
        surrogate_energy= energies,
        hybrid_score    = hybrids,
        mu_score        = mus,
        ipopt_variance  = variances,
    ))

    rho_e,  p_e  = spearmanr(df_full['surrogate_energy'], df_full['ipopt_variance'])
    rho_hy, p_hy = spearmanr(df_full['hybrid_score'],     df_full['ipopt_variance'])
    rho_mu, p_mu = spearmanr(df_full['mu_score'],         df_full['ipopt_variance'])

    print(f"\n  Spearman correlations over all feasible supports:")
    print(f"  rho(surrogate_energy, var_IPOPT) = {rho_e:.4f}  (p={p_e:.4e})")
    print(f"  rho(hybrid_score,     var_IPOPT) = {rho_hy:.4f}  (p={p_hy:.4e})")
    print(f"  rho(mu_score,         var_IPOPT) = {rho_mu:.4f}  (p={p_mu:.4e})")

    print(f"\n  Rank of the true optimum by criterion:")
    opt_bits = df_full.loc[df_full['ipopt_variance'].idxmin(), 'bits']
    opt_row  = df_full[df_full['bits'] == opt_bits].iloc[0]
    n_feas   = len(df_full)
    rank_e   = int(df_full['surrogate_energy'].rank().loc[
        df_full['bits'] == opt_bits].values[0])
    rank_hy  = int(df_full['hybrid_score'].rank(ascending=False).loc[
        df_full['bits'] == opt_bits].values[0])
    rank_mu  = int(df_full['mu_score'].rank(ascending=False).loc[
        df_full['bits'] == opt_bits].values[0])
    print(f"  Optimum ({opt_bits}): "
          f"rank_energy={rank_e}/{n_feas} | "
          f"rank_hybrid={rank_hy}/{n_feas} | "
          f"rank_mu={rank_mu}/{n_feas}")

    return df_full


if __name__ == '__main__':

    instance = INSTANCE_15
    set_global_seed(CONFIG['seed'])

    _BASELINE_CSV = "baseline_enumeration.csv"
    if os.path.exists(_BASELINE_CSV):
        print(f"[Phase 2] Loading baseline from {_BASELINE_CSV}")
        df_baseline = pd.read_csv(_BASELINE_CSV, dtype={'bits': str})
        feasible_mask  = df_baseline['feasible'] == True
        _best_var_base = float(df_baseline.loc[feasible_mask, 'variance'].min())
        _n_ipopt_base  = int(df_baseline['method'].eq('ipopt').sum())
        baseline_stats = dict(
            best_var      = _best_var_base,
            n_ipopt_calls = _n_ipopt_base,
            n_feasible    = int(feasible_mask.sum()),
        )
        print(f"  best_var={_best_var_base:.6f} | n_ipopt={_n_ipopt_base}")
    else:
        print("[Phase 2] Running the classical baseline (may take several minutes) ...")
        set_global_seed(CONFIG['seed'])
        df_baseline, baseline_stats = classical_baseline(instance)
        df_baseline.to_csv(_BASELINE_CSV, index=False)
        print(f"  Saved to {_BASELINE_CSV}")

    set_global_seed(CONFIG['seed'])
    H_cost = build_support_surrogate_hamiltonian(
        instance['N'], instance['K'],
        penalty=HAMILTONIAN_PENALTY, instance=instance)
    circuit_exp, circuit_sample = make_circuits(
        instance['N'], CONFIG['qaoa_layers'], H_cost, k=instance['K'],
        shots=CONFIG['qaoa_shots'], pl_seed=CONFIG['seed'])

    print("\n[Phase 2] Training QAOA for the attractor diagnosis ...")
    params_trained = train_qaoa(
        circuit_exp, p=CONFIG['qaoa_layers'],
        n_steps=CONFIG['qaoa_steps'], verbose=True)

    print("\n" + "=" * 70)
    print("STEP 3: QAOA ATTRACTOR DIAGNOSIS")
    print("=" * 70)

    attractor_results = diagnose_qaoa_attractor(
        circuit_sample = circuit_sample,
        params         = params_trained,
        H_cost         = H_cost,
        instance       = instance,
        baseline_df    = df_baseline,
        shots          = CONFIG['qaoa_shots'],
        top_n          = 30,
        verbose        = True,
    )

    attractor_results['support_df'].to_csv("attractor_distribution.csv", index=False)
    print("\n  -> QAOA distribution saved to attractor_distribution.csv")

    print("\n" + "=" * 70)
    print("STEP 4: SURROGATE vs IPOPT CORRELATION (all feasible supports)")
    print("=" * 70)

    df_full_corr = full_surrogate_vs_ipopt_analysis(
        H_cost       = H_cost,
        instance     = instance,
        baseline_df  = df_baseline,
    )
    df_full_corr.to_csv("surrogate_vs_ipopt_full.csv", index=False)
    print("  -> Saved to surrogate_vs_ipopt_full.csv")

    orthogonality_results = check_ranked_mu_orthogonality(
        df_full=df_full_corr,
        instance=instance,
        rho_warning_threshold=0.70,
    )
    orthogonality_results['decile_table'].to_csv(
        "ranked_mu_orthogonality.csv", index=False)
    print("  -> Decile table saved to ranked_mu_orthogonality.csv")

    print("\n" + "=" * 70)
    print("STEP 5: COMPARATIVE MATRIX v2 - 4 policies x 2 samplers")
    print("=" * 70)

    df_matrix_v2, traj_rows_v2 = run_comparative_matrix_v2(
        instance           = instance,
        baseline_stats     = baseline_stats,
        config             = CONFIG,
        seeds              = CONFIG['multiseed_seeds'],
        selection_policies = ['ranked_hybrid', 'ranked_mu',
                              'random_exactK', 'ranked_frequency'],
    )
    df_matrix_v2.to_csv("comparative_matrix_v2.csv", index=False)
    print("\n  -> Saved to comparative_matrix_v2.csv")

    print("\n" + "=" * 70)
    print("STEP 5b: CONVERGENCE CURVES NORMALIZED BY IPOPT CALLS")
    print("=" * 70)

    _max_ipopt = CONFIG['max_iter'] * CONFIG['top_eval_per_iter']
    df_convergence = build_convergence_summary(
        traj_rows=traj_rows_v2,
        max_ipopt=_max_ipopt,
        baseline_var=baseline_stats['best_var'],
    )
    df_convergence.to_csv("convergence_curves.csv", index=False)
    print(f"  -> Saved to convergence_curves.csv  "
          f"({len(df_convergence)} rows, grid 1-{_max_ipopt} IPOPT calls)")
    print(f"  Configurations in curve: "
          f"{df_convergence[['sampler_type','selection_policy']].drop_duplicates().shape[0]}")

    print("\n" + "=" * 70)
    print("COMPARATIVE SUMMARY: hybrid vs mu vs random vs frequency")
    print("=" * 70)

    summary_v2 = (
        df_matrix_v2
        .groupby(['sampler_type', 'selection_policy'])[
            ['best_var', 'gap_pct', 'ipopt_total', 'iter_best']
        ]
        .agg(['mean', 'std'])
    )
    print(summary_v2.to_string())

    print("\n  Separability test of the sampler effect under ranked_mu:")
    qa_rmu = df_matrix_v2.query(
        'sampler_type=="QAOA_surrogate" and selection_policy=="ranked_mu"'
    )['best_var']
    un_rmu = df_matrix_v2.query(
        'sampler_type=="uniform_random" and selection_policy=="ranked_mu"'
    )['best_var']

    if len(qa_rmu) >= 2 and len(un_rmu) >= 2:
        from scipy.stats import mannwhitneyu, ttest_ind
        mw  = mannwhitneyu(qa_rmu, un_rmu, alternative='two-sided')
        tt  = ttest_ind(qa_rmu, un_rmu, equal_var=False)
        print(f"  QAOA+ranked_mu  : mean={qa_rmu.mean():.6f}  std={qa_rmu.std():.6f}")
        print(f"  Unif+ranked_mu  : mean={un_rmu.mean():.6f}  std={un_rmu.std():.6f}")
        print(f"  Mann-Whitney U  : U={mw.statistic:.0f}  p={mw.pvalue:.4f}")
        print(f"  Welch t-test    : t={tt.statistic:.4f}  p={tt.pvalue:.4f}")
        identifiable = mw.pvalue < 0.05 and abs(qa_rmu.mean() - un_rmu.mean()) > 1e-4
        print(f"  Efecto sampler identificable bajo ranked_mu: {identifiable}")
    else:
        print("  Insufficient data for a statistical test.")

    print("\n[Phase 2] Run complete.")
    print("Archivos generados:")
    print("  comparative_matrix_v2.csv      - 4x2 matrix with ranked_frequency")
    print("  attractor_distribution.csv     - QAOA distribution over exact-K")
    print("  surrogate_vs_ipopt_full.csv    - surrogate vs IPOPT correlation")
    print("  ranked_mu_orthogonality.csv    - mu_score deciles vs hybrid_score")
    print("  convergence_curves.csv         - incumbent trajectories by IPOPT calls")
