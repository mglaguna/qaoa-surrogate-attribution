"""
Hybrid classical-quantum workflow for cardinality-constrained portfolio
selection with support-level QAOA sampling and exact continuous refinement.

This script implements an experimental architecture for portfolio selection
under a fixed-cardinality constraint. The goal is not to solve the continuous
Markowitz problem directly on a quantum circuit. Instead, the quantum component
is used as a discrete sampler over candidate supports, and each accepted support
is later evaluated by solving the corresponding continuous convex quadratic
program with Pyomo/IPOPT.

The central methodological distinction is:

  - QAOA operates on binary support variables y_i in {0, 1}.
  - IPOPT operates on continuous portfolio weights x_i >= 0.

Thus, the QAOA layer proposes which assets may enter the portfolio, while the
continuous optimizer determines the best weights over that proposed support.

The pipeline should therefore be interpreted as a hybrid decomposition:

  1. Discrete support generation
     A sampler proposes binary asset-selection vectors.

  2. Classical filtering and candidate selection
     Structural rules and tabu memory remove supports known to be invalid or
     already explored. A classical selection policy decides which remaining
     candidates are worth exact evaluation.

  3. Exact continuous evaluation
     For each selected support, a convex quadratic subproblem is solved exactly
     with IPOPT. Final portfolio quality is judged by this downstream QP result,
     not by the QAOA energy alone.

The QAOA Hamiltonian used in the main surrogate branch is a support-level proxy.
It combines:

  - a strong exact-cardinality penalty,
  - a return signal relative to the target return R,
  - a covariance-based support risk penalty.

This Hamiltonian is intentionally only a surrogate. It is not a reformulation of
the continuous Markowitz problem, because it assumes binary support variables
rather than continuous portfolio weights. The financial interpretation of the
Hamiltonian is therefore limited to prioritizing promising supports. The actual
risk-return feasibility and optimal variance are determined downstream by the
QP solved with IPOPT.

The ansatz is designed to preserve exact cardinality structurally. Instead of
starting from a uniform Hadamard state and using an X mixer, the circuit starts
from a computational basis state with exactly K selected assets and uses an XY
mixer. The XY mixer conserves the number of excitations, so the QAOA evolution
remains inside the exact-K subspace, up to simulator and implementation details.
This makes the cardinality constraint structural rather than merely penalized.

The script also contains an exact-cardinality Hamiltonian builder. Under the
exact-K initial state and XY mixer, a pure cardinality Hamiltonian is constant
over the reachable subspace. For that reason, it should not be interpreted as a
genuine informative QAOA objective. It is kept only as a documented reference
for future architectural comparisons, not as a central ablation baseline.

The hybrid loop is sampler-agnostic. The sampler is responsible only for
generating an observed candidate pool and associated sampling metadata. The
classical loop then applies tabu filtering, imposes the top-k exploration budget,
and selects candidates for IPOPT according to an explicit selection policy.

This separation is important for causal attribution:

  - sampler_type controls how the candidate pool is generated;
  - selection_policy controls how candidates from that pool are prioritized;
  - top_k_candidates controls the downstream exploration budget after tabu.

As a result, comparisons such as:

  - QAOA_surrogate + ranked_hybrid,
  - QAOA_surrogate + random_exactK,
  - uniform_random + ranked_hybrid,
  - uniform_random + random_exactK,

can be interpreted as attempts to separate the effect of the QAOA sampling
distribution from the effect of the classical ranking heuristic.

The ranking scores used in the classical layer are support-level heuristics.
They do not replace the continuous QP objective. They are used only to decide
which sampled and non-tabu supports should be evaluated first. The final reported
quality of a candidate is always based on the IPOPT solution of the continuous
subproblem.

The script distinguishes two notions of cardinality:

  - structural or selected cardinality:
      the number of assets allowed by the binary support y;

  - effective realized cardinality:
      the number of assets that receive strictly positive continuous weight
      after solving the QP.

By default, the QP enforces x_i <= y_i and x_i >= 0. This means that a selected
asset may still receive zero weight in the continuous optimum. The optional
parameter min_weight_selected can be used to force selected assets to receive a
minimum positive weight, but this is not enabled globally because it changes
feasibility and the economic meaning of the continuous refinement.

The structural tabu mechanism is used to avoid repeated or provably unproductive
exact solves. It stores exact no-good supports and structural rules discovered
through infeasibility diagnostics. Its purpose is to reduce repeated IPOPT calls,
not to change the definition of the optimization problem.

The classical exhaustive baseline enumerates all exact-K supports and solves
the corresponding continuous QP whenever the support passes structural
prescreening. This baseline provides the reference optimum and the reference
number of IPOPT calls used to measure hybrid savings and optimality gaps.

The uniform random sampler is a critical baseline. It uses the same hybrid loop,
the same tabu logic, the same top-k budget, and the same selection policies as
the QAOA sampler. Therefore, if QAOA and uniform sampling perform similarly, the
evidence favors the interpretation that the classical filtering and ranking
components, rather than the quantum sampler, explain the observed performance.

The function characterize_qaoa_distribution is diagnostic only. It measures
properties such as entropy, concentration, and distance from a uniform
distribution over exact-K supports. These statistics help determine whether QAOA
is producing a distribution distinguishable from uniform sampling, but they are
not the main success criterion. A distinguishable distribution is useful only if
it leads to better downstream QP outcomes.

The comparative matrix is the main experimental structure for attribution. It
crosses sampler type with classical selection policy. In the default scientific
configuration, the matrix focuses on:

  - QAOA_surrogate,
  - uniform_random,

crossed with:

  - ranked_hybrid,
  - random_exactK.

The random_exactK policy is included specifically to reduce confounding between
QAOA distributional bias and the classical ranking heuristic. It tests whether
the sampler itself produces useful candidate pools when the classical policy is
blind to the financial ranking score.

Reproducibility is handled through explicit seeding where supported. The numpy
global seed controls Python-side randomness such as parameter initialization and
uniform candidate generation. PennyLane shot-based sampling may require explicit
device-level seed support depending on the PennyLane version. The script avoids
re-seeding numpy inside the sampling path because doing so can make repeated
hybrid iterations artificially deterministic and invalidate the exploration
narrative.

Methodological interpretation:

  - The method is not an exact quantum solution to Markowitz.
  - The QAOA cost is a discrete support surrogate.
  - The continuous QP remains the authoritative evaluator.
  - The hybrid advantage, if any, must be demonstrated empirically against
    uniform random sampling under the same classical loop.
  - The main claim should be framed as controlled support generation plus
    classical exact refinement, not as end-to-end quantum portfolio optimization.

Main outputs of interest include:

  - best variance found by the hybrid loop,
  - percentage gap relative to the exhaustive classical baseline,
  - number of IPOPT calls used,
  - number of IPOPT calls avoided through structural filtering,
  - iteration at which the best solution was found,
  - number of feasible supports evaluated,
  - active_count versus selected_count for the final continuous portfolio,
  - comparative matrix results by sampler_type and selection_policy.

Recommended execution order:

  1. Generate or load the portfolio instance.
  2. Build and train the QAOA surrogate sampler.
  3. Optionally characterize the QAOA distribution as an auxiliary diagnostic.
  4. Run the exhaustive classical baseline.
  5. Run the hybrid loop with QAOA_surrogate.
  6. Run the hybrid loop with uniform_random.
  7. Run the comparative sampler_type x selection_policy matrix.
  8. Interpret results using IPOPT-refined variance and baseline gap, not QAOA
     energy alone.

Limitations to keep explicit:

  - The support surrogate ignores continuous weight optimization during QAOA.
  - Equal-support risk and return signals may not predict the optimal weighted
    portfolio after IPOPT.
  - Dense covariance matrices create dense ZZ Hamiltonians, increasing circuit
    simulation cost.
  - The XY mixer preserves cardinality but may restrict exploration depending on
    topology and circuit depth.
  - Shot noise and finite sampling can dominate candidate diversity.
  - The classical ranking policy can strongly influence performance, which is
    why the comparative matrix separates sampler effects from ranking effects.
  - Results on synthetic instances do not automatically generalize to real
    market data without additional validation.

This section defines the intended scope of the script: a conservative,
experimentally controlled hybrid architecture for studying whether a QAOA
support-surrogate sampler adds value over uniform random support generation when
both are embedded in the same classical tabu-filtered, IPOPT-refined pipeline.
"""


import time
import itertools
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pennylane as qml
from pennylane import numpy as pnp
from math import comb, log
from typing import Callable, Optional

GLOBAL_SEED = 42


def set_global_seed(seed: int):
    np.random.seed(seed)


set_global_seed(GLOBAL_SEED)

CONFIG = {
    "seed"               : GLOBAL_SEED,
    "qaoa_layers"        : 1,
    "qaoa_steps"         : 100,
    "qaoa_shots"         : 5000,
    "top_k_candidates"   : 40,
    "top_eval_per_iter"  : 2,
    "max_iter"           : 30,
    "score_alpha"        : 1.0,
    "score_beta"         : 0.05,
    "multiseed_seeds"    : list(range(10)),
}

HAMILTONIAN_PENALTY = 1.0

SOLVER_NAME = "ipopt"
SOLVER = SolverFactory(SOLVER_NAME)

if not SOLVER.available():
    raise RuntimeError(
        f"Solver '{SOLVER_NAME}' no disponible.\n"
        f"conda: conda install -c conda-forge ipopt\n"
        f"pip:   pip install idaes-pse && idaes get-extensions"
    )

SOLVER_OPTIONS: dict = {}
for opt_key, opt_val in SOLVER_OPTIONS.items():
    SOLVER.options[opt_key] = opt_val


def generate_synthetic_instance(N_assets: int = 15,
                                 K_card: int = 5,
                                 R_target: float = 13.0,
                                 seed: int = GLOBAL_SEED) -> dict:
    """Generate a parameterized synthetic portfolio instance.

    For N = 15: 5 sectors x 3 assets (original structure).
    For N > 15: sectors are extended uniformly.

    Parameters
    ----------
    N_assets  : number of assets in the universe
    K_card    : exact portfolio cardinality
    R_target  : minimum required return
    seed      : seed for the generation of returns and volatilities

    Returns
    -------
    dict with: ASSETS, SECTORS, SECTOR_OF, N, K, R, mu, sigma, cov,
               HIGH_RETURN_ASSETS, N_HIGH, N_TOTAL_COMBOS,
               N_INFEASIBLE_FAMILY, INFEASIBLE_SHARE
    """
    rng = np.random.RandomState(seed)

    sector_names = ['tech', 'healthcare', 'financial', 'energy', 'consumer',
                    'materials', 'utilities', 'industrials', 'realestate', 'comm']
    n_sectors = min(len(sector_names), max(5, N_assets // 3))
    base_per_sector = N_assets // n_sectors
    remainder = N_assets % n_sectors

    sectors = {}
    for k_s, sname in enumerate(sector_names[:n_sectors]):
        n_in = base_per_sector + (1 if k_s < remainder else 0)
        sectors[sname] = [f"{sname}_{j}" for j in range(n_in)]

    assets = [a for v in sectors.values() for a in v]
    assert len(assets) == N_assets
    sector_of = {a: s for s, v in sectors.items() for a in v}
    asset_idx = {a: i for i, a in enumerate(assets)}

    mu_vals = {}
    for sname, sassets in sectors.items():
        base = rng.uniform(7, 14)
        for a in sassets:
            mu_vals[a] = round(float(np.clip(base + rng.normal(0, 1.5), 5, 16)), 1)
    mu = pd.Series(mu_vals, index=assets)

    sigma_vals = {}
    for a in assets:
        sigma_vals[a] = round(float(np.clip(rng.uniform(1.5, 6.0), 1.0, 7.0)), 1)
    sigma = pd.Series(sigma_vals, index=assets)

    INTRA = 0.60
    INTER = 0.15
    corr = np.full((N_assets, N_assets), INTER)
    for sassets in sectors.values():
        idx = [asset_idx[a] for a in sassets]
        for i in idx:
            for j in idx:
                if i != j:
                    corr[i, j] = INTRA
    np.fill_diagonal(corr, 1.0)

    cov_np = np.outer(sigma.values, sigma.values) * corr
    cov = pd.DataFrame(cov_np, index=assets, columns=assets)

    eigvals = np.linalg.eigvalsh(cov_np)
    if eigvals.min() < -1e-10:
        raise ValueError(
            f"Covariance matrix has negative eigenvalue ({eigvals.min():.2e})."
        )

    high_return_assets = [a for a in assets if mu[a] >= R_target]
    N_high = len(high_return_assets)
    N_total = comb(N_assets, K_card)
    if N_assets - N_high >= K_card:
        N_infeasible = comb(N_assets - N_high, K_card)
    else:
        N_infeasible = 0
    infeasible_share = N_infeasible / N_total if N_total > 0 else 0.0

    print(f"\n[Instancia N={N_assets}, K={K_card}, R={R_target}]")
    print(f"  Assets with mu >= R={R_target}: {high_return_assets} ({N_high} of {N_assets})")
    print(f"  return_support_ceiling excludes: "
          f"C({N_assets-N_high},{K_card}) = {N_infeasible} of {N_total} "
          f"({infeasible_share:.2%})")

    return dict(
        ASSETS=assets, SECTORS=sectors, SECTOR_OF=sector_of,
        ASSET_IDX=asset_idx,
        N=N_assets, K=K_card, R=R_target,
        mu=mu, sigma=sigma, cov=cov,
        HIGH_RETURN_ASSETS=high_return_assets,
        N_HIGH=N_high,
        N_TOTAL_COMBOS=N_total,
        N_INFEASIBLE_FAMILY=N_infeasible,
        INFEASIBLE_SHARE=infeasible_share,
    )


SECTORS_15 = {
    'tech'      : ['tech_growth', 'tech_value',  'tech_semis'],
    'healthcare': ['pharma',      'biotech',      'medtech'   ],
    'financial' : ['banking',     'insurance',    'fintech'   ],
    'energy'    : ['oil_gas',     'renewables',   'utilities' ],
    'consumer'  : ['cons_disc',   'cons_stap',    'retail'    ],
}

ASSETS_15 = [a for v in SECTORS_15.values() for a in v]
SECTOR_OF_15 = {a: s for s, v in SECTORS_15.items() for a in v}
N_15 = len(ASSETS_15)
ASSET_IDX_15 = {a: i for i, a in enumerate(ASSETS_15)}

RETURNS_DICT_15 = {
    'tech_growth': 14.0, 'tech_value': 12.0, 'tech_semis': 13.0,
    'pharma'     : 11.0, 'biotech'   : 13.0, 'medtech'   : 10.0,
    'banking'    :  9.0, 'insurance' :  8.0, 'fintech'   : 10.0,
    'oil_gas'    :  7.0, 'renewables':  9.0, 'utilities' :  6.0,
    'cons_disc'  :  8.0, 'cons_stap' :  6.0, 'retail'    :  7.0,
}
VOLATILITY_DICT_15 = {
    'tech_growth': 5.5, 'tech_value': 4.5, 'tech_semis': 5.0,
    'pharma'     : 3.5, 'biotech'   : 5.0, 'medtech'   : 3.0,
    'banking'    : 3.0, 'insurance' : 2.5, 'fintech'   : 4.0,
    'oil_gas'    : 4.5, 'renewables': 4.0, 'utilities' : 2.5,
    'cons_disc'  : 3.0, 'cons_stap' : 1.8, 'retail'    : 2.8,
}

mu_15 = pd.Series(RETURNS_DICT_15, index=ASSETS_15)
sigma_15 = pd.Series(VOLATILITY_DICT_15, index=ASSETS_15)

_INTRA_15, _INTER_15 = 0.60, 0.15
corr_15 = np.full((N_15, N_15), _INTER_15)
for _sassets in SECTORS_15.values():
    _idx = [ASSET_IDX_15[a] for a in _sassets]
    for _i in _idx:
        for _j in _idx:
            if _i != _j:
                corr_15[_i, _j] = _INTRA_15
np.fill_diagonal(corr_15, 1.0)

cov_np_15 = np.outer(sigma_15.values, sigma_15.values) * corr_15
cov_15 = pd.DataFrame(cov_np_15, index=ASSETS_15, columns=ASSETS_15)

R_15 = 13.0
K_15 = 5

HIGH_RETURN_ASSETS_15 = [a for a in ASSETS_15 if mu_15[a] >= R_15]
N_HIGH_15 = len(HIGH_RETURN_ASSETS_15)
N_TOTAL_COMBOS_15 = comb(N_15, K_15)
N_INFEASIBLE_FAMILY_15 = comb(N_15 - N_HIGH_15, K_15) \
    if N_15 - N_HIGH_15 >= K_15 else 0
INFEASIBLE_SHARE_15 = N_INFEASIBLE_FAMILY_15 / N_TOTAL_COMBOS_15

print(f"[Instancia fija N=15, K=5, R=13]")
print(f"  Assets with mu >= R=13: {HIGH_RETURN_ASSETS_15} ({N_HIGH_15} of {N_15})")
print(f"  return_support_ceiling excludes: "
      f"C({N_15-N_HIGH_15},{K_15}) = {N_INFEASIBLE_FAMILY_15} of {N_TOTAL_COMBOS_15} "
      f"({INFEASIBLE_SHARE_15:.2%})")

INSTANCE_15 = dict(
    ASSETS=ASSETS_15, SECTORS=SECTORS_15, SECTOR_OF=SECTOR_OF_15,
    ASSET_IDX=ASSET_IDX_15,
    N=N_15, K=K_15, R=R_15,
    mu=mu_15, sigma=sigma_15, cov=cov_15,
    HIGH_RETURN_ASSETS=HIGH_RETURN_ASSETS_15,
    N_HIGH=N_HIGH_15,
    N_TOTAL_COMBOS=N_TOTAL_COMBOS_15,
    N_INFEASIBLE_FAMILY=N_INFEASIBLE_FAMILY_15,
    INFEASIBLE_SHARE=INFEASIBLE_SHARE_15,
)


def solve_subproblem(
    selection: dict,
    instance: dict,
    tee: bool = False,
    min_weight_selected: float = 0.0,
    active_tol: float = 1e-8,
) -> dict:
    """Solve the continuous QP for a fixed binary support.

    Distinguishes the exact cardinality of the set allowed by `selection` from the
    effective cardinality of the support realized by the weights. With
    min_weight_selected > 0.0, every selected asset receives a minimum positive
    weight if the problem remains feasible.

    min  x'Σx
    s.t. Σ x_i = 1
         Σ mu_i x_i >= R
         x_i <= y_i
         x_i >= min_weight_selected * y_i
         x_i >= 0

    Parameters
    ----------
    selection : {asset: 0/1}; must satisfy the exact cardinality K.
    instance  : instance dict (output of generate_synthetic_instance).
    tee       : show IPOPT output.
    min_weight_selected : minimum weight for selected assets.
    active_tol : tolerance used to count effectively active assets.

    Returns
    -------
    dict with feasibility, variance, weights and effective-support metrics.
    """
    ASSETS = instance['ASSETS']
    K      = instance['K']
    R      = instance['R']
    mu     = instance['mu']
    cov    = instance['cov']

    selected = [i for i in ASSETS if selection[i] == 1]
    assert len(selected) == K, (
        f"solve_subproblem: expected K={K} assets, received "
        f"{len(selected)}: {selected}"
    )

    if min_weight_selected < 0:
        raise ValueError("min_weight_selected must be >= 0.")

    if K * min_weight_selected > 1.0 + 1e-12:
        raise ValueError(
            f"Configuracion imposible: K * min_weight_selected = "
            f"{K * min_weight_selected:.6f} > 1."
        )

    m      = pyo.ConcreteModel()
    m.x    = pyo.Var(ASSETS, within=pyo.NonNegativeReals)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    m.full_investment = pyo.Constraint(
        expr=sum(m.x[i] for i in ASSETS) == 1)
    m.min_return = pyo.Constraint(
        expr=sum(mu[i] * m.x[i] for i in ASSETS) >= R)
    m.link_upper = pyo.ConstraintList()
    m.link_lower = pyo.ConstraintList()
    for i in ASSETS:
        y_i = selection[i]
        m.link_upper.add(m.x[i] <= y_i)
        if min_weight_selected > 0.0:
            m.link_lower.add(m.x[i] >= min_weight_selected * y_i)

    m.obj = pyo.Objective(
        expr=sum(
            m.x[i] * sum(cov.loc[i, j] * m.x[j] for j in ASSETS)
            for i in ASSETS),
        sense=pyo.minimize)

    res = SOLVER.solve(m, tee=tee)
    tc  = getattr(res.solver, 'termination_condition', None)

    if tc != pyo.TerminationCondition.optimal:
        return dict(feasible=False, variance=None, weights=None,
                    ret_realized=None, ret_slack=None, tc=str(tc),
                    active_assets=None, active_count=None, selected_count=K)

    weights      = {i: float(pyo.value(m.x[i])) for i in ASSETS}
    variance     = float(pyo.value(m.obj))
    ret_realized = float(sum(mu[i] * weights[i] for i in ASSETS))
    ret_slack    = ret_realized - R

    active_assets = [i for i in ASSETS if weights[i] > active_tol]
    active_count = len(active_assets)

    return dict(feasible=True, variance=variance, weights=weights,
                ret_realized=ret_realized, ret_slack=ret_slack, tc=str(tc),
                active_assets=active_assets, active_count=active_count,
                selected_count=K)


class DiscardReason:
    NO_GOOD_EXACT = "no_good_exact"
    STRUCT_RULE   = "structural_rule"
    NON_EXACT_K   = "non_exact_K"


class StructuralTabu:
    """Classical memory layer over candidates generated by any sampler.

    Level 1 - exact no-good: excludes bitstrings that were already evaluated.
    Level 2 - structural rules: exclude combinatorial families before the solver is called.
    """
    def __init__(self):
        self.visited_exact: set     = set()
        self.structural_rules: list = []
        self.rule_labels: list      = []
        self.stats = dict(
            no_good_hits           = 0,
            structural_hits_known  = 0,
            structural_discoveries = 0,
            non_exact_K            = 0,
            ipopt_avoided_struct   = 0,
            ipopt_avoided_prescreen= 0,
        )

    def register_visit(self, bits: str):
        self.visited_exact.add(bits)

    def add_rule(self, predicate, label: str):
        self.structural_rules.append(predicate)
        self.rule_labels.append(label)
        print(f"  [StructuralTabu] New rule: {label}")

    def classify(self, selection: dict, bits: str,
                 instance: dict) -> tuple:
        ASSETS = instance['ASSETS']
        K      = instance['K']
        card = sum(selection[i] for i in ASSETS)
        if card != K:
            self.stats['non_exact_K'] += 1
            return True, DiscardReason.NON_EXACT_K, None
        if bits in self.visited_exact:
            self.stats['no_good_hits'] += 1
            return True, DiscardReason.NO_GOOD_EXACT, None
        for pred, label in zip(self.structural_rules, self.rule_labels):
            if pred(selection):
                self.stats['structural_hits_known'] += 1
                self.stats['ipopt_avoided_struct']  += 1
                return True, DiscardReason.STRUCT_RULE, label
        return False, None, None

    def filter_and_classify(self, candidates: list,
                            instance: dict) -> tuple:
        valid, discarded = [], []
        for sel, freq, r_score, risk_score, hybrid_score in candidates:
            bits = _to_bits(sel, instance)
            is_tabu, reason, rule = self.classify(sel, bits, instance)
            if is_tabu:
                discarded.append((sel, freq, reason, rule))
            else:
                valid.append((sel, freq, r_score, risk_score, hybrid_score))
        return valid, discarded

    def n_exact(self):      return len(self.visited_exact)
    def n_structural(self): return len(self.structural_rules)
    def ipopt_avoided_total(self):
        return (self.stats['ipopt_avoided_struct']
                + self.stats['ipopt_avoided_prescreen'])


def diagnose_infeasibility(selection: dict, instance: dict) -> str:
    """Check necessary feasibility conditions WITHOUT calling IPOPT.
    Returns 'empty_selection', 'return_support_ceiling' or 'unknown'.
    """
    ASSETS = instance['ASSETS']
    R      = instance['R']
    mu     = instance['mu']
    selected = [i for i in ASSETS if selection[i] == 1]
    if not selected:
        return 'empty_selection'
    if max(mu[i] for i in selected) < R:
        return 'return_support_ceiling'
    return 'unknown'


def infer_and_register_rule(tabu: StructuralTabu, diagnosis: str,
                             instance: dict) -> bool:
    """Derive and register a structural rule from a diagnosis.
    Returns True if a new rule was registered.
    """
    ASSETS = instance['ASSETS']
    R      = instance['R']
    mu     = instance['mu']
    N      = instance['N']
    K      = instance['K']
    N_HIGH = instance['N_HIGH']
    N_INFEASIBLE_FAMILY = instance['N_INFEASIBLE_FAMILY']
    INFEASIBLE_SHARE    = instance['INFEASIBLE_SHARE']

    if diagnosis == 'return_support_ceiling':
        tag = 'return_support_ceiling'
        if any(tag in lbl for lbl in tabu.rule_labels):
            return False
        threshold = R
        rule = lambda s, t=threshold, A=ASSETS, m=mu: (
            sum(s[i] for i in A) > 0
            and max((m[i] for i in A if s[i] == 1), default=0) < t
        )
        tabu.add_rule(
            rule,
            f"return_support_ceiling(R={threshold}): "
            f"excludes C({N-N_HIGH},{K})={N_INFEASIBLE_FAMILY} supports "
            f"({INFEASIBLE_SHARE:.2%} of the search space)"
        )
        return True

    if diagnosis == 'empty_selection':
        tag = 'empty_selection'
        if any(tag in lbl for lbl in tabu.rule_labels):
            return False
        tabu.add_rule(
            lambda s, A=ASSETS: sum(s[i] for i in A) == 0,
            "empty_selection"
        )
        return True

    return False


def support_risk_score(sel: dict, instance: dict) -> float:
    """Risk proxy: mean covariance between pairs of the support.
    It is not the exact variance (weights are unknown); it is a cheap signal for
    ranking before the solver is called.
    """
    ASSETS = instance['ASSETS']
    K      = instance['K']
    cov    = instance['cov']
    chosen = [a for a in ASSETS if sel[a] == 1]
    if len(chosen) != K:
        return 0.0
    return float(cov.loc[chosen, chosen].values.mean())


def build_candidate_pool(raw_counts: dict, shots: int, instance: dict,
                         alpha: float, beta: float) -> list:
    """Build a structured candidate pool from sampler counts.

    Computes r_score, risk_score and hybrid_score with the same proxy structure as
    the V3 Hamiltonian:
      - useful return truncated with respect to R,
      - support risk based on the normalized covariance.

    Candidates are neither sorted nor truncated; only the pool fields are computed.

    Parameters
    ----------
    raw_counts : {bitstring_tuple: count}
    shots      : denominator used to normalize frequencies. For QAOA it must equal
                 the actual number of observed samples.

    Returns
    -------
    candidate_pool : list of dicts with selection, bits, freq, cardinality,
                     is_exactK, r_score, risk_score and hybrid_score.
    """
    ASSETS = instance['ASSETS']
    K      = instance['K']
    mu     = instance['mu']
    cov    = instance['cov']
    R      = instance['R']
    N      = instance['N']

    mu_vals = mu.loc[ASSETS].astype(float).values
    mu_signal = np.maximum(mu_vals - R, 0.0)
    mu_scale = float(np.max(np.abs(mu_signal)))
    if mu_scale > 1e-12:
        mu_hat = mu_signal / mu_scale
    else:
        mu_hat = np.zeros_like(mu_signal)

    cov_vals = cov.loc[ASSETS, ASSETS].astype(float).values
    cov_scale = float(np.max(np.abs(cov_vals)))
    if cov_scale > 1e-12:
        cov_hat = cov_vals / cov_scale
    else:
        cov_hat = np.zeros_like(cov_vals)

    candidate_pool = []

    for b, count in raw_counts.items():
        freq = count / shots
        sel  = {ASSETS[i]: b[i] for i in range(N)}
        card = sum(b)
        is_exactK = (card == K)

        if is_exactK:
            chosen_idx = [i for i in range(N) if b[i] == 1]
            r_score = float(sum(mu_hat[i] for i in chosen_idx))
            risk_score = float(
                sum(cov_hat[i, i] for i in chosen_idx)
                + 2.0 * sum(
                    cov_hat[i, j]
                    for pos, i in enumerate(chosen_idx)
                    for j in chosen_idx[pos + 1:]
                )
            )
            hybrid_score = alpha * r_score - beta * risk_score
        else:
            r_score = risk_score = hybrid_score = 0.0

        candidate_pool.append(dict(
            selection=sel,
            bits=b,
            count=count,
            freq=freq,
            cardinality=card,
            is_exactK=is_exactK,
            r_score=r_score,
            risk_score=risk_score,
            hybrid_score=hybrid_score,
        ))

    return candidate_pool


def rank_candidate_pool(candidate_pool: list, instance: dict,
                        policy: str = "ranked_hybrid") -> list:
    """Sort the candidate pool:
      1. exact-K candidates first,
      2. hybrid_score descending,
      3. frequency descending as tie-breaker.
    """
    if policy != "ranked_hybrid":
        raise ValueError(f"Ranking policy no soportada: {policy}")

    return sorted(candidate_pool, key=lambda c: (
        0 if c['is_exactK'] else 1,
        -c['hybrid_score'],
        -c['freq']
    ))


def prepare_top_candidates(raw_counts: dict, shots: int, instance: dict,
                           top_k: int, alpha: float,
                           beta: float) -> tuple:
    """Prepare the output expected by score_candidates.

    Uses build_candidate_pool and rank_candidate_pool and returns exactly the
    structure expected by the rest of the script.
    """
    candidate_pool = build_candidate_pool(raw_counts, shots, instance,
                                          alpha, beta)
    results = rank_candidate_pool(candidate_pool, instance)

    meta = candidate_pool_metadata(results, shots)

    top_results = [
        (c['selection'], c['freq'], c['r_score'],
         c['risk_score'], c['hybrid_score'])
        for c in results[:top_k]
    ]

    return top_results, meta


def candidate_pool_metadata(candidate_pool: list, shots: int) -> dict:
    """Compute the aggregate metrics previously produced by score_candidates."""
    n_unique_raw   = len(candidate_pool)
    n_exactK_raw   = sum(1 for c in candidate_pool if c['is_exactK'])
    n_nonexact_raw = n_unique_raw - n_exactK_raw
    total_exactK_shots = sum(c['count'] for c in candidate_pool
                             if c['is_exactK'])
    exact_probs = np.array(
        [c['count'] for c in candidate_pool if c['is_exactK']],
        dtype=float
    )
    if total_exactK_shots > 0 and len(exact_probs) > 0:
        exact_probs = exact_probs / total_exactK_shots
        entropy_exactK = float(-np.sum(exact_probs * np.log(exact_probs + 1e-15)))
        top1_exactK_freq = float(np.max(exact_probs))
    else:
        entropy_exactK = 0.0
        top1_exactK_freq = 0.0

    return dict(
        n_unique_raw      = n_unique_raw,
        n_exactK_raw      = n_exactK_raw,
        n_nonexact_raw    = n_nonexact_raw,
        n_unique_exactK_raw = n_exactK_raw,
        frac_exactK_shots = total_exactK_shots / shots,
        entropy_exactK    = entropy_exactK,
        top1_exactK_freq  = top1_exactK_freq,
    )


def score_candidates(raw_counts: dict, shots: int, instance: dict,
                     top_k: int, alpha: float,
                     beta: float) -> tuple:
    """Scoring function shared by the samplers.

    Compatibility wrapper that keeps the original output by delegating to
    prepare_top_candidates.
    """
    return prepare_top_candidates(raw_counts, shots, instance,
                                  top_k, alpha, beta)


def build_exactK_hamiltonian(n: int, k: int,
                             penalty: float = 1.0) -> qml.Hamiltonian:
    """Pure exact-cardinality Hamiltonian.

    Encodes only

        penalty * (sum_i y_i - k)^2,   with y_i = (1 - Z_i) / 2.

    Methodological role:
      - Cardinality-only cost model for architectural comparison.
      - Under an exact-K BasisState initialization and an XY mixer the evolution is
        restricted to the exact-K subspace, where this term is constant and therefore
        induces no preference among supports.
      - Kept explicitly for ablation experiments against the support surrogate.
    """
    a = penalty * (1 - 2 * k)
    b = 2.0 * penalty
    h = -a / 2.0 - (n - 1) * b / 4.0
    J = b / 4.0

    ops, coeffs = [], []
    for i in range(n):
        ops.append(qml.PauliZ(wires=i))
        coeffs.append(float(h))
    for i in range(n):
        for j in range(i + 1, n):
            ops.append(qml.PauliZ(wires=i) @ qml.PauliZ(wires=j))
            coeffs.append(float(J))

    return qml.Hamiltonian(coeffs, ops)


def build_support_surrogate_hamiltonian(
    n: int,
    k: int,
    penalty: float,
    instance: dict,
    lambda_ret: float = 1.0,
    lambda_risk: float = 1.0,
    use_truncated_return: bool = True,
) -> qml.Hamiltonian:
    """Quadratic Hamiltonian for support prioritization.

    E(y) =
        lambda_card * (sum_i y_i - k)^2
        - lambda_ret  * sum_i mu_hat_i y_i
        + lambda_risk * [sum_i Sigma_hat_ii y_i
                         + 2 sum_{i<j} Sigma_hat_ij y_i y_j]

    where y_i in {0,1} indicates selection of asset i.

    Interpretation:
      - The first term imposes exact cardinality.
      - The second rewards supports with useful return content relative to the target R.
      - The third penalizes the risk density of the support.

    Important:
      - This function does NOT represent the continuous subproblem exactly.
      - It is a support surrogate and must be validated empirically against the
        downstream optimal QP variance.
    """
    if instance is None:
        raise ValueError("instance is required and must contain mu, cov and R.")

    ASSETS = instance["ASSETS"]
    mu = instance["mu"].loc[ASSETS].astype(float).values
    cov = instance["cov"].loc[ASSETS, ASSETS].astype(float).values
    R = float(instance["R"])

    if len(ASSETS) != n:
        raise ValueError(
            f"Inconsistencia: n={n}, pero instance['ASSETS'] tiene {len(ASSETS)} assets."
        )

    mu_shift = mu - R
    if use_truncated_return:
        mu_signal = np.maximum(mu_shift, 0.0)
    else:
        mu_signal = mu_shift

    mu_scale = float(np.max(np.abs(mu_signal)))
    if mu_scale > 1e-12:
        mu_hat = mu_signal / mu_scale
    else:
        mu_hat = np.zeros_like(mu_signal)

    cov_scale = float(np.max(np.abs(cov)))
    if cov_scale > 1e-12:
        cov_hat = cov / cov_scale
    else:
        cov_hat = np.zeros_like(cov)

    z_coeff = np.zeros(n, dtype=float)
    zz_coeff = {}

    lambda_card = penalty

    a = lambda_card * (1 - 2 * k)
    b = 2.0 * lambda_card
    h_card = -a / 2.0 - (n - 1) * b / 4.0
    j_card = b / 4.0

    for i in range(n):
        z_coeff[i] += h_card

    for i in range(n):
        for j in range(i + 1, n):
            zz_coeff[(i, j)] = zz_coeff.get((i, j), 0.0) + j_card

    for i in range(n):
        c_ret = -lambda_ret * mu_hat[i]
        z_coeff[i] += -c_ret / 2.0

    for i in range(n):
        c_diag = lambda_risk * cov_hat[i, i]
        z_coeff[i] += -c_diag / 2.0

    for i in range(n):
        for j in range(i + 1, n):
            c_pair = 2.0 * lambda_risk * cov_hat[i, j]
            z_coeff[i] += -c_pair / 4.0
            z_coeff[j] += -c_pair / 4.0
            zz_coeff[(i, j)] = zz_coeff.get((i, j), 0.0) + c_pair / 4.0

    ops, coeffs = [], []

    for i in range(n):
        if abs(z_coeff[i]) > 1e-12:
            ops.append(qml.PauliZ(wires=i))
            coeffs.append(float(z_coeff[i]))

    for (i, j), coeff in zz_coeff.items():
        if abs(coeff) > 1e-12:
            ops.append(qml.PauliZ(wires=i) @ qml.PauliZ(wires=j))
            coeffs.append(float(coeff))

    return qml.Hamiltonian(coeffs, ops)


def make_circuits(
    n: int,
    p: int,
    H_cost: qml.Hamiltonian,
    k: int,
    shots: int = 5000,
    mixer_topology: str = "ring",
    pl_seed: Optional[int] = None,
):
    """Build QAOA circuits with a structural exact-cardinality constraint.

    Strategy:
      - Initial state: computational bitstring with exactly k ones.
      - Mixer: XY Hamiltonian that conserves the number of excitations.
      - Consequence: if the initial state has exact cardinality k, the whole
        evolution remains in the exact-k subspace.

    Parameters
    ----------
    n : int
        Number of qubits / assets.
    p : int
        Number of QAOA layers.
    H_cost : qml.Hamiltonian
        Cost Hamiltonian defined on binary variables.
    k : int
        Exact support cardinality.
    shots : int
        Number of shots of the sampling circuit.
    mixer_topology : str
        "ring" or "line". Defines the pairs of the XY mixer.
    pl_seed : Optional[int]
        Explicit seed for the PennyLane devices. np.random.seed does not
        necessarily control the internal PRNG of shot-based measurements.

    Returns
    -------
    circuit_exp, circuit_sample
    """
    if not (0 <= k <= n):
        raise ValueError(f"k must satisfy 0 <= k <= n; received k={k}, n={n}")

    init_state = np.array([1] * k + [0] * (n - k), dtype=int)

    if mixer_topology == "ring":
        pairs = [(i, (i + 1) % n) for i in range(n)]
    elif mixer_topology == "line":
        pairs = [(i, i + 1) for i in range(n - 1)]
    else:
        raise ValueError("mixer_topology must be 'ring' or 'line'")

    coeffs_mix = []
    ops_mix = []

    for i, j in pairs:
        coeffs_mix.extend([1.0, 1.0])
        ops_mix.append(qml.PauliX(wires=i) @ qml.PauliX(wires=j))
        ops_mix.append(qml.PauliY(wires=i) @ qml.PauliY(wires=j))

    H_mix = qml.Hamiltonian(coeffs_mix, ops_mix)

    try:
        dev_exp = qml.device("default.qubit", wires=n, seed=pl_seed)
        dev_sample = qml.device("default.qubit", wires=n, shots=shots,
                                seed=pl_seed)
    except TypeError as e:
        raise RuntimeError(
            "This script requires a PennyLane version/device that supports "
            "explicit seed=... in qml.device for shot-based sampling. "
            "Fallback reseeding with numpy was intentionally removed because "
            "it can make hybrid-loop sampling artificially deterministic "
            "across iterations."
        ) from e

    @qml.qnode(dev_exp, interface="autograd")
    def circuit_exp(params):
        qml.BasisState(init_state, wires=range(n))
        for layer in range(p):
            gamma = params[2 * layer]
            beta = params[2 * layer + 1]
            qml.ApproxTimeEvolution(H_cost, gamma, 1)
            qml.ApproxTimeEvolution(H_mix, beta, 1)
        return qml.expval(H_cost)

    @qml.qnode(dev_sample)
    def circuit_sample(params):
        qml.BasisState(init_state, wires=range(n))
        for layer in range(p):
            gamma = params[2 * layer]
            beta = params[2 * layer + 1]
            qml.ApproxTimeEvolution(H_cost, gamma, 1)
            qml.ApproxTimeEvolution(H_mix, beta, 1)
        return qml.sample(wires=range(n))

    return circuit_exp, circuit_sample


def train_qaoa(circuit_exp, p: int,
               n_steps: int = 100, stepsize: float = 0.1,
               verbose: bool = True) -> pnp.ndarray:
    """Train QAOA; the global seed controls the random initialization."""
    params = pnp.array(np.random.uniform(0, np.pi, 2 * p), requires_grad=True)
    opt    = qml.GradientDescentOptimizer(stepsize=stepsize)
    for step in range(n_steps):
        params = opt.step(lambda v: circuit_exp(v), params)
        if verbose and (step + 1) % 25 == 0:
            e = float(circuit_exp(params))
            print(f"  step {step+1}/{n_steps} | energy = {e:.5f}")
    return params


def build_qaoa_sampler(circuit_sample, instance: dict) -> Callable:
    """Build a QAOA sampler compatible with run_hybrid.

    The circuit must already be trained; params are passed on every call.
    The sampler only generates a candidate pool and metadata. The classical
    selection of the candidates to evaluate is done outside the sampler.

    Note on `shots`:
    `shots` is a nominal pipeline parameter; the actual number of samples depends
    on the device with which `circuit_sample` was built. Frequencies are therefore
    normalized by the number of observed samples (`len(raw)`), not by `shots`.
    If the two differ, a warning is printed and stored in the metadata.
    """
    def qaoa_sampler(params, shots: int, top_k: int,
                     alpha: float, beta: float) -> tuple:
        raw = circuit_sample(params)
        n_samples = len(raw)
        if shots != n_samples:
            print(
                f"[build_qaoa_sampler] Warning: requested shots={shots} "
                f"but observed samples={n_samples}. "
                "Observed samples are used to normalize frequencies."
            )
        counts = {}
        for row in raw:
            b = tuple(int(x) for x in row)
            counts[b] = counts.get(b, 0) + 1
        full_candidate_pool = build_candidate_pool(counts, n_samples, instance,
                                                   alpha, beta)
        meta = candidate_pool_metadata(full_candidate_pool, n_samples)
        meta['requested_shots'] = shots
        meta['observed_shots'] = n_samples
        meta['shots_mismatch'] = (shots != n_samples)
        return full_candidate_pool, meta

    qaoa_sampler.__name__ = "qaoa_sampler"
    return qaoa_sampler


def build_uniform_random_sampler(instance: dict) -> Callable:
    """Build a uniform random sampler over exact-K supports.

    Same output contract as the QAOA sampler. 'params' is ignored; it exists for
    signature compatibility with run_hybrid. The sampler only generates a candidate
    pool and metadata; the classical selection policy is applied in run_hybrid.

    This is the critical baseline: if run_hybrid performs equally with this sampler
    and with QAOA, the quantum component does not contribute to the search.
    """
    N      = instance['N']
    K      = instance['K']
    ASSETS = instance['ASSETS']

    def uniform_random_sampler(params=None, shots: int = 5000,
                               top_k: int = 40,
                               alpha: float = 1.0,
                               beta: float = 0.05) -> tuple:
        counts = {}
        for _ in range(shots):
            chosen_idx = np.random.choice(N, K, replace=False)
            b = tuple(1 if i in chosen_idx else 0 for i in range(N))
            counts[b] = counts.get(b, 0) + 1
        full_candidate_pool = build_candidate_pool(counts, shots, instance,
                                                   alpha, beta)
        meta = candidate_pool_metadata(full_candidate_pool, shots)
        return full_candidate_pool, meta

    uniform_random_sampler.__name__ = "uniform_random_sampler"
    return uniform_random_sampler


def characterize_qaoa_distribution(circuit_sample, params,
                                   instance: dict,
                                   shots: int = 5000) -> dict:
    """Assess whether the QAOA distribution is distinguishable from uniform sampling.
    Requires no IPOPT. Auxiliary diagnostic, not the central criterion.

    Note: `shots` is the number of requested shots, but the empirical metrics are
    normalized by the actual number of samples returned by `circuit_sample(params)`.
    """
    N = instance['N']
    K = instance['K']
    ASSETS = instance['ASSETS']

    raw = circuit_sample(params)
    n_samples = len(raw)

    if shots != n_samples:
        print(
            f"[characterize_qaoa_distribution] Warning: "
            f"requested shots={shots} but observed samples={n_samples}. "
            "Observed samples are used for the diagnostic."
        )

    counts = {}
    for row in raw:
        b = tuple(int(x) for x in row)
        counts[b] = counts.get(b, 0) + 1

    exact_counts = {b: c for b, c in counts.items() if sum(b) == K}
    total_exact  = sum(exact_counts.values())
    frac_exact   = total_exact / n_samples if n_samples > 0 else np.nan
    n_unique     = len(exact_counts)
    n_possible   = comb(N, K)

    if total_exact > 0:
        probs_qaoa = np.array(list(exact_counts.values()), dtype=float)
        probs_qaoa /= probs_qaoa.sum()
        entropy_qaoa = float(-np.sum(probs_qaoa * np.log(probs_qaoa + 1e-15)))
    else:
        entropy_qaoa = 0.0

    entropy_uniform = log(n_possible)
    entropy_ratio   = entropy_qaoa / entropy_uniform if entropy_uniform > 0 else 0.0

    if n_unique > 0:
        p_uniform = 1.0 / n_unique
        tvd = 0.5 * float(np.sum(np.abs(probs_qaoa - p_uniform)))
    else:
        tvd = 1.0

    if len(probs_qaoa) > 0:
        sorted_probs = np.sort(probs_qaoa)[::-1]
        top10_n = max(1, int(0.1 * len(sorted_probs)))
        top10_mass = float(sorted_probs[:top10_n].sum())
    else:
        top10_mass = 1.0

    verdict = ('indistinguishable_from_uniform'
               if tvd < 0.05 and entropy_ratio > 0.95
               else 'distinguishable')

    print("\n=== Characterization of the QAOA distribution ===")
    print(f"  frac_exact_K       : {frac_exact:.4f}  "
          f"(shots with exact cardinality)")
    print(f"  n_unique_exact_K   : {n_unique} of C({N},{K})={n_possible} posibles")
    print(f"  entropy_qaoa       : {entropy_qaoa:.4f}")
    print(f"  entropy_uniform    : {entropy_uniform:.4f}  (log C({N},{K}))")
    print(f"  entropy_ratio      : {entropy_ratio:.4f}  "
          f"(1.0 = identical to uniform)")
    print(f"  TVD vs. uniforme   : {tvd:.4f}  "
          f"(0.0 = identical, lower bound)")
    print(f"  top10pct_mass      : {top10_mass:.4f}  "
          f"(mass in the most frequent 10%)")
    print(f"  VEREDICTO          : {verdict}")
    if verdict == 'indistinguishable_from_uniform':
        print("  -> The QAOA component is functionally equivalent to uniform")
        print("    random sampling. The contribution is architectural (tabu).")
    else:
        print("  -> The QAOA distribution is biased with respect to uniform.")
        print("    Check whether the bias correlates with portfolio quality.")

    return dict(
        frac_exact_K     = frac_exact,
        n_unique_exact_K = n_unique,
        n_possible       = n_possible,
        entropy_qaoa     = entropy_qaoa,
        entropy_uniform  = entropy_uniform,
        entropy_ratio    = entropy_ratio,
        TVD              = tvd,
        top10pct_mass    = top10_mass,
        verdict          = verdict,
        requested_shots = shots,
        observed_shots  = n_samples,
        shots_mismatch  = (shots != n_samples),
    )


def classical_baseline(instance: dict, verbose: bool = False) -> tuple:
    """Enumerate all C(N,K) exact-K supports and solve the continuous QP for every
    support that passes the structural prescreen.

    Returns the global optimum and reference statistics for comparison with the
    hybrid loop (QAOA and uniform random).
    """
    ASSETS = instance['ASSETS']
    K      = instance['K']
    SECTORS = instance['SECTORS']

    n_total = instance['N_TOTAL_COMBOS']
    print(f"\n=== Classical baseline: C({instance['N']},{K}) = {n_total} ===")
    t0 = time.time()

    best_var      = np.inf
    best_sel      = None
    rows          = []
    n_prescreened = 0
    n_infeasible  = 0
    n_feasible    = 0
    n_ipopt_calls = 0

    for combo in itertools.combinations(ASSETS, K):
        sel  = {a: (1 if a in combo else 0) for a in ASSETS}
        bits = _to_bits(sel, instance)
        diag = diagnose_infeasibility(sel, instance)

        n_high_sel    = sum(1 for a in combo if instance['mu'][a] >= instance['R'])
        sector_counts = {s: sum(1 for a in combo
                                if instance['SECTOR_OF'][a] == s)
                         for s in SECTORS}

        if diag != 'unknown':
            n_prescreened += 1
            rows.append(dict(
                assets=str(list(combo)), bits=bits,
                feasible=False, variance=None,
                ret_realized=None, ret_slack=None,
                method='pre_screen', diagnosis=diag,
                n_high_selected=n_high_sel,
                active_assets=None,
                active_count=None,
                selected_count=K,
                sector_counts=str(sector_counts)))
            continue

        n_ipopt_calls += 1
        result = solve_subproblem(sel, instance, tee=verbose)

        if not result['feasible']:
            n_infeasible += 1
            rows.append(dict(
                assets=str(list(combo)), bits=bits,
                feasible=False, variance=None,
                ret_realized=None, ret_slack=None,
                method='ipopt', diagnosis=result['tc'],
                n_high_selected=n_high_sel,
                active_assets=None,
                active_count=result.get('active_count'),
                selected_count=result.get('selected_count'),
                sector_counts=str(sector_counts)))
        else:
            n_feasible += 1
            var = result['variance']
            rows.append(dict(
                assets=str(list(combo)), bits=bits,
                feasible=True, variance=var,
                ret_realized=result['ret_realized'],
                ret_slack=result['ret_slack'],
                method='ipopt', diagnosis='optimal',
                n_high_selected=n_high_sel,
                active_assets=str(result['active_assets']),
                active_count=result['active_count'],
                selected_count=result['selected_count'],
                sector_counts=str(sector_counts)))
            if var < best_var:
                best_var = var
                best_sel = list(combo)

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)

    print(f"  Total combinaciones      : {n_total}")
    print(f"  Pre-screened (no IPOPT)  : {n_prescreened} "
          f"({100*n_prescreened/n_total:.1f}%)")
    print(f"  IPOPT calls              : {n_ipopt_calls}")
    print(f"    Infeasible (IPOPT)     : {n_infeasible}")
    print(f"    Feasible               : {n_feasible}")
    if best_sel:
        print(f"  Global optimum           : {best_var:.6f}")
        print(f"  Optimal portfolio        : {best_sel}")
    print(f"  Tiempo                   : {elapsed:.2f}s")

    summary = dict(
        n_total       = n_total,
        best_var      = best_var if best_sel is not None else None,
        best_sel      = best_sel,
        n_prescreened = n_prescreened,
        n_ipopt_calls = n_ipopt_calls,
        n_infeasible  = n_infeasible,
        n_feasible    = n_feasible,
        elapsed       = elapsed,
    )
    return df, summary


def select_candidates(valid_candidates: list, instance: dict, policy: str,
                      top_eval_per_iter: int, rng=None) -> list:
    """Select valid candidates for exact evaluation with IPOPT.

    Methodological separation:
      - the sampler induces a generation bias on the observed pool;
      - this function applies the classical policy that decides which valid
        supports are evaluated downstream.

    Available policies:
      - ranked_hybrid: exact-K first, hybrid_score desc, freq desc.
      - ranked_frequency: exact-K first, freq desc.
      - random_exactK: random selection among valid exact-K candidates.
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
        idx = rng.choice(len(exact_candidates), size=n_pick, replace=False)
        return [exact_candidates[int(i)] for i in np.atleast_1d(idx)]

    raise ValueError(f"selection_policy no soportada: {policy}")


def build_budget_pool(valid_candidates: list, instance: dict,
                      selection_policy: str, top_k_candidates: int,
                      rng=None) -> list:
    """Build the budgeted candidate pool of the iteration.

    This function must NOT reuse select_candidates, because the hybrid loop later
    applies a second selection with top_eval_per_iter; calling select_candidates
    here would filter twice.

    Convention:
      - build_budget_pool: reduces valid_candidates to a budget pool of size
        <= top_k_candidates.
      - select_candidates: chooses from that pool the candidates actually
        evaluated by IPOPT (top_eval_per_iter).

    For deterministic policies the budget pool keeps the best prefix sorted by the
    policy. For random_exactK it is a random sample without replacement of size
    top_k_candidates drawn from the exact-K set.
    """
    if top_k_candidates <= 0:
        return []

    if selection_policy == "ranked_hybrid":
        ranked = sorted(valid_candidates, key=lambda c: (
            0 if c['is_exactK'] else 1,
            -c['hybrid_score'],
            -c['freq'],
        ))
        return ranked[:top_k_candidates]

    if selection_policy == "ranked_mu":
        exact_candidates = [c for c in valid_candidates if c['is_exactK']]
        if not exact_candidates:
            return []
        ranked = sorted(
            exact_candidates,
            key=lambda c: -mu_score_support(c['selection'], instance)
        )
        return ranked[:top_k_candidates]

    if selection_policy == "ranked_frequency":
        ranked = sorted(valid_candidates, key=lambda c: (
            0 if c['is_exactK'] else 1,
            -c['freq'],
        ))
        return ranked[:top_k_candidates]

    if selection_policy == "random_exactK":
        exact_candidates = [c for c in valid_candidates if c['is_exactK']]
        if not exact_candidates:
            return []
        if rng is None:
            rng = np.random
        n_pick = min(top_k_candidates, len(exact_candidates))
        idx = rng.choice(len(exact_candidates), size=n_pick, replace=False)
        return [exact_candidates[int(i)] for i in np.atleast_1d(idx)]

    raise ValueError(f"selection_policy no soportada: {selection_policy}")


def run_hybrid(
    sampler: Callable,
    instance: dict,
    params=None,
    max_iter: int        = 30,
    top_k_candidates: int = 40,
    shots: int           = 5000,
    top_eval_per_iter: int = 2,
    score_alpha: float   = 1.0,
    score_beta: float    = 0.05,
    baseline_stats: dict = None,
    label: str           = "",
    selection_policy: str = "ranked_hybrid",
) -> pd.DataFrame:
    """Main hybrid loop. Accepts any sampler with the standard contract.

    The same loop runs with:
      - qaoa_sampler    -> main result
      - uniform_random  -> comparison baseline

    Parameters
    ----------
    sampler           : callable (params, shots, top_k, alpha, beta) returning the
                        complete observed pool and metadata
    instance          : instance dict
    params            : sampler parameters (None for uniform)
    label             : label for logs ("QAOA" / "uniform_random" / ...)
    baseline_stats    : statistics of the classical baseline, used for the gap
    selection_policy  : classical policy used to choose valid candidates
    top_k_candidates  : budget applied after tabu and before IPOPT
    """
    ASSETS  = instance['ASSETS']
    K       = instance['K']
    N       = instance['N']
    R       = instance['R']
    sampler_name = label or getattr(sampler, '__name__', 'sampler')

    print("=" * 70)
    print(f"HYBRID STRUCTURAL-TABU | N={N}, K={K} (exact), R={R}")
    print(f"  sampler={sampler_name} | top_eval={top_eval_per_iter}")
    print(f"  selection_policy={selection_policy}")
    print(f"  score: alpha={score_alpha}, beta={score_beta}")
    print("=" * 70)

    tabu        = StructuralTabu()
    log         = []
    best_var    = np.inf
    best_sel    = None
    best_active_count = None
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

        n_exactK_prefilter   = sum(
            1 for cand in candidate_pool if cand['is_exactK'])
        n_nonexact_prefilter = len(candidate_pool) - n_exactK_prefilter

        valid, discarded = [], []
        for cand in candidate_pool:
            bits = _to_bits(cand['selection'], instance)
            is_tabu, reason, rule = tabu.classify(
                cand['selection'], bits, instance)
            if is_tabu:
                discarded.append((cand, reason, rule))
            else:
                valid.append(cand)

        n_no_good = sum(1 for _, r, _ in discarded
                        if r == DiscardReason.NO_GOOD_EXACT)
        n_struct  = sum(1 for _, r, _ in discarded
                        if r == DiscardReason.STRUCT_RULE)
        n_nonK    = sum(1 for _, r, _ in discarded
                        if r == DiscardReason.NON_EXACT_K)
        n_exactK_valid = sum(1 for cand in valid if cand['is_exactK'])

        print(f"  Muestreo: unique={sampling_meta['n_unique_raw']} | "
              f"exactK={sampling_meta['n_exactK_raw']} | "
              f"frac_exactK={sampling_meta['frac_exactK_shots']:.4f} | "
              f"entropy_exactK={sampling_meta.get('entropy_exactK', 0.0):.4f} | "
              f"top1_exactK={sampling_meta.get('top1_exactK_freq', 0.0):.4f}")
        print(f"  Observed pool: {len(candidate_pool)} | "
              f"exactK={n_exactK_prefilter} | nonK={n_nonexact_prefilter}")
        print(f"  Descartados: no-good={n_no_good} | "
              f"structural={n_struct} | nonK={n_nonK}")
        print(f"  Valid: {len(valid)} (exactK: {n_exactK_valid})")

        if not valid:
            print("  Search space exhausted by tabu. Stopping.")
            break

        budget_pool = build_budget_pool(
            valid, instance, selection_policy, top_k_candidates,
            rng=np.random)
        n_exactK_budget = sum(1 for cand in budget_pool if cand['is_exactK'])
        print(f"  Budget top-k: {len(budget_pool)} | "
              f"exactK={n_exactK_budget} | top_k={top_k_candidates}")

        if not budget_pool:
            print("  No candidates within the budget. Stopping.")
            break

        to_evaluate = select_candidates(
            budget_pool, instance, selection_policy, top_eval_per_iter,
            rng=np.random)

        if not to_evaluate:
            print("  Selection policy chose no candidates. Stopping.")
            break

        for rank, chosen in enumerate(to_evaluate):
            chosen_sel = chosen['selection']
            chosen_freq = chosen['freq']
            chosen_rscore = chosen['r_score']
            chosen_risk = chosen['risk_score']
            chosen_hybrid = chosen['hybrid_score']

            bits  = _to_bits(chosen_sel, instance)
            names = _selected_names(chosen_sel, instance)
            print(f"  [{rank}] {names} | "
                  f"freq={chosen_freq:.4f} | hybrid={chosen_hybrid:.3f}")

            tabu.register_visit(bits)

            diag = diagnose_infeasibility(chosen_sel, instance)
            if diag != 'unknown':
                is_new = infer_and_register_rule(tabu, diag, instance)
                if is_new:
                    tabu.stats['structural_discoveries'] += 1
                tabu.stats['ipopt_avoided_prescreen'] += 1
                print(f"    Pre-screen: {diag} - IPOPT skipped"
                      + (" [new rule]" if is_new else ""))
                log.append(_entry(
                    it, rank, bits, names, sampler_name,
                    sampling_meta, n_exactK_prefilter, n_nonexact_prefilter,
                    n_exactK_valid, n_no_good, n_struct, n_nonK,
                    chosen_freq, chosen_rscore, chosen_risk, chosen_hybrid,
                    'infeasible_prescreen', None, None,
                    None, None,
                    best_var, ipopt_total, tabu,
                    round(time.time() - t_iter, 3)))
                continue

            ipopt_total += 1
            result = solve_subproblem(chosen_sel, instance)

            if not result['feasible']:
                print(f"    IPOPT: infeasible ({result['tc']})")
                log.append(_entry(
                    it, rank, bits, names, sampler_name,
                    sampling_meta, n_exactK_prefilter, n_nonexact_prefilter,
                    n_exactK_valid, n_no_good, n_struct, n_nonK,
                    chosen_freq, chosen_rscore, chosen_risk, chosen_hybrid,
                    'infeasible_ipopt', None, None,
                    result.get('active_count'), result.get('selected_count'),
                    best_var, ipopt_total, tabu,
                    round(time.time() - t_iter, 3)))
                continue

            var   = result['variance']
            slack = result['ret_slack']
            print(f"    IPOPT: variance={var:.6f} | ret_slack={slack:.4f}")

            if var < best_var - 1e-8:
                best_var = var
                best_sel = chosen_sel.copy()
                best_active_count = result.get('active_count')
                print(f"    *** Best actualizado: {best_var:.6f} ***")

            log.append(_entry(
                it, rank, bits, names, sampler_name,
                sampling_meta, n_exactK_prefilter, n_nonexact_prefilter,
                n_exactK_valid, n_no_good, n_struct, n_nonK,
                chosen_freq, chosen_rscore, chosen_risk, chosen_hybrid,
                'feasible', round(var, 8), round(slack, 6),
                result.get('active_count'), result.get('selected_count'),
                best_var, ipopt_total, tabu,
                round(time.time() - t_iter, 3)))

    print("\n" + "=" * 70)
    print(f"FINAL SOLUTION [{sampler_name}]")
    print("=" * 70)

    if best_sel:
        names   = _selected_names(best_sel, instance)
        sectors = _selection_sectors(best_sel, instance)
        print(f"  Assets   : {names}")
        print(f"  Sectores : {dict((s, v) for s, v in sectors.items() if v > 0)}")
        print(f"  Variance : {best_var:.6f}")
        if best_active_count is not None:
            print(f"  Effective assets  : {best_active_count}")

        if baseline_stats and baseline_stats.get('best_var') is not None:
            ref   = baseline_stats['best_var']
            gap   = (best_var - ref) / ref * 100
            saved = baseline_stats['n_ipopt_calls'] - ipopt_total
            print(f"  Baseline : {ref:.6f} | gap = {gap:+.4f}%")
            print(f"  IPOPT - hybrid: {ipopt_total} | "
                  f"baseline: {baseline_stats['n_ipopt_calls']} | "
                  f"saved: {saved} "
                  f"({100*saved/max(baseline_stats['n_ipopt_calls'], 1):.1f}%)")
    else:
        print("  No feasible solution found.")

    s = tabu.stats
    print(f"\n  Memoria tabu:")
    print(f"    No-good exactos  : {tabu.n_exact()}")
    print(f"    Reglas estr.     : {tabu.n_structural()}")
    for lbl in tabu.rule_labels:
        print(f"      . {lbl}")
    print(f"  Contabilidad:")
    print(f"    ipopt_calls_used          : {ipopt_total}")
    print(f"    no_good_hits              : {s['no_good_hits']}")
    print(f"    structural_hits_known     : {s['structural_hits_known']}")
    print(f"    structural_discoveries    : {s['structural_discoveries']}")
    print(f"    ipopt_avoided_struct      : {s['ipopt_avoided_struct']}")
    print(f"    ipopt_avoided_prescreen   : {s['ipopt_avoided_prescreen']}")
    print(f"    ipopt_avoided_total       : {tabu.ipopt_avoided_total()}")

    df = pd.DataFrame(log)
    if not df.empty:
        show_cols = [
            'iter', 'rank', 'sampler', 'estado', 'variance', 'best_cumul',
            'ipopt_total', 'ipopt_avoided_total',
            'n_structural_discards', 'freq_chosen', 'hybrid_score_chosen',
            'active_count', 'selected_count', 'iter_time_s',
        ]
        print("\nIteration log:")
        print(df[show_cols].to_string(index=False))
    return df


def xy_exactk_reference_factory(seed: int, instance: dict,
                                config: dict) -> tuple:
    """XY exact-K reference factory.

    Keeps the exact-K initial state and the XY mixer, but must not be read as a
    QAOA trained with an informative cost: under an exact-cardinality cost the
    energy is constant on the exact-K subspace, so this branch is only a reference
    for mixing dynamics without a useful cost bias.
    """
    set_global_seed(seed)
    H = build_exactK_hamiltonian(
        instance['N'], instance['K'], penalty=HAMILTONIAN_PENALTY)
    c_exp, c_sample = make_circuits(
        instance['N'], config['qaoa_layers'], H, k=instance['K'],
        shots=config['qaoa_shots'], pl_seed=seed)

    params_reference = pnp.array(
        np.random.uniform(0, np.pi, 2 * config['qaoa_layers']),
        requires_grad=False)

    sampler = build_qaoa_sampler(c_sample, instance)
    return sampler, params_reference


def qaoa_surrogate_factory(seed: int, instance: dict,
                           config: dict) -> tuple:
    """QAOA surrogate factory.

    Uses build_support_surrogate_hamiltonian (the V3 branch with a useful-return
    signal and support risk).
    """
    set_global_seed(seed)
    H = build_support_surrogate_hamiltonian(
        instance['N'], instance['K'], penalty=HAMILTONIAN_PENALTY,
        instance=instance)
    c_exp, c_sample = make_circuits(
        instance['N'], config['qaoa_layers'], H, k=instance['K'],
        shots=config['qaoa_shots'], pl_seed=seed)
    p_trained = train_qaoa(c_exp, p=config['qaoa_layers'],
                           n_steps=config['qaoa_steps'], verbose=False)
    sampler = build_qaoa_sampler(c_sample, instance)
    return sampler, p_trained


def uniform_factory(seed: int, instance: dict,
                    config: dict) -> tuple:
    """Uniform random factory.

    The seed controls the sampling; there is no training and there are no parameters.
    """
    set_global_seed(seed)
    sampler = build_uniform_random_sampler(instance)
    return sampler, None


def run_comparative_matrix(instance: dict, baseline_stats: dict,
                           config: dict, seeds: list = None,
                           include_xy_reference: bool = False) -> pd.DataFrame:
    """Run a comparative matrix that attributes sampler bias vs. classical selection policy.

    Factors:
      - sampler_type: QAOA_surrogate, uniform_random
      - selection_policy: ranked_hybrid, random_exactK

    The XY exact-K reference can be enabled with include_xy_reference=True, but it
    is not interpreted as a QAOA optimized with an informative cost.

    Returns a consolidated DataFrame with one row per seed and configuration.
    """
    if seeds is None:
        seeds = config.get('multiseed_seeds', list(range(10)))

    sampler_factories = [
        ('QAOA_surrogate', qaoa_surrogate_factory),
        ('uniform_random', uniform_factory),
    ]
    if include_xy_reference:
        sampler_factories.append(
            ('XY_exactK_reference', xy_exactk_reference_factory))

    selection_policies = ['ranked_hybrid', 'random_exactK']

    common_kwargs = dict(
        max_iter          = config['max_iter'],
        top_k_candidates  = config['top_k_candidates'],
        shots             = config['qaoa_shots'],
        top_eval_per_iter = config['top_eval_per_iter'],
        score_alpha       = config['score_alpha'],
        score_beta        = config['score_beta'],
    )

    rows = []
    print("\n" + "=" * 70)
    print("COMPARATIVE MATRIX: sampler_type x selection_policy")
    print("=" * 70)

    for sampler_type, factory in sampler_factories:
        for selection_policy in selection_policies:
            label = f"{sampler_type}__{selection_policy}"
            print(f"\n--- Configuracion: {sampler_type} | {selection_policy} ---")

            for seed in seeds:
                print(f"\n--- Seed {seed} ---")
                sampler, params = factory(seed=seed, instance=instance,
                                          config=config)
                df_run = run_hybrid(
                    sampler          = sampler,
                    instance         = instance,
                    params           = params,
                    baseline_stats   = baseline_stats,
                    label            = label,
                    selection_policy = selection_policy,
                    **common_kwargs,
                )
                metrics = summarize_run(df_run, baseline_stats)
                metrics['sampler_type'] = sampler_type
                metrics['selection_policy'] = selection_policy
                metrics['seed'] = seed
                rows.append(metrics)

                print(f"  -> best_var={metrics['best_var']:.6f} | "
                      f"gap={metrics['gap_pct']:.4f}% | "
                      f"ipopt_total={metrics['ipopt_total']}")

    df_matrix = pd.DataFrame(rows)
    ordered_cols = [
        'sampler_type', 'selection_policy', 'seed',
        'best_var', 'gap_pct', 'ipopt_total', 'ipopt_saved',
        'iter_best', 'n_feasible',
    ]
    df_matrix = df_matrix[ordered_cols]

    summary_cols = ['best_var', 'gap_pct', 'ipopt_total',
                    'ipopt_saved', 'iter_best', 'n_feasible']
    summary = (df_matrix
               .groupby(['sampler_type', 'selection_policy'])[summary_cols]
               .agg(['mean', 'std']))

    print("\n" + "=" * 70)
    print("COMPARATIVE MATRIX SUMMARY")
    print("=" * 70)
    print(summary.to_string())

    return df_matrix


def summarize_run(df: pd.DataFrame, baseline_stats: dict) -> dict:
    """Extract the key metrics of a single run for multi-seed aggregation."""
    if df.empty:
        return dict(
            best_var=np.inf, gap_pct=np.inf,
            ipopt_total=0, ipopt_saved=0,
            iter_best=np.nan, n_feasible=0,
        )

    feasible = df[df['estado'] == 'feasible']
    if feasible.empty:
        best_var  = np.inf
        iter_best = np.nan
    else:
        best_var  = float(feasible['variance'].min())
        iter_best = int(feasible.loc[feasible['variance'].idxmin(), 'iter'])

    ref         = baseline_stats.get('best_var') or np.inf
    gap_pct     = (best_var - ref) / ref * 100 if ref < np.inf else np.inf
    ipopt_total = int(df['ipopt_total'].max()) if 'ipopt_total' in df else 0
    ipopt_saved = baseline_stats.get('n_ipopt_calls', 0) - ipopt_total

    return dict(
        best_var    = best_var,
        gap_pct     = gap_pct,
        ipopt_total = ipopt_total,
        ipopt_saved = ipopt_saved,
        iter_best   = iter_best,
        n_feasible  = len(feasible),
    )


def run_experiment_multiseed(
    sampler_factory: Callable,
    instance: dict,
    baseline_stats: dict,
    seeds: list = None,
    label: str  = "",
    **hybrid_kwargs,
) -> pd.DataFrame:
    """Run run_hybrid over several seeds and aggregate the results.

    Parameters
    ----------
    sampler_factory : callable(seed) -> sampler. For QAOA it must include the
                      internal training of the circuit with that seed.
                      For uniform it simply ignores the seed (uses np.random).
    instance        : instance dict
    baseline_stats  : dict from classical_baseline, used to compute gaps
    seeds           : list of integers (default: range(10))
    label           : descriptive label of the sampler
    **hybrid_kwargs : additional arguments for run_hybrid

    Returns
    -------
    pd.DataFrame with per-seed metrics plus a summary row (mean ± std).
    """
    if seeds is None:
        seeds = list(range(10))

    rows = []
    print(f"\n{'='*70}")
    print(f"MULTI-SEED EXPERIMENT: {label} | {len(seeds)} seeds")
    print(f"{'='*70}")

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        set_global_seed(seed)
        sampler, params = sampler_factory(seed=seed)
        df_run = run_hybrid(
            sampler        = sampler,
            instance       = instance,
            params         = params,
            baseline_stats = baseline_stats,
            label          = label,
            **hybrid_kwargs,
        )
        metrics = summarize_run(df_run, baseline_stats)
        metrics['seed'] = seed
        rows.append(metrics)
        print(f"  -> best_var={metrics['best_var']:.6f} | "
              f"gap={metrics['gap_pct']:.4f}% | "
              f"ipopt_total={metrics['ipopt_total']}")

    df = pd.DataFrame(rows)

    numeric_cols = ['best_var', 'gap_pct', 'ipopt_total',
                    'ipopt_saved', 'iter_best', 'n_feasible']
    summary = {'seed': 'summary'}
    for col in numeric_cols:
        summary[f'{col}_mean'] = df[col].mean()
        summary[f'{col}_std']  = df[col].std()

    print(f"\n{'='*70}")
    print(f"MULTI-SEED SUMMARY: {label}")
    print(f"{'='*70}")
    for col in numeric_cols:
        print(f"  {col:15s}: "
              f"{summary[f'{col}_mean']:.4f} ± {summary[f'{col}_std']:.4f}")

    return df


def _to_bits(s: dict, instance: dict) -> str:
    return ''.join(str(s[i]) for i in instance['ASSETS'])


def _selected_names(s: dict, instance: dict) -> list:
    return [i for i in instance['ASSETS'] if s[i] == 1]


def _selection_sectors(s: dict, instance: dict) -> dict:
    counts = {sec: 0 for sec in instance['SECTORS']}
    for a in instance['ASSETS']:
        if s[a] == 1:
            counts[instance['SECTOR_OF'][a]] += 1
    return counts


def _entry(it, rank, bits, names, sampler_name,
           sampling_meta, n_exactK_prefilter, n_nonexact_prefilter,
           n_exactK_valid, n_no_good, n_struct, n_nonK,
           freq, rscore, risk, hybrid,
           estado, variance, ret_slack,
           active_count, selected_count,
           best_var, ipopt_total, tabu, iter_time):
    s = tabu.stats
    return {
        'iter'                    : it,
        'rank'                    : rank,
        'sampler'                 : sampler_name,
        'bits'                    : bits,
        'assets'                  : str(names),
        'n_unique_raw'            : sampling_meta['n_unique_raw'],
        'n_exactK_raw'            : sampling_meta['n_exactK_raw'],
        'n_nonexact_raw'          : sampling_meta['n_nonexact_raw'],
        'frac_exactK_shots'       : sampling_meta['frac_exactK_shots'],
        'n_exactK_prefilter'      : n_exactK_prefilter,
        'n_nonexact_prefilter'    : n_nonexact_prefilter,
        'n_exactK_valid'          : n_exactK_valid,
        'n_no_good_discards'      : n_no_good,
        'n_structural_discards'   : n_struct,
        'n_nonK_discards'         : n_nonK,
        'freq_chosen'             : round(freq, 5),
        'r_score_chosen'          : round(rscore, 3),
        'risk_score_chosen'       : round(risk, 4),
        'hybrid_score_chosen'     : round(hybrid, 4),
        'estado'                  : estado,
        'variance'                : variance,
        'ret_slack'               : ret_slack,
        'active_count'            : active_count,
        'selected_count'          : selected_count,
        'best_cumul'              : round(best_var, 8) if best_var < np.inf else None,
        'ipopt_total'             : ipopt_total,
        'ipopt_avoided_struct'    : s['ipopt_avoided_struct'],
        'ipopt_avoided_prescreen' : s['ipopt_avoided_prescreen'],
        'ipopt_avoided_total'     : tabu.ipopt_avoided_total(),
        'no_good_hits'            : s['no_good_hits'],
        'structural_hits_known'   : s['structural_hits_known'],
        'structural_discoveries'  : s['structural_discoveries'],
        'tabu_exact_cuts'         : tabu.n_exact(),
        'tabu_struct_rules'       : tabu.n_structural(),
        'iter_time_s'             : iter_time,
    }


if __name__ == '__main__':

    instance = INSTANCE_15


    set_global_seed(CONFIG['seed'])

    H_cost = build_support_surrogate_hamiltonian(
        instance['N'], instance['K'], penalty=HAMILTONIAN_PENALTY,
        instance=instance)
    circuit_exp, circuit_sample = make_circuits(
        instance['N'], CONFIG['qaoa_layers'], H_cost, k=instance['K'],
        shots=CONFIG['qaoa_shots'], pl_seed=CONFIG['seed'])

    params_trained = train_qaoa(
        circuit_exp, p=CONFIG['qaoa_layers'],
        n_steps=CONFIG['qaoa_steps'])

    dist_analysis = characterize_qaoa_distribution(
        circuit_sample, params_trained, instance,
        shots=CONFIG['qaoa_shots'])

    set_global_seed(CONFIG['seed'])
    df_baseline, baseline_stats = classical_baseline(instance)
    df_baseline.to_csv("baseline_enumeration.csv", index=False)


    def qaoa_factory(seed: int):
        set_global_seed(seed)
        H = build_support_surrogate_hamiltonian(
            instance['N'], instance['K'], penalty=HAMILTONIAN_PENALTY,
            instance=instance)
        c_exp, c_sample = make_circuits(
            instance['N'], CONFIG['qaoa_layers'], H, k=instance['K'],
            shots=CONFIG['qaoa_shots'], pl_seed=seed)
        p_trained = train_qaoa(c_exp, p=CONFIG['qaoa_layers'],
                               n_steps=CONFIG['qaoa_steps'], verbose=False)
        sampler = build_qaoa_sampler(c_sample, instance)
        return sampler, p_trained

    def legacy_uniform_factory(seed: int):
        set_global_seed(seed)
        sampler = build_uniform_random_sampler(instance)
        return sampler, None

    common_kwargs = dict(
        max_iter         = CONFIG['max_iter'],
        top_k_candidates = CONFIG['top_k_candidates'],
        shots            = CONFIG['qaoa_shots'],
        top_eval_per_iter= CONFIG['top_eval_per_iter'],
        score_alpha      = CONFIG['score_alpha'],
        score_beta       = CONFIG['score_beta'],
    )

    df_qaoa_multi = run_experiment_multiseed(
        sampler_factory = qaoa_factory,
        instance        = instance,
        baseline_stats  = baseline_stats,
        seeds           = CONFIG['multiseed_seeds'],
        label           = "QAOA",
        **common_kwargs,
    )
    df_qaoa_multi.to_csv("multiseed_qaoa.csv", index=False)

    df_uniform_multi = run_experiment_multiseed(
        sampler_factory = legacy_uniform_factory,
        instance        = instance,
        baseline_stats  = baseline_stats,
        seeds           = CONFIG['multiseed_seeds'],
        label           = "uniform_random",
        **common_kwargs,
    )
    df_uniform_multi.to_csv("multiseed_uniform.csv", index=False)

    print("\n" + "=" * 70)
    print("COMPARISON: QAOA vs. uniform random")
    print("=" * 70)

    metrics_compare = ['best_var', 'gap_pct', 'ipopt_total', 'ipopt_saved']
    for metric in metrics_compare:
        q_mean = df_qaoa_multi[metric].mean()
        q_std  = df_qaoa_multi[metric].std()
        u_mean = df_uniform_multi[metric].mean()
        u_std  = df_uniform_multi[metric].std()
        print(f"  {metric:15s}: "
              f"QAOA={q_mean:.4f}±{q_std:.4f}  "
              f"Uniform={u_mean:.4f}±{u_std:.4f}")

    print("\n" + "=" * 70)
    print("ATRIBUTION STUDY: sampler_type x selection_policy")
    print("=" * 70)

    df_matrix = run_comparative_matrix(
        instance       = instance,
        baseline_stats = baseline_stats,
        config         = CONFIG,
        seeds          = CONFIG['multiseed_seeds'],
    )
    df_matrix.to_csv("comparative_matrix.csv", index=False)

    print("\nShort summary of comparative_matrix.csv")
    summary = (
        df_matrix
        .groupby(['sampler_type', 'selection_policy'])[
            ['best_var', 'gap_pct', 'ipopt_total', 'ipopt_saved']
        ]
        .agg(['mean', 'std'])
    )
    print(summary.to_string())
