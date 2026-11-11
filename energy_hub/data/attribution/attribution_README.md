# Attribution experiment: is the S0->S1 crossover a surrogate effect or a budget effect?

## What this is

The original canonical campaign (`data/canonical_5k/runs/S1@15k`, i.e. the S1
arm as first run, 3 restarts x 5,000 shots = 15,000 total shots) showed S1
overtaking S0 as the best surrogate-guided arm by downstream budget B=60. This
directory isolates whether that crossover is driven by S1's surrogate
Hamiltonian, or by S1 simply having a 3x larger sampling budget than every
other arm (S0, S2, Uniform all use 5,000 total shots).

Two controls were run to separate these explanations:

- **Control A** (`runs/Control_A/`): the S1 surrogate re-run with `restarts=1`
  (5,000 total shots), keeping the arm **name** "S1" so that its sub-seed
  derivation is identical to restart 0 of the original 15k-shot run. This is
  **not an independent sample** -- it isolates what a single restart, on its
  own, already contained.
- **Control B** (`runs/Control_B/`): a genuinely **independent** 5,000-shot
  draw, run under a new arm name ("S1_5k", surrogate_type="S1") so it gets its
  own, unrelated sub-seed sequence. This is the scientifically decisive
  control -- it asks what happens when S1 is given the *same* budget as
  everyone else, sampled freshly rather than reused.

## Finding, stated in neutral terms

The apparent S0->S1 crossover in the 15k-shot regime is attributable to S1's
3x larger sampling budget, not to its surrogate Hamiltonian being intrinsically
better at directing search:

- **Support-space coverage** (median unique exact-K supports sampled per
  seed) drops from 4,032.5 at 15k shots to 1,371.0 in Control A (-66.0%) and
  1,207.5 in Control B (-70.1%) when the budget is reduced to 5,000 shots --
  neither control fully collapses to S0's 775.5, but both lose most of S1's
  coverage advantage.
- **Seed-paired win/tie/loss against S0 at B=60** (the budget at which the
  original crossover appears): S1@15k wins 8/10 seeds; Control A -- which
  reuses the same restart-0 data, so it is not an independent test -- also
  wins 8/10 (expected, since it shares its sample with the original run);
  **Control B, the independent test, is a 5-0-5 coin flip**. The same 5-0-5
  pattern holds at B=100.
- **The surrogate's own ranking quality does not degrade with budget.**
  rho(E,V) -- the Spearman correlation between S1's surrogate energy and the
  true downstream TAC, i.e. how well the surrogate's internal ranking agrees
  with the real objective -- is +0.849 at 15k shots, +0.812 in Control A, and
  +0.822 in Control B: essentially unchanged. What changes with budget is
  *coverage* (how many distinct supports get sampled and evaluated), not the
  surrogate's ability to rank the supports it does sample.

## How to read this

This is **not** a negative result about QAOA or about the S1 surrogate. It
is evidence that, **in this specific regime** (low circuit depth, noiseless
simulation, a 5,000-18,000-shot budget, an exact-K benchmark small enough to
enumerate exhaustively), a modest classical resampling advantage was doing
more work than the surrogate's structural guidance. That is a
**pre-competitive characterization of the current regime**, not a failure
mode: it tells us precisely which lever (budget vs. representation) is
currently load-bearing, which is exactly the kind of attribution this
framework was built to produce. The surrogate's ranking quality (rho(E,V))
being stable across budgets is, if anything, a point in the surrogate's
favor -- it says S1's structural signal is real and budget-independent; it is
the *coverage*, not the *ranking*, that was doing the extra work at 15k shots.

## Provenance

| Variant | Source (outside this package) | Arm name here | Independent sample? |
|---|---|---|---|
| S0 (comparator) | `energy_hub_canonical_v1` (original campaign) | S0 | -- |
| S1@15k | `energy_hub_canonical_v1` (original campaign) | S1 | Reference (3 restarts) |
| Control A | `energy_hub_s1_control_A` | S1 | No -- restart 0 of S1@15k, same sub-seed by construction |
| Control B | `energy_hub_s1_control_B` | S1_5k | Yes -- independent sub-seed, same surrogate coefficients |

See `data/attribution/attribution_summary.csv`, `attribution_pairwise_wtl.csv`,
and `attribution_correlations.csv` for the full numeric tables, and
`MANIFEST.json` at the package root for the provenance/hash record of every
file.
