# Energy Hub benchmark

Main-study protocol (EH_CANONICAL_V1; all constants in
`code/hybrid_pipeline/experiments/energy_hub_canonical_config.py`):

* Exact-cardinality design: K = 6 of N = 18 candidate technologies, T = 6 representative periods.
* Four strategies (S0, S1, S2 surrogate-guided QAOA and the Uniform exact-K control), ten seeds each
  (101-110), 5,000 shots or draws per seed.
* For each run the candidate list is formed once (frequency ranking, structural screen); the downstream
  budgets B = 10, 30, 60 and 100 are nested prefixes of that list.
* The continuous value V(y) (TAC) of each evaluated support is taken from the exhaustive reference in
  `data/reference/` (all 18,564 exact-K supports), so the campaign issues no continuous solves.

Commands for re-running the campaign, the analysis, the figures, the Ipopt residual check and the
sensitivity sweeps are in `../README.md`.
