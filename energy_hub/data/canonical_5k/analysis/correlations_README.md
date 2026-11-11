# correlations_by_arm.csv -- sign convention

Computed by pooling all 10 seeds per arm (matching
`core/figures/scripts/figure5_tac_vs_budget.py`'s method for rho_E_p/rho_p_V
and `manuscript/scripts/generate_canonical_supplement_candidates.py`'s
`spearman_energy_TAC` for rho_E_V).

- **rho_E_p** = Spearman(-E(y), p_hat(y)) over unique sampled supports for the
  arm. E is negated so that POSITIVE rho_E_p means "lower surrogate energy is
  associated with higher sampling probability" -- the favourable direction.
- **rho_p_V** = Spearman(p_hat(y), -V(y)) over the B<=100 feasible evaluated
  supports. V is negated so that POSITIVE rho_p_V means "higher sampling
  probability is associated with lower TAC" -- the favourable direction.
- **rho_E_V** = Spearman(E(y), V(y)), raw (no negation), over the same B<=100
  feasible evaluated supports. Both E and V are "lower is better" quantities,
  so a POSITIVE raw correlation between them already means preserved
  ordering -- no negation is applied here, unlike rho_E_p/rho_p_V above.

Uniform has no surrogate_energy (NaN for every row, by construction -- it is
not surrogate-guided), so its rho_E_p and rho_E_V are reported as NaN rather
than silently dropped or zeroed.
