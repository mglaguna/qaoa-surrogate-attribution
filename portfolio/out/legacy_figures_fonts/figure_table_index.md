# Manuscript Output Index

## Package Notes

- Minimum main package: Figures 1-4 and Tables 1-3.
- Optional main figure: `main_fig05_capped_budget_optional.png`, generated only with `--include-optional-main`.
- When the optional main capped-budget figure is generated, `sup_fig01_capped_budget.png` is not generated to avoid duplication.
- Supplementary figures/tables contain controls and diagnostics supporting the attribution narrative.

## Main figures

- **Figure 1**: `main_fig01_feasible_support_landscape.png`
  - Sources: baseline_enumeration.csv
  - Message: Feasible-support IPOPT variance landscape.
  - Notes/warnings: Histogram includes only the 2211 feasible supports evaluated by IPOPT, not all 3003 exact-K supports.
- **Figure 2**: `main_fig02_surrogate_alignment.png`
  - Sources: surrogate_vs_ipopt_full.csv, surrogate_correlation_table.csv
  - Message: Three surrogate proxies compared with IPOPT-refined variance.
  - Notes/warnings: The global optimum marker is matched by bitstring when available.
- **Figure 3**: `main_fig03_sampler_policy_gap.png`
  - Sources: comparative_matrix_v2.csv
  - Message: Sampler x selection-policy interaction in final gap.
  - Notes/warnings: The QAOA and Uniform panels intentionally use independent y-axis ranges; compare patterns within each panel, not absolute visual bar heights across panels.
- **Figure 4**: `main_fig04_convergence_ipopt_evaluations.png`
  - Sources: convergence_curves.csv
  - Message: Incumbent convergence by cumulative IPOPT evaluations.
  - Notes/warnings: Full-scale convergence view.

## Supplementary figures

None generated.

## Main tables

None generated.

## Supplementary tables

None generated.

