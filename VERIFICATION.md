# Verification of the released source code

Every Python file in this release was produced from the code that generated the archived results by
removing comments, translating docstrings and console messages to English, and replacing machine-specific
paths with relative ones. For each file the syntax tree of the released code was compared node by node
with that of the original: they are identical except for docstrings and for string constants that are
only used in console or exception messages. Path and file-name changes are listed below.

The cleaning and the checks below were carried out with AI assistance (Claude Code, Anthropic); the
changes were also reviewed with Codex (OpenAI), to check that nothing was broken, and by the authors.
The node-by-node syntax-tree comparison and the reproduction checks are what establish that the
released code is equivalent to the code that produced the results.

Energy Hub figures: energy_hub/figures/scripts/make_figures.py redrew 43 PNG/CSV files; every one is byte-identical to the image or data file used in the manuscript.

| Released file | Message strings changed | Declared code changes |
|---|---|---|
| `energy_hub/code/hybrid_pipeline/__init__.py` | 0 | - |
| `energy_hub/code/hybrid_pipeline/benchmarks/__init__.py` | 0 | - |
| `energy_hub/code/hybrid_pipeline/benchmarks/energy_hub/__init__.py` | 0 | - |
| `energy_hub/code/hybrid_pipeline/benchmarks/energy_hub/instance.py` | 1 | - |
| `energy_hub/code/hybrid_pipeline/benchmarks/energy_hub/model.py` | 0 | - |
| `energy_hub/code/hybrid_pipeline/benchmarks/energy_hub/surrogates.py` | 0 | - |
| `energy_hub/code/hybrid_pipeline/cli/__init__.py` | 0 | - |
| `energy_hub/code/hybrid_pipeline/cli/config.py` | 0 | - |
| `energy_hub/code/hybrid_pipeline/core/__init__.py` | 0 | - |
| `energy_hub/code/hybrid_pipeline/core/selection.py` | 0 | - |
| `energy_hub/code/hybrid_pipeline/core/support.py` | 0 | - |
| `energy_hub/code/hybrid_pipeline/core/tabu.py` | 0 | - |
| `energy_hub/code/hybrid_pipeline/experiments/__init__.py` | 0 | - |
| `energy_hub/code/hybrid_pipeline/experiments/energy_hub_canonical_analysis.py` | 0 | - |
| `energy_hub/code/hybrid_pipeline/experiments/energy_hub_canonical_campaign.py` | 0 | `"`docs/canonical_campaign/EH_CANONICAL_V1_PROTOCOL.md` for t` -> `"`energy_hub/README.md` in the release for the protocol.\n\n"` |
| `energy_hub/code/hybrid_pipeline/experiments/energy_hub_canonical_config.py` | 0 | - |
| `energy_hub/code/hybrid_pipeline/experiments/energy_hub_import_cap_sensitivity_appendix.py` | 0 | - |
| `energy_hub/code/hybrid_pipeline/experiments/energy_hub_k_sweep_appendix.py` | 0 | - |
| `energy_hub/code/hybrid_pipeline/samplers/__init__.py` | 0 | - |
| `energy_hub/code/hybrid_pipeline/samplers/qaoa.py` | 0 | - |
| `energy_hub/code/scripts/compute_correlations.py` | 0 | - |
| `energy_hub/figures/scripts/enlarge_fonts.py` | 0 | - |
| `energy_hub/figures/scripts/figS_attribution.py` | 0 | - |
| `energy_hub/figures/scripts/figure2_surrogate_landscapes.py` | 0 | `"canonical/src/hybrid_pipeline/benchmarks/energy_hub/* (fres` -> `"code/hybrid_pipeline/benchmarks/energy_hub/* (fresh enumeration)"` |
| `energy_hub/figures/scripts/figure4_candidate_flow.py` | 0 | - |
| `energy_hub/figures/scripts/figure5_tac_vs_budget.py` | 0 | - |
| `energy_hub/figures/scripts/figure6_landscape_context.py` | 0 | - |
| `energy_hub/figures/scripts/generate_canonical_supplement_candidates.py` | 0 | text written to the figure README replaced (regenerated description of Figures 7, 11, 12) |
| `energy_hub/figures/scripts/_common.py` | 0 | - |
| `energy_hub/figures/scripts/figure1_pipeline_schematic.py` | 0 | - |
| `energy_hub/figures/scripts/figure3_sampling_concentration.py` | 0 | - |
| `energy_hub/figures/scripts/figure7_engineering_fingerprints.py` | 0 | `"canonical/src/hybrid_pipeline (fresh per-support feature co` -> `"code/hybrid_pipeline (fresh per-support feature computation)"` |
| `energy_hub/figures/scripts/generate_energy_hub_superstructure.py` | 0 | - |
| `energy_hub/ipopt_residuals/resolve_reference.py` | 0 | `CODE`: absolute local path replaced by `os.path.join(EH_ROOT, "code")`<br>`DATA`: absolute local path replaced by `os.path.join(EH_ROOT, "data")` |
| `portfolio/code/Quantico-cardinalidad.py` | 66 | - |
| `portfolio/code/Fase2.py` | 92 | - |
| `portfolio/code/Fase3_separacion_efectos.py` | 12 | - |
| `portfolio/code/recompute_mixer_ext.py` | 0 | - |
| `portfolio/code/recompute_s11_depth.py` | 0 | - |
| `portfolio/code/regenerate_manuscript_outputs_legacy.py` | 0 | - |
| `portfolio/run_canonical.py` | 0 | - |
| `portfolio/summarize_canonical.py` | 0 | - |
| `portfolio/assemble_and_plot_legacy.py` | 0 | `LEG`: absolute local path replaced by `os.path.join(here, "data")` |
| `portfolio/make_figures_canonical.py` | 0 | `os.path.join(m2, "major2_runs.csv")` -> `os.path.join(m2, "runs.csv")` |
| `initialization_controls/initialization_controls.py` | 0 | `CODE`: absolute local path replaced by `os.path.join(ROOT, "energy_hub", "code")`<br>`DATA`: absolute local path replaced by `os.path.join(ROOT, "energy_hub", "data")`<br>`PF_CODE`: absolute local path replaced by `os.path.join(ROOT, "portfolio", "code")`<br>`IPOPT_BIN`: absolute local path replaced by `os.path.join(os.path.dirname(sys.executable), "Library", "bin")`<br>`os.path.join(out_dir, "major2_runs.csv")` -> `os.path.join(out_dir, "runs.csv")`<br>`os.path.join(out_dir, "major2_reproduction_check.csv")` -> `os.path.join(out_dir, "reproduction_check.csv")` |

## Reproduction checks run on the released code

All checks used the environment in `environment.yml` with one BLAS/OpenMP thread, on the released files
(not on the original working copies), writing to a separate directory and comparing with the archived files.

| Check | Released code exercised | Result |
|---|---|---|
| Energy Hub main study, strategy S0, seed 101, full sampling and evaluation (`run_campaign`) | `energy_hub/code/hybrid_pipeline` (instance, surrogates, QAOA sampler, campaign) | `unique_supports.csv` and `continuous_evaluations.csv` byte-identical to `data/canonical_5k/runs/S0_101/`; only runtime, timestamp and the configuration hash (which includes the path string of the reference file) differ |
| Stage-alignment correlations | `code/scripts/compute_correlations.py` on `data/canonical_5k` | output byte-identical to the archived `analysis/` files |
| Energy Hub figures | `energy_hub/figures/scripts/make_figures.py` | all 43 redrawn PNG and CSV files byte-identical to the published figures and plotted data |
| Ipopt residuals of the exhaustive reference, 5,931 supports | `energy_hub/ipopt_residuals/resolve_reference.py`, `summarize_residuals.py` | all 5,931 per-support records identical except runtime; `resolve_results.csv` rebuilt byte-identical |
| Portfolio QAOA training, sampling and Ipopt refinement (mixer-topology job C, seeds 0-9) | `portfolio/code/recompute_mixer_ext.py` with `Quantico-cardinalidad.py` | the three output CSV files byte-identical to `portfolio/out/mixer_C/` |
| Portfolio SI tables | `portfolio/summarize_canonical.py` | `out/summary/` byte-identical |
| Portfolio SI figures S1-S13 and figure data | `assemble_and_plot_legacy.py` (with `enlarge_fonts.py`), `make_figures_canonical.py` | `legacy_figure_data/` and every PNG byte-identical |
| Initialization controls, analysis step | `initialization_controls/initialization_controls.py analyze` | `runs.csv` and `reproduction_check.csv` byte-identical; unmodified configuration reproduces the archived Energy Hub samples in 30 of 30 runs |

Not re-run for this release (code verified only by the syntax-tree comparison above): the remaining
portfolio jobs of `run_canonical.py` (each takes 10-40 min or more), the sensitivity sweeps of SI S1.1/S1.2
(exhaustive references for K = 2, 4, 8 and three import caps), and the sampling step of the initialization
controls (its unmodified configuration was checked against the archive when the controls were run).

