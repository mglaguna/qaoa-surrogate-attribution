# Surrogate attribution in hybrid quantum–classical QAOA: code and data

**Authors:** Milán A. Moraila-Anchondo<sup>a</sup>, Francisco Javier López-Flores<sup>a</sup>, Marcos Daniel Gonzalez Llanes<sup>a</sup>, Oscar Martín Hernández Calderón<sup>a</sup>, Eusiel Rubio-Castro<sup>a</sup>, Maria G. Laguna-Martinez<sup>a,&#42;</sup>

<sup>a</sup> Graduate Program in Chemical Engineering, Chemical and Biological Sciences Department, Universidad Autónoma de Sinaloa, Culiacán, Sinaloa 80000, Mexico

<sup>&#42;</sup> Corresponding author: Maria G. Laguna-Martinez, mglaguna@uas.edu.mx

Code, data and figures behind the manuscript *Surrogate Attribution in Hybrid Quantum–Classical QAOA for
Discrete–Continuous Energy Hub Design* (under review). Two fully enumerable discrete–continuous
benchmarks are included:

| Folder | Benchmark | Manuscript |
|---|---|---|
| `energy_hub/` | Exact-cardinality Energy Hub design (N = 18 technologies, K = 6; 18,564 supports, 5,801 feasible) | Sections 2–6, Figures 1–12, SI Section S1 |
| `portfolio/` | Cardinality-constrained mean–variance portfolio (N = 15, K = 5; 3,003 supports, 2,211 feasible) | Section 7, SI Section S2 |
| `initialization_controls/` | Initialization controls of the exact-K QAOA samplers, both benchmarks | Sections 5–7, SI Section S3 |

Every result can be checked against an exhaustive reference: the continuous value of every exact-K support
is known.

## Environment

```
conda env create -f environment.yml
conda activate qaoa-attribution
```

All results were produced on Windows 10 (x86-64) with Python 3.9.23, PennyLane 0.38.0, NumPy 1.26.4,
autograd 1.8.0, Pyomo 6.9.3 and Ipopt 3.14.19 (`environment.yml` lists every pinned version).

**Run every QAOA computation with one BLAS/OpenMP thread.** The training of the portfolio circuit is
sensitive to the floating-point reduction order, so the trained angles change with the number of threads:

```
set OMP_NUM_THREADS=1 & set MKL_NUM_THREADS=1 & set OPENBLAS_NUM_THREADS=1 & set NUMEXPR_NUM_THREADS=1
```

(`export ...=1` on Linux/macOS). The Energy Hub results are reproduced bit for bit under this setting.

## Energy Hub (`energy_hub/`)

```
energy_hub/
  code/hybrid_pipeline/   instance, fixed-support model, surrogates S0/S1/S2, QAOA sampler, campaign, analysis
  code/scripts/           compute_correlations.py
  data/reference/         exhaustive exact-K reference (all 18,564 supports; feasibility and TAC)
  data/canonical_5k/      main study: S0, S1, S2 and Uniform, 10 seeds each, 5,000 shots or draws per seed
  data/attribution/       15,000-shot S1 configuration vs. independent 5,000-shot draws (Figure 8, SI S1.3)
  data/appendix_sensitivity/  K sweep (SI S1.1) and grid-import-cap sweep (SI S1.2)
  figures/                Figures 1-12 (main/, supplement/), plotted data (data/), scripts (scripts/)
  ipopt_residuals/        Ipopt tolerances and constraint residuals of the exhaustive reference
```

From `energy_hub/` (with `code/` on `PYTHONPATH`, or as in the commands below):

1. Re-run one strategy-seed unit of the main study and compare it with the archive (about one minute):
   ```
   python -c "import sys; sys.path.insert(0,'code'); from pathlib import Path; from hybrid_pipeline.experiments import energy_hub_canonical_config as cfg, energy_hub_canonical_campaign as c; cfg.EXHAUSTIVE_REFERENCE_PATH = Path('data/reference/energy_hub_exhaustive_classical_rows.csv'); print(c.run_campaign(Path('rerun'), resume=False, arms=['S0'], seeds=[101])['final_status'])"
   ```
   `rerun/runs/S0_101/unique_supports.csv` and `continuous_evaluations.csv` are byte-identical to
   `data/canonical_5k/runs/S0_101/`. (Run metadata such as runtime and the configuration hash, which includes
   the path of the reference file, differ.)
2. Analysis tables (`data/canonical_5k/analysis/`):
   `python -c "import sys; sys.path.insert(0,'code'); from pathlib import Path; from hybrid_pipeline.experiments import energy_hub_canonical_analysis as a; a.run_analysis(Path('data/canonical_5k'))"`
3. Stage-alignment correlations: `python code/scripts/compute_correlations.py data/canonical_5k`
4. Figures: `cd figures/scripts && python make_figures.py` (see the table in that script for the
   correspondence with the manuscript figures).
5. Ipopt residuals of the reference (about 90 s with 16 workers):
   `python ipopt_residuals/resolve_reference.py OUT 16` and
   `python ipopt_residuals/summarize_residuals.py OUT results.csv`.
6. Sensitivity sweeps (SI S1.1, S1.2):
   `python code/hybrid_pipeline/experiments/energy_hub_k_sweep_appendix.py --output-dir OUT_K` and
   `python code/hybrid_pipeline/experiments/energy_hub_import_cap_sensitivity_appendix.py --output-dir OUT_CAP`
   (set `PYTHONPATH=code`).

## Portfolio (`portfolio/`)

```
portfolio/
  code/            benchmark, hybrid loop and experiments (Quantico-cardinalidad.py, Fase2.py,
                   Fase3_separacion_efectos.py, recompute_*.py), figure script, baseline_enumeration.csv
  data/            surrogate_correlation_table.csv (classical input of SI Figure S2)
  out/             canonical one-thread recompute: one folder per experiment, summary/ (SI tables),
                   legacy_figure_data/ and legacy_figures_fonts/ (SI Figures S1-S5, S7-S9),
                   run_record.json (environment and input hashes), MANIFEST_SHA256.json
  figures/         SI Figures S6, S10a, S10b, S11, S12, S13
  run_canonical.py, summarize_canonical.py, assemble_and_plot_legacy.py, make_figures_canonical.py
```

From `portfolio/`:

1. All QAOA-dependent experiments, seeds 0-9 (nine jobs run in parallel, each with one thread; allow
   one to several hours): `python run_canonical.py OUT`
2. SI tables: `python summarize_canonical.py OUT`
3. SI Figures S1-S5 and S7-S9: `python assemble_and_plot_legacy.py OUT`; the SI versions use a minimum
   text size of 8 pt:
   `python ../energy_hub/figures/scripts/enlarge_fonts.py code/regenerate_manuscript_outputs_legacy.py --main --figures --data-dir OUT/legacy_figure_data --out-dir FIGS`
   (and the same with `--supp`).
4. SI Figures S6 and S10-S13: `python make_figures_canonical.py OUT ../initialization_controls/out FIGS`

`run_canonical.py` looks for Ipopt in the `Library/bin` folder of the active conda environment (Windows
layout); on other systems put `ipopt` on the `PATH`.

## Initialization controls (`initialization_controls/`)

`initialization_controls.py` re-runs the exact-K QAOA samplers of both benchmarks with (i) the initial support
only (zero depth), (ii) alternative initial supports and (iii) consistent asset-to-qubit permutations, with
everything else identical to the archived runs. Values come from the exhaustive references; no continuous
solves are issued.

```
python initialization_controls.py eh  OUT 8      # Energy Hub, S0/S1/S2, seeds 101-110
python initialization_controls.py pf  OUT 8      # portfolio, seeds 0-9
python initialization_controls.py analyze OUT    # writes OUT/runs.csv and OUT/reproduction_check.csv
```

`out/` holds the per-run records (`*.json`), `runs.csv` and `reproduction_check.csv`; the unmodified
configuration reproduces the archived Energy Hub samples in 30 of 30 runs.

## Integrity and provenance

* `MANIFEST_SHA256.json` lists the SHA-256 of every file; check it with `python verify_manifest.py`.
* `VERIFICATION.md` documents how the released code was derived from the code that produced the archived
  results (comments removed, docstrings and console messages in English, relative paths) and the
  reproduction checks that were run on the released code.
* `portfolio/out/run_record.json` and `portfolio/out/MANIFEST_SHA256.json` are the records of the original
  canonical run; they hash the code as it was executed, before the cleaning described in `VERIFICATION.md`.

## Known limitations

* A few console messages of the released code are still in Spanish (`energy_hub/code/hybrid_pipeline/samplers/qaoa.py`,
  `energy_hub/code/hybrid_pipeline/core/tabu.py`, `portfolio/code/Quantico-cardinalidad.py`,
  `portfolio/make_figures_canonical.py`). The `run.log` files of the original portfolio runs are also in
  Spanish. This does not affect any result.
* `portfolio/code/Quantico-cardinalidad.py`, `Fase2.py` and `Fase3_separacion_efectos.py` keep the names of the
  original working scripts; the first contains a hyphen and is loaded by path, not imported.
* In `portfolio/code/Quantico-cardinalidad.py`, `select_candidates` refers to `mu_score_support`, which is
  defined in `Fase2.py`. That branch (policy `ranked_mu` of the original `run_hybrid` loop) is not exercised
  by the released workflows, which use the `Fase2.py` implementation; the code is kept as in the original.

## AI assistance

Parts of the development of this code, of its documentation and of the verification of the released
files were assisted by AI tools. All AI-assisted outputs were reviewed and validated by the authors,
who take full responsibility for the content.

* **Claude Code** (Anthropic): coding assistance and technical verification, including the cleaning of
  the released code (comment removal, English docstrings and console messages, relative paths) checked
  against the original by syntax-tree comparison, the analysis, figure and release-build scripts, the
  documentation (`README.md`, `VERIFICATION.md`), and audits of the results (recomputations and
  consistency checks between the manuscript, the Supplementary Information and the data).
* **Codex** (OpenAI): review of the code changes, whether written with Claude Code or by the authors,
  to check that nothing was broken.

The AI tools did not alter the numerical results, which are those of the archived runs; they were used
to help audit them. The results can be checked with `MANIFEST_SHA256.json` and the reproduction checks
in `VERIFICATION.md`.

## License

Code: MIT License (`LICENSE`). Data, results and figures: Creative Commons Attribution 4.0 International
(`LICENSE-DATA.md`).

## Citation

See `CITATION.cff`.
