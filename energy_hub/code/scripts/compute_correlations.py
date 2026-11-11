"""Compute rho(E,p), rho(p,V), rho(E,V) per arm from a data/<campaign>/ directory's
aggregated unique_supports.csv and continuous_evaluations.csv.

Sign convention (see data/canonical_5k/analysis/correlations_README.md):
  rho_E_p = Spearman(-E(y), p_hat(y))   over unique sampled supports.
  rho_p_V = Spearman(p_hat(y), -V(y))   over B<=100 feasible evaluated supports.
  rho_E_V = Spearman(E(y), V(y))        raw, over the same evaluated supports.
Positive values always mean "preserved ordering toward lower energy / lower TAC".

Usage (run from the energy_hub/ directory):
    python code/scripts/compute_correlations.py data/canonical_5k
    python code/scripts/compute_correlations.py data/attribution/S1_15k_campaign_slice
"""
import sys
from pathlib import Path

import pandas as pd


def compute(data_dir: Path, arms: list[str]) -> pd.DataFrame:
    unique = pd.read_csv(data_dir / "unique_supports.csv")
    evals = pd.read_csv(data_dir / "continuous_evaluations.csv")

    unique["empirical_probability"] = pd.to_numeric(unique["empirical_probability"], errors="coerce")
    unique["surrogate_energy"] = pd.to_numeric(unique["surrogate_energy"], errors="coerce")
    evals["TAC"] = pd.to_numeric(evals["TAC"], errors="coerce")
    evals["nlp_feasible"] = evals["nlp_feasible"].astype(str).str.lower().eq("true")

    rows = []
    for arm in arms:
        sampled = unique[(unique["arm"] == arm) & unique["empirical_probability"].notna() & unique["surrogate_energy"].notna()]
        evaluated = evals[
            (evals["arm"] == arm)
            & (evals["evaluation_rank"] <= 100)
            & evals["nlp_feasible"]
            & evals["sampling_frequency"].notna()
            & evals["TAC"].notna()
        ]
        rho_E_p = float((-sampled["surrogate_energy"]).corr(sampled["empirical_probability"], method="spearman")) if len(sampled) > 1 else float("nan")
        if len(evaluated) > 1:
            rho_p_V = float(evaluated["sampling_frequency"].corr(-evaluated["TAC"], method="spearman"))
            rho_E_V = float(evaluated["surrogate_energy"].corr(evaluated["TAC"], method="spearman"))
        else:
            rho_p_V = float("nan")
            rho_E_V = float("nan")
        rows.append(
            {
                "arm": arm,
                "rho_E_p": rho_E_p,
                "rho_p_V": rho_p_V,
                "rho_E_V": rho_E_V,
                "n_unique_sampled": int(len(sampled)),
                "n_evaluated_feasible_B100": int(len(evaluated)),
            }
        )
    return pd.DataFrame(rows)


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(__doc__)
        return 1
    data_dir = Path(argv[0])
    unique = pd.read_csv(data_dir / "unique_supports.csv")
    arms = sorted(unique["arm"].unique().tolist())
    df = compute(data_dir, arms)
    print(df.to_string(index=False))
    out = data_dir / "analysis" / "correlations_by_arm.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
