"""Collect the per-support records written by resolve_reference.py into one table and print the summary.

Adds the archived reference values (feasible_ref, TAC_ref) from data/reference/.
Usage (from energy_hub/): python ipopt_residuals/summarize_residuals.py OUT_DIR RESULTS_CSV
"""
import glob
import json
import os
import sys

import pandas as pd

out_dir, results_csv = sys.argv[1], sys.argv[2]
rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out_dir, "*.json")))]
df = pd.DataFrame(rows)
ref = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "reference",
                               "energy_hub_exhaustive_classical_rows.csv"), dtype={"support_bitstring": str})
ref = ref[["support_bitstring", "feasible", "TAC"]].rename(columns={"feasible": "feasible_ref", "TAC": "TAC_ref"})
df = df.merge(ref, on="support_bitstring", how="left")
df.to_csv(results_csv, index=False)

feas = df[df.termination == "optimal"]
print(f"supports re-solved: {len(df)}; reported optimal: {len(feas)}; other: {len(df) - len(feas)}")
print("Ipopt exit messages:", df.ipopt_exit.value_counts().to_dict())
rel = ((feas.TAC - feas.TAC_ref).abs() / feas.TAC_ref.abs())
print(f"max relative TAC difference to the archived reference: {rel.max():.3e}")
for col in ("nlp_error_unscaled", "constr_viol_unscaled", "dual_inf_unscaled", "complementarity_unscaled",
            "max_abs_residual", "max_rel_residual"):
    print(f"max {col}: {feas[col].max():.3e}")
print(f"most negative variable value: {feas.min_var_value.min():.3e}")
inf = df[df.termination != "optimal"]
if len(inf):
    print(f"infeasible by Ipopt: {len(inf)}; LP minimum total violation: min {inf.lp_min_total_violation.min():.3e}")
