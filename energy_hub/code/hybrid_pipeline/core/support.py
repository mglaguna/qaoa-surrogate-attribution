"""Support representation helpers."""

import numpy as np


def _to_bits(s: dict, instance: dict) -> str:
    return "".join(str(s[i]) for i in instance["ASSETS"])


def _selected_names(s: dict, instance: dict) -> list:
    return [i for i in instance["ASSETS"] if s[i] == 1]


def _selection_sectors(s: dict, instance: dict) -> dict:
    counts = {sec: 0 for sec in instance["SECTORS"]}
    for a in instance["ASSETS"]:
        if s[a] == 1:
            counts[instance["SECTOR_OF"][a]] += 1
    return counts


def candidate_pool_metadata(candidate_pool: list, shots: int) -> dict:
    """Compute the aggregate metrics of a candidate pool."""
    n_unique_raw = len(candidate_pool)
    n_exactK_raw = sum(1 for c in candidate_pool if c["is_exactK"])
    n_nonexact_raw = n_unique_raw - n_exactK_raw
    total_exactK_shots = sum(c["count"] for c in candidate_pool if c["is_exactK"])
    exact_probs = np.array(
        [c["count"] for c in candidate_pool if c["is_exactK"]],
        dtype=float,
    )
    if total_exactK_shots > 0 and len(exact_probs) > 0:
        exact_probs = exact_probs / total_exactK_shots
        entropy_exactK = float(-np.sum(exact_probs * np.log(exact_probs + 1e-15)))
        top1_exactK_freq = float(np.max(exact_probs))
    else:
        entropy_exactK = 0.0
        top1_exactK_freq = 0.0

    return dict(
        n_unique_raw=n_unique_raw,
        n_exactK_raw=n_exactK_raw,
        n_nonexact_raw=n_nonexact_raw,
        n_unique_exactK_raw=n_exactK_raw,
        frac_exactK_shots=total_exactK_shots / shots,
        entropy_exactK=entropy_exactK,
        top1_exactK_freq=top1_exactK_freq,
    )


to_bits = _to_bits
selected_names = _selected_names
selection_sectors = _selection_sectors
