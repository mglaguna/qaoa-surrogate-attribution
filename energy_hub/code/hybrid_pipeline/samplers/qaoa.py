"""QAOA Hamiltonian, circuit, training and diagnostic utilities."""

from __future__ import annotations

from math import comb, log
from typing import Callable, Optional

import numpy as np

from hybrid_pipeline.cli.config import HAMILTONIAN_PENALTY
from hybrid_pipeline.core.selection import build_candidate_pool
from hybrid_pipeline.core.support import candidate_pool_metadata

try:
    import pennylane as qml
    import pennylane.numpy as pnp
except Exception:
    qml = None
    pnp = None


def require_pennylane():
    if qml is None or pnp is None:
        raise RuntimeError(
            "PennyLane no esta disponible. Instala pennylane para usar QAOA; "
            "los modos de enumeracion/sanity pueden ejecutarse sin QAOA."
        )


def build_exactK_hamiltonian(n: int, k: int, penalty: float = 1.0):
    """
    Build the exact-cardinality Hamiltonian ``penalty * (sum_i y_i - k)^2``.
    """
    require_pennylane()
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
):
    """
    Build the support-prioritization surrogate Hamiltonian.
    """
    require_pennylane()
    if instance is None:
        raise ValueError("instance es obligatorio y debe contener mu, cov y R.")

    ASSETS = instance["ASSETS"]
    if instance.get("BENCHMARK") == "energy_hub":
        if len(ASSETS) != n:
            raise ValueError(
                f"Inconsistencia: n={n}, pero instance['ASSETS'] tiene {len(ASSETS)} tecnologias."
            )
        d = instance["ENERGY_HUB_DATA"]
        singleton_scores = []
        for a in ASSETS:
            heat = d["Smax"][a] * float(d["rho"].loc[a].max()) * max(0.0, d["etaQ"][a])
            cool = d["Smax"][a] * float(d["rho"].loc[a].max()) * max(0.0, d["etaC"][a])
            elec = d["Smax"][a] * float(d["rho"].loc[a].mean()) * max(0.0, d["etaE"][a])
            cost = d["CRF"] * d["a"][a] + 0.1 * d["alpha"][a]
            singleton_scores.append(heat + cool + 0.5 * elec - cost)
        scale = max([abs(v) for v in singleton_scores] + [1.0])
        utility = np.array(singleton_scores, dtype=float) / scale

        z_coeff = np.zeros(n, dtype=float)
        zz_coeff = {}
        for i in range(n):
            z_coeff[i] += -lambda_ret * utility[i] / 2.0

        group_of = d["group_of"]
        redundancy_weight = 0.08
        for i in range(n):
            for j in range(i + 1, n):
                if group_of[ASSETS[i]] == group_of[ASSETS[j]]:
                    c_pair = redundancy_weight
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

    mu = instance["mu"].loc[ASSETS].astype(float).values
    cov = instance["cov"].loc[ASSETS, ASSETS].astype(float).values
    R = float(instance["R"])

    if len(ASSETS) != n:
        raise ValueError(
            f"Inconsistencia: n={n}, pero instance['ASSETS'] tiene {len(ASSETS)} activos."
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
    H_cost,
    k: int,
    shots: int = 5000,
    mixer_topology: str = "ring",
    pl_seed: Optional[int] = None,
):
    """
    Build exact-cardinality QAOA circuits with an XY mixer.
    """
    require_pennylane()
    if not (0 <= k <= n):
        raise ValueError(f"k debe satisfacer 0 <= k <= n; recibido k={k}, n={n}")

    init_state = np.array([1] * k + [0] * (n - k), dtype=int)

    if mixer_topology == "ring":
        pairs = [(i, (i + 1) % n) for i in range(n)]
    elif mixer_topology == "line":
        pairs = [(i, i + 1) for i in range(n - 1)]
    else:
        raise ValueError("mixer_topology debe ser 'ring' o 'line'")

    coeffs_mix = []
    ops_mix = []
    for i, j in pairs:
        coeffs_mix.extend([1.0, 1.0])
        ops_mix.append(qml.PauliX(wires=i) @ qml.PauliX(wires=j))
        ops_mix.append(qml.PauliY(wires=i) @ qml.PauliY(wires=j))

    H_mix = qml.Hamiltonian(coeffs_mix, ops_mix)

    try:
        dev_exp = qml.device("default.qubit", wires=n, seed=pl_seed)
        dev_sample = qml.device("default.qubit", wires=n, shots=shots, seed=pl_seed)
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


def train_qaoa(
    circuit_exp,
    p: int,
    n_steps: int = 100,
    stepsize: float = 0.1,
    verbose: bool = True,
    rng=None,
):
    """Train the QAOA angles with the configured optimizer and initialization."""
    require_pennylane()
    _rng = rng if rng is not None else np.random
    params = pnp.array(_rng.uniform(0, np.pi, 2 * p), requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=stepsize)
    for step in range(n_steps):
        params = opt.step(lambda v: circuit_exp(v), params)
        if verbose and (step + 1) % 25 == 0:
            e = float(circuit_exp(params))
            print(f"  step {step+1}/{n_steps} | energy = {e:.5f}")
    return params


def characterize_qaoa_distribution(circuit_sample, params, instance: dict, shots: int = 5000) -> dict:
    """
    Diagnose whether a QAOA sampling distribution is distinguishable from uniform.
    """
    N = instance["N"]
    K = instance["K"]

    raw = circuit_sample(params)
    n_samples = len(raw)

    if shots != n_samples:
        print(
            f"[characterize_qaoa_distribution] Advertencia: "
            f"shots solicitados={shots} pero muestras observadas={n_samples}. "
            "Se usarán las muestras observadas para el diagnóstico."
        )

    counts = {}
    for row in raw:
        b = tuple(int(x) for x in row)
        counts[b] = counts.get(b, 0) + 1

    exact_counts = {b: c for b, c in counts.items() if sum(b) == K}
    total_exact = sum(exact_counts.values())
    frac_exact = total_exact / n_samples if n_samples > 0 else np.nan
    n_unique = len(exact_counts)
    n_possible = comb(N, K)

    probs_qaoa = np.array([])
    if total_exact > 0:
        probs_qaoa = np.array(list(exact_counts.values()), dtype=float)
        probs_qaoa /= probs_qaoa.sum()
        entropy_qaoa = float(-np.sum(probs_qaoa * np.log(probs_qaoa + 1e-15)))
    else:
        entropy_qaoa = 0.0

    entropy_uniform = log(n_possible)
    entropy_ratio = entropy_qaoa / entropy_uniform if entropy_uniform > 0 else 0.0

    if n_unique > 0 and n_possible > 0:
        p_uniform_teorica = 1.0 / n_possible
        tvd_obs = 0.5 * float(np.sum(np.abs(probs_qaoa - p_uniform_teorica)))
        tvd_unobs = 0.5 * (n_possible - n_unique) * p_uniform_teorica
        tvd = tvd_obs + tvd_unobs
    else:
        tvd = 1.0

    if len(probs_qaoa) > 0:
        sorted_probs = np.sort(probs_qaoa)[::-1]
        top10_n = max(1, int(0.1 * len(sorted_probs)))
        top10_mass = float(sorted_probs[:top10_n].sum())
    else:
        top10_mass = 1.0

    verdict = (
        "indistinguishable_from_uniform"
        if tvd < 0.10 and entropy_ratio > 0.90
        else "distinguishable"
    )

    print("\n=== Caracterización de la distribución QAOA ===")
    print(f"  frac_exact_K       : {frac_exact:.4f}  (shots que caen en cardinalidad exacta)")
    print(f"  n_unique_exact_K   : {n_unique} de C({N},{K})={n_possible} posibles")
    print(f"  entropy_qaoa       : {entropy_qaoa:.4f}")
    print(f"  entropy_uniform    : {entropy_uniform:.4f}  (log C({N},{K}))")
    print(f"  entropy_ratio      : {entropy_ratio:.4f}  (1.0 = idéntico a uniforme)")
    print(f"  TVD vs. uniforme   : {tvd:.4f}  (0.0 = identico a C({N},{K}) uniforme teorica)")
    print(f"  top10pct_mass      : {top10_mass:.4f}  (masa en el 10% más frecuente)")
    print(f"  VEREDICTO          : {verdict}")
    if verdict == "indistinguishable_from_uniform":
        print("  -> El componente QAOA es funcionalmente equivalente a muestreo")
        print("    aleatorio uniforme. La contribución es arquitectural (tabu).")
    else:
        print("  -> La distribución QAOA está sesgada respecto a la uniforme.")
        print("    Verificar si el sesgo correlaciona con calidad del portafolio.")

    return dict(
        frac_exact_K=frac_exact,
        n_unique_exact_K=n_unique,
        n_possible=n_possible,
        entropy_qaoa=entropy_qaoa,
        entropy_uniform=entropy_uniform,
        entropy_ratio=entropy_ratio,
        TVD=tvd,
        top10pct_mass=top10_mass,
        verdict=verdict,
        requested_shots=shots,
        observed_shots=n_samples,
        shots_mismatch=(shots != n_samples),
    )


def build_qaoa_sampler(circuit_sample, instance: dict, rng=None, surrogate: dict | None = None) -> Callable:
    """
    Build a QAOA sampler compatible with run_hybrid.
    """

    def qaoa_sampler(params, shots: int, top_k: int, alpha: float, beta: float) -> tuple:
        raw = circuit_sample(params)
        n_samples = len(raw)
        if shots != n_samples:
            print(
                f"[build_qaoa_sampler] Advertencia: shots solicitados={shots} "
                f"pero muestras observadas={n_samples}. "
                "Se usarán las muestras observadas para normalizar frecuencias."
            )
        counts = {}
        for row in raw:
            b = tuple(int(x) for x in row)
            counts[b] = counts.get(b, 0) + 1
        full_candidate_pool = build_candidate_pool(
            counts,
            n_samples,
            instance,
            alpha,
            beta,
            surrogate=surrogate,
            surrogate_type=surrogate.get("surrogate_type") if surrogate else None,
        )
        meta = candidate_pool_metadata(full_candidate_pool, n_samples)
        meta["requested_shots"] = shots
        meta["observed_shots"] = n_samples
        meta["shots_mismatch"] = shots != n_samples
        return full_candidate_pool, meta

    qaoa_sampler.__name__ = "qaoa_sampler"
    return qaoa_sampler


def build_hamiltonian_from_qubo(surrogate: dict):
    """Build a PennyLane Hamiltonian from an Energy Hub QUBO surrogate."""
    require_pennylane()
    from hybrid_pipeline.benchmarks.energy_hub.surrogates import (
        qubo_to_ising_coefficients,
    )

    ising = qubo_to_ising_coefficients(surrogate)
    ops, coeffs = [], []
    for idx, asset in enumerate(surrogate["assets"]):
        coeff = float(ising["linear_z"].get(asset, 0.0))
        if abs(coeff) > 1e-12:
            ops.append(qml.PauliZ(wires=idx))
            coeffs.append(coeff)
    asset_idx = {asset: idx for idx, asset in enumerate(surrogate["assets"])}
    for (i, j), coeff in ising["quadratic_zz"].items():
        coeff = float(coeff)
        if abs(coeff) > 1e-12:
            ops.append(qml.PauliZ(wires=asset_idx[i]) @ qml.PauliZ(wires=asset_idx[j]))
            coeffs.append(coeff)
    if not ops:
        ops.append(qml.Identity(wires=0))
        coeffs.append(0.0)
    return qml.Hamiltonian(coeffs, ops)


def xy_exactk_reference_factory(seed: int, instance: dict, config: dict) -> tuple:
    """
    XY exact-K reference factory.
    """
    require_pennylane()
    rng = np.random.RandomState(seed)
    H = build_exactK_hamiltonian(
        instance["N"], instance["K"], penalty=HAMILTONIAN_PENALTY
    )
    c_exp, c_sample = make_circuits(
        instance["N"],
        config["qaoa_layers"],
        H,
        k=instance["K"],
        shots=config["qaoa_shots"],
        pl_seed=seed,
    )

    params_reference = pnp.array(
        rng.uniform(0, np.pi, 2 * config["qaoa_layers"]),
        requires_grad=False,
    )

    sampler = build_qaoa_sampler(c_sample, instance)
    return sampler, params_reference


def qaoa_surrogate_factory(seed: int, instance: dict, config: dict) -> tuple:
    """
    QAOA surrogate factory.
    """
    require_pennylane()
    rng = np.random.RandomState(seed)
    H = build_support_surrogate_hamiltonian(
        instance["N"],
        instance["K"],
        penalty=HAMILTONIAN_PENALTY,
        instance=instance,
    )
    c_exp, c_sample = make_circuits(
        instance["N"],
        config["qaoa_layers"],
        H,
        k=instance["K"],
        shots=config["qaoa_shots"],
        pl_seed=seed,
    )
    p_trained = train_qaoa(
        c_exp,
        p=config["qaoa_layers"],
        n_steps=config["qaoa_steps"],
        verbose=False,
        rng=rng,
    )
    sampler = build_qaoa_sampler(c_sample, instance)
    return sampler, p_trained


def energy_hub_surrogate_factory(
    seed: int,
    instance: dict,
    config: dict,
    surrogate_type: str,
) -> tuple:
    """
    Explicit Energy Hub S0/S1/S2 QAOA surrogate factory.
    """
    require_pennylane()
    if instance.get("BENCHMARK") != "energy_hub":
        raise ValueError("energy_hub_surrogate_factory only supports Energy Hub instances.")
    from hybrid_pipeline.benchmarks.energy_hub.surrogates import (
        build_energy_hub_s0,
        build_energy_hub_s1,
        build_energy_hub_s2,
    )

    builders = {
        "S0": build_energy_hub_s0,
        "S1": build_energy_hub_s1,
        "S2": build_energy_hub_s2,
    }
    key = surrogate_type.upper()
    if key not in builders:
        raise ValueError(f"Unknown Energy Hub surrogate_type: {surrogate_type}")
    surrogate = builders[key](instance)
    H = build_hamiltonian_from_qubo(surrogate)
    rng = np.random.RandomState(seed)
    c_exp, c_sample = make_circuits(
        instance["N"],
        config["qaoa_layers"],
        H,
        k=instance["K"],
        shots=config["qaoa_shots"],
        pl_seed=seed,
    )
    p_trained = train_qaoa(
        c_exp,
        p=config["qaoa_layers"],
        n_steps=config["qaoa_steps"],
        verbose=False,
        rng=rng,
    )
    sampler = build_qaoa_sampler(c_sample, instance, surrogate=surrogate)
    return sampler, p_trained, surrogate
