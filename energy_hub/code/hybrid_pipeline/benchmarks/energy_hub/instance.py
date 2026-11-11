"""Instance construction for the Energy Hub benchmark.

Only the catalog and instance factory are extracted in Commit 2. Functional
methods from the monolith such as ``structural_filter``, ``evaluate_support``,
``surrogate_score`` and ``solve_full_miqp`` are deliberately pending migration.
"""

from math import comb

import pandas as pd

from hybrid_pipeline.cli.config import GLOBAL_SEED


class EnergyHubInstanceBuilder:
    """Build the fixed Energy Hub catalog and benchmark instance."""

    def __init__(
        self,
        N: int = 18,
        K: int = 6,
        T: int = 6,
        seed: int = GLOBAL_SEED,
        *,
        allow_supplementary_k: bool = False,
    ):
        """``allow_supplementary_k`` is an explicit, opt-in escape hatch for
        supplementary/appendix studies only (e.g. a K-sensitivity sweep). It
        still requires (N, T) == (18, 6) -- only K may vary -- and every
        existing call site (the frozen EH_CANONICAL_V1 campaign included)
        leaves it at the default False, so the K=6 canonical benchmark's
        behavior is completely unchanged.
        """
        if allow_supplementary_k:
            if (N, T) != (18, 6):
                raise ValueError(
                    "allow_supplementary_k only permits varying K; N and T must remain (18, 6)."
                )
        elif (N, K, T) != (18, 6, 6):
            raise ValueError("EnergyHubBenchmark is fixed at N=18, K=6, T=6.")
        self.N = N
        self.K = K
        self.T = T
        self.seed = seed
        self.data = self._build_catalog()

    def _build_catalog(self) -> dict:
        techs = [f"tech_{i:02d}" for i in range(1, self.N + 1)]
        groups = {
            "cogen": techs[0:6],
            "renewable": techs[6:10],
            "thermal": techs[10:14],
            "cooling": techs[14:18],
        }
        group_of = {t: g for g, vals in groups.items() for t in vals}
        idx = {t: i for i, t in enumerate(techs)}

        etaE = pd.Series(
            [
                0.46,
                0.42,
                0.38,
                0.50,
                0.34,
                0.40,
                0.95,
                0.90,
                0.88,
                0.82,
                0.05,
                0.04,
                0.03,
                0.02,
                0.10,
                0.08,
                0.06,
                0.04,
            ],
            index=techs,
        )
        etaQ = pd.Series(
            [
                0.42,
                0.48,
                0.55,
                0.30,
                0.62,
                0.36,
                0.05,
                0.08,
                0.12,
                0.04,
                0.92,
                0.88,
                0.84,
                0.78,
                0.08,
                0.10,
                0.06,
                0.04,
            ],
            index=techs,
        )
        etaC = pd.Series(
            [
                0.12,
                0.08,
                0.06,
                0.10,
                0.05,
                0.15,
                0.02,
                0.03,
                0.04,
                0.02,
                0.10,
                0.06,
                0.08,
                0.04,
                0.95,
                0.88,
                0.82,
                0.76,
            ],
            index=techs,
        )
        Smax = pd.Series(
            [
                90,
                85,
                80,
                70,
                75,
                65,
                60,
                55,
                50,
                45,
                85,
                80,
                75,
                70,
                80,
                75,
                70,
                65,
            ],
            index=techs,
            dtype=float,
        )
        a = pd.Series(
            [
                210,
                190,
                175,
                230,
                160,
                205,
                140,
                125,
                115,
                105,
                95,
                90,
                84,
                78,
                130,
                122,
                116,
                110,
            ],
            index=techs,
            dtype=float,
        )
        b = pd.Series(
            [
                5.8,
                5.4,
                5.0,
                6.2,
                4.8,
                5.6,
                7.4,
                7.0,
                6.6,
                6.1,
                3.4,
                3.2,
                3.0,
                2.8,
                4.6,
                4.3,
                4.1,
                3.9,
            ],
            index=techs,
            dtype=float,
        )
        alpha = pd.Series(
            [
                26,
                25,
                24,
                29,
                23,
                27,
                3,
                4,
                5,
                6,
                17,
                16,
                15,
                14,
                13,
                12,
                11,
                10,
            ],
            index=techs,
            dtype=float,
        )
        beta = pd.Series(
            [
                0.034,
                0.032,
                0.030,
                0.036,
                0.028,
                0.033,
                0.010,
                0.011,
                0.012,
                0.013,
                0.018,
                0.017,
                0.016,
                0.015,
                0.021,
                0.020,
                0.019,
                0.018,
            ],
            index=techs,
            dtype=float,
        )

        rho = pd.DataFrame(
            [
                [0.92, 0.88, 0.84, 0.95, 0.80, 0.86],
                [0.88, 0.90, 0.86, 0.82, 0.84, 0.89],
                [0.80, 0.82, 0.88, 0.90, 0.84, 0.86],
                [0.95, 0.93, 0.90, 0.88, 0.86, 0.92],
                [0.78, 0.80, 0.82, 0.84, 0.86, 0.88],
                [0.84, 0.86, 0.88, 0.86, 0.84, 0.82],
                [0.45, 0.70, 0.95, 0.85, 0.55, 0.35],
                [0.50, 0.75, 0.90, 0.80, 0.60, 0.40],
                [0.60, 0.65, 0.75, 0.70, 0.62, 0.58],
                [0.35, 0.50, 0.80, 0.82, 0.55, 0.30],
                [0.92, 0.90, 0.88, 0.86, 0.90, 0.94],
                [0.86, 0.88, 0.90, 0.92, 0.88, 0.84],
                [0.80, 0.84, 0.88, 0.90, 0.86, 0.82],
                [0.78, 0.82, 0.86, 0.88, 0.84, 0.80],
                [0.88, 0.84, 0.78, 0.80, 0.86, 0.90],
                [0.82, 0.80, 0.76, 0.78, 0.82, 0.86],
                [0.78, 0.76, 0.74, 0.76, 0.80, 0.84],
                [0.74, 0.72, 0.70, 0.72, 0.76, 0.80],
            ],
            index=techs,
            columns=[f"t{t+1}" for t in range(self.T)],
        )

        periods = [f"t{t+1}" for t in range(self.T)]
        tau = pd.Series([520, 610, 740, 690, 580, 430], index=periods, dtype=float)
        DE = pd.Series([92, 108, 126, 118, 102, 86], index=periods, dtype=float)
        DQ = pd.Series([72, 68, 55, 48, 62, 78], index=periods, dtype=float)
        DC = pd.Series([40, 54, 76, 82, 58, 35], index=periods, dtype=float)
        cE = pd.Series([72, 84, 118, 130, 96, 68], index=periods, dtype=float)

        return dict(
            techs=techs,
            groups=groups,
            group_of=group_of,
            idx=idx,
            periods=periods,
            etaE=etaE,
            etaQ=etaQ,
            etaC=etaC,
            Smax=Smax,
            a=a,
            b=b,
            alpha=alpha,
            beta=beta,
            rho=rho,
            tau=tau,
            DE=DE,
            DQ=DQ,
            DC=DC,
            cE=cE,
            CRF=0.11,
        )

    def generate_instance(self) -> dict:
        d = self.data
        n_total = comb(self.N, self.K)
        return dict(
            BENCHMARK="energy_hub",
            OBJECTIVE_NAME="TAC",
            ASSETS=d["techs"],
            SECTORS=d["groups"],
            SECTOR_OF=d["group_of"],
            ASSET_IDX=d["idx"],
            N=self.N,
            K=self.K,
            R=0.0,
            N_TOTAL_COMBOS=n_total,
            N_HIGH=self.N,
            N_INFEASIBLE_FAMILY=0,
            INFEASIBLE_SHARE=0.0,
            energy_hub=self,
            ENERGY_HUB_DATA=d,
        )


def build_energy_hub_instance() -> dict:
    """Return a fresh fixed Energy Hub benchmark instance."""
    return EnergyHubInstanceBuilder().generate_instance()
