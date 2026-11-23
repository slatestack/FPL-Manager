"""Optimiser constants (Section 6). Persisted with every run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

AGGRESSION = {"CONSERVATIVE": 1.25, "BALANCED": 1.0, "AGGRESSIVE": 0.75}


@dataclass(frozen=True)
class OptimizerParams:
    horizon: int = 5                         # gameweeks the transfer plan covers (Section 6)
    gamma: float = 0.85                      # temporal discount
    bench_epsilon: float = 0.08              # bench utility weight
    bench_weights: tuple[float, float, float, float] = (0.1, 1.0, 0.6, 0.3)  # GK slot, outfield slots 1..3
    vice_weight: float = 0.01                # tiny tie-break so the vice is a sensible starter
    hit_cost: int = 4
    max_per_club: int = 3                    # FPL rule; only small test leagues change it
    max_free_transfers: int = 5
    max_hits_per_gw: int = 1                 # from system_config; plans never exceed what execution allows
    ft_terminal_value: float = 1.0           # value of each FT banked beyond the horizon (pts, undiscounted)

    # Chip activation thresholds (BALANCED), scaled by the aggression profile.
    tc_min_gain: float = 12.0
    bb_min_bench_xp: float = 18.0
    bb_min_bench_play: float = 0.85
    fh_min_gain: float = 16.0
    fh_max_fielded: int = 8
    wc_min_gain: float = 22.0
    wc_min_unavailable: int = 3
    use_expiring_chips: bool = True          # a chip about to expire is used if it adds anything at all
    aggression: str = "BALANCED"

    # Candidate pool (dominated players are dropped before solving)
    pool_per_position: dict[str, int] = field(default_factory=lambda: {"GKP": 12, "DEF": 35, "MID": 40, "FWD": 25})
    pool_value_per_position: int = 8

    solver: str = "CBC"                      # CBC, SCIP and HiGHS all reach the same optimum; CBC is fastest here
    time_limit_seconds: int = 90
    relative_gap: float = 1e-4

    @property
    def threshold_scale(self) -> float:
        return AGGRESSION.get(self.aggression, 1.0)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
