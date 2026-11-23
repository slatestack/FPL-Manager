"""Every tunable constant of the xP engine in one place (persisted with each run for auditability)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelParams:
    horizon: int = 5

    # Team strength (Poisson attack/defence)
    team_xg_weight: float = 0.6            # target = w*xG + (1-w)*goals
    team_half_life_days: float = 150.0
    team_prev_season_factor: float = 0.6   # extra down-weight per season back
    team_ridge: float = 4.0
    promoted_prior: float = -0.15          # log-rate prior for attack & defence of promoted sides
    odds_weight: float = 0.7               # blend of bookmaker-implied vs model goal rates
    scorer_odds_weight: float = 0.6        # blend of bookmaker anytime-scorer odds vs model goal share
    scorer_odds_min_players: int = 8       # a team needs this many priced players for its odds to be used

    # Player rates
    rate_half_life_days: float = 365.0
    rate_prev_season_factor: float = 0.7
    goal_xg_weight: float = 0.8            # numerator blend: xG vs actual goals
    assist_xa_weight: float = 0.7          # numerator blend: calibrated xA vs actual assists
    share_prior_strength: float = 3.0      # in team-xG-on-pitch units (~2 matches)
    per90_prior_strength: float = 3.0      # in 90-minute units, for saves / DC / bps
    discipline_prior_strength: float = 15.0
    p60_prior_strength: float = 4.0        # in starts

    # Minutes classifiers (LightGBM)
    minutes_rounds: int = 300
    minutes_learning_rate: float = 0.05
    minutes_leaves: int = 15
    minutes_min_leaf: int = 100

    # Bonus model
    bonus_rounds: int = 300
    bonus_learning_rate: float = 0.05
    bonus_leaves: int = 31
    bonus_min_leaf: int = 40
    bonus_min_season: int = 2025           # BPS includes defensive actions from 2025/26
    bonus_simulations: int = 300

    seed: int = 20260
    threads: int = 1                       # LightGBM determinism

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
