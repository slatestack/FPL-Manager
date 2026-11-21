from __future__ import annotations

import numpy as np
import pandas as pd


def recency_weights(
    kickoff: pd.Series, season_idx: pd.Series, as_of: pd.Timestamp, current_idx: int, half_life_days: float, prev_factor: float
) -> np.ndarray:
    days = (as_of - kickoff).dt.total_seconds().to_numpy() / 86400.0
    days = np.clip(days, 0.0, None)
    w = 0.5 ** (days / half_life_days)
    gap = np.clip(current_idx - season_idx.to_numpy(), 0, None)
    return w * prev_factor**gap
