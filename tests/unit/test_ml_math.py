from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom, poisson

from app.ml.availability import availability, return_date
from app.ml.engine import aggregate_gameweeks, expected_floor_div, prob_at_least
from app.ml.minutes import anchor_features
from app.ml.params import ModelParams
from app.ml.team_strength import fit_team_ratings


@pytest.mark.parametrize("mu", [0.3, 1.4, 4.0, 9.0])
@pytest.mark.parametrize("n", [2, 3])
def test_expected_floor_div_matches_brute_force(mu, n):
    x = np.arange(0, 200)
    brute = float(np.sum((x // n) * poisson.pmf(x, mu)))
    assert expected_floor_div(np.array([mu]), n)[0] == pytest.approx(brute, abs=1e-9)


def test_prob_at_least_poisson_and_negbin():
    mu = np.array([8.0, 8.0])
    thr = np.array([10.0, 10.0])
    size = np.array([np.inf, 5.0])
    p = prob_at_least(mu, thr, size)
    assert p[0] == pytest.approx(poisson.sf(9, 8.0))
    assert p[1] == pytest.approx(nbinom.sf(9, 5.0, 5.0 / 13.0))
    assert p[1] > p[0], "overdispersion fattens the upper tail"


def _synthetic_fixtures(att: dict[int, float], dfn: dict[int, float], n_rounds: int = 6, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = list(att)
    rows, fid = [], 0
    start = pd.Timestamp("2026-08-15", tz="UTC")
    for r in range(n_rounds):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                lh = np.exp(0.2 + 0.15 + att[h] - dfn[a])
                la = np.exp(0.2 + att[a] - dfn[h])
                fid += 1
                rows.append(dict(season="2026-27", season_idx=2026, fixture_id=fid, fixture_uid=f"x:{fid}", gameweek=r + 1,
                                 home_code=h, away_code=a, kickoff=start + pd.Timedelta(days=fid), finished=True,
                                 home_goals=rng.poisson(lh), away_goals=rng.poisson(la), home_xg=lh, away_xg=la))
    return pd.DataFrame(rows)


def test_team_ratings_recover_strength_ordering():
    att = {1: 0.4, 2: 0.1, 3: -0.1, 4: -0.4}
    dfn = {1: 0.3, 2: 0.0, 3: 0.0, 4: -0.3}
    fx = _synthetic_fixtures(att, dfn)
    r = fit_team_ratings(fx, fx["kickoff"].max() + pd.Timedelta(days=1), 2026, ModelParams(team_ridge=0.5))
    assert sorted(r.att, key=r.att.get) == [4, 3, 2, 1]
    assert r.att[1] - r.att[4] == pytest.approx(0.8, abs=0.25)
    lh, la = r.rates(1, 4)
    assert lh > 2 * la


def test_anchor_features_never_look_ahead():
    base = pd.DataFrame({"player_code": 7, "minutes": [90, 90, 0, 90, 20], "starts": [1, 1, 0, 1, 0],
                         "season_idx": 2026})
    changed = base.copy()
    changed.loc[3:, ["minutes", "starts"]] = [[0, 0], [0, 0]]
    a, b = anchor_features(base), anchor_features(changed)
    cols = ["s_last1", "s_mean3", "s_mean5", "m_mean5", "gap"]
    # Anchors 0..3 only see rows 0..2, which are identical in both frames.
    pd.testing.assert_frame_equal(a.loc[a.row_pos <= 3, cols], b.loc[b.row_pos <= 3, cols])
    assert a.loc[a.row_pos == 3, "s_mean3"].item() == pytest.approx(2 / 3)
    assert a.loc[a.row_pos == 5, "gap"].item() == 1


def test_return_date_parsing():
    ref = datetime(2026, 11, 21, tzinfo=timezone.utc)
    assert str(return_date("Knee injury - Expected back 05 Dec", ref)) == "2026-12-05"
    assert str(return_date("Hamstring injury - Expected back 10 Jan", ref)) == "2027-01-10"
    assert str(return_date("Suspended until 29 Nov", ref)) == "2026-11-29"
    assert return_date("Knock - 75% chance of playing", ref) is None


def test_availability_profiles():
    ref = datetime(2026, 11, 21, tzinfo=timezone.utc)
    deadlines = {11: datetime(2026, 11, 21, 11, tzinfo=timezone.utc), 12: datetime(2026, 11, 28, 11, tzinfo=timezone.utc),
                 13: datetime(2026, 12, 2, 18, tzinfo=timezone.utc), 14: datetime(2026, 12, 6, 11, tzinfo=timezone.utc)}
    players = pd.DataFrame([
        dict(player_id=1, status="a", chance_of_playing=None, news=""),
        dict(player_id=2, status="d", chance_of_playing=75, news="Knock - 75% chance of playing"),
        dict(player_id=3, status="i", chance_of_playing=0, news="Knee injury - Expected back 05 Dec"),
        dict(player_id=4, status="u", chance_of_playing=0, news="Has left the club"),
        dict(player_id=5, status="s", chance_of_playing=0, news="Suspended until 29 Nov"),
    ])
    a = availability(players, deadlines, ref).pivot(index="player_id", columns="gameweek", values="p_avail")
    assert a.loc[1].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert a.loc[2].tolist() == pytest.approx([0.75, 0.875, 0.9375, 0.96875])
    assert a.loc[3].tolist() == [0.0, 0.0, 0.0, 0.9]
    assert a.loc[4].tolist() == [0.0] * 4
    assert a.loc[5].tolist() == [0.0, 0.0, 0.9, 1.0]


def test_aggregate_sums_double_gameweeks_and_zeroes_blanks():
    pf = pd.DataFrame({
        "key": [1, 1, 1], "gameweek": [11, 12, 12],
        "xp": [5.0, 4.0, 3.5], "xp_appearance": 2.0, "xp_attacking": 1.0, "xp_defensive": 1.0, "xp_bonus": 0.5,
        "xp_deductions": -0.1, "expected_minutes": 80.0, "expected_goals": 0.3, "expected_assists": 0.2,
        "p_play": [0.9, 0.9, 0.8], "p_start": [0.85, 0.85, 0.7], "p_clean_sheet": 0.3,
    })
    players = pd.DataFrame({"key": [1, 2]})
    gw = aggregate_gameweeks(pf, players, [11, 12, 13]).set_index(["key", "gameweek"])
    assert gw.loc[(1, 12), "xp"] == pytest.approx(7.5)
    assert gw.loc[(1, 12), "n_fixtures"] == 2
    assert gw.loc[(1, 12), "prob_play"] == pytest.approx(1 - 0.1 * 0.2)
    assert gw.loc[(1, 13), "xp"] == 0 and gw.loc[(1, 13), "n_fixtures"] == 0
    assert gw.loc[(2, 11), "xp"] == 0


def _pf(n_players: int, odds: list[float] | None):
    import numpy as np
    rows = []
    for i in range(n_players):
        rows.append(dict(key=i, fixture_uid="s:1", team_code=1, position="MID", lam_team=2.0, share_g=0.08,
                         p_s60=0.8, p_slt=0.1, p_sub=0.05, m_s60=88.0, m_slt=55.0, m_sub=20.0))
    pf = pd.DataFrame(rows)
    so = None if odds is None else pd.DataFrame({"key": range(len(odds)), "fixture_uid": "s:1", "median_decimal_odds": odds})
    return pf, so


def test_scorer_odds_remove_margin_and_keep_team_total():
    from app.ml.engine import OWN_GOAL_SHARE, apply_scorer_odds

    odds = [1.9, 2.5, 3.2, 4.0, 5.5, 7.0, 9.0, 13.0, 17.0, 21.0, 26.0]
    pf, so = _pf(11, odds)
    out = apply_scorer_odds(pf, so, ModelParams(scorer_odds_weight=1.0))
    assert out["scorer_odds_used"].all()
    p_play = out.p_s60 + out.p_slt + out.p_sub
    emin = (out.p_s60 * out.m_s60 + out.p_slt * out.m_slt + out.p_sub * out.m_sub) / p_play
    team_goals = (p_play * out.share_g * out.lam_team * emin / 90).sum()
    assert team_goals == pytest.approx(2.0 * (1 - OWN_GOAL_SHARE), rel=1e-6), "margin removed: totals match the team rate"
    assert list(out.share_g) == sorted(out.share_g, reverse=True), "shorter odds, bigger share"


def test_scorer_odds_blend_and_guards():
    from app.ml.engine import apply_scorer_odds

    pf, so = _pf(11, [3.0] * 11)
    half = apply_scorer_odds(pf, so, ModelParams(scorer_odds_weight=0.5))
    full = apply_scorer_odds(pf, so, ModelParams(scorer_odds_weight=1.0))
    assert half.share_g.iloc[0] == pytest.approx(0.5 * full.share_g.iloc[0] + 0.5 * 0.08)
    few, so_few = _pf(11, [3.0] * 5)          # too few priced players: leave the model alone
    assert not apply_scorer_odds(few, so_few, ModelParams()).scorer_odds_used.any()
    none, _ = _pf(11, None)
    assert apply_scorer_odds(none, None, ModelParams()).share_g.eq(0.08).all()
