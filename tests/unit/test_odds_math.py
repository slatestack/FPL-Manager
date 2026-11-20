import math

import pytest

from app.ingestion import odds_math


def test_implied_probabilities_remove_overround():
    probs = odds_math.implied_probabilities([1.75, 3.9, 4.6])
    assert sum(probs) == pytest.approx(1.0)
    assert probs[0] > probs[2] > 0
    assert odds_math.overround([1.75, 3.9, 4.6]) == pytest.approx(1 / 1.75 + 1 / 3.9 + 1 / 4.6 - 1)


def test_implied_probabilities_reject_bad_prices():
    with pytest.raises(ValueError):
        odds_math.implied_probabilities([1.0, 3.0])


def test_outcome_probabilities_are_a_distribution():
    h, d, a, over = odds_math.outcome_probabilities(1.5, 1.2)
    assert h + d + a == pytest.approx(1.0, abs=1e-6)
    assert 0 < over < 1


@pytest.mark.parametrize(("lh", "la"), [(1.6, 1.1), (2.6, 0.5), (0.9, 1.4)])
def test_fit_recovers_known_goal_rates(lh, la):
    h, d, a, over = odds_math.outcome_probabilities(lh, la)
    fit = odds_math.fit_goal_rates(h, d, a, over)
    assert fit.lambda_home == pytest.approx(lh, abs=0.02)
    assert fit.lambda_away == pytest.approx(la, abs=0.02)
    assert fit.p_cs_home == pytest.approx(math.exp(-la), abs=0.01)


def test_fit_without_totals_market():
    h, d, a, _ = odds_math.outcome_probabilities(1.8, 0.9)
    fit = odds_math.fit_goal_rates(h, d, a)
    assert fit.lambda_home == pytest.approx(1.8, abs=0.1)
    assert fit.lambda_away == pytest.approx(0.9, abs=0.1)


def test_consensus_averages_and_normalises():
    assert odds_math.consensus([[0.5, 0.3, 0.2], [0.6, 0.2, 0.2]]) == pytest.approx([0.55, 0.25, 0.2])
