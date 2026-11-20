from datetime import datetime, timedelta, timezone

import pytest

from app.ingestion.schedule import Phase, next_poll_delay, polling_phase

DEADLINE = datetime(2026, 11, 21, 11, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("before", "phase", "delay"),
    [
        (timedelta(days=3), Phase.STANDARD, timedelta(hours=6)),
        (timedelta(hours=26), Phase.STANDARD, timedelta(hours=2)),        # clipped to T-24h boundary
        (timedelta(hours=24), Phase.PRE_DEADLINE, timedelta(minutes=30)),
        (timedelta(hours=2, minutes=10), Phase.PRE_DEADLINE, timedelta(minutes=10)),  # clipped to T-2h
        (timedelta(hours=1), Phase.CRITICAL, timedelta(minutes=2)),
        (timedelta(minutes=11), Phase.CRITICAL, timedelta(seconds=60)),    # floor
        (timedelta(minutes=10), Phase.LOCKED, timedelta(minutes=11)),      # resume after deadline
        (timedelta(seconds=-30), Phase.PASSED, timedelta(minutes=5)),
    ],
)
def test_phases_and_delays(before, phase, delay):
    now = DEADLINE - before
    assert polling_phase(now, DEADLINE) is phase
    assert next_poll_delay(now, DEADLINE) == delay


def test_no_deadline_polls_every_six_hours():
    now = datetime(2027, 6, 1, tzinfo=timezone.utc)
    assert polling_phase(now, None) is Phase.NO_DEADLINE
    assert next_poll_delay(now, None) == timedelta(hours=6)


def test_never_sleeps_through_a_window_boundary():
    """Simulate the worker from 4 days out: every window must be entered on time."""
    now = DEADLINE - timedelta(days=4)
    visited = []
    while now < DEADLINE:
        visited.append((polling_phase(now, DEADLINE), DEADLINE - now))
        now += next_poll_delay(now, DEADLINE)
    first_pre = next(r for p, r in visited if p is Phase.PRE_DEADLINE)
    first_crit = next(r for p, r in visited if p is Phase.CRITICAL)
    assert first_pre == timedelta(hours=24)
    assert first_crit == timedelta(hours=2)
    crit_gaps = [a[1] - b[1] for a, b in zip(visited, visited[1:]) if a[0] is Phase.CRITICAL and b[0] is Phase.CRITICAL]
    assert crit_gaps and max(crit_gaps) <= timedelta(minutes=2)
