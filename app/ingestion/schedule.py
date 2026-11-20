"""Deadline-aware polling cadence (Section 4.2). Pure functions, no I/O."""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum

STANDARD_INTERVAL = timedelta(hours=6)
PRE_DEADLINE_INTERVAL = timedelta(minutes=30)
CRITICAL_INTERVAL = timedelta(minutes=2)
POST_DEADLINE_RETRY = timedelta(minutes=5)
MIN_SLEEP = timedelta(seconds=60)

PRE_DEADLINE_START = timedelta(hours=24)
CRITICAL_START = timedelta(hours=2)
CRITICAL_END = timedelta(minutes=10)


class Phase(str, Enum):
    NO_DEADLINE = "no_deadline"      # season over / not yet scheduled
    STANDARD = "standard"            # > 24h before deadline
    PRE_DEADLINE = "pre_deadline"    # T-24h .. T-2h
    CRITICAL = "critical"            # T-2h .. T-10m
    LOCKED = "locked"                # T-10m .. T (execution window; no further data changes acted on)
    PASSED = "passed"                # deadline passed, waiting for FPL to roll to the next GW


def polling_phase(now: datetime, deadline: datetime | None) -> Phase:
    if deadline is None:
        return Phase.NO_DEADLINE
    remaining = deadline - now
    if remaining <= timedelta(0):
        return Phase.PASSED
    if remaining <= CRITICAL_END:
        return Phase.LOCKED
    if remaining <= CRITICAL_START:
        return Phase.CRITICAL
    if remaining <= PRE_DEADLINE_START:
        return Phase.PRE_DEADLINE
    return Phase.STANDARD


def next_poll_delay(now: datetime, deadline: datetime | None) -> timedelta:
    """How long to sleep before the next core poll.

    Sleeps are clipped so the worker wakes exactly when it enters a faster window
    (e.g. it never sleeps through the T-24h boundary on a 6h cadence).
    """
    phase = polling_phase(now, deadline)
    if phase is Phase.NO_DEADLINE:
        return STANDARD_INTERVAL
    if phase is Phase.PASSED:
        return POST_DEADLINE_RETRY
    assert deadline is not None
    remaining = deadline - now
    if phase is Phase.STANDARD:
        delay = min(STANDARD_INTERVAL, remaining - PRE_DEADLINE_START)
    elif phase is Phase.PRE_DEADLINE:
        delay = min(PRE_DEADLINE_INTERVAL, remaining - CRITICAL_START)
    elif phase is Phase.CRITICAL:
        delay = min(CRITICAL_INTERVAL, remaining - CRITICAL_END)
    else:  # LOCKED: resume just after the deadline, when FPL rolls to the next gameweek
        delay = remaining + MIN_SLEEP
    return max(MIN_SLEEP, delay)
