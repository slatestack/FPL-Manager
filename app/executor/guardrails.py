"""Circuit breakers (Section 7.3) evaluated against live data immediately before a commit.

Spec guardrails: kill switch, hit cap, price/budget drift, duplicate execution. Added:
deadline cutoff, plan freshness, plan-vs-live squad parity (you changed your team by hand after
the plan was made) and a free-transfer recheck against the live ``my-team`` count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


@dataclass
class GuardrailReport:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.checks.append(Check(name, ok, detail))

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    def summary(self) -> str:
        return "; ".join(f"{c.name}: {c.detail}" for c in self.failures) or "all checks passed"

    def as_list(self) -> list[dict[str, Any]]:
        return [c.__dict__ for c in self.checks]


def evaluate(
    *,
    auto_execute_enabled: bool,
    respect_kill_switch: bool,
    now: datetime,
    deadline: datetime,
    cutoff: timedelta,
    plan_created_at: datetime,
    plan_max_age: timedelta,
    plan_gameweek: int,
    next_gameweek: int,
    expected_squad: set[int],
    live_squad: set[int],
    outs: list[int],
    ins: list[int],
    chip: str | None,
    live_free_transfers: int | None,
    max_hits: int,
    live_bank: int,
    live_selling: dict[int, int],
    now_cost: dict[int, int],
    duplicate_success: bool,
    chip_status: str | None,
) -> GuardrailReport:
    r = GuardrailReport()
    if respect_kill_switch:
        r.add("kill_switch", auto_execute_enabled, "auto-execution enabled" if auto_execute_enabled else "auto_execute_enabled is false")
    r.add("deadline", now < deadline - cutoff and plan_gameweek == next_gameweek,
          f"{(deadline - now).total_seconds() / 60:.1f} min to GW{next_gameweek} deadline (cutoff {cutoff.total_seconds() / 60:.0f}), plan for GW{plan_gameweek}")
    age = now - plan_created_at
    r.add("plan_fresh", age <= plan_max_age, f"plan is {age.total_seconds() / 60:.0f} min old (max {plan_max_age.total_seconds() / 60:.0f})")
    r.add("squad_parity", expected_squad == live_squad,
          "live squad matches the plan's starting point" if expected_squad == live_squad
          else f"live squad differs: +{sorted(live_squad - expected_squad)} -{sorted(expected_squad - live_squad)}")
    unlimited = chip in ("wildcard", "freehit")
    if unlimited:
        r.add("hit_cap", True, f"{chip}: transfers are free")
    else:
        ft = live_free_transfers if live_free_transfers is not None else 0
        hits = max(0, len(ins) - ft)
        r.add("hit_cap", hits <= max_hits, f"{len(ins)} transfers with {ft} free -> {hits} hit(s), cap {max_hits}")
    missing = [p for p in outs if p not in live_selling] + [p for p in ins if p not in now_cost]
    if missing:
        r.add("budget", False, f"no live price for {missing}")
    else:
        bank_after = live_bank + sum(live_selling[p] for p in outs) - sum(now_cost[p] for p in ins)
        r.add("budget", bank_after >= 0, f"bank after transfers £{bank_after / 10:.1f}m at live prices")
    r.add("duplicate", not duplicate_success, "already executed with an identical payload" if duplicate_success else "first execution of this payload")
    if chip:
        r.add("chip_available", chip_status in ("available", "active"), f"{chip} status: {chip_status}")
    return r
