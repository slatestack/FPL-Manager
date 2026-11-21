"""The manager's current position, reconstructed from public FPL endpoints.

Phase 5 will read the authenticated ``my-team`` endpoint, which reports selling prices, free
transfers and chip status directly. Until then they are derived from public data:

* **Squad / bank**: ``entry/{id}/event/{gw}/picks`` for the last played gameweek (the
  previous one if a Free Hit was active, since the squad reverts).
* **Selling prices**: FPL keeps half of any rise (rounded down) and passes on falls in full.
  The purchase price is the ``element_in_cost`` of the latest transfer in, or the player's
  price in the manager's first gameweek for original picks.
* **Free transfers**: replayed from ``entry/{id}/history``: one more each gameweek, capped at
  5, unchanged by a Wildcard/Free Hit week. The replay is checked against the hit points FPL
  actually charged.
* **Chips**: a chip definition is available if the next gameweek is inside its window and no
  chip of that name was played inside the same window.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

MAX_FREE_TRANSFERS = 5
HIT_COST = 4


@dataclass
class ChipWindow:
    name: str
    start_event: int
    stop_event: int


@dataclass
class TeamState:
    entry_id: int
    next_gameweek: int
    squad: list[int]
    bank: int                                   # tenths of £m
    selling_price: dict[int, int]
    purchase_price: dict[int, int]
    free_transfers: int
    chips_available: list[ChipWindow] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def squad_value(self) -> int:
        return sum(self.selling_price[p] for p in self.squad)

    def chip(self, name: str) -> ChipWindow | None:
        return next((c for c in self.chips_available if c.name == name), None)

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        d = asdict(self)
        d["selling_price"] = {str(k): v for k, v in self.selling_price.items()}
        d["purchase_price"] = {str(k): v for k, v in self.purchase_price.items()}
        return d


def selling_price(purchase: int, now: int) -> int:
    return purchase + (now - purchase) // 2 if now > purchase else now


def replay_free_transfers(
    history: Sequence[Mapping[str, Any]], chips: Sequence[Mapping[str, Any]], next_gameweek: int, started_event: int
) -> tuple[int, list[str]]:
    """Free transfers available for ``next_gameweek``; warnings if the replay disagrees with FPL's charges."""
    chip_at = {c["event"]: c["name"] for c in chips}
    by_event = {h["event"]: h for h in history}
    warnings = []
    ft = 1  # entering the gameweek after the entry's first
    for ev in range(started_event + 1, next_gameweek):
        h = by_event.get(ev)
        if h is None:
            continue
        used = int(h.get("event_transfers", 0))
        if chip_at.get(ev) in ("wildcard", "freehit"):
            continue  # free transfers are kept, no extra one is added
        expected_cost = HIT_COST * max(0, used - ft)
        charged = int(h.get("event_transfers_cost", 0))
        if charged != expected_cost:
            warnings.append(f"GW{ev}: replay expected {expected_cost} hit points, FPL charged {charged}")
        ft = min(MAX_FREE_TRANSFERS, max(0, ft - used) + 1)
    return ft, warnings


def available_chips(
    definitions: Sequence[Mapping[str, Any]], played: Sequence[Mapping[str, Any]], next_gameweek: int
) -> list[ChipWindow]:
    out = []
    for d in definitions:
        start, stop = int(d["start_event"]), int(d["stop_event"])
        if not (start <= next_gameweek <= stop):
            continue
        if any(p["name"] == d["name"] and start <= int(p["event"]) <= stop for p in played):
            continue
        out.append(ChipWindow(d["name"], start, stop))
    return out


def derive_team_state(
    *,
    entry_id: int,
    next_gameweek: int,
    entry: Mapping[str, Any],
    history: Mapping[str, Any],
    transfers: Sequence[Mapping[str, Any]],
    picks: Mapping[str, Any],
    chip_definitions: Sequence[Mapping[str, Any]],
    now_cost: Mapping[int, int],
    first_gw_price: Mapping[int, int],
) -> TeamState:
    squad = [int(p["element"]) for p in picks["picks"]]
    bank = int(picks["entry_history"]["bank"])
    started = int(entry.get("started_event") or 1)

    latest_in: dict[int, int] = {}
    for t in sorted(transfers, key=lambda t: (t["event"], t.get("time", ""))):
        latest_in[int(t["element_in"])] = int(t["element_in_cost"])
    purchase, selling = {}, {}
    warnings = []
    for p in squad:
        bought = latest_in.get(p, first_gw_price.get(p))
        if bought is None:
            bought = now_cost[p]
            warnings.append(f"purchase price unknown for {p}; using current price")
        purchase[p] = bought
        selling[p] = selling_price(bought, now_cost[p])

    ft, ft_warnings = replay_free_transfers(history.get("current", []), history.get("chips", []), next_gameweek, started)
    chips = available_chips(chip_definitions, history.get("chips", []), next_gameweek)
    return TeamState(
        entry_id=entry_id, next_gameweek=next_gameweek, squad=squad, bank=bank, selling_price=selling,
        purchase_price=purchase, free_transfers=ft, chips_available=chips, warnings=warnings + ft_warnings,
    )


async def fetch_team_state(fpl, pool, entry_id: int, next_gameweek: int, *, free_transfers: int | None = None) -> tuple[TeamState, dict[str, Any]]:
    """Fetch the public endpoints and derive the state. Returns (state, raw inputs for the audit log)."""
    entry = await fpl.entry(entry_id)
    history = await fpl.entry_history(entry_id)
    transfers = await fpl.entry_transfers(entry_id)
    last = next_gameweek - 1
    chip_at = {c["event"]: c["name"] for c in history.get("chips", [])}
    picks_event = last - 1 if chip_at.get(last) == "freehit" else last
    picks = await fpl.entry_picks(entry_id, picks_event)
    # The bank shown for a Free Hit week is the Free Hit squad's; the reverted squad keeps the earlier bank.
    rows = await pool.fetch("SELECT player_id, now_cost FROM players")
    now_cost = {r["player_id"]: r["now_cost"] for r in rows}
    started = int(entry.get("started_event") or 1)
    first = await pool.fetch(
        """SELECT DISTINCT ON (player_id) player_id, value FROM player_gw_history
           WHERE gameweek_id >= $1 AND value IS NOT NULL ORDER BY player_id, gameweek_id, kickoff_time""",
        started,
    )
    first_gw_price = {r["player_id"]: r["value"] for r in first}
    # Squad players who never played have no stored history: fetch theirs directly.
    for pid in (int(p["element"]) for p in picks["picks"]):
        if pid in first_gw_price:
            continue
        summary = await fpl.element_summary(pid)
        rows_ = [h for h in summary.get("history", []) if int(h["round"]) >= started and h.get("value") is not None]
        if rows_:
            first_gw_price[pid] = int(min(rows_, key=lambda h: (h["round"], h["kickoff_time"] or ""))["value"])
    chip_defs = [dict(r) for r in await pool.fetch("SELECT name, start_event, stop_event FROM chip_definitions")]
    state = derive_team_state(
        entry_id=entry_id, next_gameweek=next_gameweek, entry=entry, history=history, transfers=transfers,
        picks=picks, chip_definitions=chip_defs, now_cost=now_cost, first_gw_price=first_gw_price,
    )
    if free_transfers is not None:
        state.free_transfers = free_transfers
        state.warnings.append(f"free transfers overridden to {free_transfers}")
    raw = {"picks_event": picks_event, "picks": picks.get("picks"), "entry_history": picks.get("entry_history"),
           "history_chips": history.get("chips"), "transfers_count": len(transfers)}
    return state, raw


def team_state_from_my_team(my_team, entry_id: int, next_gameweek: int, chip_definitions: Sequence[Mapping[str, Any]]) -> TeamState:
    """Exact state from the authenticated ``my-team`` endpoint (Phase 5)."""
    chips = []
    for d in chip_definitions:
        start, stop = int(d["start_event"]), int(d["stop_event"])
        if start <= next_gameweek <= stop and my_team.chip_status(d["name"]) == "available":
            chips.append(ChipWindow(d["name"], start, stop))
    warnings = []
    ft = my_team.free_transfers
    if ft is None:
        warnings.append(f"transfers are unlimited this gameweek ({my_team.active_chip or 'chip'} active)")
        ft = MAX_FREE_TRANSFERS
    return TeamState(
        entry_id=entry_id, next_gameweek=next_gameweek, squad=sorted(my_team.squad), bank=my_team.bank,
        selling_price={p.element: p.selling_price for p in my_team.picks},
        purchase_price={p.element: p.purchase_price for p in my_team.picks},
        free_transfers=ft, chips_available=chips, warnings=warnings,
    )
