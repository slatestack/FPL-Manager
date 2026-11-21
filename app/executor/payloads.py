"""Map an optimiser decision onto FPL's write payloads (shapes taken from the FPL web app).

* ``POST transfers/``: {chip, entry, event, transfers: [{element_in, element_out, purchase_price, selling_price}]}
* ``POST my-team/{entry}/``: {chip, picks: [{element, position, is_captain, is_vice_captain}]}
  Positions 1-11 are the starting XI (goalkeeper first, then DEF, MID, FWD); 12 is the bench
  goalkeeper and 13-15 the outfield bench in substitution order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

POS_ORDER = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
TRANSFER_CHIPS = ("wildcard", "freehit")
TEAM_CHIPS = ("bboost", "3xc")


def pair_transfers(outs: Sequence[int], ins: Sequence[int], position: Mapping[int, str]) -> list[tuple[int, int]]:
    """Pair each outgoing player with an incoming one of the same position."""
    if len(outs) != len(ins):
        raise ValueError("unbalanced transfers")
    pairs = []
    remaining = list(ins)
    for o in sorted(outs, key=lambda p: (POS_ORDER[position[p]], p)):
        match = next((i for i in sorted(remaining) if position[i] == position[o]), None)
        if match is None:
            raise ValueError(f"no incoming {position[o]} to replace {o}")
        remaining.remove(match)
        pairs.append((o, match))
    return pairs


def transfers_payload(
    *, entry: int, event: int, outs: Sequence[int], ins: Sequence[int], chip: str | None,
    position: Mapping[int, str], selling_price: Mapping[int, int], now_cost: Mapping[int, int],
) -> dict[str, Any]:
    return {
        "chip": chip if chip in TRANSFER_CHIPS else None,
        "entry": entry,
        "event": event,
        "transfers": [
            {"element_in": i, "element_out": o, "purchase_price": int(now_cost[i]), "selling_price": int(selling_price[o])}
            for o, i in pair_transfers(outs, ins, position)
        ],
    }


def picks_payload(
    *, starting_xi: Sequence[int], bench: Sequence[int], captain: int, vice_captain: int, chip: str | None,
    position: Mapping[int, str],
) -> dict[str, Any]:
    if len(starting_xi) != 11 or len(bench) != 4:
        raise ValueError("need 11 starters and 4 bench players")
    if position[bench[0]] != "GKP":
        raise ValueError("first bench slot must be the goalkeeper")
    xi = sorted(starting_xi, key=lambda p: (POS_ORDER[position[p]], p))
    ordered = xi + list(bench)
    return {
        "chip": chip if chip in TEAM_CHIPS else None,
        "picks": [
            {"element": p, "position": k, "is_captain": p == captain, "is_vice_captain": p == vice_captain}
            for k, p in enumerate(ordered, start=1)
        ],
    }
