from app.sim.scoring import score_gameweek

POS = {1: "GKP", 2: "DEF", 3: "DEF", 4: "DEF", 5: "MID", 6: "MID", 7: "MID", 8: "MID", 9: "FWD", 10: "FWD", 11: "FWD",
       12: "GKP", 13: "DEF", 14: "MID", 15: "FWD"}
XI = list(range(1, 12))       # 3-4-3
BENCH = [12, 13, 14, 15]
PTS = {p: 2 for p in POS} | {9: 10}
MIN = {p: 90 for p in POS}


def test_plain_gameweek_with_captain():
    s = score_gameweek(XI, BENCH, 9, 5, PTS, MIN, POS)
    assert s.points == 2 * 10 + 10 + 10 and s.autosubs == []   # 10 starters x2, Haaland-style 10 doubled


def test_captain_blank_passes_armband_to_vice():
    # The captain (FWD 9) plays no minutes: the vice gets the armband, and the first bench player
    # (DEF 13) comes on because the resulting 4-4-2 is still a legal formation.
    s = score_gameweek(XI, BENCH, 9, 5, PTS | {9: 0}, MIN | {9: 0}, POS)
    assert s.captain_used == 5 and s.autosubs == [(9, 13)]
    assert s.points == 10 * 2 + 2 + 2   # 10 starters + sub, vice doubled


def test_autosub_respects_formation_and_bench_order():
    # A defender misses out in a 3-4-3: the first bench player (DEF 13) keeps 3 at the back.
    s = score_gameweek(XI, BENCH, 9, 5, PTS, MIN | {2: 0}, POS)
    assert s.autosubs == [(2, 13)]
    # If bench DEF also didn't play, the MID can't come on (only 2 DEF left): no valid sub.
    s = score_gameweek(XI, BENCH, 9, 5, PTS, MIN | {2: 0, 13: 0}, POS)
    assert s.autosubs == []


def test_goalkeeper_only_replaced_by_goalkeeper():
    s = score_gameweek(XI, BENCH, 9, 5, PTS, MIN | {1: 0}, POS)
    assert s.autosubs == [(1, 12)]


def test_chips_and_hits():
    tc = score_gameweek(XI, BENCH, 9, 5, PTS, MIN, POS, chip="3xc")
    assert tc.points == 20 + 30
    bb = score_gameweek(XI, BENCH, 9, 5, PTS | {2: 0}, MIN | {2: 0}, POS, chip="bboost", hits=1)
    assert bb.autosubs == [] and bb.points == (9 * 2 + 10) + 4 * 2 + 10 - 4
