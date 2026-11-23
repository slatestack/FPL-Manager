"""The season simulator on the real 2025/26 sample (4 clubs, GW1-4)."""

from app.ml.data import load_model_data
from app.ml.params import ModelParams
from app.optimizer.params import OptimizerParams
from app.sim.season import SimConfig, simulate
from tests.integration.test_predictions import _ingest_archive


async def test_simulation_runs_and_obeys_fpl_rules(pool, fpl):
    from app.ingestion import pipeline

    await pipeline.ingest_core(pool, fpl)
    await _ingest_archive(pool)
    async with pool.acquire() as conn:
        data = await load_model_data(conn)
    params = ModelParams(minutes_rounds=40, bonus_rounds=40, bonus_simulations=50)
    opt = OptimizerParams(max_per_club=4)   # the sample has only 4 clubs: 3 per club can't fill 15 places
    res = simulate(data, SimConfig(season="2025-26", first_gw=2, last_gw=4, horizon=2), params, opt)
    gws = res.gameweeks
    assert [g.gameweek for g in gws] == [2, 3, 4]
    assert all(g.bank >= 0 for g in gws)
    assert gws[0].transfers == [] and gws[0].free_transfers == 1, "initial squad, then 1 free transfer"
    for prev, g in zip(gws, gws[1:]):
        if g.chip not in ("wildcard", "freehit"):
            assert g.free_transfers == min(5, max(0, prev.free_transfers - len(g.transfers)) + 1)
            assert g.hits == max(0, len(g.transfers) - prev.free_transfers)
    assert res.total == sum(g.points for g in gws) and res.total > 0
    base = simulate(data, SimConfig(season="2025-26", first_gw=2, last_gw=4, horizon=2, transfers=False), params, opt)
    assert all(not g.transfers and not g.chip for g in base.gameweeks)
