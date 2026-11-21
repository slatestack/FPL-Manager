"""Shutdown behaviour of the worker loop: stop interrupts data refreshes, never a commit."""

import asyncio
from datetime import timedelta

from app.ingestion.worker import IngestionWorker


class SlowWorker(IngestionWorker):
    def __init__(self, scheduler=None):
        self._stop = asyncio.Event()
        self.scheduler = scheduler
        self.finished = False
        self.cancelled = False

    async def tick(self):
        try:
            await asyncio.sleep(30)          # e.g. a source retrying with backoff
            self.finished = True
            return timedelta(minutes=30)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class Committing:
    committing = True


async def test_stop_cancels_a_refresh_cycle_quickly():
    w = SlowWorker()
    run = asyncio.create_task(w.run_forever())
    await asyncio.sleep(0.05)
    w.stop()
    await asyncio.wait_for(run, timeout=1)
    assert w.cancelled and not w.finished


async def test_stop_waits_for_a_commit_in_progress():
    class ShortCommitWorker(SlowWorker):
        async def tick(self):
            await asyncio.sleep(0.2)
            self.finished = True
            return timedelta(minutes=30)

    w = ShortCommitWorker(scheduler=Committing())
    run = asyncio.create_task(w.run_forever())
    await asyncio.sleep(0.05)
    w.stop()
    await asyncio.wait_for(run, timeout=2)
    assert w.finished, "the commit cycle ran to completion"
