"""Runtime supervisor for agent sessions.

It owns:
  - the Session (everything that has happened in this conversation)
  - a list of Signals (each one produces a SignalReading)
  - a policy mapping signal-name -> intervention-name
  - an escalation counter (N consecutive supervised steps with trips => halt)

API surface:
  record_user / record_assistant / record_memory  - push turns onto the session
  supervise() / supervise_async()                  - run all signals, decide an intervention
  save(path)                                       - write the trajectory as JSON
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable

from sidecar.trajectory import Session, SignalReading
from sidecar.signals import (
    Signal, GoalDriftSignal, SycophancySignal,
    PersonaCollapseSignal, MemoryTrustSignal,
)
from sidecar.interventions import reanchor, memory_prune, escalate, InterventionResult

log = logging.getLogger(__name__)

DEFAULT_POLICY = {
    "goal_drift": "reanchor",
    "sycophancy": "reanchor",
    "persona_collapse": "reanchor",
    "memory_trust": "memory_prune",
    "goal_alignment_judge": "reanchor",
    "persona_judge": "reanchor",
    "sycophancy_judge": "reanchor",
}


class Supervisor:
    def __init__(
        self,
        constitution: str,
        signals: list[Signal] | None = None,
        policy: dict[str, str] | None = None,
        anchor_terms: list[str] | None = None,
        escalate_after_n_trips: int = 3,
        store: "SessionStore | None" = None,
    ):
        self.session = Session(constitution=constitution)
        self.signals = signals or [
            GoalDriftSignal(),
            SycophancySignal(),
            PersonaCollapseSignal(anchor_terms=anchor_terms or []),
            MemoryTrustSignal(),
        ]
        self.policy = {**DEFAULT_POLICY, **(policy or {})}
        self.escalate_after_n_trips = escalate_after_n_trips
        self._consecutive_trips = 0
        self.store = store
        self._handlers: dict[str, Callable] = {
            "reanchor": reanchor,
            "memory_prune": memory_prune,
            "escalate": escalate,
        }

    # ---- Recording ---------------------------------------------------------

    def record_user(self, content: str) -> None:
        self.session.add_turn("user", content)
        self._persist()

    def record_assistant(self, content: str) -> None:
        self.session.add_turn("assistant", content)
        self._persist()

    def record_memory(self, content: str, source: str = "unknown") -> None:
        self.session.add_turn("memory", content, source=source)
        self._persist()

    # ---- Supervision (sync wrapper around async core) ---------------------

    def supervise(self) -> tuple[list[SignalReading], InterventionResult | None]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.supervise_async())
        # Fallback for "supervise() called from inside a running loop": run on a
        # dedicated thread.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(asyncio.run, self.supervise_async()).result()

    async def supervise_async(
        self,
    ) -> tuple[list[SignalReading], InterventionResult | None]:
        readings = await asyncio.gather(
            *(s.measure_async(self.session) for s in self.signals),
            return_exceptions=False,
        )
        readings = list(readings)
        self.session.readings.extend(readings)
        log.debug(
            "supervise: %s",
            [(r.name, round(r.value, 2), r.tripped) for r in readings],
        )

        tripped = [r for r in readings if r.tripped]
        if not tripped:
            self._consecutive_trips = 0
            self._persist()
            return readings, None

        self._consecutive_trips += 1
        if self._consecutive_trips >= self.escalate_after_n_trips:
            result = self._handlers["escalate"](
                self.session,
                triggered_by=",".join(r.name for r in tripped),
            )
            self._persist()
            return readings, result

        primary = max(tripped, key=lambda r: r.value - r.threshold)
        action = self.policy.get(primary.name, "reanchor")
        result = self._handlers[action](self.session, triggered_by=primary.name)
        self._persist()
        return readings, result

    # ---- Persistence ------------------------------------------------------

    def _persist(self) -> None:
        if self.store is not None:
            self.store.upsert(self.session)

    def save(self, path: str) -> None:
        self.session.save(path)
