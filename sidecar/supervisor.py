"""Runtime supervisor for agent sessions.

The Supervisor owns the Session, runs every Signal after each turn, and
applies an Intervention when one trips. It is framework-agnostic — wrap
it around any chat loop. The Anthropic-specific :class:`SupervisedAgent`
lives in :mod:`sidecar.adapters.anthropic_adapter` and is built on top
of this class.

Lifecycle::

    sup = Supervisor(constitution="...")
    while True:
        sup.record_user(user_msg)
        reply = my_agent.respond(user_msg)
        sup.record_assistant(reply)
        readings, intervention = await sup.supervise_async()
        if intervention is not None:
            apply(intervention)         # caller decides how
            if intervention.halt:
                break

Three consecutive supervised steps that trip a signal escalate the
session — :class:`Intervention` of type ``escalate`` with ``halt=True``.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Callable

from sidecar.trajectory import Session, SignalReading
from sidecar.signals import (
    Signal, GoalDriftSignal, SycophancySignal,
    PersonaCollapseSignal, MemoryTrustSignal,
)
from sidecar.interventions import (
    reanchor, memory_prune, escalate, InterventionResult,
)

log = logging.getLogger(__name__)

#: Default mapping from signal name to intervention name. Override per-instance
#: by passing ``policy=`` to :class:`Supervisor`.
DEFAULT_POLICY: dict[str, str] = {
    "goal_drift": "reanchor",
    "sycophancy": "reanchor",
    "persona_collapse": "reanchor",
    "memory_trust": "memory_prune",
    "goal_alignment_judge": "reanchor",
    "persona_judge": "reanchor",
    "sycophancy_judge": "reanchor",
}


class Supervisor:
    """Watches an agent's trajectory and intervenes when a signal trips.

    Args:
        constitution: The system prompt / behavioural contract the agent
            is supposed to obey. Used by signals as the reference point
            for drift and persona stability.
        signals: List of :class:`~sidecar.signals.Signal` instances. If
            omitted, the four default heuristic signals are installed.
        policy: Mapping from signal name → intervention name. Merged with
            :data:`DEFAULT_POLICY` (caller wins on conflicts).
        anchor_terms: Constitution-distinctive terms for the default
            ``PersonaCollapseSignal`` (e.g. the agent's name, brand).
        escalate_after_n_trips: How many *consecutive* supervised steps
            with at least one tripped signal trigger an ``escalate``
            intervention. Defaults to 3.
        store: Optional :class:`~sidecar.store.SessionStore`. If provided,
            the full Session is upserted on every recorded turn,
            reading, and intervention.
    """

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
        """Synchronous wrapper around :meth:`supervise_async`. Safe to call
        from inside or outside an event loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.supervise_async())
        # Called from inside a running loop: run on a dedicated thread so we
        # don't deadlock.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(asyncio.run, self.supervise_async()).result()

    async def supervise_async(
        self,
    ) -> tuple[list[SignalReading], InterventionResult | None]:
        """Run all signals concurrently and decide on an intervention.

        Returns a tuple ``(readings, intervention)``. When no signal trips,
        ``intervention`` is ``None`` and the consecutive-trip counter is
        reset. Otherwise the highest-margin tripped signal is routed to
        the policy-mapped intervention; reaching ``escalate_after_n_trips``
        consecutive supervised steps with trips returns an ``escalate``
        intervention with ``halt=True``.
        """
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
