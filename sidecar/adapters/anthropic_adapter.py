"""SupervisedAgent — drop-in wrapper around the Anthropic Messages API.

The agent is supervised at runtime AND learns over time. Two new things
beyond the basic supervisor:

  1. If a LessonStore is provided, the agent retrieves relevant prior lessons
     at session start (or before each turn) and injects them as a
     `prior_lessons` system block. This is how the agent "remembers having
     been pressured before."

  2. After a session ends — or whenever you call `therapize()` — the
     Reflector reviews drifted turns and writes new Lessons to the store.

Usage:
    store = LessonStore("./lessons.db")
    agent = SupervisedAgent(constitution=..., lessons=store)

    while True:
        msg = input("> ")
        try:
            print(await agent.send(msg))
        except AgentHalted:
            break

    # End-of-session therapy: turn failures into Lessons for next time.
    await agent.therapize()
"""
from __future__ import annotations

import logging
from typing import Any

from sidecar.supervisor import Supervisor
from sidecar.interventions import InterventionResult
from sidecar.reflection import Reflector, Lesson
from sidecar.lessons import LessonStore, _to_fts_query  # noqa: F401

log = logging.getLogger(__name__)


class AgentHalted(RuntimeError):
    """Raised when the supervisor escalates and halts the session."""


class SupervisedAgent:
    def __init__(
        self,
        constitution: str,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
        signals: list | None = None,
        anchor_terms: list[str] | None = None,
        store: "SessionStore | None" = None,
        lessons: LessonStore | None = None,
        reflector: Reflector | None = None,
        client: Any | None = None,
        n_lessons_in_context: int = 4,
    ):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("anthropic package required for SupervisedAgent") from e
        self._supervisor = Supervisor(
            constitution=constitution,
            signals=signals,
            anchor_terms=anchor_terms,
            store=store,
        )
        self.model = model
        self.max_tokens = max_tokens
        self._client = client or anthropic.AsyncAnthropic()
        self._messages: list[dict] = []
        self._pending_extra_system: str | None = None
        self.lessons = lessons
        self.reflector = reflector or Reflector(model=model, client=client)
        self.n_lessons_in_context = n_lessons_in_context
        from sidecar.reflection import _hash
        self._constitution_hash = _hash(constitution)
        self._lessons_block: str | None = self._initial_lessons_block()

    # ---- properties -------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self.supervisor.session.id

    @property
    def supervisor(self) -> Supervisor:
        return self._supervisor

    # ---- conversation -----------------------------------------------------

    async def send(self, user_message: str) -> str:
        self.supervisor.record_user(user_message)
        self._messages.append({"role": "user", "content": user_message})

        # Refresh lessons block if we have a store and meaningful new context.
        if self.lessons is not None and len(self._messages) <= 2:
            self._lessons_block = self._initial_lessons_block(
                seed_user=user_message,
            )

        system = [{
            "type": "text",
            "text": self.supervisor.session.constitution,
            "cache_control": {"type": "ephemeral"},
        }]
        if self._lessons_block:
            system.append({"type": "text", "text": self._lessons_block})
        if self._pending_extra_system:
            system.append({"type": "text", "text": self._pending_extra_system})
            self._pending_extra_system = None

        resp = await self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=self._messages,
        )
        reply = "".join(b.text for b in resp.content if b.type == "text")
        self.supervisor.record_assistant(reply)
        self._messages.append({"role": "assistant", "content": reply})

        readings, intervention = await self.supervisor.supervise_async()
        if intervention is not None:
            self._apply_intervention(intervention)
            if intervention.halt:
                raise AgentHalted(intervention.note)

        return reply

    async def ingest_memory(self, content: str, source: str = "unknown") -> None:
        """Record a retrieved memory and screen it BEFORE the agent sees it."""
        self.supervisor.record_memory(content, source=source)
        readings, intervention = await self.supervisor.supervise_async()
        if intervention is not None:
            self._apply_intervention(intervention)
            if intervention.drop_last_memory:
                self.supervisor.session.turns = [
                    t for i, t in enumerate(self.supervisor.session.turns)
                    if not (i == len(self.supervisor.session.turns) - 1
                            and t.role == "memory")
                ]

    # ---- therapy ----------------------------------------------------------

    async def therapize(self) -> list[Lesson]:
        """Reflect on the current session and write Lessons to the store."""
        new_lessons = await self.reflector.reflect(self.supervisor.session)
        if self.lessons is not None and new_lessons:
            self.lessons.add_many(new_lessons)
        return new_lessons

    # ---- internals --------------------------------------------------------

    def _initial_lessons_block(self, seed_user: str | None = None) -> str | None:
        if self.lessons is None:
            return None
        context = seed_user or self.supervisor.session.constitution
        relevant = self.lessons.retrieve(
            context=context,
            constitution_hash=self._constitution_hash,
            k=self.n_lessons_in_context,
        )
        if not relevant:
            return None
        body = "\n\n".join(l.to_prompt_block() for l in relevant)
        return (
            "PRIOR LESSONS — you have been pressured like this before. "
            "Apply these lessons before responding:\n\n" + body
        )

    def _apply_intervention(self, result: InterventionResult) -> None:
        if result.extra_system_message:
            self._pending_extra_system = result.extra_system_message
        log.info("intervention applied: type=%s note=%s",
                 result.type, result.note)

    def save(self, path: str) -> None:
        self.supervisor.save(path)
