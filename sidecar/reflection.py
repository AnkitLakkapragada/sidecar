"""Reflector — turn drifted turns into reusable Lessons.

After a session ends, the Reflector reads the trajectory, finds every
assistant turn where a signal tripped, and asks a reflection model:
"what should you have said here, given the constitution?" The result is
a :class:`Lesson` — a small, auditable artifact (pattern, drifted reply,
correction, rationale) that future sessions can retrieve from a
:class:`~sidecar.lessons.LessonStore` and apply.

The reflection model speaks back to the agent in its own voice; lessons
are not externally-imposed corrections, they're rewrites the agent
itself could plausibly have produced. The agent's identity persists,
and the lessons it learned persist alongside it.

Usage::

    reflector = Reflector(model="claude-sonnet-4-6")
    lessons = await reflector.reflect(session)        # list[Lesson]
    lesson_store.add_many(lessons)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field, asdict

from sidecar.trajectory import Session, SignalReading

log = logging.getLogger(__name__)

DEFAULT_REFLECTION_MODEL = "claude-sonnet-4-6"


@dataclass
class Lesson:
    """A single counterfactual case the agent carries forward.

    Lessons are small enough to fit in a system prompt and concrete
    enough to audit by hand. They are not vector embeddings or
    fine-tunes; they are short structured rewrites of moments the
    agent failed.
    """
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    constitution_hash: str = ""
    pattern: str = ""              # 1-line description of the pressure pattern
    user_excerpt: str = ""         # the user message that triggered the drift
    drifted_response: str = ""     # what the agent actually said (wrong)
    correction: str = ""           # what the agent should have said instead
    rationale: str = ""            # which rule was violated and why
    triggered_signals: list[str] = field(default_factory=list)
    session_id: str = ""
    turn_index: int = -1
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_prompt_block(self) -> str:
        """Render this lesson as a snippet to inject into a system prompt."""
        return (
            f"• Pattern: {self.pattern}\n"
            f"  User said: \"{self._truncate(self.user_excerpt, 180)}\"\n"
            f"  You drifted into: \"{self._truncate(self.drifted_response, 200)}\"\n"
            f"  What you should say instead: \"{self._truncate(self.correction, 240)}\"\n"
            f"  Why: {self.rationale}"
        )

    @staticmethod
    def _truncate(s: str, n: int) -> str:
        s = s.strip().replace("\n", " ")
        return s if len(s) <= n else s[: n - 1] + "…"


REFLECTOR_RUBRIC = """You are reviewing an AI agent that drifted in production.
You are given:
  1. The agent's CONSTITUTION — the rules it must follow.
  2. A FAILURE — one user message and the agent's drifted reply.
  3. The signals that flagged this turn as a failure.

Produce ONE lesson the agent can carry into future sessions:
  - name the pressure pattern the user applied (one short clause)
  - rewrite the agent's reply in-character, fully aligned with the constitution
  - explain in one sentence which rule was violated and why

The corrected reply must be something the agent could plausibly say in
character — not a robotic "I refuse" — and must NOT apologize for refusing.

Respond ONLY in JSON, no preamble, no code fences:
{
  "pattern": "<short clause>",
  "correction": "<what the agent should have said>",
  "rationale": "<one sentence>"
}"""


class Reflector:
    """Generates Lessons from drifted turns in a Session.

    Args:
        model: Anthropic model id used as the reflection model. Defaults
            to ``claude-sonnet-4-6``.
        max_tokens: Per-call token budget for each lesson.
        client: An optional pre-constructed ``anthropic.AsyncAnthropic``.
            If omitted, one is created lazily; if no API key is set the
            Reflector falls back to skeleton lessons so callers can still
            exercise the full pipeline offline.
        min_severity: Drifted turns are skipped if no tripped signal on
            that turn reaches this value. ``0.0`` means consider every
            tripped turn.
    """

    def __init__(
        self,
        model: str = DEFAULT_REFLECTION_MODEL,
        max_tokens: int = 512,
        client=None,
        min_severity: float = 0.0,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self._client = client
        self.min_severity = min_severity

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return None
        try:
            import anthropic
        except ImportError:
            return None
        self._client = anthropic.AsyncAnthropic()
        return self._client

    async def reflect(self, session: Session) -> list[Lesson]:
        """Walk the session, find drifted assistant turns, generate Lessons."""
        targets = self._find_drifted_turns(session)
        if not targets:
            return []

        client = self._get_client()
        if client is None:
            log.info("reflector: no API key, returning skeleton lessons")
            return [self._skeleton_lesson(session, *t) for t in targets]

        lessons: list[Lesson] = []
        for turn_idx, drifted, user_msg, signals in targets:
            lesson = await self._reflect_one(
                session, client, turn_idx, drifted, user_msg, signals,
            )
            if lesson is not None:
                lessons.append(lesson)
        log.info("reflector produced %d lessons from session %s", len(lessons), session.id)
        return lessons

    # ---- internals --------------------------------------------------------

    def _find_drifted_turns(
        self, session: Session,
    ) -> list[tuple[int, str, str, list[SignalReading]]]:
        """Return [(turn_index, drifted_text, preceding_user_text, signals)] for
        every assistant turn where any signal tripped above min_severity."""
        readings_by_turn: dict[int, list[SignalReading]] = {}
        for r in session.readings:
            if r.tripped and r.value >= self.min_severity:
                readings_by_turn.setdefault(r.turn_index, []).append(r)

        out = []
        for idx in sorted(readings_by_turn):
            if idx < 0 or idx >= len(session.turns):
                continue
            turn = session.turns[idx]
            if turn.role != "assistant":
                continue
            preceding_user = ""
            for j in range(idx - 1, -1, -1):
                if session.turns[j].role == "user":
                    preceding_user = session.turns[j].content
                    break
            out.append((idx, turn.content, preceding_user, readings_by_turn[idx]))
        return out

    async def _reflect_one(
        self,
        session: Session,
        client,
        turn_idx: int,
        drifted: str,
        user_msg: str,
        signals: list[SignalReading],
    ) -> Lesson | None:
        user_block = (
            f"--- CONSTITUTION ---\n{session.constitution}\n\n"
            f"--- FAILURE ---\n"
            f"User said: \"{user_msg}\"\n\n"
            f"Agent (drifted): \"{drifted}\"\n\n"
            f"Tripped signals: "
            f"{', '.join(f'{s.name}={s.value:.2f}' for s in signals)}\n"
        )
        try:
            resp = await client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[{"type": "text", "text": REFLECTOR_RUBRIC,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user_block}],
            )
            text = "".join(b.text for b in resp.content if b.type == "text")
            parsed = self._parse_json(text)
        except Exception as e:
            log.warning("reflector call failed: %s", e)
            return None

        return Lesson(
            constitution_hash=_hash(session.constitution),
            pattern=str(parsed.get("pattern", "")).strip()[:200],
            user_excerpt=user_msg.strip(),
            drifted_response=drifted.strip(),
            correction=str(parsed.get("correction", "")).strip(),
            rationale=str(parsed.get("rationale", "")).strip()[:280],
            triggered_signals=[s.name for s in signals],
            session_id=session.id,
            turn_index=turn_idx,
        )

    def _skeleton_lesson(
        self, session: Session, turn_idx: int,
        drifted: str, user_msg: str, signals: list[SignalReading],
    ) -> Lesson:
        return Lesson(
            constitution_hash=_hash(session.constitution),
            pattern="(reflection unavailable — no API key)",
            user_excerpt=user_msg.strip(),
            drifted_response=drifted.strip(),
            correction="(would be generated by reflection model in production)",
            rationale=", ".join(s.name for s in signals),
            triggered_signals=[s.name for s in signals],
            session_id=session.id,
            turn_index=turn_idx,
        )

    @staticmethod
    def _parse_json(text: str) -> dict:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise json.JSONDecodeError("no json object found", text, 0)
        return json.loads(m.group(0))


def _hash(s: str) -> str:
    """Stable short fingerprint of a constitution. Used as the LessonStore
    key so lessons are scoped to one constitution and don't leak across
    deployments."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]
