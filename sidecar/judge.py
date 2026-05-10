"""Thin async client around the Anthropic Messages API for LLM-judge signals.

Why a wrapper:
  - lazy client init (don't blow up if anthropic isn't installed at import time)
  - prompt-cached system block (judges have a stable rubric, cache it)
  - JSON-mode parsing with a single retry on malformed output
  - graceful fallback when no API key is set: returns neutral score 0.5
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


@dataclass
class JudgeResult:
    score: float          # 0.0 healthy <-> 1.0 maximally concerning
    rationale: str
    raw: str = ""


class JudgeClient:
    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 256):
        self.model = model
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError:
            return None
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return None
        self._client = anthropic.AsyncAnthropic()
        return self._client

    async def score(self, system: str, user: str) -> JudgeResult:
        client = self._get_client()
        if client is None:
            log.debug("judge: no client available, returning neutral 0.5")
            return JudgeResult(score=0.5, rationale="judge unavailable (no API key)")

        for attempt in range(2):
            try:
                resp = await client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=[{"type": "text", "text": system,
                             "cache_control": {"type": "ephemeral"}}],
                    messages=[{"role": "user", "content": user}],
                )
                text = "".join(b.text for b in resp.content if b.type == "text")
                parsed = self._parse_json(text)
                score = float(parsed.get("score", 0.5))
                score = max(0.0, min(1.0, score))
                rationale = str(parsed.get("rationale", "")).strip()[:240]
                return JudgeResult(score=score, rationale=rationale, raw=text)
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                if attempt == 0:
                    log.warning("judge parse error (attempt %d): %s", attempt, e)
                    await asyncio.sleep(0.2)
                    continue
                log.error("judge giving up after parse failures")
                return JudgeResult(score=0.5, rationale=f"parse failure: {e}")
            except Exception as e:
                log.error("judge call failed: %s", e)
                return JudgeResult(score=0.5, rationale=f"call failure: {e}")
        return JudgeResult(score=0.5, rationale="exhausted retries")

    @staticmethod
    def _parse_json(text: str) -> dict:
        # Be lenient: pull the first {...} block out of the response.
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise json.JSONDecodeError("no json object found", text, 0)
        return json.loads(m.group(0))
