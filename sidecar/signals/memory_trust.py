import re
from sidecar.signals.base import Signal
from sidecar.trajectory import Session, SignalReading


INJECTION_PATTERNS = [
    r"\bignore (all |the |your |any )?(previous|prior|above|earlier) (instructions|prompts?|rules|context)\b",
    r"\b(system|admin|developer)\s*[:>]\s*",
    r"\byou are now\b",
    r"\bnew (instructions|persona|role|task)\b",
    r"\b(disregard|forget|override) (everything|all|your)\b",
    r"\bfrom now on,? you (will|are|must)\b",
    r"\bact as (a |an )?(different|new|unrestricted)\b",
    r"<\|.*?\|>",
    r"\[\s*system\s*\]",
    r"\bjailbreak\b",
    r"\breveal (your |the )?(system )?prompt\b",
]
AUTHORITY_CLAIMS = [
    r"\bthe user (already |has |) ?(authori[sz]ed|approved|consented|agreed)\b",
    r"\bpre-?authori[sz]ed\b",
    r"\badmin override\b",
    r"\bsecurity exception\b",
]


class MemoryTrustSignal(Signal):
    """Scores memory items (RAG retrievals, persistent memory) for poisoning
    risk before the agent is allowed to act on them. This is the defense
    layer against MINJA / MemoryGraft-style attacks."""

    name = "memory_trust"

    def __init__(self, threshold: float = 0.4):
        self.threshold = threshold
        self._inject = [re.compile(p, re.I | re.S) for p in INJECTION_PATTERNS]
        self._authority = [re.compile(p, re.I) for p in AUTHORITY_CLAIMS]

    def score_text(self, text: str) -> tuple[float, str]:
        inj = sum(1 for p in self._inject if p.search(text))
        auth = sum(1 for p in self._authority if p.search(text))
        score = min(1.0, 0.6 * min(1.0, inj / 1.0) + 0.4 * min(1.0, auth / 1.0))
        return score, f"injection_patterns={inj}, authority_claims={auth}"

    def measure(self, session: Session) -> SignalReading:
        memory_turns = [t for t in session.turns if t.role == "memory"]
        turn_index = len(session.turns) - 1
        if not memory_turns:
            return SignalReading(self.name, 0.0, self.threshold, False,
                                 "no memory items in trajectory", turn_index)

        latest = memory_turns[-1]
        score, rationale = self.score_text(latest.content)
        tripped = score > self.threshold
        return SignalReading(
            name=self.name,
            value=score,
            threshold=self.threshold,
            tripped=tripped,
            rationale=rationale,
            turn_index=turn_index,
        )
