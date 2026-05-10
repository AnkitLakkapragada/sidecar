from abc import ABC, abstractmethod
import asyncio
from sidecar.trajectory import Session, SignalReading


class Signal(ABC):
    """A signal produces a SignalReading from a Session.

    Heuristic signals (regex, TF-IDF) override `measure` synchronously.
    LLM-judge signals override `measure_async` directly so they don't
    block on a thread.
    """

    name: str
    threshold: float
    cost_class: str = "cheap"   # cheap | judge - informs gating

    @abstractmethod
    def measure(self, session: Session) -> SignalReading: ...

    async def measure_async(self, session: Session) -> SignalReading:
        return await asyncio.to_thread(self.measure, session)
