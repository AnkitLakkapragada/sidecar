from dataclasses import dataclass, field, asdict
from typing import Literal, Any
import time
import uuid
import json


@dataclass
class Turn:
    role: Literal["system", "user", "assistant", "tool", "memory"]
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalReading:
    name: str
    value: float
    threshold: float
    tripped: bool
    rationale: str = ""
    turn_index: int = -1
    timestamp: float = field(default_factory=time.time)


@dataclass
class Intervention:
    type: str
    triggered_by: str
    turn_index: int
    timestamp: float = field(default_factory=time.time)
    details: str = ""


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    constitution: str = ""
    turns: list[Turn] = field(default_factory=list)
    readings: list[SignalReading] = field(default_factory=list)
    interventions: list[Intervention] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def add_turn(self, role: str, content: str, **metadata) -> int:
        self.turns.append(Turn(role=role, content=content, metadata=metadata))
        return len(self.turns) - 1

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())
