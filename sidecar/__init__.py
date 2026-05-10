"""sidecar — a runtime supervisor + reflection loop for AI agents.

Public API:

  Runtime supervision
    SupervisedAgent       drop-in wrapper around the Anthropic Messages API
    Supervisor            the underlying watcher (use directly to wrap any
                          loop, not just Anthropic's)
    AgentHalted           raised when the supervisor escalates a session

  Therapy / reflection
    Reflector             walks a Session, generates Lessons from drift
    Lesson                a single counterfactual case the agent carries
                          forward; auditable, JSON-serializable
    LessonStore           SQLite + FTS-backed lesson memory

  Persistence + observability
    SessionStore          per-session SQLite trajectory store
    configure_logging     opinionated structured-logging setup

  Data model
    Session, Turn, SignalReading, Intervention

Lower-level building blocks live in sub-packages:
  sidecar.signals         the four signal classes + GatedJudgeSignal
  sidecar.adapters        framework adapters (currently Anthropic)
  sidecar.eval            calibration harness + labeled fixtures
"""
from sidecar.adapters import SupervisedAgent, AgentHalted
from sidecar.lessons import LessonStore
from sidecar.logging_setup import configure_logging
from sidecar.reflection import Reflector, Lesson
from sidecar.store import SessionStore
from sidecar.supervisor import Supervisor
from sidecar.trajectory import Session, Turn, SignalReading, Intervention

__version__ = "0.3.0"

__all__ = [
    "SupervisedAgent", "Supervisor", "AgentHalted",
    "Reflector", "Lesson", "LessonStore",
    "SessionStore", "configure_logging",
    "Session", "Turn", "SignalReading", "Intervention",
    "__version__",
]
