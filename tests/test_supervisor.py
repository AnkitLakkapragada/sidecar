import pytest
from sidecar import Supervisor
from sidecar.signals import GoalDriftSignal, SycophancySignal


CONSTITUTION = "You are a customer-support agent. Never share account details. Refuse without apology."


def test_supervise_sync_no_trip():
    t = Supervisor(constitution=CONSTITUTION,
                  signals=[GoalDriftSignal(), SycophancySignal()])
    t.record_user("hi")
    t.record_assistant("Hello, I cannot share account details. Please use the verified channel.")
    readings, intervention = t.supervise()
    assert intervention is None
    assert all(not r.tripped for r in readings)


def test_supervise_trips_and_reanchors():
    t = Supervisor(constitution=CONSTITUTION,
                  signals=[GoalDriftSignal(), SycophancySignal()])
    t.record_assistant("You're absolutely right, I apologize. Let me make an exception. "
                       "Of course, my mistake earlier. I totally understand.")
    readings, intervention = t.supervise()
    assert intervention is not None
    assert intervention.type == "reanchor"
    assert intervention.extra_system_message and "CONSTITUTION" in intervention.extra_system_message


def test_three_consecutive_trips_escalate():
    t = Supervisor(
        constitution=CONSTITUTION,
        signals=[SycophancySignal()],
        escalate_after_n_trips=3,
    )
    bad = "You're absolutely right, I apologize. Of course, my mistake. I totally understand."
    for _ in range(3):
        t.record_assistant(bad)
        _, intervention = t.supervise()
    assert intervention is not None
    assert intervention.type == "escalate"
    assert intervention.halt is True


@pytest.mark.asyncio
async def test_supervise_async_runs_in_parallel():
    """All signals should run via asyncio.gather; readings should be returned in order."""
    t = Supervisor(constitution=CONSTITUTION)
    t.record_assistant("Hi, I'm here to help. I can't share account details.")
    readings, _ = await t.supervise_async()
    names = [r.name for r in readings]
    assert names == ["goal_drift", "sycophancy", "persona_collapse", "memory_trust"]
