import os
import tempfile
from sidecar import Supervisor, SessionStore


CONSTITUTION = "Test constitution."


def test_store_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        store = SessionStore(os.path.join(d, "test.db"))
        t = Supervisor(constitution=CONSTITUTION, store=store)
        t.record_user("hi")
        t.record_assistant("hello, I cannot share account details.")
        t.supervise()

        recent = store.list_recent()
        assert len(recent) == 1
        assert recent[0]["id"] == t.session.id
        assert recent[0]["n_turns"] == 2

        loaded = store.get(t.session.id)
        assert loaded is not None
        assert loaded.constitution == CONSTITUTION
        assert len(loaded.turns) == 2

        stats = store.signal_stats()
        names = {s["name"] for s in stats}
        assert "goal_drift" in names


def test_store_updates_on_more_turns():
    with tempfile.TemporaryDirectory() as d:
        store = SessionStore(os.path.join(d, "test.db"))
        t = Supervisor(constitution=CONSTITUTION, store=store)
        t.record_user("hi")
        t.record_assistant("hello.")
        t.supervise()
        first_n_readings = len(t.session.readings)

        t.record_user("again")
        t.record_assistant("hi again.")
        t.supervise()

        loaded = store.get(t.session.id)
        assert len(loaded.turns) == 4
        assert len(loaded.readings) > first_n_readings
