"""Two safety features that have to compose: the timeout that abandons a call, and STOP.

`safety.tool_timeout_s` abandons a call that runs too long - by design, because a
tool stuck inside a third-party library cannot be cancelled. The command it
started, however, is still a real process, still running, and may still be about
to write a file. The kill switch's promise ("terminates any command that is still
running") has to cover exactly that: work whose *caller* gave up.

The two mechanisms were built in different rounds and, until this file, were only
tested apart. The sequence here is the one that matters:

    gate call times out (2 s)  ->  tool thread abandoned, process still alive
    user presses STOP          ->  the abandoned command must die, not finish

The failure mode being guarded against is quiet and nasty: "timed out and was
abandoned" read as "nothing is running any more", while a command keeps writing
files in the background, immune to a stop that only looks at the current call.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from core.killswitch import KillSwitch
from core.logger import get_logger
from core.permissions import PermissionGate, ScriptedConfirmer, STOPPED, TIMEOUT
from tools import build_registry

#: Long enough that no test ever waits for it to finish on its own.
LONG_S = 60
#: How long the gate lets a call run before abandoning it.
CALL_TIMEOUT_S = 2.0


@pytest.fixture()
def killswitch(cfg, activity) -> KillSwitch:
    return KillSwitch(cfg, activity=activity, log=get_logger("killswitch"))


@pytest.fixture()
def registry(cfg, activity, killswitch):
    return build_registry(cfg, None, {"activity": activity, "killswitch": killswitch})


@pytest.fixture()
def gate(cfg, activity, registry, killswitch) -> PermissionGate:
    cfg.set("safety.tool_timeout_s", CALL_TIMEOUT_S)
    return PermissionGate(cfg=cfg, activity=activity, log=get_logger("permissions"),
                          services={"activity": activity, "registry": registry,
                                    "killswitch": killswitch},
                          confirmer=ScriptedConfirmer(approve_all=True), registry=registry)


@pytest.fixture()
def sandbox(cfg) -> Path:
    path = cfg.resolve_path(cfg.get("files.sandbox_dir"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _slow_command(marker: str, seconds: int = LONG_S) -> str:
    """An allowlisted command that sleeps, then writes a file. Nothing else."""
    return (
        f'python3 -c "import time, pathlib; time.sleep({seconds}); '
        f"pathlib.Path('{marker}').write_text('the abandoned command finished')\""
    )


def _wait_for_marker(path: Path, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.05)
    return False


# ---------------------------------------------------------------------------
# the sequence
# ---------------------------------------------------------------------------
def test_a_timed_out_call_leaves_its_process_alive_and_registered(gate, killswitch, sandbox) -> None:
    """The premise: 'abandoned' describes the caller, not the work."""
    outcome = gate.execute("shell.run", {"command": _slow_command("still-running.txt"),
                                         "timeout_s": LONG_S})

    assert outcome.decision == TIMEOUT and not outcome.ok
    assert "abandoned" in outcome.content
    assert killswitch.running_processes(), (
        "the abandoned call's command is not registered, so STOP could not reach it"
    )


def test_stop_kills_the_work_of_an_abandoned_call(gate, killswitch, sandbox) -> None:
    """The promise: STOP terminates any command still running, whoever gave up on it."""
    marker = sandbox / "abandoned-finished.txt"
    gate.execute("shell.run", {"command": _slow_command(marker.name), "timeout_s": LONG_S})
    assert killswitch.running_processes(), "nothing to kill: the premise test failed"

    event = killswitch.trigger("test: stop everything", source="test")

    assert event.extra.get("terminated_processes"), f"STOP terminated nothing (event={event})"
    assert not event.extra.get("unstopped_processes"), "a process survived the stop"
    assert not killswitch.running_processes(), "the registry still lists the process"

    # and no late write: the command must be dead, not merely detached
    assert not _wait_for_marker(marker, 1.5), "the abandoned command finished anyway"


def test_the_stop_is_recorded_with_what_it_terminated(gate, killswitch, activity, cfg) -> None:
    """A stop that leaves no trace is a stop nobody can audit."""
    gate.execute("shell.run", {"command": _slow_command("audit-marker.txt"), "timeout_s": LONG_S})
    killswitch.trigger("test: stop everything", source="test")

    trail = ""
    for key in ("logging.activity_log", "logging.activity_jsonl"):
        path = cfg.resolve_path(cfg.get(key))
        if path.exists():
            trail += path.read_text(encoding="utf-8", errors="replace")
    assert "terminated" in trail.lower(), "the stop and its effect are not in the audit trail"


def test_a_call_made_while_stopped_is_refused_before_it_starts(gate, killswitch, sandbox) -> None:
    """Second lock on the same door: the gate itself refuses while stopped.

    handle_line already keeps a frozen session's text away from the model, so the
    model cannot drive a call here. But the gate is the one place *every* call
    passes through, so a UI button, a worker thread or a future caller must not be
    able to run a tool after the user pressed stop either. Nothing is started, so
    there is nothing left to kill - which is strictly better than starting the
    command and terminating it a moment later.
    """
    killswitch.trigger("test: stop before the call", source="test")
    marker = sandbox / "never-should-exist.txt"

    outcome = gate.execute("shell.run", {"command": _slow_command(marker.name), "timeout_s": LONG_S})

    assert outcome.decision == STOPPED and not outcome.ok
    assert "kill switch" in (outcome.content or "").lower()
    assert "resume" in (outcome.content or "").lower(), "the model is not told how the stop ends"
    assert killswitch.running_processes() == [], "a command was started despite the stop"
    assert not _wait_for_marker(marker, 0.5)


def test_resume_gives_the_gate_back(gate, killswitch, sandbox, cfg) -> None:
    """...and the way out works, or 'refused until resume' would be a trap."""
    killswitch.trigger("test: stop", source="test")
    assert gate.execute("clock.now", {}).decision == STOPPED

    killswitch.resume("test over")

    (sandbox / "back.txt").write_text("hello\n")
    outcome = gate.execute("shell.run", {"command": "cat back.txt"})
    assert outcome.ok, outcome.content


def test_the_stop_is_recorded_when_it_refuses_a_call(gate, killswitch, cfg) -> None:
    killswitch.trigger("test: stop", source="test")
    gate.execute("shell.run", {"command": "echo hi"})

    trail = ""
    for key in ("logging.activity_log", "logging.activity_jsonl"):
        path = cfg.resolve_path(cfg.get(key))
        if path.exists():
            trail += path.read_text(encoding="utf-8", errors="replace")
    assert "the kill switch is engaged" in trail, "a refused call was not written down"
    assert "stopped" in trail


def test_a_normal_message_while_stopped_never_reaches_the_model(cfg, activity, monkeypatch) -> None:
    """The user-facing half: frozen text is consumed locally, with the way out.

    This is what the class docstring promises ("new tasks are refused") and it is
    implemented in main.py rather than the gate: a stop command that had to
    survive a model round trip would not be a stop command, and neither would the
    refusal.
    """
    import main as main_module

    app = main_module.Garvis(cfg, main_module.build_parser().parse_args([]))
    called: list[str] = []
    monkeypatch.setattr(app.brain, "respond", lambda text, **kw: called.append(text))
    app.killswitch.trigger("test: stop", source="test")

    app.handle_line("write a 10 page report about penguins")

    assert called == [], f"a stopped session sent work to the model: {called}"
    assert app.killswitch.frozen is True, "the stop silently lifted itself"


# ---------------------------------------------------------------------------
# ...and the two regimes where they must NOT interact
# ---------------------------------------------------------------------------
def test_a_normal_call_leaves_nothing_registered(gate, killswitch, sandbox) -> None:
    """No leak on the happy path: the registry is empty once a call returns."""
    (sandbox / "notes.txt").write_text("hello\n")
    outcome = gate.execute("shell.run", {"command": "cat notes.txt"})

    assert outcome.ok, outcome.content
    assert killswitch.running_processes() == [], "a finished process stayed registered"


def test_a_command_that_hits_its_own_timeout_is_killed_by_the_tool(gate, killswitch, sandbox) -> None:
    """The shell's own timeout path: it kills its process and says so."""
    outcome = gate.execute("shell.run", {"command": _slow_command("own-timeout.txt", seconds=30),
                                         "timeout_s": 1})

    assert not outcome.ok
    assert "timed out" in outcome.content and "killed" in outcome.content
    assert outcome.decision != TIMEOUT, (
        "the gate abandoned the call instead of letting the tool clean up - the tool's own "
        "timeout (1s) is inside the call timeout, so this should be an ordinary tool failure"
    )
    assert killswitch.running_processes() == [], "the tool's timeout left a process registered"


def test_the_gate_still_works_after_abandoning_a_call(gate, sandbox) -> None:
    """An abandoned thread must not wedge the gate (the lock is released, the next
    call runs, and a normal tool still answers)."""
    gate.execute("shell.run", {"command": _slow_command("wedge.txt"), "timeout_s": LONG_S})

    (sandbox / "after.txt").write_text("fine\n")
    outcome = gate.execute("shell.run", {"command": "cat after.txt"})

    assert outcome.ok, f"the gate stopped working after a timeout: {outcome.content}"
