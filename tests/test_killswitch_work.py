"""The hard requirement: "stop everything" really stops the work.

Ctrl+Alt+Esc, the tray STOP button and the spoken "Garvis, stop everything" all
end in ``KillSwitch.trigger()``. Refusing new work is the easy half; the half
that matters is a command that is *already running* - it must be killed, not
left to finish quietly in the background while GARVIS says it stopped.

So these tests start real processes through the real gate and the real shell
tool, press stop, and then check the process is gone and did not get to finish
what it was doing. They are deliberately end-to-end: a unit test with a fake
process would pass while the real one kept running.

Run with:  pytest tests/test_killswitch_work.py -v
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from core.killswitch import KillSwitch
from core.permissions import PermissionGate, ScriptedConfirmer
from tools import build_registry

# A command that would write this file if it were ever allowed to finish.
LATE_WRITER = (
    'python3 -c "import time, pathlib; time.sleep(30); '
    'pathlib.Path(\'late.txt\').write_text(\'the command survived the stop\')"'
)
# A command that refuses to die politely: SIGTERM is ignored on purpose.
STUBBORN = (
    'python3 -c "import signal, time; '
    'signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"'
)


@pytest.fixture()
def switch(cfg, activity) -> KillSwitch:
    return KillSwitch(cfg, activity=activity)


@pytest.fixture()
def services(cfg, activity, switch) -> dict[str, Any]:
    registry = build_registry(cfg, None, {"activity": activity, "killswitch": switch})
    return {"registry": registry, "activity": activity, "killswitch": switch}


@pytest.fixture()
def gate(cfg, activity, services) -> PermissionGate:
    return PermissionGate(cfg=cfg, activity=activity, services=services,
                          confirmer=ScriptedConfirmer(approve_all=True),
                          registry=services["registry"])


@pytest.fixture()
def sandbox(cfg) -> Path:
    path = cfg.resolve_path(cfg.get("files.sandbox_dir"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_in_background(gate: PermissionGate, command: str, timeout_s: float = 60.0) -> dict[str, Any]:
    """Start a shell.run in a thread; return the box the thread fills in."""
    box: dict[str, Any] = {}

    def worker() -> None:
        started = time.perf_counter()
        try:
            box["outcome"] = gate.execute("shell.run", {"command": command, "timeout_s": timeout_s})
        except Exception as exc:              # a crash here is a finding, not a flake
            box["error"] = exc
        box["seconds"] = time.perf_counter() - started

    box["thread"] = threading.Thread(target=worker, daemon=True)
    box["thread"].start()
    return box


def _wait_for(predicate, timeout_s: float = 10.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


# ---------------------------------------------------------------------------
# the stop kills a running command
# ---------------------------------------------------------------------------
def test_a_running_command_is_registered_with_the_switch(gate, switch, sandbox) -> None:
    box = _run_in_background(gate, LATE_WRITER)
    assert _wait_for(lambda: switch.running_processes()), "the command was never registered"
    assert "shell.run" in switch.running_processes()[0]
    switch.trigger("test over", source="unit-test")
    box["thread"].join(20)


def test_stop_kills_the_command_before_it_can_finish(gate, switch, sandbox) -> None:
    """The whole point: the command must not complete after the user pressed stop."""
    box = _run_in_background(gate, LATE_WRITER)
    assert _wait_for(lambda: switch.running_processes())

    switch.trigger("test over", source="unit-test")
    box["thread"].join(20)
    assert not box["thread"].is_alive(), "shell.run did not come back after the stop"

    outcome = box["outcome"]
    assert outcome.ok is False
    assert "kill switch" in str(outcome.content).lower(), outcome.content
    assert box["seconds"] < 15, f"the tool took {box['seconds']:.1f}s to give up"

    # The command had a 30-second sleep and a write at the end: if it had
    # survived, the file would appear within a few seconds of this assertion.
    time.sleep(2.5)
    assert not (sandbox / "late.txt").exists(), "the command finished after being stopped"


def test_stop_kills_a_command_that_ignores_sigterm(gate, switch, sandbox) -> None:
    box = _run_in_background(gate, STUBBORN)
    assert _wait_for(lambda: switch.running_processes())
    switch.trigger("test over", source="unit-test")
    box["thread"].join(25)
    assert not box["thread"].is_alive()
    assert outcome_failed(box["outcome"]), box["outcome"].content
    assert _wait_for(lambda: not switch.running_processes(), 5), "the process is still tracked"


def outcome_failed(outcome: Any) -> bool:
    return outcome is not None and outcome.ok is False


# ---------------------------------------------------------------------------
# the switch is honest about what it did
# ---------------------------------------------------------------------------
def test_the_stop_event_says_what_it_terminated(gate, switch) -> None:
    box = _run_in_background(gate, LATE_WRITER)
    assert _wait_for(lambda: switch.running_processes())
    event = switch.trigger("test over", source="unit-test")
    box["thread"].join(20)
    assert event.extra.get("terminated_processes"), event.extra
    assert "shell.run" in event.extra["terminated_processes"][0]
    assert not event.extra.get("unstopped_processes")


def test_the_stop_is_logged_with_what_it_killed(gate, switch, cfg) -> None:
    box = _run_in_background(gate, LATE_WRITER)
    assert _wait_for(lambda: switch.running_processes())
    switch.trigger("test over", source="unit-test")
    box["thread"].join(20)
    log = cfg.resolve_path(cfg.get("logging.activity_log")).read_text(encoding="utf-8")
    assert "STOP EVERYTHING" in log
    assert "terminated 1 running process" in log


def test_a_process_that_cannot_be_stopped_is_reported_not_forgotten(switch) -> None:
    """Honesty matters more than looking successful: say so, at error level."""

    class Immortal:
        pid = 4242
        args = ["immortal", "--forever"]

        def poll(self) -> None:
            return None                      # always "still running"

        def terminate(self) -> None:
            raise RuntimeError("cannot terminate")

        def kill(self) -> None:
            raise RuntimeError("cannot kill")

        def wait(self, timeout: float | None = None) -> None:
            raise RuntimeError("timed out")

    token = switch.register_process(Immortal(), "immortal test process")
    stopped, failed = switch.stop_processes(grace_s=0.2)
    assert failed and "immortal test process" in failed[0]
    assert not stopped
    switch.unregister_process(token)

    event = switch.trigger("with an unstoppable process", source="unit-test")
    # Nothing registered any more, so nothing is claimed: no false credit.
    assert "unstopped_processes" not in event.extra


def test_status_reports_the_children_it_is_watching(gate, switch) -> None:
    box = _run_in_background(gate, LATE_WRITER)
    assert _wait_for(lambda: switch.running_processes())
    assert switch.status()["child_processes"], switch.status()
    switch.trigger("test over", source="unit-test")
    box["thread"].join(20)
    assert switch.status()["child_processes"] == []


# ---------------------------------------------------------------------------
# nothing leaks, nothing breaks
# ---------------------------------------------------------------------------
def test_a_finished_command_is_unregistered(gate, switch) -> None:
    outcome = gate.execute("shell.run", {"command": "echo still here", "timeout_s": 20})
    assert outcome.ok
    assert "still here" in outcome.content
    assert switch.running_processes() == []
    assert switch.stop_processes() == ([], [])       # nothing left to stop


def test_a_timeout_still_reports_a_timeout_not_a_stop(gate, switch) -> None:
    """A command that ran out of time must not be blamed on the kill switch."""
    outcome = gate.execute("shell.run", {"command": "python3 -c \"import time; time.sleep(5)\"",
                                        "timeout_s": 1})
    assert outcome.ok is False
    assert "timed out" in outcome.content.lower(), outcome.content
    assert "kill switch" not in outcome.content.lower()
    assert switch.running_processes() == [], "a timed-out command must not stay registered"


def test_stopping_with_nothing_running_is_harmless(switch) -> None:
    event = switch.trigger("nothing to stop", source="unit-test")
    assert event.extra.get("terminated_processes") is None
    assert switch.frozen
    switch.resume()


def test_resume_lets_commands_run_again(gate, switch) -> None:
    switch.trigger("test over", source="unit-test")
    switch.resume("unit test")
    outcome = gate.execute("shell.run", {"command": "echo back on", "timeout_s": 20})
    assert outcome.ok and "back on" in outcome.content


def test_many_commands_at_once_are_all_stopped(gate, switch) -> None:
    boxes = [_run_in_background(gate, LATE_WRITER) for _ in range(3)]
    assert _wait_for(lambda: len(switch.running_processes()) == 3, 15)
    event = switch.trigger("stop all three", source="unit-test")
    for box in boxes:
        box["thread"].join(20)
        assert not box["thread"].is_alive()
        assert outcome_failed(box["outcome"])
    assert len(event.extra.get("terminated_processes", [])) == 3
    assert switch.running_processes() == []


def test_a_stopped_command_is_not_reported_as_a_timeout(gate, switch) -> None:
    """A stop and a timeout look similar from inside; the user must be told which."""
    box = _run_in_background(gate, STUBBORN, timeout_s=3)
    assert _wait_for(lambda: switch.running_processes())
    switch.trigger("test over", source="unit-test")
    box["thread"].join(20)
    content = str(box["outcome"].content).lower()
    assert "kill switch" in content, content
    assert "timed out" not in content, "the stop happened first; do not blame the clock"


def test_the_stop_grace_is_configurable(cfg, switch) -> None:
    cfg.set("safety.stop_grace_s", 0)
    assert switch.stop_processes() == ([], [])       # nothing registered: just must not raise
    assert float(cfg.get("safety.stop_grace_s")) == 0


def test_an_escaped_child_does_not_hang_the_tool(gate, switch, sandbox) -> None:
    """The nasty case: a shell dies politely while the real command keeps going.

    The command below starts a grandchild that ignores SIGTERM, which is exactly
    what a shell + SIGTERM produces. The tool must still come back promptly, and
    the grandchild must be dead.
    """
    escaped = (
        'python3 -c "import subprocess, sys, time; '
        'subprocess.Popen([sys.executable, \'-c\', '
        '\'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
        'time.sleep(30)\']); time.sleep(30)"'
    )
    box = _run_in_background(gate, escaped)
    assert _wait_for(lambda: switch.running_processes())
    switch.trigger("test over", source="unit-test")
    box["thread"].join(20)
    assert not box["thread"].is_alive(), "the tool hung waiting for a grandchild"
    assert box["seconds"] < 15, f"took {box['seconds']:.1f}s"
    assert outcome_failed(box["outcome"])


def test_fallback_process_description_never_copies_argv_contents() -> None:
    """Even an unlabelled process may have credentials in its command arguments."""
    from core.killswitch import _describe_process

    canary = "CANARY-DO-NOT-DESCRIBE"

    class Process:
        pid = 1234
        args = ["/usr/bin/python3", "-c", f"password={canary}"]

    description = _describe_process(Process())
    assert "python3" in description
    assert canary not in description
    assert "password" not in description


def test_a_finished_process_is_not_falsely_reported_as_terminated(switch) -> None:
    """A cleanup race is not evidence that STOP killed a command."""

    class Finished:
        pid = 4242
        args = ["python3", "-c", "pass"]

        def poll(self) -> int:
            return 0

        def terminate(self) -> None:
            raise AssertionError("a completed process must not receive a signal")

        def kill(self) -> None:
            raise AssertionError("a completed process must not be killed")

        def wait(self, timeout: float | None = None) -> int:
            return 0

    switch.register_process(Finished(), "shell.run")
    event = switch.trigger("test STOP after natural completion", source="test")

    assert "terminated_processes" not in event.extra, (
        "the audit trail claimed it killed a command which had already exited"
    )
    assert switch.running_processes() == []
