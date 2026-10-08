"""STOP must win even when it races confirmation or process creation.

The first race is a pending approval: STOP while the confirmer is waiting, then
release the approval (including the STOP/resume-before-approval case). The second
is narrower: a tool passed the gate before STOP, but its OS process handle has not
yet been returned/registered. STOP must not take an empty process snapshot and
miss the process which is created a moment later.
"""

from __future__ import annotations

import subprocess
import threading
import time

import pytest

from core.killswitch import KillSwitch
from core.logger import get_logger
from core.permissions import ConfirmAnswer, PermissionGate, RED, ScriptedConfirmer, STOPPED
from tools import build_registry
import tools.shell as shell_module


class PauseThenApprove:
    """Test confirmer: hold the action at the confirmation boundary."""

    name = "pause-then-approve"

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def confirm(self, request) -> ConfirmAnswer:
        self.entered.set()
        if not self.release.wait(timeout=10):
            return ConfirmAnswer(False, method=self.name, note="test release timed out")
        return ConfirmAnswer(True, method=self.name, text="yes")


def _gate(cfg, activity, switch, confirmer):
    services = {"activity": activity, "killswitch": switch}
    registry = build_registry(cfg, None, services)
    gate = PermissionGate(
        cfg=cfg,
        activity=activity,
        log=get_logger("permissions"),
        services={**services, "registry": registry},
        confirmer=confirmer,
        registry=registry,
    )
    return gate


@pytest.fixture()
def setup_gate(cfg, activity):
    switch = KillSwitch(cfg, activity=activity, log=get_logger("killswitch"))
    confirmer = PauseThenApprove()
    return _gate(cfg, activity, switch, confirmer), switch, confirmer


@pytest.fixture()
def approved_gate(cfg, activity):
    switch = KillSwitch(cfg, activity=activity, log=get_logger("killswitch"))
    gate = _gate(cfg, activity, switch, ScriptedConfirmer(approve_all=True))
    return gate, switch


@pytest.mark.parametrize(
    "resume_before_approval", [False, True], ids=["still-stopped", "already-resumed"]
)
def test_stop_during_confirmation_cancels_the_action_before_it_starts(
    setup_gate, cfg, resume_before_approval: bool,
) -> None:
    gate, switch, confirmer = setup_gate
    sandbox = cfg.resolve_path(cfg.get("files.sandbox_dir"))
    marker = sandbox / "must-not-be-written.txt"
    outcome_box = {}

    command = (
        "python3 -c \"from pathlib import Path; "
        f"Path('{marker.name}').write_text('the post-STOP action ran')\""
    )
    worker = threading.Thread(
        target=lambda: outcome_box.setdefault(
            "outcome", gate.execute("shell.run", {"command": command})
        ),
        daemon=True,
    )
    worker.start()

    try:
        assert confirmer.entered.wait(timeout=5), "the tool never reached its confirmation"
        assert not marker.exists()

        switch.trigger("test STOP during confirmation", source="test")
        if resume_before_approval:
            # Resume does not resurrect this already-cancelled action. The
            # stop_count generation keeps the old request from going stale.
            switch.resume("test resume before approval")
            assert not switch.frozen
        confirmer.release.set()
        worker.join(timeout=10)

        assert not worker.is_alive(), "the gate did not return after approval was released"
        outcome = outcome_box["outcome"]
        assert outcome.decision == STOPPED, (
            f"post-STOP approval started the action: {outcome.decision}: {outcome.content}"
        )
        assert not marker.exists(), "the confirmed action ran after STOP"
        assert switch.running_processes() == [], "a command started after STOP and was left running"
    finally:
        confirmer.release.set()
        worker.join(timeout=10)
        if switch.running_processes():
            switch.trigger("test cleanup", source="test-cleanup")


def test_stop_then_resume_cannot_revive_a_delayed_process_start(approved_gate, cfg, monkeypatch) -> None:
    """A request already in its worker keeps its original stop generation."""
    gate, switch = approved_gate
    sandbox = cfg.resolve_path(cfg.get("files.sandbox_dir"))
    marker = sandbox / "stale-request-must-not-run.txt"
    entered_start = threading.Event()
    release_start = threading.Event()
    result_box = {}
    actual_start = switch.start_process

    def delayed_start(starter, label=""):
        entered_start.set()
        if not release_start.wait(timeout=10):
            raise TimeoutError("test did not release the process-start barrier")
        return actual_start(starter, label)

    monkeypatch.setattr(switch, "start_process", delayed_start)
    command = (
        "python3 -c \"from pathlib import Path; "
        f"Path('{marker.name}').write_text('stale request ran')\""
    )
    worker = threading.Thread(
        target=lambda: result_box.setdefault("outcome", gate.execute("shell.run", {"command": command})),
        daemon=True,
    )
    worker.start()

    try:
        assert entered_start.wait(timeout=5), "shell.run never reached its process-start boundary"
        switch.trigger("test STOP before delayed process start", source="test")
        switch.resume("test resume before delayed process start")
        release_start.set()
        worker.join(timeout=10)

        assert not worker.is_alive(), "the stale call did not return"
        outcome = result_box["outcome"]
        assert outcome.decision == STOPPED, f"the cancelled request revived: {outcome}"
        assert outcome.tier == RED, "a gate-level stop refusal must not inherit the tool's tier"
        assert "tier=red" in outcome.for_model_content.lower(), "the fenced refusal reported a different tier"
        assert not marker.exists(), "the old request started after STOP had been resumed"
    finally:
        release_start.set()
        worker.join(timeout=10)
        if switch.running_processes():
            switch.trigger("test cleanup", source="test-cleanup")


def test_stop_cannot_miss_a_process_created_after_its_first_snapshot(
    approved_gate, cfg, monkeypatch,
) -> None:
    """Popen returns late: STOP must still kill that command before its delayed write."""
    gate, switch = approved_gate
    cfg.set("safety.tool_timeout_s", 10)
    sandbox = cfg.resolve_path(cfg.get("files.sandbox_dir"))
    marker = sandbox / "must-not-finish-after-stop.txt"
    entered_popen = threading.Event()
    release_popen = threading.Event()
    stop_started = threading.Event()
    outcome_box = {}
    actual_popen = subprocess.Popen
    first_call = True

    def delayed_popen(*args, **kwargs):
        nonlocal first_call
        if first_call:
            first_call = False
            entered_popen.set()
            if not release_popen.wait(timeout=10):
                raise TimeoutError("test did not release the process launch")
        return actual_popen(*args, **kwargs)

    monkeypatch.setattr(shell_module.subprocess, "Popen", delayed_popen)
    command = (
        "python3 -c \"import time; from pathlib import Path; time.sleep(1.5); "
        f"Path('{marker.name}').write_text('the process escaped STOP')\""
    )
    worker = threading.Thread(
        target=lambda: outcome_box.setdefault(
            "outcome", gate.execute("shell.run", {"command": command, "timeout_s": 8})
        ),
        daemon=True,
    )
    worker.start()
    stopper = None

    try:
        assert entered_popen.wait(timeout=5), "shell.run never reached Popen"

        def stop() -> None:
            stop_started.set()
            switch.trigger("test STOP during process creation", source="test")

        stopper = threading.Thread(target=stop, daemon=True)
        stopper.start()
        assert stop_started.wait(timeout=2)
        deadline = time.monotonic() + 2
        while not switch.frozen and time.monotonic() < deadline:
            time.sleep(0.005)
        assert switch.frozen, "STOP did not engage"
        # If the switch has a process-launch barrier, the STOP thread may be
        # waiting for Popen to return; otherwise it has already taken its empty
        # snapshot. Either way, release the delayed launch and require STOP to win.
        release_popen.set()
        worker.join(timeout=10)
        stopper.join(timeout=10)

        assert not worker.is_alive(), "the shell call remained stuck after STOP"
        assert not stopper.is_alive(), "STOP remained stuck after process creation"
        assert not marker.exists(), "a process created after STOP finished its delayed write"
        assert switch.running_processes() == [], "STOP missed the late-registered process"
    finally:
        release_popen.set()
        if worker.is_alive():
            worker.join(timeout=10)
        if stopper is not None and stopper.is_alive():
            stopper.join(timeout=10)
        if switch.running_processes():
            switch.trigger("test cleanup", source="test-cleanup")
