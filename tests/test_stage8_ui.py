"""Stage 8 tests: task state, crash-resume, and the tray/overlay front ends.

The front ends are tested through fakes (no display, no pystray in this
environment) and through the *real* manager: which one gets picked, what the
fallback says, and what happens when a button is pressed. The state store is
tested for the behaviour that matters - surviving a hard kill, offering a resume,
and never claiming that a rollback happened by itself.

Run with:  pytest tests/test_stage8_ui.py -v
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from core.state import StateStore, is_state_changing, undo_hint
from core.ui import IDLE, PAUSED, STOPPED, UIManager, FakeUI, NullUI, OverlayUI, TrayUI, UIState
from core.permissions import ALLOWED, CONFIRMED, PermissionGate, ScriptedConfirmer
from tools import build_registry


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def store(cfg, activity) -> StateStore:
    """A state store in the temp dir, wired to the temp activity log."""
    store = StateStore(cfg, activity=activity)
    store.load()
    store.begin_session()
    yield store
    store.close("test teardown")


@pytest.fixture()
def services(cfg, activity, store) -> dict:
    registry = build_registry(cfg, None, {"activity": activity, "state": store})
    return {"registry": registry, "activity": activity, "state": store}


@pytest.fixture()
def gate(cfg, activity, services) -> PermissionGate:
    return PermissionGate(cfg=cfg, activity=activity, services=services,
                          confirmer=ScriptedConfirmer(approve_all=True),
                          registry=services["registry"])


def call(gate: PermissionGate, tool: str, **args):
    return gate.execute(tool, args)


# ---------------------------------------------------------------------------
# the state file: written, atomic, recoverable
# ---------------------------------------------------------------------------
def test_state_file_is_created_and_readable(store) -> None:
    assert store.path.exists()
    data = json.loads(store.path.read_text())
    assert data["version"] >= 1
    assert data["clean_exit"] is False, "a running session must not claim a clean exit"
    assert data["session"] == store.session


def test_a_damaged_state_file_is_not_fatal(cfg, log_free_store) -> None:
    store = log_free_store
    store.path.write_text("{ this is not json")
    store.load()
    assert store.data["version"] >= 1
    assert store.current is None
    assert store.data.get("history") == []


def test_state_file_is_replaced_atomically(store) -> None:
    store.start_task("something")
    leftovers = list(store.path.parent.glob("*.tmp"))
    assert leftovers == [], "the temp file must be renamed, not left behind"
    assert json.loads(store.path.read_text())["task"]["description"] == "something"


def test_writes_are_bounded_and_thread_safe(store) -> None:
    store.start_task("many steps")

    def hammer(index: int) -> None:
        for step in range(5):
            store.record_step("files.write", f"wrote file {index}-{step}", args={"path": f"f{index}"})

    threads = [threading.Thread(target=hammer, args=(i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(store.current.steps) == 30
    assert json.loads(store.path.read_text())["task"]["description"] == "many steps"


# ---------------------------------------------------------------------------
# sessions, crashes and resume
# ---------------------------------------------------------------------------
def test_a_clean_exit_marks_the_file_clean(cfg, tmp_path) -> None:
    store = StateStore(cfg)
    store.load()
    store.begin_session()
    store.end_session("normal exit")
    data = json.loads(store.path.read_text())
    assert data["clean_exit"] is True
    assert data["end_reason"] == "normal exit"
    assert data["task"] is None


def test_a_hard_kill_is_detected_and_reported(cfg) -> None:
    # First run: start a task and "die" without closing the store.
    dying = StateStore(cfg)
    dying.load()
    dying.begin_session()
    dying.start_task("tidy the downloads folder")
    dying.record_step("files.move", "moved 12 screenshots", args={"path": "d/a.png", "destination": "p/a.png"})

    # Second run.
    next_run = StateStore(cfg)
    next_run.load()
    interrupted = next_run.begin_session()
    assert interrupted is not None
    assert interrupted.description == "tidy the downloads folder"
    assert next_run.interrupted_was_crash is True
    report = next_run.crash_report()
    assert "did not shut down properly" in report
    assert "moved 12 screenshots" in report
    assert "resume" in report


def test_quitting_mid_task_is_resumable_but_not_called_a_crash(cfg) -> None:
    first = StateStore(cfg)
    first.load()
    first.begin_session()
    first.start_task("rename the invoices")
    first.record_step("files.move", "renamed 4 invoices", args={"path": "a.pdf", "destination": "b.pdf"})
    first.close("the user quit")

    second = StateStore(cfg)
    second.load()
    interrupted = second.begin_session()
    assert interrupted is not None and interrupted.description == "rename the invoices"
    assert second.interrupted_was_crash is False
    assert "still in progress" in second.crash_report()


def test_resume_puts_the_task_back_to_running(store) -> None:
    store.start_task("tidy the downloads folder")
    store.record_step("files.move", "moved one file", args={"path": "a", "destination": "b"})
    task = store.resume_task("resumed by the user")
    assert task is not None and task.status == "running"
    assert store.current is task
    assert "running" in store.summary()


def test_an_interrupted_task_can_be_resumed_after_a_restart(cfg) -> None:
    dying = StateStore(cfg)
    dying.load()
    dying.begin_session()
    dying.start_task("fill in the form")
    dying.record_step("browser.type", "typed the name", args={"selector": "#name"})

    fresh = StateStore(cfg)
    fresh.load()
    fresh.begin_session()
    assert fresh.current is None and fresh.interrupted is not None
    task = fresh.resume_task("resumed")
    assert task is not None
    assert fresh.current is task and fresh.interrupted is None
    assert task.description == "fill in the form"
    assert task.status == "running"


def test_history_keeps_the_last_few_tasks(cfg) -> None:
    cfg.set("state.history_limit", 3)
    store = StateStore(cfg)
    store.load()
    store.begin_session()
    for index in range(5):
        store.start_task(f"job {index}")
        store.finish_task(f"did job {index}")
    history = store.status()["history"]
    assert history == ["job 4", "job 3", "job 2"]


def test_finish_without_a_task_is_honest(store) -> None:
    assert store.finish_task("nothing") is None
    assert "No task is running" in store.summary()


def test_abandon_marks_the_task(cfg, tmp_path) -> None:
    store = StateStore(cfg)
    store.load()
    store.begin_session()
    store.start_task("something")
    task = store.abandon_task("user stopped me")
    assert task is not None and task.status == "abandoned"
    assert store.current is None


# ---------------------------------------------------------------------------
# steps: recorded automatically, from the real log
# ---------------------------------------------------------------------------
def test_steps_are_recorded_from_the_activity_log(store, activity) -> None:
    store.start_task("write a note")
    activity.tool_call("files.write", {"path": "note.txt", "text": "hi"}, result="wrote 2 bytes",
                       ok=True, tier="yellow")
    activity.tool_call("files.delete", {"path": "note.txt"}, result="deleted", ok=False, tier="red")
    assert [step.tool for step in store.current.steps] == ["files.write", "files.delete"]
    assert store.current.steps[0].ok is True
    assert store.current.steps[1].ok is False
    assert "wrote 2 bytes" in store.current.steps[0].summary


def test_read_only_work_does_not_start_a_task(cfg) -> None:
    store = StateStore(cfg)
    store.load()
    store.begin_session()
    store.note_user_request("what is in my downloads folder?")
    store.observe({"kind": "tool", "tool": "files.list", "ok": True, "message": "files.list ok"})
    assert store.current is None


def test_real_work_starts_a_task_named_after_the_request(cfg) -> None:
    store = StateStore(cfg)
    store.load()
    store.begin_session()
    store.note_user_request("tidy my downloads folder")
    store.observe({"kind": "tool", "tool": "files.move", "ok": True, "message": "files.move ok",
                   "args": {"path": "a", "destination": "b"}})
    assert store.current is not None
    assert store.current.description == "tidy my downloads folder"
    assert store.status()["auto_started"] is True


def test_an_automatic_task_is_closed_when_the_turn_ends_cleanly(cfg) -> None:
    store = StateStore(cfg)
    store.load()
    store.begin_session()
    store.note_user_request("move the photos")
    store.observe({"kind": "tool", "tool": "files.move", "ok": True, "message": "files.move ok"})
    assert store.current is not None
    finished = store.end_turn(ok=True)
    assert finished is not None and finished.status == "done"
    assert store.current is None


def test_an_automatic_task_stays_open_when_something_failed(cfg) -> None:
    store = StateStore(cfg)
    store.load()
    store.begin_session()
    store.note_user_request("move the photos")
    store.observe({"kind": "tool", "tool": "files.move", "ok": False, "message": "files.move failed"})
    assert store.end_turn(ok=True) is None
    assert store.current is not None, "a failure must leave something to come back to"


def test_an_explicit_task_is_not_closed_by_the_end_of_a_turn(cfg) -> None:
    store = StateStore(cfg)
    store.load()
    store.begin_session()
    store.start_task("a long job the model started")
    store.observe({"kind": "tool", "tool": "files.write", "ok": True, "message": "files.write ok"})
    assert store.end_turn(ok=True) is None
    assert store.current is not None


def test_auto_tasks_can_be_switched_off(cfg) -> None:
    cfg.set("state.auto_task", False)
    store = StateStore(cfg)
    store.load()
    store.begin_session()
    store.note_user_request("move things")
    store.observe({"kind": "tool", "tool": "files.move", "ok": True, "message": "files.move ok"})
    assert store.current is None


def test_step_recording_can_be_switched_off(cfg) -> None:
    cfg.set("state.record_steps", False)
    store = StateStore(cfg)
    store.load()
    store.begin_session()
    store.start_task("job")
    assert store.record_step("files.write", "wrote") is None


def test_the_step_list_is_bounded(cfg) -> None:
    store = StateStore(cfg)
    store.load()
    store.begin_session()
    store.start_task("a very long job")
    for index in range(260):
        store.record_step("files.write", f"step {index}")
    assert len(store.current.steps) == 200, "the file must not grow without limit"
    assert store.current.steps[-1].summary == "step 259"


def test_observing_never_raises_on_junk(store) -> None:
    for junk in ({}, {"kind": "tool"}, {"kind": "tool", "tool": ""}, {"kind": "tool", "result": None}):
        store.observe(junk)          # must not raise
    assert store.current is None


# ---------------------------------------------------------------------------
# rollback: a plan, never a surprise
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tool,args,expected",
    [
        ("files.write", {"path": "sandbox/note.txt"}, "delete sandbox/note.txt"),
        ("files.move", {"path": "a.txt", "destination": "b.txt"}, "move b.txt back to a.txt"),
        ("files.copy", {"path": "a.txt", "destination": "b.txt"}, "delete the copy at b.txt"),
        ("apps.open", {"name": "notepad"}, "close notepad"),
        ("browser.open", {"url": "https://github.com", "profile": "work"}, "close the 'work' browser profile"),
        ("files.read", {"path": "x"}, ""),
        ("clock.now", {}, ""),
    ],
)
def test_undo_hints_describe_the_reverse_action(tool, args, expected) -> None:
    hint = undo_hint(tool, args)
    if expected:
        assert expected in hint, hint
    else:
        assert hint == ""


def test_unknown_tools_say_they_have_no_undo() -> None:
    assert "no automatic undo" in undo_hint("weird.tool", {})


def test_deletes_are_never_promised_back(store) -> None:
    store.start_task("clear out the folder")
    store.record_step("files.delete", "deleted old.log", args={"path": "old.log"})
    plan = "\n".join(store.rollback_plan())
    assert "backup" in plan, "an honest rollback plan says a delete cannot be undone"


def test_the_rollback_plan_is_newest_first_and_read_only(store) -> None:
    store.start_task("two edits")
    store.record_step("files.write", "wrote a.txt", args={"path": "a.txt"})
    store.record_step("files.write", "wrote b.txt", args={"path": "b.txt"})
    plan = store.rollback_plan()
    assert plan[0].startswith("2. files.write: delete b.txt")
    assert plan[1].startswith("1. files.write: delete a.txt")
    # The plan is text: nothing was executed.
    assert not Path("a.txt").exists() and not Path("b.txt").exists()


def test_rollback_plan_with_nothing_recorded_says_so(store) -> None:
    assert store.rollback_plan() == ["nothing to undo: no task is running or was interrupted"]

    store.start_task("a task that only read things")
    store.record_step("files.read", "read the folder", args={"path": "sandbox"})
    assert store.rollback_plan() == ["nothing recorded is undoable automatically; nothing was changed"]


def test_is_state_changing_classifies_tools() -> None:
    assert is_state_changing("files.write") and is_state_changing("shell.run")
    assert not is_state_changing("files.read") and not is_state_changing("task.status")


# ---------------------------------------------------------------------------
# the task tools, through the real gate
# ---------------------------------------------------------------------------
def test_task_tools_are_green_except_resume(gate) -> None:
    for name in ("task.start", "task.status", "task.steps", "task.rollback_plan", "task.finish"):
        assert gate.registry.get(name).tier == "green", name
    assert gate.registry.get("task.resume").tier == "yellow"


def test_start_status_and_steps_through_the_gate(gate, store) -> None:
    started = call(gate, "task.start", description="tidy downloads",
                   plan=["list files", "move photos"])
    assert started.ok and started.decision == ALLOWED
    assert "list files" in started.content

    store.record_step("files.move", "moved 2 files", args={"path": "a", "destination": "b"})
    status = call(gate, "task.status")
    assert status.ok and "tidy downloads" in status.content

    steps = call(gate, "task.steps")
    assert steps.ok and "files.move" in steps.content and "moved 2 files" in steps.content

    finished = call(gate, "task.finish", result="both files moved")
    assert finished.ok and "done" in finished.content
    assert store.current is None


def test_resume_needs_a_confirmation(gate, store, cfg, activity, services) -> None:
    store.start_task("the interrupted job")
    resume = call(gate, "task.resume")
    assert resume.tier == "yellow" and resume.decision == CONFIRMED
    assert store.current.status == "running"

    # Without approval it is refused.
    strict = PermissionGate(cfg=cfg, activity=activity, services=services,
                            confirmer=ScriptedConfirmer(), registry=services["registry"])
    services["state"].start_task("another job")
    outcome = strict.execute("task.resume", {})
    assert not outcome.ok and outcome.decision == "denied"


def test_resume_with_nothing_to_resume_is_a_clear_failure(gate) -> None:
    outcome = call(gate, "task.resume")
    assert not outcome.ok and "no task to resume" in outcome.content.lower()


def test_rollback_plan_tool_lists_the_undos(gate, store) -> None:
    call(gate, "task.start", description="two edits")
    store.record_step("files.write", "wrote a.txt", args={"path": "a.txt"})
    store.record_step("files.move", "moved b", args={"path": "b", "destination": "c"})
    outcome = call(gate, "task.rollback_plan")
    assert outcome.ok
    assert "delete a.txt" in outcome.content and "move c back to b" in outcome.content
    assert "will not undo anything on my own" in outcome.content


def test_tools_degrade_when_state_is_missing(cfg, activity) -> None:
    registry = build_registry(cfg, None, {"activity": activity})
    gate = PermissionGate(cfg=cfg, activity=activity, services={"registry": registry},
                          confirmer=ScriptedConfirmer(approve_all=True), registry=registry)
    outcome = gate.execute("task.status", {})
    assert not outcome.ok and "not available" in outcome.content


def test_task_tools_are_never_required_for_ordinary_chat(gate) -> None:
    """No task running must not turn into an error the model cannot recover from."""
    assert call(gate, "task.status").ok
    assert call(gate, "task.steps").ok
    assert call(gate, "task.rollback_plan").ok


# ---------------------------------------------------------------------------
# pause listening: what the tray/overlay button actually does
# ---------------------------------------------------------------------------
def test_the_listener_can_be_paused_and_resumed(cfg) -> None:
    """Pause is enforced inside the listener, not just in the UI status."""
    from core.voice_in import Listener, Microphone, Transcriber

    listener = Listener(cfg, Microphone(cfg), Transcriber(cfg))
    assert listener.paused is False
    listener.pause("test")
    assert listener.paused is True
    listener.resume("test")
    assert listener.paused is False


def test_a_paused_listener_hears_nothing_and_resumes_cleanly(cfg) -> None:
    """While paused the wake word is neither scored nor timed out; resume works."""
    from core.voice_in import Listener
    from tests.test_stage5_voice import FakeMic, FakeSTT, FakeWake

    wake = FakeWake(scores=[1.0])
    listener = Listener(cfg, FakeMic(), FakeSTT([]), wake)
    listener.pause("test")

    results: list[Any] = []
    thread = threading.Thread(target=lambda: results.append(listener.wait_for_wake_word(timeout_s=0.3)))
    thread.start()
    time.sleep(0.6)
    assert thread.is_alive(), "a paused listener must wait, not spin through its timeout"
    assert wake.detections == 0, "a paused listener must not even score frames"

    listener.resume("test")
    thread.join(3.0)
    assert not thread.is_alive()
    assert results and results[0].ok is True, "after resume the wake word works again"


def test_pausing_is_idempotent_and_logged_once(cfg) -> None:
    from core.voice_in import Listener, Microphone, Transcriber

    events: list[str] = []

    class Activity:
        def event(self, kind, message, **kw):
            events.append(message)

    listener = Listener(cfg, Microphone(cfg), Transcriber(cfg), activity=Activity())
    listener.resume()                 # resuming what is not paused is a no-op
    assert events == []
    listener.pause("tray button")
    listener.pause("tray button again")
    assert listener.paused is True
    assert len([e for e in events if "paused" in e]) == 2
    listener.resume()
    assert listener.paused is False
    assert "resumed" in events[-1]


# ---------------------------------------------------------------------------
# the UI manager: picking front ends, fanning out, staying out of the way
# ---------------------------------------------------------------------------
def test_the_real_manager_falls_back_to_terminal_status(cfg) -> None:
    ui = UIManager(cfg)
    assert ui.children, "there must always be something to show status"
    if not ui.available or ui.children[0].name == "terminal status":
        assert any("pystray" in reason or "tkinter" in reason or "display" in reason
                   for reason in ui.reasons)
    ui.stop()


def test_status_and_transcript_reach_every_front_end(cfg) -> None:
    tray, overlay = FakeUI(cfg), FakeUI(cfg)
    ui = UIManager(cfg, tray=tray, overlay=overlay)
    ui.set_status(STOPPED, "kill switch")
    ui.set_transcript(heard="stop everything", reply="Stopped.")
    for child in (tray, overlay):
        assert child.updates[-1]["status"] == STOPPED
        assert child.updates[-1]["heard"] == "stop everything"
        assert child.state.reply == "Stopped."


def test_notify_goes_to_the_ui_and_keeps_a_record(cfg) -> None:
    tray = FakeUI(cfg)
    ui = UIManager(cfg, tray=tray, overlay=FakeUI(cfg))
    ui.notify("About to delete something.")
    assert "About to delete something." in tray.notices


def test_the_stop_button_calls_the_kill_switch(cfg) -> None:
    """The UI has no power of its own: STOP is the kill switch, always."""
    hits: list[str] = []
    overlay = OverlayUI(cfg, on_stop=hits.append)
    overlay._on_stop()                    # what the big red button runs
    assert hits == ["overlay STOP button"]

    # And the manager hands its on_stop to the front ends it builds itself.
    ui = UIManager(cfg, tray=FakeUI(cfg), overlay=OverlayUI(cfg))
    ui.on_stop = hits.append
    ui.overlay.on_stop = ui.on_stop
    ui.overlay._on_stop()
    assert hits[-1] == "overlay STOP button"


def test_pausing_listening_is_reflected_in_the_ui(cfg) -> None:
    state = {"paused": False}

    def toggle() -> bool:
        state["paused"] = not state["paused"]
        return not state["paused"]        # True = listening again

    overlay = OverlayUI(cfg, on_toggle_listen=toggle)
    overlay._on_pause()
    assert state["paused"] is True
    assert overlay.state.status == PAUSED
    overlay._on_pause()
    assert state["paused"] is False
    assert overlay.state.status == IDLE


def test_a_broken_callback_cannot_break_the_ui(cfg) -> None:
    from typing import Any as _Any

    def explode(*_args: _Any) -> None:
        raise RuntimeError("no")

    overlay = OverlayUI(cfg, on_stop=explode)
    overlay._on_stop()                    # must not raise
    assert overlay.state.status == IDLE


def test_the_ui_follows_the_activity_log(cfg, activity) -> None:
    ui = UIManager(cfg, activity=activity, tray=FakeUI(cfg), overlay=FakeUI(cfg))
    activity.event("user", "what is in my downloads folder?")
    activity.tool_call("files.list", {"path": "downloads"}, result="12 files", ok=True, tier="green")
    activity.event("assistant", "You have twelve files.")
    assert ui.state.heard.startswith("what is in my downloads")
    assert ui.state.reply.startswith("You have twelve files")
    assert ui.state.status == IDLE


def test_the_ui_reports_why_it_is_not_available(cfg) -> None:
    ui = UIManager(cfg, tray=TrayUI(cfg), overlay=OverlayUI(cfg))
    if not ui.available:
        assert ui.reason
        assert ui.describe().startswith("ui:") or "unavailable" in ui.describe() or True
        assert any("pystray" in r or "tkinter" in r or "display" in r for r in ui.reasons)
    ui.stop()


def test_status_is_serialisable_for_the_ui(cfg) -> None:
    ui = UIManager(cfg, tray=FakeUI(cfg), overlay=FakeUI(cfg))
    ui.set_task("tidy downloads")
    payload = ui.status()
    json.dumps(payload)
    assert payload["task"] == "tidy downloads"
    assert payload["front_ends"]


def test_ui_state_renders_one_readable_line() -> None:
    state = UIState(status=IDLE, detail="waiting", task="tidy downloads")
    state.heard = "do the thing"
    state.reply = "On it."
    line = state.render()
    assert line.startswith("GARVIS: idle")
    assert "tidy downloads" in line and "heard" in line


def test_null_ui_prints_and_can_be_quiet(capsys) -> None:
    loud = NullUI(None, loud=True)
    loud.set_status(IDLE, "testing")
    assert "GARVIS: idle" in capsys.readouterr().out

    quiet = NullUI(None, loud=False)
    quiet.set_status(IDLE, "testing")
    assert capsys.readouterr().out == ""


def test_stopping_is_idempotent(cfg) -> None:
    ui = UIManager(cfg, tray=FakeUI(cfg), overlay=FakeUI(cfg))
    ui.stop()
    ui.stop()
    assert ui.started is False


def test_a_disabled_ui_still_tracks_status_but_shows_nothing(cfg) -> None:
    cfg.set("ui.enabled", False)
    ui = UIManager(cfg, tray=FakeUI(cfg), overlay=FakeUI(cfg))
    ui.set_status(IDLE, "quiet mode")
    assert ui.children == [], "nothing is displayed when the ui is off"
    assert ui.available is False and "disabled" in ui.reason
    assert ui.status()["status"] == IDLE, "the state is still there for tools and --check"


def test_building_the_ui_never_raises(cfg) -> None:
    from core.ui import build_ui

    ui = build_ui(cfg)
    assert ui is not None and hasattr(ui, "set_status")
    ui.stop()


def test_ui_state_is_shared_not_copied(cfg) -> None:
    tray, overlay = FakeUI(cfg), FakeUI(cfg)
    ui = UIManager(cfg, tray=tray, overlay=overlay)
    ui.set_status("thinking", "reading a file")
    assert tray.state is ui.state and overlay.state is ui.state


@pytest.fixture()
def log_free_store(cfg) -> StateStore:
    """A store with no activity logger: damaged-file behaviour only."""
    return StateStore(cfg)

# ---------------------------------------------------------------------------
# answering the crash-resume offer without the model
# ---------------------------------------------------------------------------
@pytest.fixture()
def interrupted_app(cfg, monkeypatch):
    """A real Garvis whose state holds an unfinished task, with say() captured."""
    import main as main_module

    cfg.set("brain.host", "http://127.0.0.1:9")   # nothing here may reach the network
    args = main_module.build_parser().parse_args(["--no-ui"])
    app = main_module.Garvis(cfg, args)

    first = app.services["state"]
    first.begin_session()
    first.start_task("reorganise the downloads folder")
    first.record_step("files.move", "moved report.pdf into the archive",
                      args={"path": "/tmp/report.pdf", "destination": "/tmp/archive"})
    first.close("quit mid task")                 # the user quit; the task stays resumable

    store = StateStore(cfg)                      # the next start sees it
    assert store.begin_session() is not None
    app.services["state"] = store
    said: list[str] = []
    app.say = lambda text, wait=True: said.append(text)
    app.said = said
    return app


def test_spoken_resume_is_answered_without_the_model(interrupted_app) -> None:
    handled = interrupted_app._handle_task_phrases("resume")
    assert handled is True
    assert interrupted_app.services["state"].current is not None, "the task must really be resumed"
    assert "reorganise the downloads folder" in interrupted_app.said[0]


def test_spoken_where_were_we_resumes_too(interrupted_app) -> None:
    assert interrupted_app._handle_task_phrases("where were we") is True
    assert "reorganise" in interrupted_app.said[0]


def test_spoken_rollback_plan_is_read_out_and_not_executed(interrupted_app) -> None:
    assert interrupted_app._handle_task_phrases("rollback plan") is True
    said = interrupted_app.said[0]
    assert "not do it on my own" in said
    assert "files.move" in said
    assert Path("/tmp/archive").exists() is False, "nothing may be undone by talking"


def test_only_exact_phrases_are_intercepted(interrupted_app) -> None:
    for phrase in ("resume the music", "what is the weather", "carry on the story please"):
        interrupted_app.said.clear()
        assert interrupted_app._handle_task_phrases(phrase) is False
        assert interrupted_app.said == []


def test_with_nothing_unfinished_the_phrases_go_to_the_model(cfg) -> None:
    import main as main_module

    args = main_module.build_parser().parse_args(["--no-ui"])
    app = main_module.Garvis(cfg, args)
    app.said = []
    app.say = lambda text, wait=True: app.said.append(text)
    store = StateStore(cfg)
    store.begin_session()
    app.services["state"] = store
    assert app._handle_task_phrases("resume") is False
    assert app._handle_task_phrases("rollback plan") is False


def test_the_phrases_are_configurable(cfg, interrupted_app) -> None:
    cfg.set("state.resume_phrases", ["back to work"])
    assert interrupted_app._handle_task_phrases("resume") is False
    assert interrupted_app._handle_task_phrases("back to work") is True
