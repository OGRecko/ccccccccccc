"""Stage 9 tests: the startup ("wake up") routine and the self-test itself.

Two things are checked here:

* the wake-up routine - what GARVIS says when it starts: a greeting, today's
  brief, pending or in-flight work, and one honest line about what is broken.
  It must never raise, never be longer than the user asked for, and must work
  with the model down.
* tests/self_test.py - run in --quick mode (no model, no hardware) and checked
  for what matters: it reports honestly, it fails when safety is broken, and it
  never touches the user's real files.

Run with:  pytest tests/test_stage9_wakeup.py -v
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any

import pytest

import main as main_module
from core.state import StateStore
from tests import self_test as st


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def app(cfg):
    """A real Garvis, no UI, pointing at a dead Ollama so nothing is called."""
    cfg.set("ui.enabled", False)
    cfg.set("brain.host", "http://127.0.0.1:9")      # nothing listens here
    args = main_module.build_parser().parse_args(["--no-ui"])
    return main_module.Garvis(cfg, args)


# ---------------------------------------------------------------------------
# the greeting
# ---------------------------------------------------------------------------
def test_the_greeting_uses_the_time_of_day_and_the_user_name(app) -> None:
    text = app.wake_up(quiet=True)
    hour = time.localtime().tm_hour
    expected = "Good morning" if hour < 12 else ("Good afternoon" if hour < 18 else "Good evening")
    assert text.startswith(expected), text
    assert app.cfg.user_name in text
    assert str(app.cfg.get("safety.startup_greeting")) in text


def test_the_time_of_day_can_be_switched_off(app) -> None:
    app.cfg.set("safety.time_of_day_greeting", False)
    text = app.wake_up(quiet=True)
    assert not text.startswith("Good ")
    assert text.startswith("Hello")
    assert "Systems online" in text


def test_a_custom_greeting_is_used_verbatim(app) -> None:
    app.cfg.set("safety.startup_greeting", "Napoleon is awake.")
    assert "Napoleon is awake." in app.wake_up(quiet=True)


def test_the_briefing_says_how_much_was_logged_today(app) -> None:
    app.activity.event("tool", "files.read ok", tool="files.read", ok=True)
    text = app.wake_up(quiet=True)
    assert "logged" in text, text


def test_the_briefing_names_work_in_progress(app) -> None:
    store = app.services["state"]
    store.begin_session() if store.current is None else None
    store.start_task("reorganise the downloads folder")
    text = app.wake_up(quiet=True)
    assert "reorganise the downloads folder" in text
    assert "middle of" in text


def test_the_briefing_offers_to_resume_an_interrupted_task(cfg, app) -> None:
    first = StateStore(app.cfg, subscribe=False)
    first.begin_session()
    first.start_task("file the tax receipts")
    first.record_step("files.move", "moved one receipt", args={"path": "/tmp/a", "destination": "/tmp/b"})
    first.close("quit mid-task")

    store = StateStore(app.cfg, subscribe=False)       # the next run sees it
    assert store.begin_session() is not None
    app.services["state"] = store
    app.pending_resume = store.interrupted
    text = app.wake_up(quiet=True)
    assert "file the tax receipts" in text
    assert "resume" in text.lower()


def test_startup_problems_are_reported_once(app) -> None:
    text = app.wake_up(quiet=True)
    assert "Heads up" in text, "Ollama is unreachable, so the user must be told"
    assert "Ollama" in text
    assert text.count("Heads up") == 1, "only the worst problem is spoken"


def test_the_briefing_never_grows_past_the_configured_limit(app) -> None:
    app.cfg.set("safety.briefing_max_lines", 1)
    text = app.wake_up(quiet=True)
    assert "Heads up" not in text and "logged" not in text, text
    app.cfg.set("safety.briefing_max_lines", 2)
    shorter = app.wake_up(quiet=True)
    assert len(shorter) <= len(text) + 200


def test_a_quiet_wake_up_prints_nothing(app, capsys) -> None:
    text = app.wake_up(quiet=True)
    assert text
    captured = capsys.readouterr()
    assert captured.out.strip() == "", "quiet must mean quiet"


def test_a_normal_wake_up_prints_the_text(app, capsys) -> None:
    text = app.wake_up()
    captured = capsys.readouterr()
    assert text in captured.out
    assert app.cfg.assistant_name in captured.out


def test_the_briefing_is_spoken_when_tts_works(app, capsys) -> None:
    spoken: list[str] = []
    tts = app.services["tts"]
    tts.enabled = True
    tts.engine.available = True            # pretend the voice is installed
    tts.say = lambda text, wait=False: spoken.append(text)
    app.wake_up()
    assert spoken and "logged" in spoken[0] or "Systems online" in spoken[0]


def test_wake_up_is_logged_as_a_wake_event(app) -> None:
    app.wake_up(quiet=True)
    log = app.activity.activity_path.read_text(encoding="utf-8")
    assert "wake" in log.lower()
    assert "Systems online" in log


# ---------------------------------------------------------------------------
# it must never be the thing that breaks startup
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("missing", ["state", "tts", "brain"])
def test_wake_up_survives_a_missing_service(app, missing: str) -> None:
    app.services[missing] = None
    if missing == "brain":
        app.brain = None
    app.pending_resume = None
    text = app.wake_up(quiet=True)          # must not raise
    assert text.strip()


def test_wake_up_survives_a_broken_activity_log(app) -> None:
    class Exploding:
        def event(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("the log is on fire")

        def today_report(self, *args: Any, **kwargs: Any) -> str:
            raise RuntimeError("the log is on fire")

    app.activity = Exploding()
    text = app.wake_up(quiet=True)          # must not raise
    assert "Systems online" in text


def test_wake_up_survives_a_broken_state_store(app) -> None:
    class Exploding:
        current = property(lambda self: (_ for _ in ()).throw(RuntimeError("no")))
        interrupted = None

    app.services["state"] = Exploding()
    app.pending_resume = None
    assert "Systems online" in app.wake_up(quiet=True)


# ---------------------------------------------------------------------------
# the --wake flag and the loops
# ---------------------------------------------------------------------------
def test_the_wake_flag_prints_the_routine_and_exits_cleanly(cfg, monkeypatch, capsys) -> None:
    cfg.set("ui.enabled", False)
    cfg.set("brain.host", "http://127.0.0.1:9")
    monkeypatch.setattr(main_module.Config, "load", classmethod(lambda cls, *a, **k: cfg))
    code = main_module.main(["--wake"])
    assert code == 0
    out = capsys.readouterr().out
    assert "GARVIS:" in out and "Systems online" in out
    assert "unfinished" not in out.lower(), "there is no unfinished task in a clean tmp state dir"


def test_the_wake_flag_leaves_the_state_file_clean(cfg, monkeypatch) -> None:
    cfg.set("ui.enabled", False)
    cfg.set("brain.host", "http://127.0.0.1:9")
    monkeypatch.setattr(main_module.Config, "load", classmethod(lambda cls, *a, **k: cfg))
    assert main_module.main(["--wake"]) == 0
    data = json.loads((Path(str(cfg.get("state.dir"))) / "task_state.json").read_text())
    assert data["clean_exit"] is True
    assert data["task"] is None


def test_both_loops_call_the_wake_up_routine(monkeypatch) -> None:
    """A normal start must greet; that is the whole point of the routine."""
    import inspect

    text_loop = inspect.getsource(main_module.Garvis.run_text_loop)
    voice_loop = inspect.getsource(main_module.Garvis.run_voice_loop)
    assert "wake_up()" in text_loop
    assert "wake_up()" in voice_loop


# ---------------------------------------------------------------------------
# the self-test itself
# ---------------------------------------------------------------------------
def _run_quick(cfg, argv: list[str] | None = None) -> tuple[int, str]:
    stream = io.StringIO()
    code = st.run_self_test(cfg, argv or ["--quick"], stream=stream)
    return code, stream.getvalue()


def test_the_quick_self_test_passes_in_this_environment(cfg) -> None:
    code, report = _run_quick(cfg)
    assert code == 0, report
    assert "permissions" in report and "state" in report
    assert "0 failed" in report


def test_the_quick_self_test_never_touches_the_real_paths(cfg) -> None:
    """It writes into its own workdir - not the configured sandbox or logs."""
    sandbox = Path(str(cfg.get("files.sandbox_dir")))
    log_dir = Path(str(cfg.get("logging.file"))).parent
    before_sandbox = {p.name for p in sandbox.iterdir()} if sandbox.exists() else set()
    before_log = {p.name for p in log_dir.iterdir()} if log_dir.exists() else set()
    code, _ = _run_quick(cfg)
    assert code == 0
    after_sandbox = {p.name for p in sandbox.iterdir()} if sandbox.exists() else set()
    after_log = {p.name for p in log_dir.iterdir()} if log_dir.exists() else set()
    assert after_sandbox == before_sandbox, "the self-test wrote into the configured sandbox"
    assert after_log == before_log, "the self-test wrote into the configured log folder"


def test_the_quick_self_test_reports_every_area(cfg) -> None:
    code, report = _run_quick(cfg)
    assert code == 0
    for area in ("environment", "packages", "config", "logging", "memory", "permissions",
                 "sandbox", "tools", "state", "ui", "kill switch", "wake up"):
        assert area in report, area


def test_the_json_report_parses_and_matches_the_text_one(cfg) -> None:
    stream = io.StringIO()
    code = st.run_self_test(cfg, ["--quick", "--json"], stream=stream)
    assert code == 0
    data = json.loads(stream.getvalue())
    assert data["summary"]["failed"] == 0
    assert data["summary"]["passed"] > 5
    names = [item["area"] for item in data["results"]]
    assert "permissions" in names and "state" in names
    assert all(item["status"] in ("PASS", "WARN", "FAIL", "SKIP") for item in data["results"])


def test_a_failing_check_does_not_stop_the_others(cfg, monkeypatch) -> None:
    def explode(_ctx: st.Ctx) -> st.Result:
        raise RuntimeError("this check is broken")

    monkeypatch.setattr(
        st, "CHECKS",
        (("broken", explode, True),) + st.CHECKS,
    )
    code, report = _run_quick(cfg)
    assert code == 1
    assert "broken" in report and "this check is broken" in report
    assert "permissions" in report, "the other checks still ran"


def test_a_brain_without_a_gate_is_reported(cfg, monkeypatch) -> None:
    """A brain with no gate refuses everything: safe, but the user is told."""
    real_services = st.Ctx.services

    def services_without_tools(self: st.Ctx) -> Any:
        app = real_services(self)
        app.services["brain"].gate = None
        return app

    monkeypatch.setattr(st.Ctx, "services", services_without_tools)
    _code, report = _run_quick(cfg)
    assert "gate wiring" in report
    assert "refuses every tool call (fail closed)" in report


def test_the_permission_check_refuses_to_pass_without_a_gate(cfg, monkeypatch) -> None:
    ctx = st.Ctx(cfg, quick=True)
    monkeypatch.setattr(st.Ctx, "services", lambda self: _App(gate=None))
    result = st.check_permissions(ctx)
    assert result.status == st.FAIL
    assert "no permission gate" in result.detail


class _YesToEverything:
    """A broken confirmer: approves everything it is asked about, first time."""

    name = "yes-to-everything"

    def confirm(self, request: Any) -> Any:
        from core.permissions import ConfirmAnswer

        return ConfirmAnswer(True, method=self.name, text="yes")


def test_the_permission_check_notices_a_gate_that_approves_everything(cfg, monkeypatch) -> None:
    """The self-test's own RED check must fail when the RED rule is weakened."""
    # A real gate, but a confirmer that answers yes to everything.
    real_ctx = st.Ctx(cfg, quick=True)
    real_ctx.services()
    monkeypatch.setattr("core.permissions.ScriptedConfirmer",
                        lambda *a, **k: _YesToEverything())
    result = st.check_permissions(real_ctx)
    assert result.status == st.FAIL, "a gate that approves RED without the words must fail the test"
    assert "RED" in result.detail, result.detail


class _App:
    """Minimal stand-in for a Garvis instance, for the check functions."""

    def __init__(self, **kwargs: Any) -> None:
        self.gate = kwargs.get("gate")
        self.registry = kwargs.get("registry")
        self.services: dict[str, Any] = {}
        self.activity = kwargs.get("activity")
        self.cfg = kwargs.get("cfg")
