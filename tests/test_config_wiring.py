"""Every knob that was wired up in this round must actually change behaviour.

tests/test_config_coverage.py proves a key is *read* somewhere; that is not the
same as proving it does anything. Each test here flips one setting and checks the
difference in what happens, because these four were all dead or wrong before:

* ``apps.protected`` - was ignored (the code used a hardcoded list);
* ``self_test.check_*`` - were ignored (every check always ran);
* ``voice_out.sentence_chunking`` - was ignored (always streamed sentences);
* ``state.resume_on_start`` - was ignored (always announced unfinished work);
* ``brain.cloud_fallback.use_only_if_local_down`` - was *inverted* (false turned
  the fallback off entirely); its test lives in tests/test_cloud_fallback.py.
"""

from __future__ import annotations

from typing import Any

import pytest

import main as main_module
from tools import build_registry


# ---------------------------------------------------------------------------
# apps.protected: the user can protect more processes, never fewer
# ---------------------------------------------------------------------------
def _apps_close(cfg, activity):
    registry = build_registry(cfg, None, {"activity": activity})
    tool = registry.get("apps.close")
    assert tool is not None, "apps.close is missing from the registry"
    return tool


def test_a_process_the_user_protected_cannot_be_closed(cfg, activity) -> None:
    cfg.set("apps.protected", ["mytool"])
    tool = _apps_close(cfg, activity)

    result = tool.func(name="mytool")

    assert not result.ok, "apps.protected is read but does not stop the close"
    assert "protected" in result.content
    assert "mytool" in result.content


def test_the_users_protect_list_cannot_unprotect_the_built_ins(cfg, activity) -> None:
    """`protected: []` must not make sshd (or GARVIS itself) killable."""
    cfg.set("apps.protected", [])
    tool = _apps_close(cfg, activity)

    for dangerous in ("sshd", "explorer", "garvis", "python3"):
        result = tool.func(name=dangerous)
        assert not result.ok, f"{dangerous} became closable by emptying apps.protected"
        assert "protected" in result.content


def test_the_protect_list_is_lowercased_and_trimmed(cfg, activity) -> None:
    """People write 'MyTool' and ' mytool ' - both must mean the same process."""
    cfg.set("apps.protected", ["  MyTool  "])
    tool = _apps_close(cfg, activity)
    assert not tool.func(name="MYTOOL").ok
    assert not tool.func(name="mytool.exe").ok


def test_an_unprotected_process_is_still_allowed_to_be_closed(cfg, activity) -> None:
    """The guard must not block everything: that would be a different bug."""
    tool = _apps_close(cfg, activity)
    safe = tool.func(name="some-random-app-that-is-not-running")
    # No running process matches, so this is the "nothing to do" success path -
    # importantly not the refusal path.
    assert safe.ok, safe.content
    assert "protected" not in safe.content


# ---------------------------------------------------------------------------
# self_test.check_*: a disabled check is skipped *visibly*
# ---------------------------------------------------------------------------
def test_a_disabled_self_test_check_is_reported_as_skipped(cfg) -> None:
    import io

    from tests import self_test as st

    cfg.set("self_test.check_tray", False)
    stream = io.StringIO()
    code = st.run_self_test(cfg, ["--quick"], stream=stream)
    report = stream.getvalue()

    assert code == 0, report
    assert "self_test.check_tray: false" in report, "the switched-off check is not visible"
    assert "ui" in report


def test_a_disabled_check_does_not_run_its_function(cfg, monkeypatch) -> None:
    """Skipping has to mean skipping: the check must not run and then be hidden."""
    import io

    from tests import self_test as st

    calls: list[str] = []
    original = st.check_tray if hasattr(st, "check_tray") else None
    cfg.set("self_test.check_tray", False)

    def spy(ctx):
        calls.append("ui")
        return original(ctx) if original else st._ok("ui", "spy")

    monkeypatch.setattr(st, "check_ui", spy, raising=False)
    # CHECKS holds the function object, so patch the tuple entry that matters.
    monkeypatch.setattr(
        st, "CHECKS",
        tuple((name, spy if name == "ui" else fn, quick) for name, fn, quick in st.CHECKS),
        raising=False,
    )
    stream = io.StringIO()
    code = st.run_self_test(cfg, ["--quick"], stream=stream)

    assert code == 0
    assert calls == [], "the disabled check ran anyway"
    assert "disabled in config.yaml" in stream.getvalue()


# ---------------------------------------------------------------------------
# voice_out.sentence_chunking: whole reply vs sentence by sentence
# ---------------------------------------------------------------------------
def _speak_reply(cfg, mock_ollama, monkeypatch, *, chunking: bool) -> list[str]:
    from core.voice_out import VoiceOut
    from tests.mock_ollama import text_chunks
    from tests.test_stage5_voice import FakeEngine, FakePlayer

    cfg.set("brain.host", mock_ollama.url)
    cfg.set("voice_out.sentence_chunking", chunking)
    mock_ollama.queue_raw(
        text_chunks("All systems nominal. Nothing to report.", size=4) + [{"done": True}]
    )

    app = main_module.Garvis(cfg, main_module.build_parser().parse_args([]))
    engine, player = FakeEngine(), FakePlayer()
    voice = VoiceOut(cfg, engine=engine, player=player)
    app.services["tts"] = voice
    app.voice_mode = False
    try:
        app.handle_line("status?")
        voice.wait(timeout=10)
    finally:
        voice.close()
    return list(engine.said)


def test_sentence_chunking_speaks_as_the_answer_arrives(cfg, activity, mock_ollama) -> None:
    said = _speak_reply(cfg, mock_ollama, None, chunking=True)
    assert len(said) >= 2, f"expected sentence-sized chunks, got {said}"


def test_sentence_chunking_off_speaks_the_reply_in_one_piece(cfg, activity, mock_ollama) -> None:
    said = _speak_reply(cfg, mock_ollama, None, chunking=False)
    assert len(said) == 1, f"expected one whole reply, got {said}"
    assert "All systems nominal" in said[0] and "Nothing to report" in said[0]


# ---------------------------------------------------------------------------
# state.resume_on_start: quiet at startup, still audited
# ---------------------------------------------------------------------------
class _FakeTask:
    description = "clear the downloads folder"

    def summarize(self) -> str:
        return "clear the downloads folder (2 steps recorded)"


class _FakeState:
    def crash_report(self) -> str:
        return "last run was interrupted while running: clear the downloads folder"

    def resume_task(self, reason: str) -> Any:
        self.resume_reason = reason
        return _FakeTask()


def _app_with_unfinished_work(cfg, state: _FakeState):
    app = main_module.Garvis(cfg, main_module.build_parser().parse_args([]))
    app.services["state"] = state
    app.pending_resume = _FakeTask()
    return app


def _log_text(cfg) -> str:
    text = ""
    for key in ("logging.activity_log", "logging.activity_jsonl"):
        path = cfg.resolve_path(cfg.get(key))
        if path.exists():
            text += path.read_text(encoding="utf-8", errors="replace")
    return text


def test_unfinished_work_is_announced_by_default(cfg, activity, capsys) -> None:
    app = _app_with_unfinished_work(cfg, _FakeState())
    app.report_interruption()
    out = capsys.readouterr().out
    assert "unfinished business" in out
    assert "clear the downloads folder" in out


def test_resume_on_start_false_keeps_startup_quiet_but_still_audits(cfg, activity, capsys) -> None:
    """Silence is a preference; the audit trail is not. A crash must be logged."""
    cfg.set("state.resume_on_start", False)
    app = _app_with_unfinished_work(cfg, _FakeState())

    app.report_interruption()

    assert "unfinished business" not in capsys.readouterr().out
    assert "recovered after an unclean exit" in _log_text(cfg), (
        "the crash was hidden from the audit trail as well as the screen"
    )


def test_an_explicit_resume_still_reports_when_the_knob_is_off(cfg, activity, capsys) -> None:
    """--resume is an explicit request: it must win over a quiet default."""
    cfg.set("state.resume_on_start", False)
    state = _FakeState()
    app = _app_with_unfinished_work(cfg, state)
    app.args.resume = True

    app.report_interruption()

    out = capsys.readouterr().out
    assert "unfinished business" in out
    assert "resuming" in out
    assert getattr(state, "resume_reason", ""), "the task was not resumed"
    assert app.pending_resume is None


@pytest.mark.parametrize("flag", ["state.resume_on_start"])
def test_the_knob_still_defaults_to_true_in_the_shipped_config(cfg, flag: str) -> None:
    assert cfg.get(flag) is True
