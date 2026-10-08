"""Three failures in a row stop tool use - and the user must be able to lift that.

`safety.max_consecutive_errors` is documented as "GARVIS stops and asks". The
first half worked; the second half did not. The counter zeroed itself only when a
call *succeeded*, but a tripped breaker refuses every call before it runs, so no
call could ever succeed: three transient failures (a flaky network, a model
inventing bad arguments) bricked every tool for the rest of the session,
including reads, while the message told the user it was "waiting" for something
they had no way to do. There was no reset method at all - `dir(gate)` had nothing
that looked like one.

The fix splits the two audiences:

* the **model** cannot clear it - a breaker a model can reset is not a breaker,
  and this one exists to stop a failing model from hammering away inside a turn;
* the **user** clears it by saying anything at all, which is exactly what "stops
  and asks" means, and the fact that it was cleared is shown and audited.

This file proves both halves, plus the two things that must *not* change: the
breaker still holds within a turn, and no tool can reach it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import main as main_module
from core.logger import get_logger
from core.permissions import ABORTED, ALLOWED, PermissionGate, ScriptedConfirmer
from tools.base import GREEN, Tool, ToolRegistry, ToolResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# a registry with one flaky tool and one healthy tool
# ---------------------------------------------------------------------------
@pytest.fixture()
def registry(effects: dict[str, Any]) -> ToolRegistry:
    reg = ToolRegistry()

    def flaky() -> ToolResult:
        effects["flaky_calls"] = effects.get("flaky_calls", 0) + 1
        raise RuntimeError("temporary failure")

    def healthy() -> ToolResult:
        effects["healthy_calls"] = effects.get("healthy_calls", 0) + 1
        return ToolResult.success("all good")

    reg.register(Tool(name="net.fetch", description="flaky network tool",
                      parameters={"type": "object", "properties": {}},
                      func=flaky, tier=GREEN, readonly=True, timeout_s=5))
    reg.register(Tool(name="clock.now", description="healthy read-only tool",
                      parameters={"type": "object", "properties": {}},
                      func=healthy, tier=GREEN, readonly=True))
    return reg


@pytest.fixture()
def effects() -> dict[str, Any]:
    return {}


@pytest.fixture()
def gate(cfg, activity, registry, effects) -> PermissionGate:
    cfg.set("safety.max_consecutive_errors", 3)
    # `log` is passed the way main.py passes it, so the developer log is wired.
    return PermissionGate(cfg=cfg, activity=activity, log=get_logger("permissions"),
                          services={"activity": activity, "registry": registry},
                          confirmer=ScriptedConfirmer(approve_all=True), registry=registry)


def _trip(gate: PermissionGate) -> None:
    for _ in range(3):
        gate.execute("net.fetch", {})
    assert gate.breaker_tripped(), "the breaker did not trip after three failures"


# ---------------------------------------------------------------------------
# the breaker itself (the half that always worked)
# ---------------------------------------------------------------------------
def test_the_breaker_trips_after_the_configured_number_of_failures(gate) -> None:
    _trip(gate)
    outcome = gate.execute("clock.now", {})
    assert outcome.decision == ABORTED and not outcome.ok


def test_a_healthy_call_before_the_limit_clears_the_count(gate) -> None:
    gate.execute("net.fetch", {})
    gate.execute("clock.now", {})
    gate.execute("net.fetch", {})
    assert not gate.breaker_tripped(), "one success between failures must reset the count"
    assert gate.execute("clock.now", {}).decision == ALLOWED


def test_it_stays_closed_inside_the_turn(gate, effects) -> None:
    """No self-healing: retrying must not sneak past the breaker."""
    _trip(gate)
    attempts_when_tripped = effects["flaky_calls"]  # includes the single retry per call

    for _ in range(5):
        assert gate.execute("clock.now", {}).decision == ABORTED
        assert gate.execute("net.fetch", {}).decision == ABORTED

    assert effects.get("healthy_calls") in (None, 0), "a tool ran while the breaker was tripped"
    assert effects["flaky_calls"] == attempts_when_tripped, (
        "the failing tool ran again after the breaker had tripped"
    )


def test_the_model_is_told_what_happened_and_how_it_ends(gate) -> None:
    """The old text said 'waiting for the user' - a promise nothing could keep."""
    _trip(gate)
    outcome = gate.execute("clock.now", {})
    text = outcome.content.lower()
    assert "too many consecutive failures" in text
    assert "anything they say next clears this" in text, (
        "the model is not told how the situation ends, so it cannot explain it either"
    )


# ---------------------------------------------------------------------------
# the user's way out
# ---------------------------------------------------------------------------
def test_reset_breaker_clears_a_tripped_breaker(gate) -> None:
    _trip(gate)

    assert gate.reset_breaker("test") is True

    assert not gate.breaker_tripped()
    assert gate.execute("clock.now", {}).ok, "tool use did not come back"


def test_reset_breaker_is_a_noop_when_nothing_is_wrong(gate) -> None:
    assert gate.reset_breaker("test") is False
    assert not gate.breaker_tripped()


def test_counting_starts_again_after_a_reset(gate, effects) -> None:
    _trip(gate)
    gate.reset_breaker("test")
    gate.execute("net.fetch", {})          # 1 failure
    gate.execute("clock.now", {})          # clears the count
    gate.execute("net.fetch", {})          # 1 failure again
    assert not gate.breaker_tripped(), "the counter was not reset, it was merely lowered"


def test_the_reset_is_written_to_the_audit_trail(gate, cfg, activity) -> None:
    _trip(gate)
    gate.reset_breaker("you sent a new message")

    trail = ""
    for key in ("logging.activity_log", "logging.activity_jsonl"):
        path = cfg.resolve_path(cfg.get(key))
        if path.exists():
            trail += path.read_text(encoding="utf-8", errors="replace")
    assert "breaker" in trail.lower(), "lifting a failure lockout left no trace"
    assert "you sent a new message" in trail


def test_the_reset_reason_reaches_the_developer_log(gate, cfg) -> None:
    _trip(gate)
    gate.reset_breaker("spelled out for the log")
    log = cfg.resolve_path(cfg.get("logging.file")).read_text(encoding="utf-8", errors="replace")
    assert "failure breaker cleared" in log


# ---------------------------------------------------------------------------
# through the app: a user's message is what lifts it
# ---------------------------------------------------------------------------
def _stub_reply() -> SimpleNamespace:
    return SimpleNamespace(reply="ok", error=None, interrupted=False, tool_calls=[],
                           iterations=1, duration_ms=1.0, model="stub", provider="stub")


def _app(cfg, registry: ToolRegistry):
    """A real Garvis whose gate uses this file's registry, so tripping works."""
    app = main_module.Garvis(cfg, main_module.build_parser().parse_args([]))
    app.brain.respond = lambda text, **kw: _stub_reply()  # never call a model in tests
    app.gate.registry = registry
    app.gate.services["registry"] = registry
    cfg.set("safety.max_consecutive_errors", 3)
    return app


def test_a_user_message_lifts_the_breaker(cfg, activity, registry, capsys) -> None:
    app = _app(cfg, registry)
    gate = app.gate
    assert gate is not None
    _trip(gate)

    app.handle_line("what happened?")

    assert not gate.breaker_tripped()
    assert gate.execute("clock.now", {}).ok, "tool use stayed broken after the user spoke"
    assert "back on" in capsys.readouterr().out, "the user was not told tool use came back"


def test_an_empty_line_does_not_lift_it(cfg, activity, registry) -> None:
    """Nothing was said, so nothing was asked."""
    app = _app(cfg, registry)
    _trip(app.gate)
    app.handle_line("   ")
    assert app.gate.breaker_tripped()


def test_an_untripped_breaker_says_nothing(cfg, activity, registry, capsys) -> None:
    """No noise on the normal path."""
    app = _app(cfg, registry)
    app.handle_line("hello")
    assert "back on" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# the model must not be able to lift it
# ---------------------------------------------------------------------------
def test_no_tool_can_reach_the_breaker() -> None:
    """A tripwire, the same shape as the fence one: only the user path resets it.

    If a tool (or a service the model can drive) ever calls reset_breaker, the
    breaker stops being a breaker - a model that keeps failing could simply clear
    its own limit and carry on.
    """
    offenders: list[str] = []
    for path in sorted(PROJECT_ROOT.glob("core/*.py")) + sorted(PROJECT_ROOT.glob("tools/*.py")):
        if path.name == "permissions.py":
            continue  # the definition itself
        if "reset_breaker" in path.read_text(encoding="utf-8"):
            offenders.append(path.name)
    assert not offenders, (
        f"{', '.join(offenders)} can reach reset_breaker; only a user action may call it "
        f"(main.py, at the start of a message the user sent)"
    )


def test_the_gate_offers_no_model_visible_reset(gate) -> None:
    """No tool name suggests a reset, and the gate exposes no tool-shaped hook."""
    names = gate.registry.names()
    assert not any("reset" in name or "breaker" in name for name in names), names
    assert not any("breaker" in schema.get("function", {}).get("name", "")
                   for schema in gate.registry.schemas()), (
        "a reset-like tool is visible to the model"
    )
