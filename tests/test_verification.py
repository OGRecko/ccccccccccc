"""Requirement: state-changing actions are verified, and a tool cannot bluff.

The interesting question is not "does the gate call a verify function" - it is
"can a tool claim it did something and get away with it". So these tests put
deliberately lying tools in the registry (they return success without doing
anything) and check what the model and the user end up seeing.

The boundary is tested too, on purpose. Verification here means re-checking the
end state after the call; a README does not exist for "I sent that email", so a
tool with nothing re-checkable is *reported as unchecked* rather than quietly
counted as verified. Pretending otherwise would be the dishonest option.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.permissions import (
    FLAGGED,
    UNVERIFIED_NOTE,
    VERIFY_FAILED_PREFIX,
    PermissionGate,
    ScriptedConfirmer,
)
from tools import build_registry
from tools.base import GREEN, RED, YELLOW, Tool, ToolResult, ToolRegistry


# ---------------------------------------------------------------------------
# a registry of liars: every one of them reports success and does nothing
# ---------------------------------------------------------------------------
@pytest.fixture()
def effects() -> dict[str, object]:
    return {}


@pytest.fixture()
def liars(cfg, effects: dict[str, object]) -> ToolRegistry:
    registry = ToolRegistry()

    def write_nothing(path: str = "") -> ToolResult:
        effects["claimed"] = path
        return ToolResult.success(f"Wrote {path}.", display="wrote it")

    def mkdir_nothing(path: str = "") -> ToolResult:
        return ToolResult.success(f"Created folder {path}.")

    def delete_nothing(path: str = "") -> ToolResult:
        return ToolResult.success(f"Deleted {path}.")

    def move_nothing(source: str = "", destination: str = "") -> ToolResult:
        return ToolResult.success(f"Moved {source} to {destination}.")

    def email_nothing(to: str = "", body: str = "") -> ToolResult:
        return ToolResult.success(f"Email sent to {to}.")

    props = {"type": "object", "properties": {}}
    registry.register(Tool(name="files.write", description="write a file", parameters=props,
                           func=write_nothing, tier=YELLOW, path_args=("path",)))
    registry.register(Tool(name="files.mkdir", description="make a folder", parameters=props,
                           func=mkdir_nothing, tier=YELLOW, path_args=("path",)))
    registry.register(Tool(name="files.delete", description="delete a path", parameters=props,
                           func=delete_nothing, tier=RED, path_args=("path",)))
    registry.register(Tool(name="files.move", description="move a path", parameters=props,
                           func=move_nothing, tier=YELLOW, path_args=("source", "destination")))
    registry.register(Tool(name="email.send", description="send an email", parameters=props,
                           func=email_nothing, tier=YELLOW, content_args=("body",)))
    return registry


@pytest.fixture()
def liar_gate(cfg, activity, liars, effects) -> PermissionGate:
    services = {"activity": activity, "registry": liars, "effects": effects}
    return PermissionGate(cfg=cfg, activity=activity, services=services,
                          confirmer=ScriptedConfirmer(approve_all=True), registry=liars)


@pytest.fixture()
def real_gate(cfg, activity) -> PermissionGate:
    registry = build_registry(cfg, None, {"activity": activity})
    services = {"activity": activity, "registry": registry}
    return PermissionGate(cfg=cfg, activity=activity, services=services,
                          confirmer=ScriptedConfirmer(approve_all=True), registry=registry)


@pytest.fixture()
def sandbox(cfg) -> Path:
    path = cfg.resolve_path(cfg.get("files.sandbox_dir"))
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# the happy path: real work, real evidence
# ---------------------------------------------------------------------------
def test_a_real_write_is_verified_with_evidence(real_gate, sandbox: Path) -> None:
    outcome = real_gate.execute("files.write", {"path": "notes.txt", "text": "hello"})
    assert outcome.ok, outcome.content
    assert outcome.verified, "a real write must come back verified"
    assert "notes.txt exists" in outcome.verified
    assert "bytes" in outcome.verified
    assert "[checked]" in outcome.content, "the evidence must be in what the model reads"
    assert (sandbox / "notes.txt").read_text() == "hello"


def test_a_real_mkdir_copy_move_delete_are_verified(real_gate, sandbox: Path) -> None:
    """The four other file tools must also produce evidence, not silence."""
    assert real_gate.execute("files.mkdir", {"path": "docs"}).verified
    (sandbox / "docs" / "a.txt").write_text("one")
    copied = real_gate.execute("files.copy", {"source": "docs/a.txt", "destination": "docs/copy.txt"})
    assert copied.ok and copied.verified, copied.verified
    moved = real_gate.execute("files.move", {"source": "docs/copy.txt", "destination": "docs/b.txt"})
    assert moved.verified and "b.txt exists" in moved.verified and "copy.txt is gone" in moved.verified
    deleted = real_gate.execute("files.delete", {"path": "docs/b.txt"})
    assert deleted.ok and deleted.verified and "b.txt is gone" in deleted.verified
    assert not (sandbox / "docs" / "b.txt").exists()


# ---------------------------------------------------------------------------
# the liar: claimed it, did not do it
# ---------------------------------------------------------------------------
def test_a_write_that_never_happened_is_reported_as_failed(liar_gate, sandbox: Path) -> None:
    """The whole point: success message, no file -> the user must not hear success."""
    outcome = liar_gate.execute("files.write", {"path": "ghost.txt", "text": "x"})
    assert not outcome.ok, "the gate believed a tool that wrote nothing"
    assert outcome.decision == FLAGGED
    assert outcome.verified.startswith(VERIFY_FAILED_PREFIX)
    assert not (sandbox / "ghost.txt").exists()
    # ...and the model is told in the very text it gets back, not in a side channel.
    assert "VERIFY FAILED" in outcome.for_model_content
    assert "the tool said it worked" in outcome.for_model_content


def test_the_failed_claim_beats_the_tools_own_success_story(liar_gate) -> None:
    outcome = liar_gate.execute("files.write", {"path": "ghost.txt", "text": "x"})
    assert "Wrote ghost.txt" not in outcome.content
    assert outcome.error.startswith(VERIFY_FAILED_PREFIX)


@pytest.mark.parametrize(
    "tool,args,expected",
    [
        ("files.mkdir", {"path": "ghostdir"}, "after the mkdir"),
        ("files.delete", {"path": "never-there.txt"}, "nothing was deleted"),
        ("files.move", {"source": "a.txt", "destination": "b.txt"}, "nothing to move"),
    ],
)
def test_every_lying_file_tool_is_caught(liar_gate, tool: str, args: dict, expected: str) -> None:
    outcome = liar_gate.execute(tool, args)
    assert not outcome.ok and outcome.verified.startswith(VERIFY_FAILED_PREFIX)
    assert expected in outcome.verified


def test_the_failure_is_in_the_log_as_a_failure(liar_gate, activity, cfg) -> None:
    """A lie that is only visible in memory is not visible enough."""
    liar_gate.execute("files.write", {"path": "ghost.txt", "text": "x"})
    text = ""
    for key in ("logging.activity_log", "logging.activity_jsonl"):
        path = cfg.resolve_path(cfg.get(key))
        if path.exists():
            text += path.read_text(encoding="utf-8", errors="replace")
    assert VERIFY_FAILED_PREFIX in text, "the log does not record that the claim failed"


# ---------------------------------------------------------------------------
# the honest boundary: things that cannot be re-checked are said to be unchecked
# ---------------------------------------------------------------------------
def test_a_non_file_tool_is_not_secretly_trusted(liar_gate) -> None:
    """We cannot see inside an email tool - so we must not imply we did."""
    outcome = liar_gate.execute("email.send", {"to": "a@b.c", "body": "hi"})
    assert outcome.ok, "nothing here can contradict the tool's own report"
    assert outcome.verified is None
    assert UNVERIFIED_NOTE in outcome.content
    assert UNVERIFIED_NOTE in outcome.for_model_content


def test_a_lying_file_tool_is_never_told_it_is_unverified(liar_gate) -> None:
    """The two notes are different, and the failure note must win."""
    outcome = liar_gate.execute("files.delete", {"path": "never-there.txt"})
    assert UNVERIFIED_NOTE not in outcome.content
    assert outcome.verified.startswith(VERIFY_FAILED_PREFIX)


def test_the_unverified_reminder_is_configurable(cfg, activity, liars) -> None:
    """A user who finds the reminder noisy can turn it off; the checks stay on."""
    cfg.set("permissions.note_unverified", False)
    services = {"activity": activity, "registry": liars}
    quiet = PermissionGate(cfg=cfg, activity=activity, services=services,
                           confirmer=ScriptedConfirmer(approve_all=True), registry=liars)
    outcome = quiet.execute("email.send", {"to": "a@b.c", "body": "hi"})
    assert UNVERIFIED_NOTE not in outcome.content
    # ...and the reminder being off does not switch off the actual checking.
    lied = quiet.execute("files.write", {"path": "ghost.txt", "text": "x"})
    assert not lied.ok and lied.verified.startswith(VERIFY_FAILED_PREFIX)


# ---------------------------------------------------------------------------
# the screenshot hook (what GUI/browser actions verify against)
# ---------------------------------------------------------------------------
def test_the_verifier_hook_is_used_for_non_file_tools(cfg, activity, liars) -> None:
    """The hook used to be dead code - nothing ever registered it."""
    calls: list[tuple[str, dict]] = []

    def verifier(tool_name: str, args: dict) -> str:
        calls.append((tool_name, dict(args)))
        return "screen looked at (screenshot 1234x768)"

    services = {"activity": activity, "registry": liars, "verifier": verifier}
    gate = PermissionGate(cfg=cfg, activity=activity, services=services,
                          confirmer=ScriptedConfirmer(approve_all=True), registry=liars)
    outcome = gate.execute("email.send", {"to": "a@b.c", "body": "hi"})
    assert calls and calls[0][0] == "email.send"
    assert outcome.verified == "screen looked at (screenshot 1234x768)"
    assert "screen looked at" in outcome.content
    assert UNVERIFIED_NOTE not in outcome.content


def test_a_broken_verifier_does_not_break_the_turn(cfg, activity, liars) -> None:
    def verifier(tool_name: str, args: dict) -> str:
        raise RuntimeError("no screen attached")

    services = {"activity": activity, "registry": liars, "verifier": verifier}
    gate = PermissionGate(cfg=cfg, activity=activity, services=services,
                          confirmer=ScriptedConfirmer(approve_all=True), registry=liars)
    outcome = gate.execute("email.send", {"to": "a@b.c", "body": "hi"})
    assert outcome.ok, "a verification problem must not turn a real action into a failure"
    assert "verification hook failed" in (outcome.verified or "")
    assert "no screen attached" in (outcome.verified or "")


def test_readonly_tools_are_not_verified_or_reminded(cfg, activity) -> None:
    """GREEN reads change nothing, so there is nothing to check and nothing to say."""
    registry = ToolRegistry()
    registry.register(Tool(name="files.read", description="read", parameters={"type": "object", "properties": {}},
                           func=lambda path="": ToolResult.success("contents"), tier=GREEN,
                           readonly=True, path_args=("path",)))
    services = {"activity": activity, "registry": registry}
    gate = PermissionGate(cfg=cfg, activity=activity, services=services, registry=registry)
    outcome = gate.execute("files.read", {"path": "whatever.txt"})
    assert outcome.ok and outcome.verified is None
    assert UNVERIFIED_NOTE not in outcome.content


# ---------------------------------------------------------------------------
# the wiring the app actually uses: browser actions, through the real hook
# ---------------------------------------------------------------------------
class _FakeBrowser:
    """Stands in for BrowserManager: only .screenshot() is needed here."""

    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.calls: list[tuple[str, str]] = []

    def screenshot(self, profile: str = "", name: str = ""):
        self.calls.append((profile, name))
        if not self.ok:
            return _DriverResult(False, "profile 'default' is not open")
        return _DriverResult(True, "screenshot saved to shots/after-click.png",
                             {"path": "shots/after-click.png"})


class _DriverResult:
    def __init__(self, ok: bool, message: str, data: dict | None = None) -> None:
        self.ok = ok
        self.message = message
        self.data = data or {}


def _browser_gate(cfg, activity, browser, name: str = "browser.click"):
    """A gate with one state-changing browser tool and the real verifier hook."""
    from core.browser import after_action_note

    registry = ToolRegistry()
    registry.register(Tool(name=name, description="click", parameters={"type": "object", "properties": {}},
                           func=lambda selector="": ToolResult.success(f"clicked {selector}"),
                           tier=YELLOW, path_args=()))
    services = {"activity": activity, "registry": registry, "browser": browser,
                "verifier": lambda tool, args: after_action_note(services, tool, args)}
    return PermissionGate(cfg=cfg, activity=activity, services=services,
                          confirmer=ScriptedConfirmer(approve_all=True), registry=registry)


def test_a_browser_action_is_verified_with_a_screenshot(cfg, activity) -> None:
    browser = _FakeBrowser()
    gate = _browser_gate(cfg, activity, browser)
    outcome = gate.execute("browser.click", {"selector": "#save"})
    assert outcome.ok
    assert browser.calls == [("", "after-click")], "the page was not captured afterwards"
    assert outcome.verified and "after-click.png" in outcome.verified
    assert "compare it with what was expected" in outcome.content
    assert UNVERIFIED_NOTE not in outcome.content


def test_a_browser_action_with_no_page_open_is_reported_unchecked(cfg, activity) -> None:
    """A failed screenshot is not evidence - say 'unchecked', do not imply success."""
    browser = _FakeBrowser(ok=False)
    gate = _browser_gate(cfg, activity, browser)
    outcome = gate.execute("browser.click", {"selector": "#save"})
    assert outcome.ok, "the click itself still succeeded"
    assert outcome.verified is None
    assert UNVERIFIED_NOTE in outcome.content


def test_the_browser_hook_ignores_non_browser_tools(cfg, activity) -> None:
    """Installing the browser hook must not make every tool 'verified'."""
    from core.browser import after_action_note

    assert after_action_note({"browser": _FakeBrowser()}, "shell.run", {}) is None
    assert after_action_note({}, "browser.click", {}) is None


def test_verification_looks_where_the_tool_actually_wrote(cfg, activity, tmp_path) -> None:
    """The other half of the same bug: checking the wrong folder would report a
    perfectly good write as a failed one. The gate must use the tool's own rule."""
    other = tmp_path / "shell-cwd"
    other.mkdir()
    cfg.set("shell.default_cwd", str(other))
    registry = build_registry(cfg, None, {"activity": activity})
    gate = PermissionGate(cfg=cfg, activity=activity,
                          services={"activity": activity, "registry": registry},
                          confirmer=ScriptedConfirmer(approve_all=True), registry=registry)

    outcome = gate.execute("files.write", {"path": "note.txt", "text": "hi"})

    sandbox = cfg.resolve_path(cfg.get("files.sandbox_dir"))
    assert outcome.ok, outcome.content
    assert outcome.verified and "note.txt exists" in outcome.verified
    assert (sandbox / "note.txt").exists()
    assert not (other / "note.txt").exists()
