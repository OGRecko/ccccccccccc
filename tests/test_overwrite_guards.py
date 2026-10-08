"""Destroying data needs the RED protocol - and a model cannot shop for a cheaper one.

`files.move` and `files.copy` onto an existing target have always been RED
("a move can silently destroy the target"), and the tier table in README.md lists
overwrite as RED. `files.write` with `mode='overwrite'` was not: it stayed YELLOW,
protected only by a `confirm=true` flag **the model chooses to pass**. Probed
end to end, "CHAPTER 1: four years of work" became "lol" after one yes.

That flag is exactly why the guard cannot live in the tool: whether an action is
destructive is not the model's decision, and a tool that can be argued into
overwriting is not protection. The promotion now sits in the gate's guard, the
same way copy's does, so the tier is decided by the *state of the world* (the
file exists) plus the model's stated intent (overwrite + acknowledge).

What must stay true after the change:

* creating a file, appending to a file and create_only are still one-question
  YELLOW actions - the guard must not make ordinary work painful;
* a model that does not acknowledge the overwrite gets a plain failure, not a
  prompt that was never going to be allowed to delete anything;
* the RED protocol still yields a working overwrite when the user really means it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.permissions import DENIED, RED, YELLOW, GateOutcome, PermissionGate, ScriptedConfirmer
from tools import build_registry
from tools.base import Tool, ToolRegistry


@pytest.fixture()
def registry(cfg, activity) -> ToolRegistry:
    return build_registry(cfg, None, {"activity": activity})


@pytest.fixture()
def services(cfg, activity, registry) -> dict:
    return {"activity": activity, "registry": registry}


@pytest.fixture()
def sandbox(cfg) -> Path:
    path = cfg.resolve_path(cfg.get("files.sandbox_dir"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _gate(cfg, activity, services, confirmer) -> PermissionGate:
    return PermissionGate(cfg=cfg, activity=activity, services=services,
                          confirmer=confirmer, registry=services["registry"])


def call(gate: PermissionGate, name: str, **args) -> GateOutcome:
    return gate.execute(name, args)


# ---------------------------------------------------------------------------
# the guard itself
# ---------------------------------------------------------------------------
def test_replacing_an_existing_file_is_promoted_to_red(cfg, activity, services, sandbox) -> None:
    confirmer = ScriptedConfirmer(approve_all=True)
    gate = _gate(cfg, activity, services, confirmer)
    (sandbox / "thesis.txt").write_text("four years of work")

    outcome = call(gate, "files.write", path="thesis.txt", text="lol",
                   mode="overwrite", confirm=True)

    assert outcome.tier == RED, "replacing a file was not treated as destructive"
    # ...and the user is told why, in the words they will read before approving.
    reasons = " ".join(confirmer.requests[0].reasons).lower()
    assert "replace" in reasons, confirmer.requests[0].reasons


def test_a_single_yes_cannot_destroy_a_file(cfg, activity, services, sandbox) -> None:
    """The anti-bypass test: the weakest approval still cannot lose data."""
    original = "CHAPTER 1: four years of work"
    (sandbox / "thesis.txt").write_text(original)
    gate = _gate(cfg, activity, services, ScriptedConfirmer(default_text="yes"))

    outcome = call(gate, "files.write", path="thesis.txt", text="lol",
                   mode="overwrite", confirm=True)

    assert outcome.decision == DENIED, f"a yes-only user approved a clobber ({outcome.decision})"
    assert (sandbox / "thesis.txt").read_text() == original, "the file was destroyed anyway"


def test_repeating_the_exact_action_plus_the_confirm_word_works(cfg, activity, services,
                                                               sandbox, monkeypatch) -> None:
    """The honest path must still work - a guard that blocks everything is a bug."""
    from core.permissions import ConsoleConfirmer

    (sandbox / "thesis.txt").write_text("draft")
    steps = iter(["files.write path=thesis.txt text=final mode=overwrite", "confirm"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(steps))
    gate = _gate(cfg, activity, services, ConsoleConfirmer(interactive=False))

    outcome = call(gate, "files.write", path="thesis.txt", text="final",
                   mode="overwrite", confirm=True)

    assert outcome.ok, outcome.content
    assert (sandbox / "thesis.txt").read_text() == "final"
    assert outcome.verified, "the overwrite was not verified afterwards"


# ---------------------------------------------------------------------------
# ordinary work must not get harder
# ---------------------------------------------------------------------------
def test_creating_a_new_file_is_still_yellow(cfg, activity, services, sandbox) -> None:
    gate = _gate(cfg, activity, services, ScriptedConfirmer(approve_all=True))
    outcome = call(gate, "files.write", path="fresh.txt", text="hello")
    assert outcome.tier == YELLOW and outcome.ok


def test_appending_is_still_yellow(cfg, activity, services, sandbox) -> None:
    (sandbox / "log.md").write_text("first\n")
    gate = _gate(cfg, activity, services, ScriptedConfirmer(approve_all=True))

    outcome = call(gate, "files.write", path="log.md", text="second", mode="append")

    assert outcome.tier == YELLOW and outcome.ok
    assert (sandbox / "log.md").read_text() == "first\nsecond\n"


def test_create_only_is_still_yellow_and_still_refuses(cfg, activity, services, sandbox) -> None:
    (sandbox / "keep.txt").write_text("original")
    gate = _gate(cfg, activity, services, ScriptedConfirmer(approve_all=True))

    blocked = call(gate, "files.write", path="keep.txt", text="other", mode="create_only")
    assert blocked.tier == YELLOW
    assert not blocked.ok and "create_only" in blocked.content
    assert (sandbox / "keep.txt").read_text() == "original"

    fresh = call(gate, "files.write", path="new.txt", text="hello", mode="create_only")
    assert fresh.ok


def test_an_unacknowledged_overwrite_fails_without_asking(cfg, activity, services, sandbox) -> None:
    """confirm=false: nothing is at risk, so nobody should be interrupted."""
    (sandbox / "keep.txt").write_text("original")
    confirmer = ScriptedConfirmer(approve_all=True)
    gate = _gate(cfg, activity, services, confirmer)

    outcome = call(gate, "files.write", path="keep.txt", text="clobber", mode="overwrite")

    assert not outcome.ok
    assert outcome.tier == YELLOW, "a call that cannot destroy anything is not RED"
    assert "already exists" in outcome.content and "append" in outcome.content
    assert (sandbox / "keep.txt").read_text() == "original"
    assert confirmer.requests, "the ordinary confirmation still happened (it is a YELLOW write)"
    assert all(request.tier == YELLOW for request in confirmer.requests), (
        "the user was asked for a RED approval for a write that was going to be refused"
    )


# ---------------------------------------------------------------------------
# the three tools that can destroy a file must agree
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,args",
    [
        ("files.write", {"path": "target.txt", "text": "new", "mode": "overwrite", "confirm": True}),
        ("files.move", {"source": "source.txt", "destination": "target.txt"}),
        ("files.copy", {"source": "source.txt", "destination": "target.txt", "overwrite": True}),
    ],
)
def test_every_clobbering_tool_is_red(cfg, activity, services, sandbox, name, args) -> None:
    (sandbox / "target.txt").write_text("valuable")
    (sandbox / "source.txt").write_text("replacement")
    gate = _gate(cfg, activity, services, ScriptedConfirmer(default_text="yes"))

    outcome = call(gate, name, **args)

    assert outcome.tier == RED, f"{name} can destroy a file at {outcome.tier}"
    assert outcome.decision == DENIED
    assert (sandbox / "target.txt").read_text() == "valuable", f"{name} destroyed the target"


def test_replacing_a_directory_is_not_promoted_by_the_guard(cfg, activity, services, sandbox) -> None:
    """The guard checks for a file, so a folder target keeps its own honest error.

    (The tool refuses to write over a folder; the guard must not turn that into a
    RED dance first.)
    """
    (sandbox / "docs").mkdir()
    gate = _gate(cfg, activity, services, ScriptedConfirmer(approve_all=True))

    outcome = call(gate, "files.write", path="docs", text="x", mode="overwrite", confirm=True)

    assert not outcome.ok
    assert outcome.tier == YELLOW
    assert "folder" in outcome.content


def test_the_guard_does_not_fire_for_paths_outside_the_allowlist(cfg, activity, services,
                                                                tmp_path) -> None:
    """Blocked-in-the-end must stay blocked: the allowlist decides, not the guard."""
    outside = tmp_path / "outside.txt"
    outside.write_text("not yours")
    gate = _gate(cfg, activity, services, ScriptedConfirmer(approve_all=True))

    outcome = call(gate, "files.write", path=str(outside), text="x",
                   mode="overwrite", confirm=True)

    assert not outcome.ok
    assert outcome.decision in ("denied", "blocked")
    assert outside.read_text() == "not yours"


# ---------------------------------------------------------------------------
# the same rule for a tool the tests do not know about
# ---------------------------------------------------------------------------
def test_a_future_tool_getting_this_wrong_is_visible(cfg, activity) -> None:
    """A tripwire of the shape used elsewhere: the guard lives in the tool config.

    If a new tool declares itself capable of destroying something, it must either
    carry a guard or declare the tier honestly - this checks the *mechanism* is
    available and used, not that every future tool is perfect.
    """
    registry = ToolRegistry()
    registry.register(Tool(
        name="files.write_unsafe", description="pretends to be a file tool",
        parameters={"type": "object", "properties": {}}, func=lambda **_: None,
        tier=YELLOW, path_args=("path",), path_base="sandbox",
    ))
    gate = PermissionGate(cfg=cfg, activity=activity,
                          services={"activity": activity, "registry": registry},
                          confirmer=ScriptedConfirmer(approve_all=True), registry=registry)

    # No guard: the gate can only go by the declared tier. That is the honest
    # limit of what a gate can do, and it is why the tools declare their guards.
    outcome = call(gate, "files.write_unsafe", path="anything.txt")
    assert outcome.tier == YELLOW, (
        "a tool with no guard is classified by its declared tier; guards are part of the "
        "tool's contract (see tools/files.py)"
    )
