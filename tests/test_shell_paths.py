"""The shell could walk around every file allowlist. Two halves: close what can be closed, say the rest.

Probed through the real gate, `cat <a file in no allowlist>` ran at YELLOW and
the contents came back, while `files.read` of the same path was blocked. The
command checker validated the *executable* of each segment, the blocked patterns,
substitution and redirect targets - never the arguments. `cat ~/.ssh/id_rsa` was
the worst version of it: one allowlisted program away from the private key the
README promises is never read.

The fix closes the class that *can* be closed from the outside - arguments that
name a protected file (keys, browser cookie stores, GARVIS's own config) - using
the same matcher the file tools use, so the promise "GARVIS's own files are
outside every allowlist" holds for the shell too.

What cannot be closed while an interpreter may run is stated instead of faked:
`python3 -c` can reach anything the user's account can. `test_the_open_boundary_is_
still_open_and_documented` below pins that reality on purpose - it is a tripwire
that fails if someone quietly claims otherwise, and it exists so the README's
honesty stays checkable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.permissions import BLOCKED, DENIED, GateOutcome, PermissionGate, ScriptedConfirmer
from tools import build_registry


@pytest.fixture()
def registry(cfg, activity):
    return build_registry(cfg, None, {"activity": activity})


@pytest.fixture()
def sandbox(cfg) -> Path:
    path = cfg.resolve_path(cfg.get("files.sandbox_dir"))
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture()
def gate(cfg, activity, registry) -> PermissionGate:
    """A gate whose user says yes to everything - the weakest realistic approval."""
    services = {"activity": activity, "registry": registry}
    return PermissionGate(cfg=cfg, activity=activity, services=services,
                          confirmer=ScriptedConfirmer(approve_all=True), registry=registry)


def run(gate: PermissionGate, command: str) -> GateOutcome:
    return gate.execute("shell.run", {"command": command})


def _refused(outcome: GateOutcome) -> bool:
    return outcome.decision in (BLOCKED, DENIED) and not outcome.ok


def _ran(outcome: GateOutcome) -> bool:
    """The gate let it through. NOT the same as ok: `grep` exits 1 on no match,
    and a tool that ran and failed is still a tool the gate allowed."""
    return outcome.decision not in (BLOCKED, DENIED)


# ---------------------------------------------------------------------------
# the half that can be closed: arguments naming a protected file
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "template",
    [
        "cat {target}",
        "head -n 5 {target}",
        "tail {target}",
        "wc -l {target}",
        "grep -rn API_KEY {target}",
        "python3 -c \"print(open('{target}').read())\"",   # the token is still visible
        "cp {target} /tmp/steal",
        "cat --file={target}",
    ],
)
def test_a_command_naming_a_secret_file_never_runs(gate, sandbox, tmp_path, template) -> None:
    key = tmp_path / "id_rsa"
    key.write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nNOT-FOR-YOU\n")
    outcome = run(gate, template.format(target=key))

    assert _refused(outcome), f"ran: {outcome.decision} / ok={outcome.ok}"
    assert "NOT-FOR-YOU" not in (outcome.content or ""), "the key's contents came back"
    assert "protected" in outcome.content


@pytest.mark.parametrize(
    "command",
    [
        "cat ~/.ssh/id_rsa",
        "cat ~/.aws/credentials",
        "head ~/.netrc",
        "cat ./config.yaml",
        "cat .env",
        "cat .env.local",
        "head src/config.yaml",
        "find . -name id_rsa",
        "find . -name '*.pem'",
        "ls ~/.ssh/id_ed25519",
        "cat ../../etc/passwd.pem",
    ],
)
def test_path_shaped_arguments_are_checked_like_the_file_tools_check_paths(gate, command) -> None:
    outcome = run(gate, command)
    assert _refused(outcome), f"ran: {command} ({outcome.decision})"


def test_a_var_assignment_pointing_at_a_key_is_refused(gate, tmp_path) -> None:
    key = tmp_path / "server.key"
    key.write_text("-----BEGIN RSA PRIVATE KEY-----")
    outcome = run(gate, f"KEYFILE={key} cat $KEYFILE")
    assert _refused(outcome)


# ---------------------------------------------------------------------------
# ordinary work keeps working (an over-blocking guard is its own bug)
# ---------------------------------------------------------------------------
# NOTE: the ids are deliberately neutral. pytest puts a parametrized id into the
# tmp_path, so an id containing "password" or "secret" would make the test's own
# folder match files.denied_globs and the command would be refused for the wrong
# reason - the same trap that bit tests/test_log_privacy.py.
@pytest.mark.parametrize(
    "command",
    [
        "grep -rn password src/",
        "grep -rn token README.md",
        "echo hello world",
        "ls -la .",
        "python3 -c 'print(1 + 1)'",
        "cat notes.txt",
    ],
    ids=["word-a", "word-b", "echo", "list", "calc", "cat-file"],
)
def test_ordinary_commands_are_untouched(gate, sandbox, command) -> None:
    (sandbox / "notes.txt").write_text("shopping list\n")
    (sandbox / "src").mkdir(exist_ok=True)
    (sandbox / "src" / "notes.txt").write_text("mentions the password and token words freely\n")
    (sandbox / "README.md").write_text("mentions password and token freely\n")
    outcome = run(gate, command)
    assert _ran(outcome), f"{command} was refused: {outcome.content}"
    assert "protected" not in (outcome.content or ""), f"{command} was refused as a secret read"


def test_a_bare_word_that_is_not_a_file_name_is_prose(gate, sandbox) -> None:
    """The distinction that makes this usable: the word, not the file."""
    (sandbox / "src").mkdir(exist_ok=True)
    (sandbox / "src" / "notes.txt").write_text("the password is checked here\n")
    assert _ran(run(gate, "grep -rn password src/")), "a prose word was treated as a file name"
    # ...while the same word as part of a *path* is a file name, and that is refused.
    assert _refused(run(gate, "grep -rn x ./password-notes.txt"))


# ---------------------------------------------------------------------------
# even a fully agreeing user cannot approve it
# ---------------------------------------------------------------------------
def test_no_approval_lifts_the_secret_block(gate, tmp_path) -> None:
    key = tmp_path / "private.pem"
    key.write_text("KEY MATERIAL")
    outcome = run(gate, f"cat {key}")
    assert outcome.decision == BLOCKED, "a secret read was confirmable"
    assert "KEY MATERIAL" not in (outcome.content or "")


# ---------------------------------------------------------------------------
# built-ins hold even if the user empties the deny list
# ---------------------------------------------------------------------------
def test_the_built_in_names_survive_an_empty_deny_list(cfg, activity, registry, tmp_path) -> None:
    cfg.set("files.denied_globs", [])
    gate = PermissionGate(cfg=cfg, activity=activity, services={"activity": activity, "registry": registry},
                          confirmer=ScriptedConfirmer(approve_all=True), registry=registry)

    for command in ("cat config.yaml", "cat .env", "cat id_rsa", "ls -la secret.pem"):
        assert _refused(run(gate, command)), f"emptying the deny list un-protected: {command}"


def test_config_extension_patterns_flow_into_the_shell(cfg, activity, registry, tmp_path) -> None:
    """Adding a pattern to config.yaml must protect the shell too, or they drift."""
    cfg.set("files.denied_globs", [*cfg.get("files.denied_globs", []), "**/*.vault"])
    gate = PermissionGate(cfg=cfg, activity=activity,
                          services={"activity": activity, "registry": registry},
                          confirmer=ScriptedConfirmer(approve_all=True), registry=registry)
    assert _refused(run(gate, "cat my.vault"))


# ---------------------------------------------------------------------------
# the honest boundary: what the shell still is not
# ---------------------------------------------------------------------------
def test_the_stated_shell_boundary_is_still_exactly_that(gate, tmp_path) -> None:
    """A tripwire, on purpose.

    An allowlisted interpreter can read and write anything the user's account can;
    no argument-scanning can change that. This test asserts the *current, stated*
    reality: a path that is not a protected pattern and not outside a pattern the
    shell enforces still runs. If someone later implements real containment, this
    test fails, and the failure is the reminder to update README.md's "What it
    cannot do" and shell.allowlist's comment at the same time.
    """
    # A neutral folder: pytest names this directory after the test, and a name
    # matching files.denied_globs would make the command refused for the wrong
    # reason (it would look like the fence working when it is the pattern list).
    folder = tmp_path / "ordinary"
    folder.mkdir()
    plain = folder / "notes.txt"
    plain.write_text("ordinary data outside every allowlist")
    outcome = run(gate, f"cat {plain}")

    assert _ran(outcome), "the shell became more contained than the documentation claims"
    assert "ordinary data" in outcome.content


def test_the_config_check_warns_about_the_shell_boundary(cfg) -> None:
    """The user is told, in --check and the self-test, not left to infer it."""
    warnings = cfg.validate()
    assert any("files.allowed_read/allowed_write do not bound the shell" in w for w in warnings), (
        "the shell's weaker boundary is no longer surfaced"
    )


def test_removing_the_file_capable_programs_clears_the_warning(cfg) -> None:
    """The warning is actionable, not permanent noise."""
    cfg.set("shell.allowlist", ["echo", "date", "whoami", "hostname"])
    assert not any("do not bound the shell" in w for w in cfg.validate())


def test_the_readme_states_the_boundary() -> None:
    """Documentation is part of the mechanism here (see the tripwire above)."""
    text = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    lowered = text.lower()
    assert "do not bound the shell" in lowered or "not a hard boundary" in lowered or "weaker" in lowered, (
        "README.md no longer says the shell allowlist is a weaker boundary"
    )
