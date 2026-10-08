"""The log must never contain a credential. Tested with canaries, not by reading code.

"Never read, log, or display passwords" is a promise about a file on disk, so
these tests write files containing canaries, drive the real tools through the
real gate, and then read the log back looking for the canary. A test that only
checks the redaction function in isolation would pass while `files.read` wrote
a whole file into the log.

Two halves:

* **must never appear** - credentials written in the shapes people actually use
  (`password = x`, `the token is x`, `my pin is 4321`), the key formats of the
  services people hold keys for, and secrets placed deep inside content that is
  read, searched or cat-ed.
* **must still work** - ordinary text survives, the model still receives the
  full content (only the log is bounded), and the log still answers "what did
  you do today?".

Run with:  pytest tests/test_log_privacy.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.logger import ActivityLogger, configure_activity_logger
from core.permissions import PermissionGate, ScriptedConfirmer
from tools import build_registry

CANARY = "CANARY-7f3a9d"
# Things a person would be genuinely upset to find in a log file.
SECRETS = {
    "plain phrasing": f"the password is {CANARY}",
    "equals phrasing": f"password = {CANARY}",
    "colon phrasing": f"api_key: {CANARY}",
    "token phrasing": f"my token is {CANARY}",
    "pin phrasing": f"my pin is 4321{CANARY[:2]}",
    "json-ish": '{"secret": "' + CANARY + '"}',
    "github token": "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8",
    "openai key": "sk-live-" + "9f3a9d7c1b2e4f6a8c0d2e4f6a8c0d2e",
    "aws key": "AKIAIOSFODNN7EXAMPLE",
    "private key": "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA" + CANARY + "\n-----END RSA PRIVATE KEY-----",
}


@pytest.fixture()
def log_path(cfg) -> Path:
    return cfg.resolve_path(cfg.get("logging.activity_log"))


@pytest.fixture()
def registry(cfg, activity) -> object:
    return build_registry(cfg, None, {"activity": activity})


@pytest.fixture()
def gate(cfg, activity, registry) -> PermissionGate:
    services = {"activity": activity, "registry": registry}
    return PermissionGate(cfg=cfg, activity=activity, services=services,
                          confirmer=ScriptedConfirmer(approve_all=True), registry=registry)


@pytest.fixture()
def sandbox(cfg) -> Path:
    path = cfg.resolve_path(cfg.get("files.sandbox_dir"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_logs(cfg) -> str:
    """Everything on disk that claims to be a log."""
    text = ""
    for key in ("logging.activity_log", "logging.activity_jsonl"):
        path = cfg.resolve_path(cfg.get(key))
        if path.exists():
            text += path.read_text(encoding="utf-8", errors="replace")
    return text


# ---------------------------------------------------------------------------
# the shapes people actually write
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,secret", sorted(SECRETS.items()))
def test_a_logged_message_never_keeps_the_secret(activity, cfg, name: str, secret: str) -> None:
    """The last line of defence: anything routed through the logger."""
    activity.event("user", f"here it is, {secret}")
    activity.event("assistant", secret)
    activity.tool_call("files.read", {"path": "x"}, result=secret, ok=True)
    log = read_logs(cfg)
    for part in secret.split():
        if len(part) > 6:
            assert part not in log, f"{name}: {part!r} reached the log"


def test_the_canary_never_reaches_the_log_through_a_file_read(gate, cfg, sandbox: Path) -> None:
    """The real path: a file with a credential, read by the tool the model calls."""
    (sandbox / "notes.txt").write_text(f"shopping\npassword = {CANARY}\n")
    outcome = gate.execute("files.read", {"path": "notes.txt"})
    assert outcome.ok
    assert CANARY not in read_logs(cfg), "files.read wrote a credential into the log"


def test_the_canary_never_reaches_the_log_through_a_search(gate, cfg, sandbox: Path) -> None:
    """Search results contain the matching line - which is where the secret is."""
    (sandbox / "notes.txt").write_text(f"line one\npassword = {CANARY}\nline three\n")
    outcome = gate.execute("files.search", {"query": "password", "path": "."})
    assert outcome.ok
    assert CANARY not in read_logs(cfg)


def test_the_canary_never_reaches_the_log_through_a_command(gate, cfg, sandbox: Path) -> None:
    """`cat` is allowlisted, and its output would otherwise be logged in full."""
    (sandbox / "notes.txt").write_text(f"password = {CANARY}\n")
    outcome = gate.execute("shell.run", {"command": "cat notes.txt"})
    assert outcome.ok, outcome.content
    assert CANARY not in read_logs(cfg)


def test_content_far_inside_a_file_cannot_reach_the_log(gate, cfg, sandbox: Path) -> None:
    """Bounding the log - not pattern matching - is what makes this true."""
    body = "\n".join(f"line {n}" for n in range(1, 60))
    body += f"\nthe password is {CANARY}\n" + "\n".join(f"line {n}" for n in range(61, 200))
    (sandbox / "big.txt").write_text(body)
    assert gate.execute("files.read", {"path": "big.txt"}).ok
    log = read_logs(cfg)
    assert CANARY not in log
    assert "line 100" not in log, "the log kept file content; it should keep one line"


def test_a_secret_in_command_output_is_not_kept(gate, cfg, sandbox: Path) -> None:
    """Output of an allowlisted command that prints a key on its third line."""
    (sandbox / "keys.txt").write_text(f"first\nsecond\nthe token is {CANARY}\n")
    assert gate.execute("shell.run", {"command": "cat keys.txt"}).ok
    assert CANARY not in read_logs(cfg)


# ---------------------------------------------------------------------------
# the log still has to be a log
# ---------------------------------------------------------------------------
def test_the_full_content_still_reaches_the_model_and_the_user(gate, sandbox: Path) -> None:
    """The log is bounded; the answer is not. Otherwise this would be censorship."""
    (sandbox / "notes.txt").write_text(f"line one\npassword = {CANARY}\nline three\n")
    outcome = gate.execute("files.read", {"path": "notes.txt"})
    assert "line three" in outcome.content, "the tool's own output must be complete"
    assert CANARY in outcome.content, "the user asked for this file; they get it"


def test_ordinary_text_is_left_alone(activity, cfg) -> None:
    """Over-redacting is a cost, so it has to be a small one - but only that."""
    activity.event("user", "read the notes file and summarise it")
    activity.event("assistant", "Done: 3 files moved into 2 folders.")
    log = read_logs(cfg)
    assert "read the notes file and summarise it" in log
    assert "3 files moved into 2 folders" in log
    assert "[REDACTED]" not in log


def test_the_ambiguous_phrases_are_redacted_on_purpose(activity, cfg) -> None:
    """Documented trade-off: "the token is invalid" reads like a credential.

    Redacting it loses a little log detail; not redacting it would mean guessing
    when "token is ..." is harmless. Safety wins, and the choice is recorded here
    so nobody "fixes" it later by accident.
    """
    activity.event("system", "the token is invalid")
    assert "token is invalid" not in read_logs(cfg)


def test_the_log_still_answers_what_happened(gate, cfg, sandbox: Path) -> None:
    """Bounded results must not cost the audit trail its usefulness."""
    (sandbox / "notes.txt").write_text("hello\n")
    gate.execute("files.write", {"path": "out.txt", "text": "hello"})
    gate.execute("files.read", {"path": "out.txt"})
    log = read_logs(cfg)
    assert "files.write" in log and "files.read" in log
    assert "out.txt" in log, "the arguments are what matter in the audit trail, and they stay"
    assert "more characters not written to this log" in log, "the log says what it did not keep"


def test_the_report_still_works(cfg, activity, gate, sandbox: Path) -> None:
    configure_activity_logger(cfg)
    (sandbox / "a.txt").write_text("x")
    gate.execute("files.read", {"path": "a.txt"})
    report = activity.today_report()
    assert "files.read" in report
    assert "Today" in report


def test_the_cap_is_configurable(cfg) -> None:
    cfg.set("logging.result_max_chars", 120)
    logger = configure_activity_logger(cfg)
    assert logger.result_max_chars == 120
    record = logger.event("tool", "files.read ok", tool="files.read",
                          result="z" * 500, ok=True)
    assert len(record["result"]) < 500
    assert "not written to this log" in record["result"]


def test_a_broken_config_pattern_cannot_disable_redaction(cfg) -> None:
    """A bad regex in config.yaml must not take the built-ins down with it."""
    cfg.set("logging.redact_patterns", ["this is not (a valid regex"])
    logger = ActivityLogger(log_dir=cfg.resolve_path(cfg.get("logging.file")).parent,
                            activity_log_name="activity_log.txt",
                            redact_patterns=cfg.get("logging.redact_patterns", []))
    assert "hunter2" not in logger.redactor("password = hunter2")
    assert logger.result_max_chars > 0


def test_removing_the_config_patterns_still_protects(cfg) -> None:
    """The built-ins are the promise; config patterns are the extras."""
    cfg.set("logging.redact_patterns", [])
    logger = ActivityLogger(log_dir=cfg.resolve_path(cfg.get("logging.file")).parent,
                            activity_log_name="activity_log.txt", redact_patterns=[])
    for text in ("password = hunter2", "the token is abc123", "my pin is 4321",
                 "ghp_" + "a" * 36, "sk-" + "b" * 32, "AKIAIOSFODNN7EXAMPLE"):
        assert logger.redactor(text) == "[REDACTED]" or "[REDACTED]" in logger.redactor(text), text
