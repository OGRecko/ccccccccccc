"""config.yaml is a contract: no dead knobs, nothing hidden, nothing unexplained.

"Everything configurable in config.yaml, with clear comments" is only true if
three things hold, and each one has broken at least once in this project:

1. **No dead knobs.** A key that nothing reads is worse than no key at all: the
   user turns it and nothing happens. This test found 22 of them (unimplemented
   hotkeys advertised in the README, a `data_dir` root that resolved nothing, a
   `log_every_decision` switch sitting on top of an always-on audit trail...).
   Every one was either wired up or deleted; this keeps it that way.
2. **No hidden knobs.** Every dotted key the code reads must appear in
   config.yaml, or the setting is discoverable only by reading the source.
3. **No unexplained knobs.** Every key carries a comment on its line or directly
   above it, apart from a reviewed list of names that say everything themselves
   (`enabled`, `engine`, a path...).

These are lints over the source, not proofs of behaviour - a comment can say the
wrong thing. The checks that a knob *changes behaviour* live with each feature
(see tests/test_cloud_fallback.py, tests/test_verification.py and friends).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
SOURCE_GLOBS = ("core/*.py", "tools/*.py", "main.py", "tests/self_test.py")

#: Leaf names allowed to go without a comment: the name is the documentation.
#: Add here only when a reader cannot reasonably wonder what the key does.
SELF_EVIDENT = {
    "enabled", "engine", "stt", "listen", "wake", "yellow_confirm", "red_confirm",
    "rollback_phrases", "also_accept", "base_url", "kokoro_voices", "profiles_dir",
    "screenshots_dir", "frames_dir", "profile_file", "project_log_file",
    "redaction_replacement", "console_style", "overlay_opacity", "startup_greeting",
    "screenshot_full_page",
}


def _sources() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for pattern in SOURCE_GLOBS
        for path in sorted(PROJECT_ROOT.glob(pattern))
    )


def _config_text() -> str:
    return CONFIG_PATH.read_text(encoding="utf-8")


def _leaf_keys(data: dict, prefix: str = "") -> list[str]:
    """Every scalar key, as a dotted path ("brain.model", "apps.allowlist")."""
    out: list[str] = []
    for key, value in (data or {}).items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.extend(_leaf_keys(value, path + "."))
        else:
            out.append(path)
    return out


# ---------------------------------------------------------------------------
# 1. no dead knobs
# ---------------------------------------------------------------------------
def test_every_key_in_config_yaml_is_read_somewhere() -> None:
    """The failure message tells you what to do: wire it up or delete it."""
    data = yaml.safe_load(_config_text())
    source = _sources()
    orphaned: list[str] = []
    for key in _leaf_keys(data):
        leaf = key.split(".")[-1]
        if key in source or f'"{leaf}"' in source or f"'{leaf}'" in source:
            continue
        orphaned.append(key)
    assert not orphaned, (
        "config.yaml has keys nothing reads, so turning them does nothing: "
        f"{', '.join(sorted(orphaned))}. Either wire them up or delete them."
    )


# ---------------------------------------------------------------------------
# 2. no hidden knobs
# ---------------------------------------------------------------------------
def test_every_key_the_code_reads_is_in_config_yaml() -> None:
    """A knob only the source knows about is not 'configurable in config.yaml'.

    Only dotted names count: bare names are reads on a section dict
    (`self.red_cfg.get("timeout_s")`), which are covered by the section's own
    keys.
    """
    data = yaml.safe_load(_config_text())
    present = set(_leaf_keys(data))
    found = set(
        re.findall(r'cfg\.get\(\s*["\']([a-zA-Z0-9_]+\.[a-zA-Z0-9_.]+)["\']', _sources())
    )
    hidden = sorted(key for key in found if key not in present)
    assert not hidden, (
        "these keys are read by the code but cannot be found in config.yaml, so "
        f"nobody can discover or change them: {', '.join(hidden)}. Add them to "
        "config.yaml (with a comment) or drop the read."
    )


# ---------------------------------------------------------------------------
# 3. no unexplained knobs
# ---------------------------------------------------------------------------
def test_every_key_is_commented_or_self_evident() -> None:
    lines = _config_text().splitlines()
    key_line = re.compile(r"^(\s+)([a-zA-Z_][a-zA-Z0-9_]*):\s*(.*)$")
    unexplained: list[str] = []
    previous_was_comment = False
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            previous_was_comment = True
            continue
        match = key_line.match(line)
        if match and "#" not in match.group(3):
            name = match.group(2)
            if not previous_was_comment and name not in SELF_EVIDENT:
                unexplained.append(f"line {number}: {stripped}")
        if stripped:
            previous_was_comment = False
    assert not unexplained, (
        "these settings have no comment saying what they do - add one, or add the "
        "name to SELF_EVIDENT in this file if the name really says everything:\n  "
        + "\n  ".join(unexplained)
    )


# ---------------------------------------------------------------------------
# the file itself
# ---------------------------------------------------------------------------
def test_the_config_parses_and_keeps_its_core_safety_defaults() -> None:
    """A malformed config would make every other config test meaningless."""
    data = yaml.safe_load(_config_text())
    assert isinstance(data, dict)
    assert data["app"]["name"] == "GARVIS"
    assert data["brain"]["cloud_fallback"]["enabled"] is False
    assert data["permissions"]["default_tier"] == "red"


def test_the_summary_matches_the_shipped_config(cfg) -> None:
    """`--check`'s one-line summary must not drift from the file it describes."""
    from core.config import describe

    text = describe(cfg)
    assert "cloud fallback   : disabled (local only)" in text
    assert "provider" in text


def test_the_shipped_config_protects_its_own_secrets() -> None:
    """The deny list is the second line of defence after the allowlists.

    It must cover the shapes people actually keep secrets in - and GARVIS's own
    files, so no tool can be talked into reading (or rewriting) the config, the
    permission gate or the kill switch.
    """
    data = yaml.safe_load(_config_text())
    globs = " ".join(str(g) for g in data["files"]["denied_globs"])
    for needed in (".ssh", ".aws", "*.key", "*.pem", ".env", "id_rsa", "Login Data",
                   "Cookies", "kdbx", "netrc", "password", "secret",
                   "config.yaml", "permissions.py", "killswitch.py", "system_prompt.md"):
        assert needed in globs, f"files.denied_globs no longer protects {needed}"


def test_the_shipped_config_still_allows_the_users_own_files() -> None:
    """The other direction: a deny list so strict it denies everything is useless.

    allowed_read/write are meant to be edited by the user to point at their own
    folders (Documents, a project directory...). Home-relative entries are the
    documented way to do that, so they must keep working - what must NOT happen
    is a shipped allowlist of "/" or "~".
    """
    data = yaml.safe_load(_config_text())
    entries = list(data["files"]["allowed_read"]) + list(data["files"]["allowed_write"])
    assert entries, "an empty allowlist means every read and write is refused"
    assert not any(str(e).strip() in ("/", "~", "C:\\", "C:/") for e in entries), (
        "the shipped allowlist must not be the whole disk"
    )
    assert any(e == "sandbox" for e in data["files"]["allowed_write"]), (
        "the sandbox must be writable out of the box"
    )


# ---------------------------------------------------------------------------
# warnings: computed, kept, and surfaced (they were silently discarded)
# ---------------------------------------------------------------------------
def test_loading_a_config_keeps_its_warnings() -> None:
    """`validate()` returns a list; the constructor used to throw it away.

    Every config warning in the project was dead because of that one missing
    assignment: main.py printed nothing at startup and the self-test's config
    check extended an empty list. The README's promise that `--check` warns about
    an enabled cloud fallback (and a bad personality, and an empty allowlist)
    simply did not happen.
    """
    from core.config import Config

    config = Config.load(CONFIG_PATH)
    assert isinstance(config.warnings, list)
    assert config.warnings, (
        "the shipped config has at least one standing caveat (the shell allowlist); an empty "
        "list here means warnings are being discarded again"
    )
    assert config.warnings == config.validate(), "the kept list drifted from a fresh check"


def test_a_deliberately_broken_config_warns_about_each_problem(cfg) -> None:
    cfg.set("brain.personality", "nonsense")
    cfg.set("voice_out.engine", "nonsense")
    cfg.set("files.allowed_read", [])
    text = " | ".join(cfg.warnings)
    assert "personality" in text
    assert "voice_out.engine" in text
    assert "allowed_read is empty" in text


def test_a_runtime_change_refreshes_the_warnings(cfg, monkeypatch) -> None:
    """`set()` is used for --model and personality swaps; the list must follow."""
    monkeypatch.delenv("GARVIS_CLOUD_API_KEY", raising=False)
    assert not any("Cloud fallback is enabled" in w for w in cfg.warnings)

    cfg.set("brain.cloud_fallback.enabled", True)

    assert any("Cloud fallback is enabled" in w for w in cfg.warnings), (
        "enabling cloud fallback at runtime did not re-run the checks"
    )


def test_the_self_test_reports_the_warnings_it_is_given(cfg) -> None:
    """The consumer side: a warning must reach the report, not just the list."""

    from tests import self_test as st

    cfg.set("brain.personality", "nonsense")
    result = st.check_config(st.Ctx(cfg, quick=True))
    assert result.status == st.WARN
    assert "personality" in result.detail
