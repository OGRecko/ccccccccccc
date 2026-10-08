"""Memory: profile + project log (stage 2).

Two plain markdown files you own and can edit by hand:

``memory/profile.md``
    Stable facts about you: name, machine specs, preferences, do-not-touch list.
    Loaded at startup in full (capped by ``config.memory.max_chars_loaded``).

``memory/project_log.md``
    Rolling working memory: what you were doing, decisions, TODOs. Only the
    *tail* is loaded (``config.memory.project_log_tail_lines``), so it can grow
    for years without eating the context window.

Both are injected into the system prompt inside an ``<untrusted_data>`` fence:
they are your notes, but GARVIS must not treat text found in a stored file as a
command. That matters because a project log can contain a pasted web snippet.
"""

from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import safety

if TYPE_CHECKING:  # avoid a runtime import cycle; we only need the type
    from .config import Config

PROFILE_TEMPLATE = """# GARVIS - User Profile
<!-- Loaded at startup. Keep it short and factual. -->

## Basics
- Name / how to address me:
- Timezone:
- Languages: English

## Machine
- OS:
- CPU:
- GPU / VRAM:
- RAM:
- Python:

## Preferences
- Personality default: standard
- Reply length: short unless I ask for detail
"""

PROJECT_LOG_TEMPLATE = """# GARVIS - Project Log
<!-- Append-only working memory. Newest entries are appended at the bottom. -->

## Ongoing
- (nothing yet)

## Log
"""


def _now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M")


class Memory:
    """Reads and writes the two memory files."""

    def __init__(
        self,
        memory_dir: Path,
        profile_file: Path | None = None,
        project_log_file: Path | None = None,
        load_profile: bool = True,
        load_project_log: bool = True,
        project_log_tail_lines: int = 80,
        max_chars_loaded: int = 8000,
    ) -> None:
        self.dir = Path(memory_dir)
        self.profile_path = Path(profile_file) if profile_file else self.dir / "profile.md"
        self.project_log_path = Path(project_log_file) if project_log_file else self.dir / "project_log.md"
        # NOTE: method names load_profile()/load_project_log() are taken, so the
        # enabling flags live under private names to avoid the collision.
        self._load_profile_enabled = bool(load_profile)
        self._load_project_log_enabled = bool(load_project_log)
        self.project_log_tail_lines = max(1, int(project_log_tail_lines))
        self.max_chars_loaded = max(500, int(max_chars_loaded))
        self._profile_cache: str | None = None
        self._log_cache: str | None = None

    # -- construction ------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: "Config") -> "Memory":
        return cls(
            memory_dir=cfg.path("memory.dir", "memory"),
            profile_file=cfg.path("memory.profile_file", "memory/profile.md"),
            project_log_file=cfg.path("memory.project_log_file", "memory/project_log.md"),
            load_profile=bool(cfg.get("memory.load_profile", True)),
            load_project_log=bool(cfg.get("memory.load_project_log", True)),
            project_log_tail_lines=int(cfg.get("memory.project_log_tail_lines", 80)),
            max_chars_loaded=int(cfg.get("memory.max_chars_loaded", 8000)),
        )

    # -- files -------------------------------------------------------------
    def ensure_files(self) -> dict[str, bool]:
        """Create the memory files from templates if missing. Returns what was made."""
        created: dict[str, bool] = {}
        self.dir.mkdir(parents=True, exist_ok=True)
        for path, template in (
            (self.profile_path, PROFILE_TEMPLATE),
            (self.project_log_path, PROJECT_LOG_TEMPLATE),
        ):
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(template, encoding="utf-8")
                created[str(path)] = True
        return created

    # -- reading -----------------------------------------------------------
    def load_profile(self, refresh: bool = False) -> str:
        if self._profile_cache is None or refresh:
            self._profile_cache = _read_text(self.profile_path, self.max_chars_loaded)
        return self._profile_cache

    def load_project_log(self, refresh: bool = False) -> str:
        if self._log_cache is None or refresh:
            text = _read_text(self.project_log_path, self.max_chars_loaded * 4)
            self._log_cache = _tail_lines(text, self.project_log_tail_lines, self.max_chars_loaded)
        return self._log_cache

    def load_all(self, refresh: bool = True) -> dict[str, str]:
        return {
            "profile": self.load_profile(refresh) if self._load_profile_enabled else "",
            "project_log": self.load_project_log(refresh) if self.load_project_log else "",
        }

    def memory_block(self) -> str:
        """The fenced block that goes into the system prompt.

        Returns an empty string when memory is switched off or empty, so callers
        can concatenate blindly.
        """
        parts: list[str] = []
        profile = self.load_profile() if self._load_profile_enabled else ""
        log = self.load_project_log() if self._load_project_log_enabled else ""
        if profile.strip():
            parts.append("## Long-term profile (memory/profile.md)\n" + profile.strip())
        if log.strip():
            parts.append("## Recent project log (memory/project_log.md)\n" + log.strip())
        if not parts:
            return ""
        body = "\n\n".join(parts)
        return safety.wrap_untrusted(
            body,
            source="memory files",
            kind="memory",
            max_chars=self.max_chars_loaded * 2,
            note="user-authored background notes; may be stale or contain pasted text from elsewhere",
        )

    # -- writing -----------------------------------------------------------
    def append_log(self, entry: str, section: str = "Log") -> Path:
        """Append one bullet at the end of a section. Creates it if missing.

        Newest entries end up at the bottom of their section, which is what you
        want for a log you scroll by hand.
        """
        self.ensure_files()
        bullet = f"- {_now_stamp()} - {entry.strip()}"
        text = self.project_log_path.read_text(encoding="utf-8") if self.project_log_path.exists() else ""
        header = f"## {section}"
        lines = text.splitlines()

        if header in [line.strip() for line in lines]:
            out: list[str] = []
            inside = False
            for line in lines:
                if line.strip() == header:
                    inside = True
                    out.append(line)
                    continue
                if inside and line.lstrip().startswith("## "):
                    # End of section: drop the bullet in just before the next header.
                    inside = False
                    out.append(bullet)
                    out.append("")
                out.append(line)
            if inside:
                out.append(bullet)
            new_text = "\n".join(out).rstrip() + "\n"
        else:
            new_text = text.rstrip() + f"\n\n{header}\n{bullet}\n"

        self.project_log_path.write_text(new_text, encoding="utf-8")
        self._log_cache = None
        return self.project_log_path

    def set_profile_field(self, field: str, value: str) -> bool:
        """Update ``- Field: old`` to ``- Field: new`` in profile.md.

        Returns False when the field does not exist (we do not invent structure
        in a file the user maintains by hand).
        """
        self.ensure_files()
        text = self.profile_path.read_text(encoding="utf-8")
        pattern = re.compile(rf"^(\s*[-*]\s*{re.escape(field)}\s*:\s*).*$", re.IGNORECASE | re.MULTILINE)
        if not pattern.search(text):
            return False
        new_text = pattern.sub(lambda m: f"{m.group(1)}{value}", text, count=1)
        self.profile_path.write_text(new_text, encoding="utf-8")
        self._profile_cache = None
        return True

    def note(self, text: str) -> str:
        """Convenience: 'remember this' -> append to the project log."""
        self.append_log(text)
        return f"Noted in the project log: {text}"

    # -- maintenance -------------------------------------------------------
    def tail_lines(self, count: int = 30) -> str:
        return _tail_lines(_read_text(self.project_log_path, 10**7), count, 10**7)

    def stats(self) -> dict[str, object]:
        def _size(path: Path) -> int:
            return path.stat().st_size if path.exists() else 0

        return {
            "profile_path": str(self.profile_path),
            "project_log_path": str(self.project_log_path),
            "profile_bytes": _size(self.profile_path),
            "project_log_bytes": _size(self.project_log_path),
            "loaded_chars": len(self.memory_block()),
        }


def _read_text(path: Path, limit: int) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(text) > limit:
        # Keep the head: the top of profile.md is the important part.
        text = text[:limit] + "\n...[truncated]..."
    return text


def _tail_lines(text: str, lines: int, max_chars: int) -> str:
    if not text:
        return ""
    tail = "\n".join(text.splitlines()[-lines:])
    if len(tail) > max_chars:
        tail = "...[truncated]...\n" + tail[-max_chars:]
    return tail
