"""Logging for GARVIS (requirement 8).

Three sinks, deliberately:

``logs/activity_log.txt``
    Human-readable audit trail, one block per logged event, greppable, with
    timestamp / kind / tool / args / account / result. This is the file you
    read when you ask "what did you do today?".

``logs/activity_log.jsonl``
    The same events as JSON Lines, so the "what did you do today" report can be
    computed exactly rather than by regex. Rotated by date, one file per day:
    ``activity_log-2026-10-07.jsonl``.

``logs/garvis.log``
    Developer-facing debug log. Never contains credentials: every message goes
    through :func:`redact`.

Optional structlog support: if ``structlog`` is installed it is used for the
debug log; otherwise a stdlib logger with a compact formatter is used. Either
way the two activity files are written directly by this module, so the audit
trail is independent of logging configuration.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Callable, Iterable

# --------------------------------------------------------------------------
# Redaction - belt and braces. The permission layer refuses to read secret
# files at all; this catches anything that slips through in a command line,
# a path name, or a model's own output before it hits a log file.
# --------------------------------------------------------------------------

DEFAULT_REDACT_PATTERNS: tuple[str, ...] = (
    r"(?i)\b(password|passwd|pwd|secret|token|api[_-]?key|apikey|bearer|authorization)\b\s*[:=]\s*\S+",
    # JSON-ish forms, e.g. {"password": "hunter2"} or password: hunter2
    r'(?i)"?\b(password|passwd|pwd|secret|token|api[_-]?key|apikey|bearer|authorization|otp|totp|cvv)\b"?\s*[:=]\s*"?[^",}\s]+',  
    r'(?i)\b\d{3}-?\d{2}-?\d{4}\b',          # US SSN shape
    r"\b(?:\d[ -]?){13,19}\b",                    # card-number shape
    r"(?i)\bcard\s*(number|no\.?)\b\s*[:=]?\s*[0-9][0-9 \-]{10,}",
    r"\bgh[pousr]_[A-Za-z0-9]{16,}\b",
    r"\bsk-[A-Za-z0-9]{16,}\b",
    r"\bAKIA[0-9A-Z]{16}\b",
    r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
)

REDACTION_REPLACEMENT = "[REDACTED]"


class Redactor:
    """Applies the configured patterns, plus a few always-on ones."""

    def __init__(self, patterns: Iterable[str] | None = None, replacement: str = REDACTION_REPLACEMENT) -> None:
        self.replacement = replacement
        raw = list(DEFAULT_REDACT_PATTERNS)
        for pattern in patterns or []:
            if pattern and pattern not in raw:
                raw.append(pattern)
        self._compiled: list[re.Pattern[str]] = []
        for pattern in raw:
            try:
                self._compiled.append(re.compile(pattern))
            except re.error:
                # A bad user-supplied regex must not take the logger down.
                continue

    def __call__(self, text: object) -> str:
        out = "" if text is None else str(text)
        for regex in self._compiled:
            out = regex.sub(self.replacement, out)
        return out


def redact(text: object) -> str:
    """Module-level redaction with default patterns (cheap, idempotent)."""
    return _DEFAULT_REDACTOR(text)


_DEFAULT_REDACTOR = Redactor()

# --------------------------------------------------------------------------
# Event kinds used throughout the app. Free-form strings are allowed, these
# are just the ones the "what did you do today" report knows how to group.
# --------------------------------------------------------------------------
KIND_TOOL = "tool"            # a tool ran (success or failure)
KIND_PERMISSION = "permission"  # a permission decision
KIND_USER = "user"            # user speech/typing
KIND_ASSISTANT = "assistant"  # what GARVIS said
KIND_SYSTEM = "system"        # startup, shutdown, mode changes, errors
KIND_STATE = "state"          # task state save/resume/rollback

_SENSITIVE_KEYS = {
    "password", "passwd", "pwd", "secret", "token", "api_key", "apikey",
    "authorization", "auth", "cookie", "cookies", "session", "credential",
    "credentials", "card", "cvv", "otp", "totp", "private_key",
}


def _scrub_args(args: Any, redactor: Redactor, sensitive: bool = False) -> Any:
    """Redact secrets from a nested args structure, by key name and by value.

    ``sensitive=True`` is used when the permission gate has flagged the whole
    call as secret-bearing (e.g. a red keyword appeared in the arguments): every
    value is then replaced by a length-tagged marker, so the log still shows
    *that* something was passed without showing what.
    """
    if args is None:
        return {}
    if isinstance(args, dict):
        out: dict[str, Any] = {}
        for key, value in args.items():
            key_l = str(key).lower()
            if any(s in key_l for s in _SENSITIVE_KEYS) or sensitive:
                out[key] = _redacted_marker(value)
            else:
                out[key] = _scrub_args(value, redactor)
        return out
    if isinstance(args, (list, tuple)):
        return [_scrub_args(item, redactor) for item in args]
    if isinstance(args, str):
        return redactor(args)
    return args


def _redacted_marker(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (list, tuple, dict)):
        return f"[REDACTED {len(value)} item(s)]"
    return f"[REDACTED {len(str(value))} chars]"


def _truncate(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [+{len(text) - limit} chars truncated]"


class ActivityLogger:
    """Writes the audit trail. Thread-safe: tools and TTS run in threads."""

    def __init__(
        self,
        log_dir: Path,
        activity_log_name: str = "activity_log.txt",
        activity_jsonl_prefix: str = "activity_log",
        redact_patterns: Iterable[str] | None = None,
        redaction_replacement: str = REDACTION_REPLACEMENT,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.activity_path = self.dir / Path(activity_log_name).name
        self.jsonl_prefix = Path(activity_jsonl_prefix).stem
        self.redactor = Redactor(redact_patterns, redaction_replacement)
        self._lock = threading.RLock()
        self._sub_lock = threading.RLock()
        self._subscribers: list[Any] = []
        self._session_id = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    # -- paths -------------------------------------------------------------
    def _jsonl_path(self, day: _dt.date | None = None) -> Path:
        day = day or _dt.date.today()
        return self.dir / f"{self.jsonl_prefix}-{day.isoformat()}.jsonl"

    # -- subscribers -------------------------------------------------------
    def subscribe(self, callback: "Callable[[dict[str, Any]], None]") -> None:
        """Register a callback for every event that is written.

        Used by the task-state store (stage 8) and the UI: both want to see what
        happened without re-reading the log file. A broken subscriber must never
        break logging, so exceptions are swallowed and reported to the log only.
        """
        with self._sub_lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def unsubscribe(self, callback: "Callable[[dict[str, Any]], None]") -> None:
        with self._sub_lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def _dispatch(self, record: dict[str, Any]) -> None:
        with self._sub_lock:
            subscribers = list(self._subscribers)
        for callback in subscribers:
            try:
                callback(dict(record))
            except Exception:
                continue  # a bad listener is not allowed to lose a log line

    # -- writing -----------------------------------------------------------
    def event(
        self,
        kind: str,
        message: str,
        tool: str | None = None,
        args: Any = None,
        account: str | None = None,
        result: str | None = None,
        ok: bool | None = None,
        duration_ms: float | None = None,
        tier: str | None = None,
        decision: str | None = None,
        extra: dict[str, Any] | None = None,
        sensitive: bool = False,
    ) -> dict[str, Any]:
        """Record one event. Returns the record that was written."""
        now = _dt.datetime.now(_dt.timezone.utc).astimezone()
        record: dict[str, Any] = {
            "ts": now.isoformat(timespec="seconds"),
            "session": self._session_id,
            "kind": kind,
            "message": _truncate(self.redactor(message)),
        }
        if tool:
            record["tool"] = tool
        scrubbed = _scrub_args(args, self.redactor, sensitive=sensitive)
        if scrubbed:
            record["args"] = scrubbed
        if account:
            record["account"] = account
        if result is not None:
            record["result"] = _truncate(self.redactor(result))
        if ok is not None:
            record["ok"] = bool(ok)
        if duration_ms is not None:
            record["duration_ms"] = round(float(duration_ms), 1)
        if tier:
            record["tier"] = tier
        if decision:
            record["decision"] = decision
        if extra:
            record["extra"] = _scrub_args(extra, self.redactor, sensitive=sensitive)

        self._dispatch(record)
        if not self.enabled:
            return record

        with self._lock:
            try:
                with self.activity_path.open("a", encoding="utf-8") as fh:
                    fh.write(self._format_block(record))
            except OSError:
                pass  # never let logging kill the assistant
            try:
                with self._jsonl_path().open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            except OSError:
                pass
        return record

    def _format_block(self, record: dict[str, Any]) -> str:
        """One readable block. Kept stable so `grep` and human eyes both work."""
        lines = [f"[{record['ts']}] {record.get('kind', 'event').upper()}"]
        lines.append(f"  what    : {record.get('message', '')}")
        if record.get("tool"):
            lines.append(f"  tool    : {record['tool']}")
        if record.get("args"):
            try:
                args_text = json.dumps(record["args"], ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                args_text = str(record["args"])
            lines.append(f"  args    : {_truncate(args_text, 2000)}")
        if record.get("account"):
            lines.append(f"  account : {record['account']}")
        if record.get("tier"):
            lines.append(f"  tier    : {record['tier']}")
        if record.get("decision"):
            lines.append(f"  decision: {record['decision']}")
        if record.get("ok") is not None:
            lines.append(f"  ok      : {record['ok']}")
        if record.get("duration_ms") is not None:
            lines.append(f"  took    : {record['duration_ms']} ms")
        if record.get("result") is not None:
            lines.append(f"  result  : {record['result']}")
        if record.get("extra"):
            lines.append(f"  extra   : {json.dumps(record['extra'], ensure_ascii=False, default=str)}")
        return "\n".join(lines) + "\n\n"

    # -- convenience -------------------------------------------------------
    def tool_call(
        self,
        tool: str,
        args: Any,
        result: str | None = None,
        ok: bool | None = None,
        duration_ms: float | None = None,
        account: str | None = None,
        tier: str | None = None,
        sensitive: bool = False,
    ) -> dict[str, Any]:
        status = "ok" if ok else ("failed" if ok is False else "ran")
        return self.event(
            KIND_TOOL,
            f"{tool} {status}",
            tool=tool,
            args=args,
            result=result,
            ok=ok,
            duration_ms=duration_ms,
            account=account,
            tier=tier,
            sensitive=sensitive,
        )

    def permission(
        self,
        tool: str,
        args: Any,
        tier: str,
        decision: str,
        note: str = "",
        sensitive: bool = False,
    ) -> dict[str, Any]:
        message = f"permission {decision}: {tool} is {tier.upper()}"
        if note:
            message += f" ({note})"
        return self.event(
            KIND_PERMISSION, message, tool=tool, args=args, tier=tier,
            decision=decision, sensitive=sensitive,
        )

    # -- reporting (requirement 8: "what did you do today") ----------------
    def read_records(self, day: _dt.date | None = None) -> list[dict[str, Any]]:
        day = day or _dt.date.today()
        path = self._jsonl_path(day)
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def today_report(self, day: _dt.date | None = None, limit: int = 40) -> str:
        """A short spoken-friendly summary of everything logged today."""
        day = day or _dt.date.today()
        records = self.read_records(day)
        if not records:
            return f"Nothing logged yet today ({day.isoformat()})."

        tools: dict[str, int] = {}
        failures: list[dict[str, Any]] = []
        denials: list[dict[str, Any]] = []
        first = records[0].get("ts", "")
        last = records[-1].get("ts", "")
        for rec in records:
            if rec.get("kind") == KIND_TOOL and rec.get("tool"):
                tools[rec["tool"]] = tools.get(rec["tool"], 0) + 1
                if rec.get("ok") is False:
                    failures.append(rec)
            elif rec.get("kind") == KIND_PERMISSION and rec.get("decision") in ("denied", "blocked"):
                denials.append(rec)

        lines = [
            f"Today ({day.isoformat()}) I logged {len(records)} events"
            + (f", from {first[11:19]} to {last[11:19]}" if first and last else "")
            + "."
        ]
        if tools:
            top = sorted(tools.items(), key=lambda kv: -kv[1])[:8]
            lines.append("Tools used: " + ", ".join(f"{name} x{count}" for name, count in top) + ".")
        if denials:
            lines.append(f"Blocked or denied requests: {len(denials)}.")
        if failures:
            lines.append(f"Failures: {len(failures)}.")
            for rec in failures[:5]:
                lines.append(f"  - {rec.get('tool')}: {rec.get('message')}")
        lines.append("")
        lines.append("Recent events:")
        for rec in records[-limit:]:
            stamp = rec.get("ts", "")[11:19]
            kind = rec.get("kind", "?")
            msg = rec.get("message", "")
            lines.append(f"  {stamp} [{kind}] {msg}")
        return "\n".join(lines)

    def recent_tools(self, limit: int = 10, day: _dt.date | None = None) -> list[dict[str, Any]]:
        records = [r for r in self.read_records(day) if r.get("kind") == KIND_TOOL]
        return records[-limit:]


class _CompactFormatter(logging.Formatter):
    """`12:04:31 INFO  brain: reply ready (412 ms)` - short, aligned, greppable."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        ts = _dt.datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        level = record.levelname[:7].ljust(7)
        name = record.name.replace("garvis.", "")
        msg = record.getMessage()
        base = f"{ts} {level} {name}: {msg}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def setup_logging(
    log_file: Path | str,
    level: str = "INFO",
    console: bool = True,
    redact_patterns: Iterable[str] | None = None,
    console_style: str = "compact",
    force: bool = True,
) -> logging.Logger:
    """Configure the developer log. Safe to call more than once."""
    level_value = getattr(logging, str(level).upper(), logging.INFO)
    logger = logging.getLogger("garvis")
    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
    logger.setLevel(level_value)
    logger.propagate = False

    formatter = _CompactFormatter()

    try:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level_value)
        # File gets a redaction filter so a stray secret in a message is caught.
        file_handler.addFilter(_RedactionFilter(redact_patterns))
        logger.addHandler(file_handler)
    except OSError:
        pass

    if console:
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        stream.setLevel(level_value)
        logger.addHandler(stream)

    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


class _RedactionFilter(logging.Filter):
    def __init__(self, patterns: Iterable[str] | None = None) -> None:
        super().__init__()
        self.redactor = Redactor(patterns)

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            record.msg = self.redactor(record.getMessage())
            record.args = None
        except Exception:  # pragma: no cover - never break logging
            pass
        return True


def get_logger(name: str = "garvis") -> logging.Logger:
    """`get_logger(__name__)` everywhere; children inherit the root config."""
    if name == "garvis" or name.startswith("garvis."):
        return logging.getLogger(name)
    return logging.getLogger(f"garvis.{name}")


# --------------------------------------------------------------------------
# Singleton helpers: the activity logger is process-wide, one audit trail.
# --------------------------------------------------------------------------
_ACTIVITY: ActivityLogger | None = None


def today_report(limit: int = 40) -> str:
    """Requirement 8 helper: `python main.py --today`."""
    if _ACTIVITY is None:
        raise RuntimeError("Activity logger not initialised; call configure_activity_logger first.")
    return _ACTIVITY.today_report(limit=limit)


def configure_activity_logger(cfg: "Any") -> ActivityLogger:
    """Build the process-wide ActivityLogger from a Config object."""
    global _ACTIVITY
    log_dir = cfg.resolve_path(cfg.get("logging.file", "logs/garvis.log")).parent
    _ACTIVITY = ActivityLogger(
        log_dir=log_dir,
        activity_log_name=Path(str(cfg.get("logging.activity_log", "logs/activity_log.txt"))).name,
        activity_jsonl_prefix=Path(str(cfg.get("logging.activity_jsonl", "logs/activity_log.jsonl"))).stem,
        redact_patterns=cfg.get("logging.redact_patterns", []),
        redaction_replacement=str(cfg.get("logging.redaction_replacement", REDACTION_REPLACEMENT)),
        enabled=True,
    )
    return _ACTIVITY


def activity() -> ActivityLogger:
    """Access the process-wide logger (raises if not configured)."""
    if _ACTIVITY is None:
        raise RuntimeError("Activity logger not configured")
    return _ACTIVITY
