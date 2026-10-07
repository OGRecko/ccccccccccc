"""Task state: survive a crash, and know what was going on (stage 8).

GARVIS is a long-running process that can be killed by a power cut, a crash, or
the user closing the terminal. This module keeps a small, written-down record of
*what task was in flight and what was already done*, so the next start can say:

    "I was interrupted while tidying your downloads folder. I had already moved
     12 files. Say resume and I will carry on, or say rollback to see the undo plan."

Design rules:

* **Written to disk after every step**, atomically (temp file + rename), so a
  crash mid-write leaves the previous good state rather than a half file.
* **Never a source of truth for permissions.** This records what happened; it
  cannot authorize anything. Undoing work goes back through the permission gate
  like any other action.
* **Rollback is a plan, not a surprise.** :meth:`rollback_plan` describes how to
  undo each recorded step, in reverse order, with the exact tool calls. GARVIS
  shows it and asks; nothing is undone behind the user's back.
* **Secrets never land here.** Step summaries come from the activity log, which
  is already redacted; the store re-checks with :func:`core.logger.redact`.

The store subscribes to the activity log, so it fills itself in as work happens
without any tool or the model having to remember to call it.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .logger import redact

VERSION = 1

#: Tools whose effect can be described as an undo. The value is a small template;
#: ``{args}`` is the recorded (already redacted) argument dict.
_UNDO_HINTS: dict[str, str] = {
    "files.write": "delete {path} if you did not have it before (check first: files.read)",
    "files.mkdir": "remove the empty folder {path}",
    "files.move": "move {destination} back to {path}",
    "files.copy": "delete the copy at {destination}",
    "files.delete": "restore {path} from your backup (I do not keep one)",
    "apps.open": "close {name}",
    "browser.open": "close the '{profile}' browser profile",
    "browser.click": "review the page and undo the click by hand",
    "browser.type": "review the field and clear what was typed",
    "shell.run": "no automatic undo: re-run the command with the opposite effect, if one exists",
    "screen.click": "no automatic undo: check what changed on screen",
    "screen.type": "no automatic undo: undo it in the focused window",
}

#: Steps that changed nothing locally and need no undo entry.
_READONLY_TOOLS = ("files.read", "files.list", "files.stat", "files.search", "shell.which",
                   "shell.info", "clock.", "activity.", "memory.read", "screen.look",
                   "screen.capture", "screen.status", "browser.read", "browser.screenshot",
                   "browser.status", "browser.profiles", "assistant.", "task.", "apps.list")


@dataclass
class Step:
    """One recorded action inside a task."""

    index: int
    tool: str
    summary: str
    ok: bool = True
    tier: str = ""
    decision: str = ""
    at: float = 0.0
    undo: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Step":
        known = {name: data[name] for name in cls.__dataclass_fields__ if name in data}
        return cls(**known)


@dataclass
class Task:
    """What GARVIS was asked to do, and how far it got."""

    description: str
    status: str = "running"          # running | done | failed | abandoned | interrupted
    plan: list[str] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    session: str = ""
    result: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["steps"] = [step.to_dict() for step in self.steps]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        steps = [Step.from_dict(item) for item in (data.get("steps") or [])]
        known = {name: data[name] for name in cls.__dataclass_fields__ if name in data}
        known["steps"] = steps
        task = cls(**known)
        return task

    # -- convenience -------------------------------------------------------
    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.started_at)

    def changes(self) -> list[Step]:
        """Steps that actually changed something (the ones worth undoing)."""
        return [step for step in self.steps if step.undo and step.ok]

    def summarize(self, limit: int = 6) -> str:
        head = f"{self.description} [{self.status}, {len(self.steps)} step(s)"
        if self.status == "running":
            head += f", running for {self.age_s / 60:.0f} min"
        head += "]"
        recent = self.steps[-limit:]
        if not recent:
            return head
        lines = [head, "  last steps:"]
        for step in recent:
            mark = "ok" if step.ok else "failed"
            lines.append(f"    {step.index}. {step.tool} ({mark}) - {step.summary[:120]}")
        return "\n".join(lines)


class StateStore:
    """Owns the state file: sessions, the current task, history and the undo plan."""

    def __init__(
        self,
        cfg: Any = None,
        activity: Any = None,
        log: Any = None,
        path: Path | str | None = None,
        subscribe: bool = True,
    ) -> None:
        self.cfg = cfg
        self.activity = activity
        self.log = log
        self.enabled = True
        self.history_limit = 10
        self.record_steps = True
        self.auto_task = True
        self._last_request = ""
        self._auto_started = False
        self._lock = threading.RLock()

        directory = self._resolve_dir(cfg)
        self.path = Path(path) if path is not None else directory / "task_state.json"
        if cfg is not None:
            try:
                self.history_limit = max(1, int(cfg.get("state.history_limit", 10) or 10))
                self.record_steps = bool(cfg.get("state.record_steps", True))
                self.auto_task = bool(cfg.get("state.auto_task", True))
            except Exception:
                pass

        self.session = time.strftime("%Y%m%d-%H%M%S")
        self.data: dict[str, Any] = self._empty()
        self.current: Task | None = None
        self.interrupted: Task | None = None      # unfinished work from the last run
        self.interrupted_was_crash = False
        self._loaded = False
        self._lock_path = self.path.with_suffix(".lock")
        if subscribe and activity is not None and hasattr(activity, "subscribe"):
            try:
                activity.subscribe(self.observe)
                self._subscribed = True
            except Exception:
                self._subscribed = False
        else:
            self._subscribed = False

    # -- storage -----------------------------------------------------------
    @staticmethod
    def _resolve_dir(cfg: Any) -> Path:
        try:
            return cfg.resolve_path(cfg.get("state.dir", "state"))
        except Exception:
            return Path("state")

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "version": VERSION,
            "clean_exit": True,
            "session": "",
            "pid": os.getpid(),
            "started_at": 0.0,
            "updated_at": 0.0,
            "task": None,
            "history": [],
        }

    def load(self) -> dict[str, Any]:
        """Read the state file. A damaged file is reported, never fatal."""
        with self._lock:
            raw: dict[str, Any] = {}
            try:
                text = self.path.read_text(encoding="utf-8")
                raw = json.loads(text) if text.strip() else {}
            except FileNotFoundError:
                raw = {}
            except (OSError, json.JSONDecodeError) as exc:
                if self.log:
                    self.log.warning("state file unreadable (%s); starting fresh", exc)
                raw = {}
            self.data = {**self._empty(), **(raw if isinstance(raw, dict) else {})}
            self._loaded = True
            return self.data

    def save(self, pending: "Task | None" = None) -> None:
        """Write the state atomically: a crash cannot leave a half-written file.

        ``pending`` is used when a task should outlive this process (the user quit
        mid-task): it goes into the file's task slot instead of the live task.
        """
        with self._lock:
            self.data["version"] = VERSION
            self.data["updated_at"] = time.time()
            task = self.current if self.current is not None else pending
            self.data["task"] = task.to_dict() if task else None
            payload = json.dumps(self.data, indent=2, ensure_ascii=False, default=str)
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(payload, encoding="utf-8")
                os.replace(tmp, self.path)
            except OSError as exc:
                if self.log:
                    self.log.warning("could not write %s: %s", self.path, exc)

    # -- sessions and crash detection --------------------------------------
    def begin_session(self) -> "Task | None":
        """Start a clean session and report a previous run that did not end.

        Returns the task that was in flight when the machine went down (if any),
        so the caller can offer to resume it. The task stays in the file until it
        is resumed or finished: starting a session (``--check``, a second window)
        must not silently throw the offer away.
        """
        with self._lock:
            if not self._loaded:
                self.load()
            previous = self.data
            self.interrupted = None
            self.interrupted_was_crash = False
            if previous:
                task_data = previous.get("task")
                unclean = not previous.get("clean_exit", True)
                if isinstance(task_data, dict):
                    try:
                        candidate = Task.from_dict(task_data)
                        # "running" means the file was last written mid-task;
                        # "interrupted" means we stopped on purpose, mid-task.
                        if candidate.status in ("running", "interrupted"):
                            candidate.status = "interrupted"
                            self.interrupted = candidate
                            self.interrupted_was_crash = unclean
                    except Exception as exc:
                        if self.log:
                            self.log.debug("could not read the interrupted task: %s", exc)
            self.session = time.strftime("%Y%m%d-%H%M%S")
            self.data = self._empty()
            self.data["session"] = self.session
            self.data["started_at"] = time.time()
            self.data["clean_exit"] = False
            self.data["history"] = list(previous.get("history") or []) if previous else []
            self.current = None
            # Keep the unfinished task on disk as the pending one, so a later
            # start (or --check) still finds it.
            self.save(pending=self.interrupted)
        if self.interrupted and self.activity is not None:
            try:
                how = "did not exit cleanly" if self.interrupted_was_crash else "stopped mid-task"
                self.activity.event(
                    "state",
                    f"previous run {how}; unfinished task: {self.interrupted.description}",
                    extra={"tool_calls": len(self.interrupted.steps)},
                )
            except Exception:
                pass
        return self.interrupted

    def end_session(self, reason: str = "normal exit") -> None:
        """Leave the file in a state the next run can act on.

        Quitting in the middle of a task is not a crash, but the work is still
        unfinished, so the task stays in the file as *interrupted* and the next
        start offers to carry on. Only a task that was already finished or
        abandoned disappears.
        """
        with self._lock:
            task = self.current
            self.data["clean_exit"] = True
            self.data["ended_at"] = time.time()
            self.data["end_reason"] = reason
            self.current = None
            pending: Task | None = None
            if task is not None and task.status == "running":
                task.status = "interrupted"
                task.result = reason
                self._push_history(task)
                pending = task
            elif task is None and self.interrupted is not None:
                # Unfinished work from the last run that was never resumed: keep
                # offering it. (A task started *in this* session wins the single
                # task slot - the file cannot hold two, and the newer one is the
                # one the user was actually working on.)
                pending = self.interrupted
            self.save(pending=pending)

    def crash_report(self) -> str:
        """A sentence for the user about the unfinished work from the last run."""
        task = self.interrupted
        if task is None:
            return ""
        age = time.time() - (task.updated_at or task.started_at)
        when = _human_age(age)
        if getattr(self, "interrupted_was_crash", False):
            lines = [f"The last run did not shut down properly ({when}) - it was in the "
                     f"middle of: {task.description}"]
        else:
            lines = [f"We stopped {when} while this was still in progress: {task.description}"]
        changes = task.changes()
        if changes:
            lines.append(f"There were {len(changes)} change(s) recorded before I stopped:")
            for step in changes[-5:]:
                lines.append(f"  - {step.tool}: {step.summary[:120]}")
            lines.append("Say 'resume' and I will carry on, or ask for the rollback plan to undo them.")
        elif task.steps:
            lines.append("Nothing that changed your files was recorded, so there is nothing to undo.")
        else:
            lines.append("I had not started doing anything yet.")
        return "\n".join(lines)

    # -- task lifecycle ----------------------------------------------------
    def note_user_request(self, text: str) -> None:
        """Remember what the user asked for, so an automatic task can be named."""
        self._last_request = redact(" ".join(str(text or "").split()))[:200]

    def end_turn(self, ok: bool = True) -> Task | None:
        """Called by main after one user turn.

        A task GARVIS started by itself is closed as soon as the turn ends
        cleanly; one that hit a failure is left running, because that is exactly
        the situation the user wants to be able to resume.
        """
        with self._lock:
            task = self.current
            if task is None or not self._auto_started:
                return None
            failed = any(not step.ok for step in task.steps)
            if ok and not failed:
                return self.finish_task("the turn finished cleanly", status="done")
            return None

    def start_task(self, description: str, plan: Iterable[str] | None = None,
                   auto: bool = False) -> Task:
        with self._lock:
            if self.current is not None and self.current.status == "running":
                self.current.status = "abandoned"
                self.current.result = "replaced by a new task"
                self._push_history(self.current)
            task = Task(
                description=redact(str(description))[:400],
                plan=[redact(str(item))[:200] for item in (plan or [])][:20],
                session=self.session,
            )
            self.current = task
            self._auto_started = bool(auto)
            self.save()
        self._event(
            "state",
            f"task started{'(automatic)' if auto else ''}: {task.description}",
            extra={"task": task.description},
        )
        return task

    def finish_task(self, result: str = "", status: str = "done") -> Task | None:
        with self._lock:
            task = self.current
            if task is None:
                return None
            task.status = status
            task.result = redact(str(result))[:400]
            task.finished_at = time.time()
            self._push_history(task)
            self.current = None
            self._auto_started = False
            self.save()
        self._event("state", f"task {status}: {task.description}", extra={"result": task.result})
        return task

    def abandon_task(self, reason: str = "stopped") -> Task | None:
        return self.finish_task(reason, status="abandoned")

    def resume_task(self, note: str = "resumed by the user") -> Task | None:
        """Re-instate the interrupted task (or the current one) as running."""
        with self._lock:
            task = self.current
            if task is None and self.interrupted is not None:
                task = self.interrupted
                self.interrupted = None
                task.status = "running"
                self.current = task
            if task is None:
                return None
            task.status = "running"
            task.result = redact(note)
            self.save()
        self._event("state", f"task resumed: {task.description}")
        return task

    def _push_history(self, task: Task) -> None:
        self.data.setdefault("history", []).insert(0, task.to_dict())
        del self.data["history"][self.history_limit:]

    # -- steps -------------------------------------------------------------
    def record_step(
        self, tool: str, summary: str = "", ok: bool = True, tier: str = "",
        decision: str = "", args: dict[str, Any] | None = None,
    ) -> Step | None:
        """Add one step to the current task. Returns None when nothing is running."""
        with self._lock:
            task = self.current
            if task is None or not self.record_steps:
                return None
            step = Step(
                index=len(task.steps) + 1,
                tool=str(tool),
                summary=redact(str(summary or ""))[:300],
                ok=bool(ok),
                tier=str(tier or ""),
                decision=str(decision or ""),
                at=time.time(),
                undo=undo_hint(str(tool), args or {}),
            )
            task.steps.append(step)
            del task.steps[:-200]                   # bound the file, keep the tail
            task.updated_at = time.time()
            self.save()
            return step

    def observe(self, record: dict[str, Any]) -> None:
        """Activity-log subscriber: fill the current task in as work happens."""
        try:
            kind = str(record.get("kind") or "")
            if kind != "tool":
                return
            status = str(record.get("message") or "").split()[-1] if record.get("message") else ""
            ok = bool(record.get("ok", True))
            if status not in ("ok", "failed", "ran"):
                ok = bool(record.get("ok", True))
            tool = str(record.get("tool") or "")
            if not tool:
                return
            summary = _first_line(record.get("result")) or _first_line(record.get("message")) or tool
            args = record.get("args") if isinstance(record.get("args"), dict) else {}
            if self.current is None and self.auto_task and is_state_changing(tool):
                # Real work started without anyone declaring a task: label it with
                # what the user just asked for, so a crash still has a story.
                self.start_task(self._last_request or "an unlabelled job", auto=True)
            self.record_step(tool, summary, ok=ok, tier=str(record.get("tier") or ""),
                             decision=str(record.get("decision") or ""), args=args)
        except Exception:
            pass  # observing must never disturb the real work

    # -- rollback ----------------------------------------------------------
    def rollback_plan(self, task: Task | None = None) -> list[str]:
        """Human-readable undo steps, newest first. Nothing is executed here."""
        task = task or self.current or self.interrupted
        if task is None:
            return ["nothing to undo: no task is running or was interrupted"]
        plan: list[str] = []
        for step in reversed(task.changes()):
            plan.append(f"{step.index}. {step.tool}: {step.undo}")
        had_delete = any(step.tool == "files.delete" and step.ok for step in task.steps)
        if had_delete:
            plan.append("note: a delete cannot be undone by me - restore from your own backup")
        if not plan:
            plan.append("nothing recorded is undoable automatically; nothing was changed")
        return plan

    # -- views -------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        with self._lock:
            task = self.current
            return {
                "session": self.session,
                "state_file": str(self.path),
                "task": task.to_dict() if task else None,
                "interrupted": self.interrupted.to_dict() if self.interrupted else None,
                "history": [item.get("description") for item in (self.data.get("history") or [])][:5],
                "steps": len(task.steps) if task else 0,
                "changes": len(task.changes()) if task else 0,
                "recording": self.record_steps,
                "auto_started": bool(self._auto_started),
                "clean_exit": bool(self.data.get("clean_exit", True)),
            }

    def summary(self) -> str:
        """One paragraph for the model or the user: what is going on right now."""
        with self._lock:
            if self.current is None:
                if self.interrupted is not None:
                    return f"No task is running. Interrupted task: {self.interrupted.summarize()}"
                history = self.data.get("history") or []
                if history:
                    return f"No task is running. Last finished: {history[0].get('description')}"
                return "No task is running. Nothing has been recorded yet."
            return self.current.summarize()

    def _event(self, kind: str, message: str, extra: dict[str, Any] | None = None) -> None:
        if self.activity is None:
            return
        try:
            self.activity.event(kind, message, extra=extra or {})
        except Exception:
            pass

    # -- lifecycle ---------------------------------------------------------
    def close(self, reason: str = "normal exit") -> None:
        try:
            self.end_session(reason)
        except Exception as exc:
            if self.log:
                self.log.debug("could not close the state store: %s", exc)

    shutdown = close


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def undo_hint(tool: str, args: dict[str, Any]) -> str:
    """Describe how to undo one call, or "" when there is nothing to undo."""
    if any(tool.startswith(prefix) for prefix in _READONLY_TOOLS):
        return ""
    template = None
    for name, text in _UNDO_HINTS.items():
        if tool == name or tool.startswith(name + "."):
            template = text
            break
    if template is None:
        return f"no automatic undo for {tool}: review what it changed"
    safe_args = {key: _short(value) for key, value in (args or {}).items()}
    try:
        return redact(template.format(**safe_args))[:300]
    except KeyError:
        # The template wants an argument this call did not carry.
        return redact(template)[:300]


def is_state_changing(tool: str) -> bool:
    """True when a tool can leave a mark (so a task is worth recording)."""
    if not tool:
        return False
    return not any(tool.startswith(prefix) for prefix in _READONLY_TOOLS)


def _short(value: Any) -> str:
    text = " ".join(str(value).split())
    return text[:80] + ("..." if len(text) > 80 else "")


def _first_line(value: Any) -> str:
    text = redact("" if value is None else str(value))
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:200]
    return ""


def _human_age(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} seconds ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f} minutes ago"
    if seconds < 172800:
        return f"{seconds / 3600:.0f} hours ago"
    return f"{seconds / 86400:.0f} days ago"


def build_state(cfg: Any, activity: Any = None, log: Any = None) -> StateStore:
    store = StateStore(cfg, activity=activity, log=log)
    store.load()
    return store
