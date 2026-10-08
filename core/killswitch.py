"""The kill switch: global hotkey, spoken phrase, and the STOP button.

One object owns the emergency stop so that every path - Ctrl+Alt+Esc, "Garvis,
stop everything", the tray button, the model calling ``assistant.stop`` - goes
through the same code. That matters: an emergency stop that behaves differently
per entry point is worse than no stop at all.

State it holds:
    ``frozen``      - everything is halted until ``resume()``. New tasks are
                      refused, TTS is paused, in-flight tools are abandoned.
    ``reason``      - why, for the log and for what GARVIS says afterwards.
    ``stop_count``  - how many times it has fired this session.

What it can and cannot do, stated honestly:
    * It cancels the model generation between tokens, drops queued speech, stops
      the audio player, halts browser profiles and screen guidance, and tells the
      tools to abandon their work.
    * It terminates child processes that tools registered with it: a shell
      command that is still running is killed, whole process group first
      (``SIGTERM``, then ``SIGKILL`` after a short grace), so a stopped command
      cannot keep writing to disk. Windows uses ``taskkill /T`` for the tree.
    * It cannot un-run something that already finished, and it cannot kill a
      Python thread: a library call already inside a third-party function (a
      screenshot backend, a Playwright action) ends when that call returns.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class StopEvent:
    at: float
    reason: str
    source: str
    frozen: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


def _is_running(proc: Any) -> bool:
    """True while a child process is alive (never raises)."""
    try:
        return proc.poll() is None
    except Exception:
        return False


def _describe_process(proc: Any) -> str:
    """A short, log-safe name for a process: what it was, not its full command."""
    args = getattr(proc, "args", None)
    pid = getattr(proc, "pid", "?")
    if isinstance(args, (list, tuple)) and args:
        name = " ".join(str(part) for part in args[:3])
    elif isinstance(args, str):
        name = args
    else:
        name = "process"
    return f"pid {pid}: {name[:60]}"


def _signal_group(proc: Any, *, gentle: bool) -> None:
    """Signal a child and its whole process group. Never raises.

    ``gentle`` sends the polite signal (SIGTERM / terminate), otherwise the
    unavoidable one (SIGKILL / taskkill /F). The tools start their children with
    ``start_new_session=True``, so the child is a group leader and this reaches
    everything it spawned.
    """
    pid = getattr(proc, "pid", None)
    if sys.platform == "win32":
        if pid is not None and shutil.which("taskkill"):
            argv = ["taskkill", "/T", "/PID", str(pid)] + ([] if gentle else ["/F"])
            try:
                subprocess.run(argv, capture_output=True, timeout=10)
                return
            except Exception:
                pass
        try:
            (proc.terminate if gentle else proc.kill)()
        except Exception:
            pass
        return

    if pid is not None and hasattr(os, "killpg"):
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM if gentle else signal.SIGKILL)
            return
        except ProcessLookupError:
            return                      # already gone: nothing to do, and not an error
        except Exception:
            pass
    try:                                 # no process groups here: signal the child itself
        (proc.terminate if gentle else proc.kill)()
    except Exception:
        pass


def _terminate(proc: Any, grace_s: float = 0.5) -> bool:
    """Stop one child process *and everything it started*. Honest about the result.

    Two things this deliberately does not assume:

    * that the direct child is the command - a shell can spawn the real work as a
      child of its own, so the *group* is signalled, not just the pid;
    * that a polite signal worked - a process is free to ignore SIGTERM, and a
      shell that dies of SIGTERM while its grandchild ignores it would leave the
      command running and, worse, holding the pipe open so the tool never
      returns. So the hard signal follows after a short grace, unconditionally.
    """
    _signal_group(proc, gentle=True)
    if grace_s > 0:
        time.sleep(min(grace_s, 0.5))
    _signal_group(proc, gentle=False)
    try:
        proc.wait(timeout=max(0.2, min(grace_s, 1.0)))
    except Exception:
        pass
    return not _is_running(proc)


class KillSwitch:
    """Global emergency stop."""

    def __init__(self, cfg: Any, activity: Any = None, log: Any = None) -> None:
        self.cfg = cfg
        self.activity = activity
        self.log = log
        self._frozen = threading.Event()
        self._lock = threading.RLock()
        self.enabled = bool(cfg.get("safety.kill_phrases", True) is not None)
        self.stop_count = 0
        self.last_stop: StopEvent | None = None
        self.history: list[StopEvent] = []
        self.kill_phrases = cfg.kill_phrases()
        self._subscribers: list[Callable[[StopEvent], None]] = []
        self._toggle_hotkeys: Any = None
        self._hotkey_backend = "none"
        # Child processes tools asked us to watch (shell commands): a stop kills
        # them for real, not just in the log.
        self._processes: dict[int, tuple[Any, str]] = {}
        self._process_lock = threading.RLock()
        self._next_process = 1

    # -- subscription ------------------------------------------------------
    def subscribe(self, callback: Callable[[StopEvent], None]) -> None:
        """Register a callback fired on every stop (TTS, brain, UI, tools)."""
        self._subscribers.append(callback)

    def _notify(self, event: StopEvent) -> None:
        for callback in list(self._subscribers):
            try:
                callback(event)
            except Exception:
                if self.log:
                    self.log.debug("kill switch subscriber failed", exc_info=True)

    # -- child processes ---------------------------------------------------
    def register_process(self, proc: Any, label: str = "") -> int:
        """Watch a child process so a stop can terminate it.

        Returns a token; the caller unregisters it when the process is gone.
        """
        with self._process_lock:
            token = self._next_process
            self._next_process += 1
            self._processes[token] = (proc, label or _describe_process(proc))
        return token

    def unregister_process(self, token: int) -> None:
        with self._process_lock:
            self._processes.pop(token, None)

    def running_processes(self) -> list[str]:
        """Labels of the registered processes that are still alive."""
        with self._process_lock:
            return [label for proc, label in self._processes.values() if _is_running(proc)]

    def stop_processes(self, grace_s: float | None = None) -> tuple[list[str], list[str]]:
        """Terminate every registered child process. Returns (stopped, failed).

        Called by ``trigger()``. A process that ignores SIGTERM gets SIGKILL;
        one that survives even that is reported as failed rather than forgotten.
        """
        if grace_s is None:
            try:
                grace_s = float(self.cfg.get("safety.stop_grace_s", 0.5) or 0.5)
            except Exception:
                grace_s = 0.5
        with self._process_lock:
            items = list(self._processes.items())
        stopped: list[str] = []
        failed: list[str] = []
        for token, (proc, label) in items:
            name = label or _describe_process(proc)
            if _is_running(proc) and not _terminate(proc, grace_s):
                failed.append(name)
            else:
                stopped.append(name)
            self.unregister_process(token)
        return stopped, failed

    # -- state -------------------------------------------------------------
    @property
    def frozen(self) -> bool:
        return self._frozen.is_set()

    def require_running(self) -> None:
        """Raise if the switch is engaged. Tools and the main loop call this."""
        if self._frozen.is_set():
            raise RuntimeError(
                f"GARVIS is stopped ({self.last_stop.reason if self.last_stop else 'kill switch'}). "
                f"Say 'resume' or press the hotkey again to continue."
            )

    def status(self) -> dict[str, Any]:
        return {
            "frozen": self.frozen,
            "child_processes": self.running_processes(),
            "stop_count": self.stop_count,
            "last_reason": self.last_stop.reason if self.last_stop else None,
            "hotkey": self.cfg.get("hotkeys.killswitch", "ctrl+alt+esc"),
            "backend": self._hotkey_backend,
            "phrases": self.kill_phrases,
        }

    # -- the stop itself ---------------------------------------------------
    def trigger(self, reason: str = "user requested", source: str = "manual") -> StopEvent:
        """Halt everything. Safe to call from any thread, repeatedly."""
        with self._lock:
            self._frozen.set()
            self.stop_count += 1
            event = StopEvent(at=time.time(), reason=reason, source=source)
            self.last_stop = event
            self.history.append(event)
            if len(self.history) > 50:
                self.history = self.history[-50:]
        # Kill anything a tool started, before telling the rest of the app: a
        # shell command still writing files is the most urgent thing to stop.
        stopped, failed = self.stop_processes()
        if stopped:
            event.extra["terminated_processes"] = stopped
            if self.log:
                self.log.warning("kill switch terminated: %s", "; ".join(stopped))
        if failed:
            event.extra["unstopped_processes"] = failed
            if self.log:
                self.log.error(
                    "kill switch could NOT stop these processes (they may still be running): %s",
                    "; ".join(failed),
                )
        if self.activity:
            detail = f"STOP EVERYTHING ({source}): {reason}"
            if stopped:
                detail += f" - terminated {len(stopped)} running process(es)"
            self.activity.event("system", detail, ok=False)
        if self.log:
            self.log.warning("KILL SWITCH (%s): %s", source, reason)
        self._notify(event)
        return event

    def resume(self, note: str = "resumed") -> None:
        """Clear the freeze. Does not restart whatever was cancelled."""
        with self._lock:
            if not self._frozen.is_set():
                return
            self._frozen.clear()
        if self.activity:
            self.activity.event("system", f"kill switch cleared: {note}")
        if self.log:
            self.log.info("kill switch cleared (%s)", note)

    def toggle(self, reason: str = "hotkey") -> bool:
        """Hotkey behaviour: freeze if running, resume if frozen."""
        if self._frozen.is_set():
            self.resume("hotkey toggle")
            return False
        self.trigger(reason, source="hotkey")
        return True

    # -- spoken phrases ----------------------------------------------------
    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase, punctuation -> spaces, collapse.

        Word-boundary matching matters here: without it "aborting the build"
        would match the phrase "abort" and stop GARVIS mid-sentence.
        """
        cleaned = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower())
        return " ".join(cleaned.split()).strip()

    def matches_kill_phrase(self, text: str) -> bool:
        """True when a transcript is a hard-stop command."""
        cleaned = self._normalize(text)
        if not cleaned:
            return False
        for phrase in self.kill_phrases:
            normalized = self._normalize(phrase)
            if not normalized:
                continue
            if cleaned == normalized or cleaned.startswith(normalized + " "):
                return True
            # After being woken, "stop" or "stop everything" on its own is unambiguous.
            if normalized.endswith("stop everything") and cleaned in ("stop", "stop everything", "stop it"):
                return True
        return False

    def matches_resume_phrase(self, text: str) -> bool:
        cleaned = self._normalize(text)
        return cleaned in (
            "resume", "garvis resume", "carry on", "garvis carry on",
            "continue", "garvis continue", "unfreeze",
        )

    # -- global hotkey -----------------------------------------------------
    def install_hotkey(self, combination: str | None = None, backend: str | None = None) -> bool:
        """Register the global hotkey. Returns False with a clear reason if not possible.

        Backends: ``keyboard`` (Windows, Linux/X11, needs root on Linux) or
        ``pynput`` (macOS). Wayland does not allow global hotkeys from user
        processes at all - in that case bind the STOP button or the spoken phrase.
        """
        combination = combination or str(self.cfg.get("hotkeys.killswitch", "ctrl+alt+esc"))
        backend = (backend or str(self.cfg.get("hotkeys.backend", "auto"))).lower()
        if backend == "none":
            return False
        if os.environ.get("WAYLAND_DISPLAY") and not os.environ.get("DISPLAY"):
            if self.log:
                self.log.warning(
                    "Wayland session detected: global hotkeys are not available. "
                    "Use the tray STOP button or say 'Garvis, stop everything'."
                )
            return False

        errors: list[str] = []
        for candidate in (["keyboard"] if backend == "auto" else [backend]):
            try:
                if candidate == "keyboard":
                    import keyboard  # noqa: PLC0415

                    keyboard.add_hotkey(combination, lambda: self.toggle(f"hotkey {combination}"))
                    self._toggle_hotkeys = keyboard
                    self._hotkey_backend = f"keyboard:{combination}"
                    if self.log:
                        self.log.info("kill switch hotkey armed: %s", combination)
                    return True
                if candidate == "pynput":
                    from pynput import keyboard as pynput_keyboard  # noqa: PLC0415

                    def on_activate() -> None:
                        self.toggle(f"hotkey {combination}")

                    listener = pynput_keyboard.GlobalHotKeys({f"<{combination.replace('+', '>+<')}>": on_activate})
                    listener.daemon = True
                    listener.start()
                    self._toggle_hotkeys = listener
                    self._hotkey_backend = f"pynput:{combination}"
                    if self.log:
                        self.log.info("kill switch hotkey armed: %s", combination)
                    return True
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")
                continue
        if self.log:
            self.log.warning(
                "could not arm the global hotkey (%s). The spoken phrase and the tray "
                "STOP button still work.", "; ".join(errors) or "no backend available",
            )
        return False

    def close(self) -> None:
        backend = self._toggle_hotkeys
        if backend is None:
            return
        try:
            if hasattr(backend, "remove_hotkey"):
                backend.remove_hotkey()
            elif hasattr(backend, "stop"):
                backend.stop()
        except Exception:
            pass
        self._toggle_hotkeys = None
