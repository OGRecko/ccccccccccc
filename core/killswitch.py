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
    * It cancels the model generation between tokens, drops queued speech,
      stops the audio player, and tells the tools to abandon their work.
    * It cannot un-run a command that already started, and it cannot kill a
      Python thread. Tools that touch the world (shell commands) are started in
      their own process group so ``stop()`` can terminate the group.
"""

from __future__ import annotations

import os
import re
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
        if self.activity:
            self.activity.event("system", f"STOP EVERYTHING ({source}): {reason}", ok=False)
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
