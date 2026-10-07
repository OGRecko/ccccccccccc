"""The face of GARVIS: a system-tray icon and a small always-on-top overlay (stage 8).

Both are optional and both degrade honestly:

* **Tray** (``pystray`` + Pillow for the icon): status line, a STOP button, resume,
  pause/resume listening, personality switching, "what did you do today?", quit.
* **Overlay** (Tkinter, which ships with Python): a small always-on-top window with
  the current status, what was heard, what GARVIS said, and a big STOP button.
* **Neither**: :class:`NullUI` prints the same status lines into the terminal, so
  GARVIS is never silent about what it is doing.

Everything here is defensive. A UI that fails to start logs *why* and gets out of
the way; it must never be the reason the assistant stops working. The UI holds no
authority either: its STOP button calls the same kill switch as Ctrl+Alt+Esc and
"Garvis, stop everything", and it can only *ask* the assistant to do things
through the callbacks main.py hands it.
"""

from __future__ import annotations

import importlib.util
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

def _say(log: Any, level: str, message: str, *args: Any) -> None:
    """Log if that object can.

    The UI is handed whatever logger the caller has: a stdlib logger, an
    ActivityLogger, or None. Anything missing or raising is ignored - a status
    line must never be able to take the assistant down.
    """
    if log is None:
        return
    method = getattr(log, level, None)
    if method is None:
        return
    try:
        method(message, *args)
    except Exception:
        pass


def _module_available(name: str) -> bool:
    """True when a module can be imported, without importing it here."""
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


#: Status values the UI knows how to show.
IDLE = "idle"
LISTENING = "listening"
THINKING = "thinking"
SPEAKING = "speaking"
STOPPED = "stopped"
PAUSED = "paused"

_ICON_COLORS = {
    IDLE: (60, 150, 220),
    LISTENING: (70, 190, 120),
    THINKING: (230, 180, 60),
    SPEAKING: (150, 120, 220),
    STOPPED: (210, 60, 60),
    PAUSED: (140, 140, 140),
}


@dataclass
class UIState:
    """What the UI is showing. Plain data, so any front end can render it."""

    status: str = IDLE
    detail: str = ""
    heard: str = ""
    reply: str = ""
    task: str = ""
    personality: str = ""
    stopper: str = "Ctrl+Alt+Esc"        # the hotkey, shown in the tooltip
    seen: list[str] = field(default_factory=list)

    def render(self) -> str:
        bits = [f"GARVIS: {self.status}"]
        if self.detail:
            bits.append(f"- {self.detail}")
        if self.task:
            bits.append(f"| task: {self.task}")
        if self.modeline:
            bits.append(f"| {self.modeline}")
        return " ".join(bits)

    @property
    def modeline(self) -> str:
        if self.heard and self.reply:
            return f"heard: {self.heard[:40]!r} -> said: {self.reply[:60]!r}"
        if self.heard:
            return f"heard: {self.heard[:60]!r}"
        if self.reply:
            return f"said: {self.reply[:80]!r}"
        return ""


# ---------------------------------------------------------------------------
# base + the always-available fallback
# ---------------------------------------------------------------------------
class BaseUI:
    """Shared bookkeeping: status, transcript, callbacks, subscribers."""

    name = "ui"

    def __init__(
        self,
        cfg: Any = None,
        log: Any = None,
        *,
        on_stop: Callable[[str], None] | None = None,
        on_resume: Callable[[], None] | None = None,
        on_toggle_listen: Callable[[], bool] | None = None,
        on_personality: Callable[[str], bool] | None = None,
        on_today: Callable[[], str] | None = None,
        on_quit: Callable[[], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.log = log
        self.state = UIState()
        self.on_stop = on_stop
        self.on_resume = on_resume
        self.on_toggle_listen = on_toggle_listen
        self.on_personality = on_personality
        self.on_today = on_today
        self.on_quit = on_quit
        self.started = False
        self.available = True
        self.reason = ""
        self._lock = threading.RLock()
        if cfg is not None:
            try:
                self.state.stopper = str(cfg.get("hotkeys.killswitch", "Ctrl+Alt+Esc"))
                self.state.personality = str(cfg.get("brain.personality", ""))
            except Exception:
                pass

    # -- what the rest of the app calls ------------------------------------
    def set_status(self, status: str, detail: str = "") -> None:
        with self._lock:
            self.state.status = status
            self.state.detail = detail
            self.state.seen.append(f"{time.strftime('%H:%M:%S')} {status} {detail}".strip())
            del self.state.seen[:-20]
        self._publish()

    def set_transcript(self, heard: str = "", reply: str = "") -> None:
        with self._lock:
            if heard:
                self.state.heard = heard[:200]
            if reply:
                self.state.reply = reply[:400]
        self._publish()

    def set_task(self, task: str) -> None:
        with self._lock:
            self.state.task = task[:200]
        self._publish()

    def set_personality(self, name: str) -> None:
        with self._lock:
            self.state.personality = name
        self._publish()

    def notify(self, text: str) -> None:
        """A short out-of-band remark (confirmations, denials, handovers)."""
        self._publish(extra=text)

    def describe(self) -> str:
        if not self.available:
            return f"{self.name} unavailable: {self.reason}"
        return f"{self.name} active ({self.state.status})"

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> bool:
        self.started = True
        return True

    def stop(self) -> None:
        self.started = False

    # -- hooks for subclasses ---------------------------------------------
    def _publish(self, extra: str = "") -> None:
        """Subclasses render here. Never raise."""

    def _safe(self, callback: Callable[..., Any], *args: Any) -> Any:
        if callback is None:
            return None
        try:
            return callback(*args)
        except Exception as exc:
            _say(self.log, "debug", "UI callback failed: %s", exc)
            return None


class NullUI(BaseUI):
    """No tray, no window: the same status information, on standard output."""

    name = "terminal status"

    def __init__(self, cfg: Any = None, log: Any = None, loud: bool = False, **kwargs: Any) -> None:
        super().__init__(cfg, log, **kwargs)
        self.loud = loud             # False keeps the terminal clean in voice mode

    def _publish(self, extra: str = "") -> None:
        if not self.loud:
            return
        text = f"  [{self.state.render()}]"
        if self.state.heard:
            text += f"\n  [heard: {self.state.heard[:120]}]"
        if extra:
            text += f"\n  [{extra}]"
        print(text, flush=True)


class FakeUI(BaseUI):
    """Test double: records everything that would have been displayed."""

    name = "fake"

    def __init__(self, cfg: Any = None, log: Any = None, **kwargs: Any) -> None:
        super().__init__(cfg, log, **kwargs)
        self.updates: list[dict[str, Any]] = []
        self.notices: list[str] = []

    def _publish(self, extra: str = "") -> None:
        self.updates.append({
            "status": self.state.status, "detail": self.state.detail,
            "heard": self.state.heard, "reply": self.state.reply,
            "task": self.state.task, "extra": extra,
        })

    def notify(self, text: str) -> None:
        self.notices.append(text)
        super().notify(text)


# ---------------------------------------------------------------------------
# overlay: Tkinter (stdlib)
# ---------------------------------------------------------------------------
class OverlayUI(BaseUI):
    """A small always-on-top window with the status and a STOP button."""

    name = "overlay"

    def __init__(self, cfg: Any = None, log: Any = None, root: Any = None, **kwargs: Any) -> None:
        super().__init__(cfg, log, **kwargs)
        self._root = root                       # injectable for tests
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._widgets: dict[str, Any] = {}
        self.show_transcript = True
        self.position = "bottom-right"
        self.opacity = 0.92
        self.always_on_top = True
        self.confirm_stop = False
        self._pause_text = "Pause listening"
        if cfg is not None:
            try:
                self.show_transcript = bool(cfg.get("ui.show_transcript", True))
                self.position = str(cfg.get("ui.overlay_position", "bottom-right"))
                self.opacity = float(cfg.get("ui.overlay_opacity", 0.92) or 0.92)
                self.always_on_top = bool(cfg.get("ui.overlay_always_on_top", True))
                self.confirm_stop = bool(cfg.get("ui.stop_button_confirm", False))
            except Exception:
                pass
        if self._root is None:
            ok, reason = self._tk_available()
            if not ok:
                self.available = False
                self.reason = reason

    @staticmethod
    def _tk_available() -> tuple[bool, str]:
        if not _module_available("tkinter"):
            return False, "tkinter is not available; on Linux install python3-tk"
        import os
        import sys

        if sys.platform not in ("darwin", "win32") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            return False, "no display is reachable (DISPLAY/WAYLAND_DISPLAY are unset)"
        return True, ""

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> bool:
        if not self.available:
            return False
        if self.started:
            return True
        self.started = True
        self._thread = threading.Thread(target=self._run, name="garvis-overlay", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self.started = False
        try:
            self._queue.put(("quit", None))
        except Exception:
            pass

    def _publish(self, extra: str = "") -> None:
        if not self.started:
            return
        try:
            self._queue.put(("update", None))
        except Exception:
            pass
        if extra:
            try:
                self._queue.put(("notice", extra))
            except Exception:
                pass

    # -- the window --------------------------------------------------------
    def _run(self) -> None:
        try:
            import tkinter as tk
        except Exception as exc:
            self.available = False
            self.reason = f"tkinter is not available ({exc})"
            return
        try:
            root = self._root if self._root is not None else tk.Tk()
            root.title("GARVIS")
            root.attributes("-topmost", bool(self.always_on_top))
            try:
                root.attributes("-alpha", self.opacity)
            except Exception:
                pass
            root.resizable(False, False)
            root.protocol("WM_DELETE_WINDOW", self._on_close)

            status = tk.Label(root, text="GARVIS: idle", font=("Segoe UI", 11, "bold"),
                              anchor="w", justify="left", fg="#dbe6f2", bg="#1b1f27")
            status.pack(fill="x", padx=10, pady=(8, 2))
            heard = tk.Label(root, text="", font=("Segoe UI", 9), anchor="w",
                             justify="left", wraplength=360, fg="#9fb3c8", bg="#1b1f27")
            heard.pack(fill="x", padx=10)
            reply = tk.Label(root, text="", font=("Segoe UI", 9), anchor="w",
                             justify="left", wraplength=360, fg="#cfe3d3", bg="#1b1f27")
            reply.pack(fill="x", padx=10, pady=(0, 6))
            button_row = tk.Frame(root, bg="#1b1f27")
            button_row.pack(fill="x", padx=10, pady=(0, 10))
            stop = tk.Button(button_row, text="STOP", bg="#c0392b", fg="white",
                             activebackground="#e74c3c", relief="flat", padx=14, pady=6,
                             command=self._on_stop)
            stop.pack(side="left")
            pause = tk.Button(button_row, text=self._pause_text, bg="#2c3440", fg="#dbe6f2",
                              activebackground="#3b4552", relief="flat", padx=10, pady=6,
                              command=self._on_pause)
            pause.pack(side="left", padx=6)
            root.configure(bg="#1b1f27")
            self._widgets = {"root": root, "status": status, "heard": heard, "reply": reply,
                             "pause": pause}
            self._place(root)
            root.after(120, self._drain)
            root.mainloop()
        except Exception as exc:                       # a broken window must not matter
            self.available = False
            self.reason = f"the overlay failed to start: {exc}"
            _say(self.log, "warning", "overlay unavailable: %s", exc)

    def _place(self, root: Any) -> None:
        """Park the window in the configured corner."""
        try:
            root.update_idletasks()
            width, height = root.winfo_reqwidth(), root.winfo_reqheight()
            screen_w = root.winfo_screenwidth()
            screen_h = root.winfo_screenheight()
            margin = 16
            spots = {
                "top-left": (margin, margin),
                "top-right": (screen_w - width - margin, margin),
                "bottom-left": (margin, screen_h - height - 60),
                "bottom-right": (screen_w - width - margin, screen_h - height - 60),
            }
            x, y = spots.get(self.position, spots["bottom-right"])
            root.geometry(f"+{max(0, x)}+{max(0, y)}")
        except Exception:
            pass

    def _drain(self) -> None:
        """Poll the update queue from the UI thread (tkinter is not thread-safe)."""
        root = self._widgets.get("root")
        if root is None:
            return
        while True:
            try:
                action, payload = self._queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break
            if action == "quit":
                try:
                    root.destroy()
                except Exception:
                    pass
                return
            if action == "update":
                self._render()
            elif action == "notice":
                self._render(notice=str(payload))
        if self.started:
            try:
                root.after(150, self._drain)
            except Exception:
                pass

    def _render(self, notice: str = "") -> None:
        try:
            status = self._widgets["status"]
            heard = self._widgets["heard"]
            reply = self._widgets["reply"]
            pause = self._widgets["pause"]
            status.configure(text=self.state.render() + (f"  [{notice}]" if notice else ""))
            if self.show_transcript:
                heard.configure(text=(f"heard: {self.state.heard[:160]}" if self.state.heard else ""))
                reply.configure(text=(f"said: {self.state.reply[:240]}" if self.state.reply else ""))
            pause.configure(text="Resume listening" if self.state.status == PAUSED else "Pause listening")
        except Exception:
            pass

    # -- buttons -----------------------------------------------------------
    def _on_stop(self) -> None:
        self._safe(self.on_stop, "overlay STOP button")

    def _on_pause(self) -> None:
        """Pause/resume the microphone. The callback returns True when listening."""
        result = self._safe(self.on_toggle_listen)
        if result is None:
            self.set_status(PAUSED, "listening is not available in this mode")
        else:
            self.set_status(IDLE if result else PAUSED, "from the overlay")

    def _on_close(self) -> None:
        # Closing the overlay hides the window; it does not stop the assistant.
        self.started = False
        try:
            self._widgets["root"].destroy()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# tray: pystray + Pillow
# ---------------------------------------------------------------------------
class TrayUI(BaseUI):
    """System-tray icon with the essentials: STOP, resume, pause, personality."""

    name = "tray"

    def __init__(self, cfg: Any = None, log: Any = None, icon_factory: Any = None, **kwargs: Any) -> None:
        super().__init__(cfg, log, **kwargs)
        self._icon = None
        self._icon_factory = icon_factory
        self._thread: threading.Thread | None = None
        self.personalities = ("standard", "sassy", "formal", "hyped", "focus", "chill")
        if self._icon_factory is None:
            ok, reason = self._deps_available()
            if not ok:
                self.available = False
                self.reason = reason

    @staticmethod
    def _deps_available() -> tuple[bool, str]:
        if not _module_available("pystray"):
            return False, "pystray is not installed; pip install pystray pillow"
        if not _module_available("PIL"):
            return False, "Pillow is not installed for the icon; pip install pillow"
        return True, ""

    # -- icon drawing ------------------------------------------------------
    def _make_icon(self, status: str = IDLE) -> Any:
        if self._icon_factory is not None:
            return self._icon_factory(status)
        from PIL import Image, ImageDraw

        color = _ICON_COLORS.get(status, _ICON_COLORS[IDLE])
        image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.ellipse((4, 4, 60, 60), fill=color)
        draw.polygon([(26, 22), (26, 42), (44, 32)], fill=(20, 22, 26))   # a small "play" glyph
        return image

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> bool:
        if not self.available:
            return False
        if self.started:
            return True
        try:
            import pystray
        except Exception as exc:
            self.available = False
            self.reason = f"pystray is not installed ({exc})"
            return False
        try:
            icon = pystray.Icon("garvis", self._make_icon(), self._tooltip(), self._menu())
            self._icon = icon
            self._thread = threading.Thread(target=self._run, name="garvis-tray", daemon=True)
            self.started = True
            self._thread.start()
            return True
        except Exception as exc:
            self.available = False
            self.reason = f"the tray icon could not be created: {exc}"
            _say(self.log, "warning", "tray unavailable: %s", exc)
            return False

    def _run(self) -> None:
        try:
            self._icon.run()          # blocks until the icon is stopped
        except Exception as exc:
            self.available = False
            self.reason = f"the tray icon stopped: {exc}"
            _say(self.log, "debug", "tray run ended: %s", exc)

    def stop(self) -> None:
        self.started = False
        try:
            if self._icon is not None:
                self._icon.stop()
        except Exception:
            pass

    def _tooltip(self) -> str:
        return f"GARVIS - {self.state.status} (stop: {self.state.stopper})"

    def _publish(self, extra: str = "") -> None:
        icon = self._icon
        if icon is None:
            return
        try:
            icon.title = self._tooltip() + (f" - {extra}" if extra else "")
            if self.state.status in _ICON_COLORS:
                try:
                    icon.icon = self._make_icon(self.state.status)
                except Exception:
                    pass
            icon.update_menu()
        except Exception:
            pass

    # -- menu --------------------------------------------------------------
    def _menu(self) -> Any:
        import pystray

        def item(text: str, action: Callable[[], Any], **kwargs: Any) -> Any:
            return pystray.MenuItem(text, lambda *_: self._safe(action), **kwargs)

        personality_items = [
            pystray.MenuItem(
                name.title(),
                (lambda name=name: lambda *_: self._choose_personality(name)),
            )
            for name in self.personalities
        ]
        return pystray.Menu(
            pystray.MenuItem(lambda *_: self.state.render(), lambda *_: None, enabled=False),
            pystray.Menu.SEPARATOR,
            item("STOP everything", lambda: self.on_stop and self.on_stop("tray STOP button"),
                 default=True),
            item("Resume", self._resume),
            item("Pause / resume listening", self._toggle_listen),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Personality", pystray.Menu(*personality_items)),
            item("What did you do today?", self._show_today),
            pystray.Menu.SEPARATOR,
            item("Quit GARVIS", self._quit),
        )

    def _resume(self) -> None:
        self._safe(self.on_resume)
        self.set_status(IDLE, "resumed from the tray")

    def _toggle_listen(self) -> None:
        result = self._safe(self.on_toggle_listen)
        if result is None:
            self.set_status(PAUSED, "listening is not available in this mode")
        else:
            self.set_status(IDLE if result else PAUSED, "from the tray")

    def _choose_personality(self, name: str) -> None:
        ok = self._safe(self.on_personality, name)
        if ok:
            self.set_personality(name)
            self.notify(f"Personality: {name}")

    def _show_today(self) -> None:
        report = self._safe(self.on_today) or "nothing logged yet"
        _say(self.log, "info", "today (from the tray):\n%s", report)
        self.notify(report.splitlines()[0] if report else "nothing logged yet")

    def _quit(self) -> None:
        self._safe(self.on_quit)
        self.stop()


# ---------------------------------------------------------------------------
# the service main.py talks to
# ---------------------------------------------------------------------------
class UIManager(BaseUI):
    """Picks the front ends that work here and keeps them all in sync."""

    name = "ui"

    def __init__(
        self,
        cfg: Any = None,
        log: Any = None,
        activity: Any = None,
        tray: BaseUI | None = None,
        overlay: BaseUI | None = None,
        subscribe: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(cfg, log, **kwargs)
        self.activity = activity
        self.enabled = bool(cfg.get("ui.enabled", True)) if cfg is not None else True
        self.want_tray = bool(cfg.get("ui.tray", True)) if cfg is not None else True
        self.want_overlay = bool(cfg.get("ui.overlay", True)) if cfg is not None else True

        self.tray = tray
        self.overlay = overlay
        self.reasons: list[str] = []
        if not self.enabled:
            # ui.enabled: false means *nothing* is displayed, injected or not.
            self.tray = self.overlay = None
            self.children = []
            self.available = False
            self.reason = "disabled in config.yaml (ui.enabled: false)"
            self.fallback = None
            return
        if self.tray is None and self.want_tray:
            self.tray = TrayUI(cfg, log, **kwargs)
        if self.overlay is None and self.want_overlay:
            self.overlay = OverlayUI(cfg, log, **kwargs)
        for candidate in (self.tray, self.overlay):
            if candidate is None:
                continue
            if not candidate.available and candidate.reason:
                self.reasons.append(f"{candidate.name}: {candidate.reason}")

        self.children = [child for child in (self.tray, self.overlay) if child and child.available]
        self.available = bool(self.children)
        if not self.children:
            self.reason = "; ".join(self.reasons) or "no front end is available"
            self.fallback = NullUI(cfg, log, loud=bool(cfg.get("ui.terminal_status", True)) if cfg else True,
                                  **kwargs)
            self.children = [self.fallback]
        else:
            self.fallback = None
        self._subscribed = False
        if subscribe and activity is not None and hasattr(activity, "subscribe"):
            try:
                activity.subscribe(self.observe)
                self._subscribed = True
            except Exception:
                pass

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        started = []
        for child in self.children:
            try:
                if child.start():
                    started.append(child.name)
            except Exception as exc:
                self.reasons.append(f"{child.name}: {exc}")
                _say(self.log, "warning", "%s failed to start: %s", child.name, exc)
        self.started = bool(started)
        self.available = bool(started)
        if self.reasons:
            for reason in self.reasons:
                _say(self.log, "info", "ui: %s", reason)
        return

    def stop(self) -> None:
        for child in self.children:
            try:
                child.stop()
            except Exception:
                pass
        self.started = False

    close = stop
    shutdown = stop

    # -- fan-out -----------------------------------------------------------
    def _publish(self, extra: str = "") -> None:
        for child in self.children:
            try:
                child.state = self.state
                child._publish(extra)
            except Exception:
                continue

    def set_status(self, status: str, detail: str = "") -> None:
        super().set_status(status, detail)

    def notify(self, text: str) -> None:
        for child in self.children:
            try:
                child.notify(text)
            except Exception:
                continue
        if self.fallback is not None:
            try:
                self.fallback.notify(text)
            except Exception:
                pass

    # -- activity subscriber ----------------------------------------------
    def observe(self, record: dict[str, Any]) -> None:
        """Keep the display honest by following the real activity log."""
        try:
            kind = str(record.get("kind") or "")
            message = str(record.get("message") or "")
            if kind == "user":
                self.set_transcript(heard=message[:200])
                self.set_status(LISTENING, "heard you")
            elif kind == "assistant":
                self.set_transcript(reply=message[:400])
                self.set_status(IDLE, "answered")
            elif kind == "tool":
                tool = str(record.get("tool") or "tool")
                ok = record.get("ok")
                self.set_status(THINKING, f"{tool} {'ok' if ok else 'failed'}")
            elif kind == "permission":
                decision = str(record.get("decision") or "")
                if decision:
                    self.notify(f"permission {decision}: {record.get('tool', '')}")
            elif kind == "state":
                self.set_task(message[:200])
        except Exception:
            pass

    # -- introspection -----------------------------------------------------
    def describe(self) -> str:
        if not self.available:
            return f"ui: none ({self.reason})"
        names = ", ".join(child.name for child in self.children)
        return f"ui: {names} ({self.state.status})"

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "front_ends": [child.name for child in self.children],
            "available": self.available,
            "state": self.state.render(),
            "status": self.state.status,
            "heard": self.state.heard,
            "reply": self.state.reply,
            "task": self.state.task,
            "personality": self.state.personality,
            "reasons": self.reasons,
            "stopper": self.state.stopper,
        }


def build_ui(cfg: Any, log: Any = None, activity: Any = None, **kwargs: Any) -> UIManager:
    """Factory used by main.py: never raises, always returns something usable."""
    try:
        return UIManager(cfg, log=log, activity=activity, **kwargs)
    except Exception as exc:
        if log:
            log.warning("ui failed to build (%s); falling back to terminal status", exc)
        fallback = UIManager.__new__(UIManager)
        BaseUI.__init__(fallback, cfg, log, **kwargs)
        fallback.enabled = False
        fallback.available = False
        fallback.reason = str(exc)
        fallback.reasons = [str(exc)]
        fallback.activity = activity
        fallback.children = [NullUI(cfg, log, loud=True, **kwargs)]
        fallback.tray = fallback.overlay = None
        fallback.fallback = fallback.children[0]
        return fallback
