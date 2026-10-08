"""Application control: list, open and close programs.

Cross-platform by design, with honest limits:

* **Windows** - ``os.startfile`` for documents/URLs, ``subprocess`` with
  ``CREATE_NEW_PROCESS_GROUP`` for executables, ``tasklist``/``taskkill`` for
  listing and closing.
* **macOS** - ``open -a`` / ``open``, ``ps`` + ``osascript quit``.
* **Linux** - ``gtk-launch`` then ``xdg-open`` then a direct exec; ``ps`` for
  listing, SIGTERM then SIGKILL for closing.

Allowlist rule: ``apps.open`` will only launch an executable whose *name* is in
``apps.allowlist`` (config.yaml). A file path or a URL is opened with the OS
default handler, which is YELLOW - the gate asks first, and the path/URL checks
still apply. ``apps.close`` is RED by default in ``config.yaml`` (it can kill
unsaved work), and it refuses to kill GARVIS itself, the shell, or anything that
looks like a system process.

None of this can do anything the permission gate has not approved: every tool
here declares its executable argument so the gate can check it.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .base import GREEN, RED, ToolRegistry, ToolResult, YELLOW

CATEGORY = "apps"
IS_WINDOWS = os.name == "nt"
IS_MAC = sys.platform == "darwin"

#: Processes we refuse to kill regardless of approval: killing them breaks the
#: session, loses work, or takes the machine down.
NEVER_KILL = {
    "system", "systemd", "init", "kernel_task", "launchd", "windowserver",
    "explorer", "winlogon", "csrss", "smss", "services", "lsass", "dwm",
    "python", "python3", "pythonw", "garvis", "main.py",
    "bash", "zsh", "sh", "pwsh", "powershell", "cmd", "conhost", "terminal",
    "gnome-shell", "plasmashell", "xfwm4", "kwin", "cinnamon", "taskbar",
    "sshd", "loginwindow", "finder",
}


def _run(command: list[str], timeout: float = 15.0) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603 - argv lists only, never shell=True
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        errors="replace",
    )


def _list_processes() -> list[tuple[int, str]]:
    """Return [(pid, name)] using the platform's own tool."""
    if IS_WINDOWS:
        completed = _run(["tasklist", "/fo", "csv", "/nh"])
        found: list[tuple[int, str]] = []
        for line in completed.stdout.splitlines():
            parts = [p.strip('"') for p in line.split('","')]
            if len(parts) >= 2:
                try:
                    found.append((int(parts[1]), parts[0]))
                except ValueError:
                    continue
        return found
    completed = _run(["ps", "-eo", "pid=,comm="])
    found = []
    for line in completed.stdout.splitlines():
        bits = line.strip().split(None, 1)
        if len(bits) == 2:
            try:
                found.append((int(bits[0]), os.path.basename(bits[1].strip())))
            except ValueError:
                continue
    return found


def register(registry: ToolRegistry, cfg: Any, log: Any = None, services: dict[str, Any] | None = None) -> None:
    services = services if services is not None else {}
    allowlist = [str(a).strip().lower() for a in (cfg.get("apps.allowlist", []) or [])]
    close_tier = str(cfg.get("apps.close_tier", "red")).lower()
    if close_tier not in ("yellow", "red"):
        close_tier = RED
    self_names = {"garvis", "main.py", "python", "python3", "pythonw"}
    # apps.protected EXTENDS the built-in list; it cannot shrink it. Configuring
    # `protected: []` does not make sshd or explorer killable - the list above
    # is the last word on the programs that keep the session (and the machine)
    # alive.
    protected = NEVER_KILL | {str(n).strip().lower() for n in (cfg.get("apps.protected", []) or []) if str(n).strip()}

    def _safe_to_kill(name: str) -> tuple[bool, str]:
        base = os.path.basename(str(name)).lower()
        if base.endswith(".exe"):
            base = base[:-4]
        if not base:
            return False, "no process name given"
        if base in protected or base in self_names:
            return False, f"'{base}' is protected: killing it would break your session or GARVIS itself"
        return True, ""

    def _guard_close(values: dict[str, Any]) -> tuple[str | None, str]:
        """Gate guard: refuse protected processes before any confirmation is asked."""
        safe, why = _safe_to_kill(str(values.get("name", "")))
        if not safe:
            return "blocked", why
        return None, ""

    # ---------------------------------------------------------------- listing
    @registry.tool(
        name="apps.list",
        description=(
            "List running programs (name and process id). Read-only. Use it to find out what is "
            "running before opening or closing something."
        ),
        parameters={
            "type": "object",
            "properties": {
                "filter": {"type": "string", "description": "Only show names containing this text."},
                "limit": {"type": "integer", "description": "Maximum rows (default 40)."},
            },
            "required": [],
        },
        tier=GREEN,
        category=CATEGORY,
        readonly=True,
        example="apps.list(filter='chrome')",
    )
    def apps_list(filter: str = "", limit: int = 40) -> ToolResult:
        try:
            processes = _list_processes()
        except Exception as exc:
            return ToolResult.failure(f"Could not list processes: {exc}")
        needle = str(filter or "").lower()
        if needle:
            processes = [p for p in processes if needle in p[1].lower()]
        limit = max(1, min(int(limit or 40), 300))
        # Collapse duplicates by name, keep the lowest pid as representative.
        seen: dict[str, list[int]] = {}
        for pid, name in processes:
            seen.setdefault(name, []).append(pid)
        rows = [f"{name}  (pids: {', '.join(str(p) for p in sorted(pids)[:5])})" for name, pids in sorted(seen.items())]
        if not rows:
            return ToolResult.success(f"No processes matched '{filter}'.")
        return ToolResult.success("\n".join(rows[:limit]), display=f"{len(rows)} program(s)")

    @registry.tool(
        name="apps.allowlist",
        description=(
            "Show which programs GARVIS is allowed to launch, and the current rules for opening and "
            "closing apps. Read-only. Useful when apps.open refuses something."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        tier=GREEN,
        category=CATEGORY,
        readonly=True,
    )
    def apps_allowlist() -> ToolResult:
        lines = [
            "launchable (apps.allowlist): " + (", ".join(allowlist) if allowlist else "(none - opening is refused)"),
            f"closing apps requires: {close_tier.upper()} approval",
            "protected from closing: " + ", ".join(sorted(protected)[:20]) + ", ...",
            f"platform: {'windows' if IS_WINDOWS else 'macos' if IS_MAC else 'linux'}",
        ]
        return ToolResult.success("\n".join(lines), display="app rules")

    # ----------------------------------------------------------------- opening
    @registry.tool(
        name="apps.open",
        description=(
            "Open a program, file, folder or URL with the operating system. Programs must be in "
            "apps.allowlist; files and URLs must be inside the allowlists too. The user approves "
            "this before it runs. On Linux/macOS, launching is 'fire and forget': we cannot see the "
            "new window, so we report the launch attempt, not the app's state."
        ),
        parameters={
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Program name (e.g. 'notepad', 'code'), file/folder path, or URL.",
                },
                "args": {
                    "type": "string",
                    "description": "Optional extra arguments, space separated (programs only).",
                },
            },
            "required": ["target"],
        },
        tier=YELLOW,
        category=CATEGORY,
        path_args=("target",),
        example="apps.open(target='code', args='notes/')",
    )
    def apps_open(target: str, args: str = "") -> ToolResult:
        name = str(target or "").strip()
        if not name:
            return ToolResult.failure("No target given.")

        # A URL: hand it to the OS handler.
        if name.lower().startswith(("http://", "https://")):
            try:
                if IS_WINDOWS:
                    os.startfile(name)  # noqa: S606
                elif IS_MAC:
                    _run(["open", name])
                else:
                    _run(["xdg-open", name])
            except Exception as exc:
                return ToolResult.failure(f"Could not open the URL: {exc}")
            return ToolResult.success(
                f"Handed {name} to your default browser. I cannot see the page from here - "
                f"use browser.* tools if you want me to read it.",
                display=f"opened URL {name[:60]}",
            )

        path = Path(os.path.expandvars(os.path.expanduser(name)))
        if path.exists():
            try:
                if IS_WINDOWS:
                    os.startfile(str(path))  # noqa: S606
                elif IS_MAC:
                    _run(["open", str(path)])
                else:
                    _run(["xdg-open", str(path)])
            except Exception as exc:
                return ToolResult.failure(f"Could not open {path}: {exc}")
            return ToolResult.success(f"Opened {path} with the default application.")

        # Otherwise: treat it as a program name and require the allowlist.
        base = os.path.basename(name).lower()
        if base.endswith(".exe"):
            base = base[:-4]
        if not allowlist:
            return ToolResult.failure(
                "apps.allowlist is empty, so I am not launching anything. Add the program name "
                "to apps.allowlist in config.yaml."
            )
        if base not in allowlist:
            return ToolResult.failure(
                f"'{base}' is not in apps.allowlist. Add it to config.yaml if you want me to "
                f"launch it, or start it yourself."
            )
        executable = shutil.which(name) or name
        argv = [executable] + (args.split() if args else [])
        try:
            if IS_WINDOWS:
                subprocess.Popen(argv, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)  # noqa: S603
            else:
                subprocess.Popen(argv, start_new_session=True, stdout=subprocess.DEVNULL,  # noqa: S603
                                 stderr=subprocess.DEVNULL)
        except Exception as exc:
            return ToolResult.failure(f"Could not launch {name}: {exc}")

        time.sleep(0.6)  # give it a moment so a crash-on-start is visible
        running = any(base in proc_name.lower() for _, proc_name in _list_processes())
        if running:
            return ToolResult.success(f"Launched {name}. Verified: it is in the process list.")
        return ToolResult.failure(
            f"I ran the launch command for {name} but it does not appear in the process list. "
            f"It may have failed on startup, or the process name differs from '{base}'."
        )

    # ----------------------------------------------------------------- closing
    @registry.tool(
        name="apps.close",
        description=(
            "Close a running program by name. Unsaved work is lost, so this needs RED approval: "
            "repeat the exact action, then say confirm. Protected processes (GARVIS itself, shells, "
            "system UI) are refused outright."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Process name, e.g. 'notepad'."},
                "force": {"type": "boolean", "description": "Force-kill instead of asking nicely."},
            },
            "required": ["name"],
        },
        tier=close_tier,
        category=CATEGORY,
        spoken_action=lambda a: f"close {a.get('name', '')}",
        guard=_guard_close,
        example="apps.close(name='notepad')",
    )
    def apps_close(name: str, force: bool = False) -> ToolResult:
        target = str(name or "").strip()
        if not target:
            return ToolResult.failure("No process name given.")
        safe, why = _safe_to_kill(target)
        if not safe:
            return ToolResult.failure(why)
        base = os.path.basename(target).lower()
        if base.endswith(".exe"):
            base = base[:-4]

        matches = [(pid, pname) for pid, pname in _list_processes() if base in pname.lower()]
        if not matches:
            return ToolResult.success(f"No running process matched '{target}'. Nothing to close.")

        killed: list[int] = []
        refused: list[str] = []
        for pid, pname in matches:
            ok, reason = _safe_to_kill(pname)
            if not ok:
                refused.append(f"{pname} (pid {pid}): {reason}")
                continue
            try:
                if IS_WINDOWS:
                    command = ["taskkill", "/PID", str(pid)] + (["/F"] if force else [])
                    completed = _run(command)
                    if completed.returncode != 0:
                        refused.append(f"{pname} (pid {pid}): {completed.stdout.strip() or completed.stderr.strip()}")
                        continue
                elif IS_MAC and not force:
                    completed = _run(["osascript", "-e", f'tell application "{pname}" to quit'])
                    if completed.returncode != 0:
                        os.kill(pid, signal.SIGTERM)
                else:
                    os.kill(pid, signal.SIGKILL if force else signal.SIGTERM)
                killed.append(pid)
            except Exception as exc:
                refused.append(f"{pname} (pid {pid}): {exc}")

        time.sleep(0.5)
        still_running = {pid for pid, _ in _list_processes()}
        survivors = [pid for pid in killed if pid in still_running]
        parts = []
        if killed:
            parts.append(f"closed {len(killed)} process(es): {', '.join(str(p) for p in killed)}")
        if survivors:
            parts.append(f"still running (they ignored the signal): {survivors}. Try force=true.")
        if refused:
            parts.append("refused:\n  " + "\n  ".join(refused))
        message = "\n".join(parts) or "Nothing was closed."
        if killed and not survivors:
            return ToolResult.success(message, display=f"closed {len(killed)} process(es)")
        if killed and survivors:
            return ToolResult.failure(message)
        return ToolResult.failure(message)

    if log:
        log.info("registered app tools (%d launchable programs)", len(allowlist))
