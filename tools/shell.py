"""Shell tools: run a command, with an allowlist, a timeout and no escape hatches.

The refusal to be clever here is deliberate. ``shell.run`` is YELLOW, and the
permission gate (core/permissions.py) does the real work:

* the first token of **every** segment must be in ``shell.allowlist``;
* ``shell.blocked_patterns`` are refused outright, no confirmation possible;
* command substitution (``$(...)``, backticks) is refused by default: it is the
  classic way to smuggle an unlisted command past a first-token check;
* output redirection is refused unless the target is inside a write-allowed folder;
* every command runs in its own process group and is registered with the kill
  switch, so "stop everything" terminates a command that is still running;
* the process runs with a timeout, captured output, and a filtered environment -
  no inherited secrets, no interactive prompts, no network tools that are not
  explicitly allowlisted.

If your command needs a shell feature that is refused, that is the design
working. Add the specific executable to ``shell.allowlist`` in config.yaml, or
do the step by hand.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any

from .base import GREEN, ToolRegistry, ToolResult, YELLOW

CATEGORY = "shell"


def register(registry: ToolRegistry, cfg: Any, log: Any = None, services: dict[str, Any] | None = None) -> None:
    services = services if services is not None else {}
    default_timeout = float(cfg.get("shell.timeout_s", 30))
    max_output = int(cfg.get("shell.max_output_chars", 20000))
    default_cwd = cfg.resolve_path(cfg.get("shell.default_cwd", cfg.get("files.sandbox_dir", "sandbox")))
    env_allow = [str(k) for k in (cfg.get("shell.env_allow", []) or [])]

    def build_env() -> dict[str, str]:
        """Only pass through variables the config allows: no inherited secrets."""
        if not env_allow:
            return {}
        env = {key: value for key, value in os.environ.items() if key in env_allow}
        env.setdefault("PATH", os.environ.get("PATH", ""))
        # Keep tools that expect a language environment working.
        env.setdefault("PYTHONIOENCODING", "utf-8")
        return env

    def _end_process(proc: Any) -> None:
        """Kill a process we started (timeout path). The group dies with it."""
        try:
            if sys.platform != "win32" and hasattr(os, "killpg"):
                os.killpg(os.getpgid(proc.pid), 15)
            else:
                proc.terminate()
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.wait(timeout=5)
        except Exception:
            try:
                if sys.platform != "win32" and hasattr(os, "killpg"):
                    os.killpg(os.getpgid(proc.pid), 9)
                else:
                    proc.kill()
            except Exception:
                pass

    def truncate(text: str) -> str:
        if len(text) <= max_output:
            return text
        half = max_output // 2
        return f"{text[:half]}\n... [{len(text) - max_output} chars omitted] ...\n{text[-half:]}"

    @registry.tool(
        name="shell.run",
        description=(
            "Run one command in the GARVIS sandbox and return its output and exit code. "
            "The executable must be on the shell allowlist in config.yaml, and the user must "
            "approve it. Use for quick, local, reversible things: listing, git status, running a "
            "script you just wrote. Do not use it for installs, network access, or anything "
            "destructive - those will be refused."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command line to run."},
                "cwd": {"type": "string", "description": "Working directory (default: sandbox)."},
                "timeout_s": {"type": "number", "description": "Kill after this many seconds."},
                "stdin_text": {"type": "string", "description": "Optional text piped to stdin."},
            },
            "required": ["command"],
        },
        tier=YELLOW,
        category=CATEGORY,
        command_args=("command",),
        path_args=("cwd",),
        spoken_action=lambda a: f"run the command {str(a.get('command', ''))[:100]}",
        example="shell.run(command='git status --short')",
    )
    def shell_run(command: str, cwd: str = "", timeout_s: float = 0.0, stdin_text: str = "") -> ToolResult:
        text = str(command or "").strip()
        if not text:
            return ToolResult.failure("No command given.")

        workdir = cfg.resolve_path(cwd) if cwd else default_cwd
        if not workdir.exists():
            try:
                workdir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                return ToolResult.failure(f"Working directory {workdir} does not exist and cannot be created: {exc}")

        timeout = float(timeout_s) if timeout_s else default_timeout
        needs_shell = any(token in text for token in ("|", ">", "<", "&", ";", "*", "?"))
        argv: list[str] | str
        if needs_shell:
            argv = text
        else:
            try:
                argv = shlex.split(text, posix=sys.platform != "win32")
            except ValueError as exc:
                return ToolResult.failure(f"Could not parse the command: {exc}")

        started = time.perf_counter()
        # Popen (not subprocess.run) so the command can be registered with the
        # kill switch: "stop everything" must actually kill a running command,
        # not just refuse to run the next one. start_new_session puts the shell
        # and everything it spawns in one process group, which is what makes
        # that termination complete.
        try:
            proc = subprocess.Popen(  # noqa: S602 - shell use is gated above
                argv,
                shell=needs_shell,
                cwd=str(workdir),
                env=build_env(),
                stdin=subprocess.PIPE if stdin_text else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                start_new_session=(sys.platform != "win32"),
            )
        except FileNotFoundError as exc:
            return ToolResult.failure(f"Executable not found: {exc}")
        except OSError as exc:
            return ToolResult.failure(f"Could not run the command: {exc}")

        switch = services.get("killswitch")
        token = switch.register_process(proc, f"shell.run: {text[:60]}") if switch is not None else None
        timed_out = False
        try:
            try:
                out_text, err_text = proc.communicate(input=(stdin_text or None), timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _end_process(proc)
                try:
                    out_text, err_text = proc.communicate(timeout=5)
                except Exception:
                    out_text, err_text = "", ""
            except Exception as exc:                      # never leave a process behind
                _end_process(proc)
                return ToolResult.failure(f"Could not run the command: {exc}")
        finally:
            if switch is not None and token is not None:
                switch.unregister_process(token)

        elapsed = (time.perf_counter() - started) * 1000
        stopped = switch is not None and getattr(switch, "frozen", False)
        if timed_out and not stopped:
            return ToolResult.failure(
                f"Command timed out after {timeout:.0f}s and was killed: {text}"
            )
        if stopped and (timed_out or proc.returncode != 0):
            # Say what actually happened, and by whose order.
            return ToolResult.failure(
                f"Stopped by the kill switch before it finished (exit code {proc.returncode}): "
                f"{text}"
            )

        stdout = truncate(out_text or "")
        stderr = truncate(err_text or "")
        parts = [f"$ {text}", f"exit code: {proc.returncode}", f"duration: {elapsed:.0f} ms"]
        if stdout.strip():
            parts.append("--- stdout ---\n" + stdout.rstrip())
        if stderr.strip():
            parts.append("--- stderr ---\n" + stderr.rstrip())
        if not stdout.strip() and not stderr.strip():
            parts.append("(no output)")

        body = "\n".join(parts)
        if proc.returncode != 0:
            # Honest reporting: a non-zero exit is a failure, not a success with notes.
            return ToolResult.failure(
                f"Command exited with code {proc.returncode}.",
                content=body,
            )
        return ToolResult.success(body, display=f"ran: {text[:60]} (exit 0)")

    @registry.tool(
        name="shell.which",
        description=(
            "Check whether an executable exists on this machine, and where. Read-only. Use it "
            "before proposing a command, so you do not suggest something that is not installed."
        ),
        parameters={
            "type": "object",
            "properties": {"program": {"type": "string", "description": "Executable name, e.g. 'ffmpeg'."}},
            "required": ["program"],
        },
        tier=GREEN,
        category=CATEGORY,
        readonly=True,
        example="shell.which(program='git')",
    )
    def shell_which(program: str) -> ToolResult:
        found = shutil.which(str(program).strip())
        if found:
            return ToolResult.success(f"{program} -> {found}")
        allowlist = cfg.shell_allowlist()
        if str(program).strip() not in allowlist:
            return ToolResult.success(
                f"{program} was not found on PATH. (It is also not in shell.allowlist, so it could "
                f"not be run even if it existed.)"
            )
        return ToolResult.success(f"{program} not found on PATH.")

    @registry.tool(
        name="shell.info",
        description=(
            "Report the shell sandbox rules: which executables are allowed, what is blocked, the "
            "working directory and timeout. Read-only. Useful when a command was refused."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        tier=GREEN,
        category=CATEGORY,
        readonly=True,
    )
    def shell_info() -> ToolResult:
        lines = [
            f"working directory: {default_cwd}",
            f"timeout: {default_timeout:.0f}s",
            f"require allowlist: {cfg.get('shell.require_allowlist', True)}",
            "allowed executables: " + ", ".join(cfg.shell_allowlist()),
            "blocked patterns:",
        ] + [f"  - {pattern}" for pattern in (cfg.get("shell.blocked_patterns", []) or [])]
        return ToolResult.success("\n".join(lines), display="shell rules")

    if log:
        log.info("registered shell tools (%d allowed executables)", len(cfg.shell_allowlist()))
