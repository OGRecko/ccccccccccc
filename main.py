#!/usr/bin/env python3
"""GARVIS - local voice-controlled assistant.

Main entry point. Stage 1 provides a text chat loop; later stages add voice in,
voice out, tools, browser, screen guidance and the tray UI. Everything is
controlled by config.yaml, and the CLI flags let you run individual pieces
without touching the config.

Usage
-----
    python main.py                     # normal startup (text loop until stage 5)
    python main.py --ask "what time is it"
    python main.py --check             # validate config + environment, then exit
    python main.py --today             # print the activity report for today
    python main.py --show-prompt       # print the assembled system prompt
    python main.py --model qwen2.5:7b  # override the model for this run
    python main.py --personality focus # start in a given tone
    python main.py --no-tools          # plain chat, tools disabled

What "healthy" looks like on startup:
    1. config loads (warnings are printed, not fatal)
    2. logs/ and memory/ exist
    3. Ollama answers, and the model in config.yaml is pulled
Then you get a `you>` prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Make `python main.py` work from any directory, and let core/tools import each other.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.brain import Brain, BrainError, BrainEvent, SentenceChunker  # noqa: E402
from core.config import Config, ConfigError, describe  # noqa: E402
from core.logger import (  # noqa: E402
    ActivityLogger,
    configure_activity_logger,
    get_logger,
    setup_logging,
)
from core.memory import Memory  # noqa: E402

VERSION = "0.1.0-stage1"
BANNER = r"""
   ____    _    ____  __     __ ___ ____
  / ___|  / \  |  _ \ \ \   / /|_ _/ ___|
 | |  _  / _ \ | |_) | \ \ / /  | |\___ \
 | |_| |/ ___ \|  _ <   \ V /   | | ___) |
  \____/_/   \_\_| \_\   \_/   |___|____/
"""


class Garvis:
    """The application object. Later stages attach voice/UI to this."""

    def __init__(self, cfg: Config, args: argparse.Namespace) -> None:
        self.cfg = cfg
        self.args = args
        self.log = get_logger("main")
        self.running = False
        self.started_at = time.time()

        self.activity = self._setup_logging()
        self.memory = Memory.from_config(cfg)
        self.memory.ensure_files()

        self.services: dict[str, Any] = {
            "cfg": cfg,
            "log": self.log,
            "activity": self.activity,
            "memory": self.memory,
            "brain": None,        # filled below
            "killswitch": None,   # stage 8
            "state": None,        # stage 8
            "tts": None,          # stage 5
            "browser": None,      # stage 6
            "ui": None,           # stage 8
        }

        from tools import build_registry

        self.registry = build_registry(cfg, self.log, self.services)
        if args.no_tools:
            self.log.warning("tools disabled by --no-tools; the model cannot act this run")

        self.gate = self._build_gate()

        self.brain = Brain(
            cfg=cfg,
            registry=None if args.no_tools else self.registry,
            gate=None if args.no_tools else self.gate,
            memory=self.memory,
            activity=self.activity,
        )
        self.services["brain"] = self.brain
        if args.personality:
            if not self.brain.set_personality(args.personality):
                self.log.warning("could not switch personality to %r", args.personality)

    # -- wiring ------------------------------------------------------------
    def _setup_logging(self) -> ActivityLogger:
        cfg = self.cfg
        log_path = cfg.resolve_path(cfg.get("logging.file", "logs/garvis.log"))
        setup_logging(
            log_path,
            level=cfg.get("logging.level", "INFO"),
            console=cfg.get("logging.console", True),
            redact_patterns=cfg.get("logging.redact_patterns", []),
            console_style=cfg.get("logging.console_style", "compact"),
        )
        return configure_activity_logger(cfg)

    def _build_gate(self) -> Any | None:
        """The permission gate is mandatory. If it is missing, we fail closed:
        the brain will refuse every tool call rather than run unguarded."""
        try:
            from core.permissions import PermissionGate  # noqa: PLC0415
        except ImportError as exc:
            self.log.error(
                "core/permissions.py is unavailable (%s). Running with the gate DISABLED: "
                "every tool call will be refused for safety.", exc,
            )
            return None
        gate = PermissionGate(
            cfg=self.cfg,
            activity=self.activity,
            services=self.services,
            log=self.log,
        )
        self.services["gate"] = gate
        return gate

    # -- startup -----------------------------------------------------------
    def preflight(self, quiet: bool = False) -> bool:
        """Check the basics before we pretend to be online. Returns True if usable."""
        problems: list[str] = []
        for warning in self.cfg.warnings:
            self.log.warning("config: %s", warning)

        if not quiet:
            print(BANNER)
            print(f"  GARVIS {VERSION} - local assistant. Ctrl+C or /quit to exit.\n")

        if self.cfg.cloud_fallback_enabled:
            print("  !! Cloud fallback is ENABLED in config.yaml: prompts may leave this machine.\n")

        if not self.brain.client.is_up():
            problems.append(
                f"Ollama is not answering at {self.cfg.ollama_host}.\n"
                f"     Fix: run `ollama serve` in another terminal."
            )
        else:
            version = self.brain.client.version()
            if not self.brain.client.has_model(self.cfg.model):
                problems.append(
                    f"Model '{self.cfg.model}' is not installed.\n"
                    f"     Fix: run `ollama pull {self.cfg.model}` "
                    f"(see `ollama list` for what you have)."
                )
            else:
                self.log.info("ollama %s ready, model %s", version or "?", self.cfg.model)

        if problems:
            print("Startup checks failed:\n")
            for problem in problems:
                print(f"  - {problem}")
            print()
            return False

        # Tool inventory is useful context, and proves registration worked.
        counts: dict[str, int] = {}
        for tool in self.registry.all():
            counts[tool.tier] = counts.get(tool.tier, 0) + 1
        self.log.info(
            "tools: %d (%s)", len(self.registry),
            ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none",
        )
        if not self.args.no_tools:
            self.log.info("permission gate: %s", "active" if self.gate else "DISABLED (fail-closed)")
        return True

    def say(self, text: str) -> None:
        """Print (stage 1) or speak (stage 5). Kept in one place for that reason."""
        print(f"\n{self.cfg.assistant_name}: {text}\n")

    # -- one turn ----------------------------------------------------------
    def handle_line(self, line: str, chunker: SentenceChunker | None = None) -> bool:
        """Process one user line. Returns False when the user wants to quit."""
        text = line.strip()
        if not text:
            return True

        if text.startswith("/"):
            return self._handle_command(text)

        self.activity.event("user", text)
        streamed: list[str] = []

        def on_event(event: BrainEvent) -> None:
            if event.kind == "delta":
                streamed.append(event.text)
                print(event.text, end="", flush=True)
                # Stage 5 speaks these chunks; printing them proves the chunker
                # splits sentences the way the voice engine will.
                if chunker is not None:
                    for sentence in chunker.feed(event.text):
                        self.log.debug("speakable chunk: %s", sentence)
            elif event.kind == "tool_call":
                print(f"\n  [tool] {event.text}({_short(event.data.get('args'))})", flush=True)
            elif event.kind == "tool_result":
                ok = event.data.get("ok")
                tier = event.data.get("tier", "?")
                mark = "ok" if ok else "denied/failed"
                print(f"  [tool result: {event.text.splitlines()[0][:160]}...] ({mark}, {tier})", flush=True)
            elif event.kind == "status":
                print(f"  [{event.text}]", flush=True)
            elif event.kind == "error":
                print(f"\n  [error] {event.text}", flush=True)

        started = time.perf_counter()
        try:
            result = self.brain.respond(text, on_event=on_event)
        except BrainError as exc:
            self.say(f"I could not reach the model. {exc}")
            self.activity.event("system", f"brain error: {exc}", ok=False)
            return True
        except KeyboardInterrupt:
            self.brain.interrupt()
            print("\n  [interrupted]")
            return True

        print()  # newline after streamed text
        if result.interrupted:
            self.say("Stopped.")
        elif result.error:
            self.say(f"That failed: {result.error}")
        elif not result.reply.strip():
            self.say("I have nothing to say to that.")

        self.activity.event(
            "assistant",
            result.reply,
            extra={
                "iterations": result.iterations,
                "duration_ms": round(result.duration_ms, 1),
                "tool_calls": [c["name"] for c in result.tool_calls],
                "provider": result.provider,
                "model": result.model,
            },
        )
        if self.cfg.get("logging.level") == "DEBUG":
            self.log.debug("turn took %.0f ms", (time.perf_counter() - started) * 1000)
        return True

    def _handle_command(self, text: str) -> bool:
        """Local slash commands: they never reach the model."""
        cmd, _, rest = text[1:].partition(" ")
        cmd = cmd.strip().lower()
        rest = rest.strip()

        if cmd in ("quit", "exit", "q", "stop"):
            return False
        if cmd in ("help", "?"):
            print(
                "\n  /help              this list\n"
                "  /reset             forget the conversation (memory files stay)\n"
                "  /prompt            print the system prompt that is being used\n"
                "  /tools             list registered tools and their tiers\n"
                "  /today             what GARVIS logged today\n"
                "  /personality NAME  standard|sassy|formal|hyped|focus|chill\n"
                "  /memory            show what memory GARVIS loaded\n"
                "  /quit              exit\n"
            )
            return True
        if cmd == "reset":
            self.brain.reset_history()
            print("  [history cleared]")
            return True
        if cmd == "prompt":
            print("\n" + self.brain.build_system_prompt() + "\n")
            return True
        if cmd == "tools":
            for category, tools in sorted(self.registry.by_category().items()):
                print(f"\n  {category}:")
                for tool in tools:
                    print(f"    {tool.name:<28} [{tool.tier}] {'(read-only)' if tool.readonly else ''}")
            print()
            return True
        if cmd == "today":
            print("\n" + self.activity.today_report() + "\n")
            return True
        if cmd == "personality":
            if not rest:
                print(f"  current: {self.brain.personality}")
            elif self.brain.set_personality(rest):
                print(f"  [personality -> {rest.lower()}]")
            else:
                print(f"  unknown mode '{rest}' (standard|sassy|formal|hyped|focus|chill)")
            return True
        if cmd == "memory":
            loaded = self.memory.load_all()
            for name, body in loaded.items():
                print(f"\n  --- {name} ({len(body)} chars) ---")
                print("  " + (body[:600].replace("\n", "\n  ") if body else "(empty)"))
            print()
            return True
        print(f"  unknown command '/{cmd}' - try /help")
        return True

    # -- loops -------------------------------------------------------------
    def run_text_loop(self) -> int:
        """Stage 1 loop: typed input, streamed replies. Voice replaces this later."""
        chunker = SentenceChunker(
            min_chars=int(self.cfg.get("voice_out.min_chunk_chars", 12)),
            max_chars=int(self.cfg.get("voice_out.max_chunk_chars", 220)),
        )
        print("Type a message. /help lists local commands.\n")
        while self.running:
            try:
                line = input(f"{self.cfg.user_name.lower()}> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            try:
                if not self.handle_line(line, chunker):
                    break
            except KeyboardInterrupt:
                self.brain.interrupt()
                print("\n  [interrupted - type /quit to exit]")
            except Exception as exc:  # keep the loop alive; log the traceback
                self.log.exception("unhandled error in text loop")
                self.activity.event("system", f"unhandled error: {exc}", ok=False)
                print(f"\n  [internal error: {exc}] (see logs/garvis.log)\n")
        return 0

    def shutdown(self) -> None:
        up = time.time() - self.started_at
        self.activity.event(
            "system",
            f"GARVIS shutting down after {up / 60:.1f} minutes",
            extra={"uptime_s": round(up, 1)},
        )
        self.log.info("shutdown after %.1f s", up)
        for service_name in ("browser", "tts", "state", "ui", "killswitch"):
            service = self.services.get(service_name)
            closer = getattr(service, "close", None) or getattr(service, "shutdown", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    self.log.debug("error closing %s", service_name, exc_info=True)


def _short(value: Any, limit: int = 120) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="garvis",
        description="GARVIS - local voice-controlled personal assistant.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python main.py\n"
            "  python main.py --ask 'summarise my project log'\n"
            "  python main.py --check\n"
            "  python main.py --today\n"
        ),
    )
    parser.add_argument("--config", default=None, help="path to config.yaml (default: ./config.yaml)")
    parser.add_argument("--ask", metavar="TEXT", default=None, help="ask one question, print the answer, exit")
    parser.add_argument("--check", action="store_true", help="validate config and environment, then exit")
    parser.add_argument("--today", action="store_true", help="print today's activity report and exit")
    parser.add_argument("--show-prompt", action="store_true", help="print the assembled system prompt and exit")
    parser.add_argument("--model", default=None, help="override brain.model for this run")
    parser.add_argument("--ollama", default=None, help="override brain.host, e.g. http://127.0.0.1:11999")
    parser.add_argument("--personality", default=None, help="start in this tone mode")
    parser.add_argument("--no-tools", action="store_true", help="chat only; the model cannot act")
    parser.add_argument("--log-level", default=None, help="DEBUG|INFO|WARNING|ERROR")
    parser.add_argument("--version", action="version", version=f"GARVIS {VERSION}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cfg = Config.load(args.config)
    except ConfigError as exc:
        print(f"Configuration error:\n  {exc}", file=sys.stderr)
        return 2

    if args.model:
        cfg.set("brain.model", args.model)
    if args.ollama:
        cfg.set("brain.host", args.ollama)
    if args.log_level:
        cfg.set("logging.level", args.log_level.upper())

    # --check and --today need no LLM, so they work on a bare machine.
    if args.today:
        configure_activity_logger(cfg)
        from core.logger import activity

        print(activity().today_report())
        return 0

    try:
        app = Garvis(cfg, args)
    except BrainError as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        return 2

    if args.check:
        print(describe(cfg))
        print()
        print("Tools registered:")
        for category, tools in sorted(app.registry.by_category().items()):
            names = ", ".join(f"{t.name}[{t.tier}]" for t in tools)
            print(f"  {category}: {names}")
        print()
        ok = app.preflight(quiet=True)
        print()
        print("RESULT:", "ready" if ok else "not ready (see above)")
        return 0 if ok else 1

    if args.show_prompt:
        print(app.brain.build_system_prompt())
        return 0

    if not app.preflight():
        return 1

    if args.ask:
        app.handle_line(args.ask)
        app.shutdown()
        return 0

    app.running = True
    app.activity.event(
        "system",
        f"GARVIS started (model={cfg.model}, personality={app.brain.personality}, "
        f"gate={'on' if app.gate else 'OFF'}, tools={len(app.registry)})",
        extra={"version": VERSION, "python": sys.version.split()[0], "pid": os.getpid()},
    )
    try:
        return app.run_text_loop()
    finally:
        app.running = False
        app.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
